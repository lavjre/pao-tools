"""Tests for Stage 6.9 attack helpers: poison_tcache_link.

The helper is the first *write-capable* glibc component, so the fake stack
gains a write path: ``AttackFakeDebugger.write_memory`` mutates the fake
gdb's byte store (so a later read sees the new bytes) and returns the exact
``DebuggerOperations.write_memory`` envelope shape (``original_hex`` /
``actual_hex`` / ``verified`` / ``revertible``).  ``dbg.writes`` records every
write call, which is how dry-run/no-op tests assert nothing was mutated.
"""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.attack_helpers import ATTACK_TCACHE_POISON_SCHEMA
from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import HeapTools, encode_safe_linked

from tests.test_glibc_arenas import (
    MAIN_ARENA_ADDR,
    MUSL_MAPS,
    ArenaFakeGdb,
    ArenaFakeInspection,
    make_arena_bytes,
)
from tests.test_glibc_regular_bins import _place_self_referential_bins
from tests.test_glibc_tcache import (
    TCACHE_ADDR,
    TcacheAwareFakeDebugger,
    _extend_maps_with_heap,
    _libc_version,
    make_tcache_bytes,
    place_tcache_chain,
)

HEAP_BASE = 0x555555559000

# Victim freed chunk lives inside the [heap] mapping, above the tcache struct
# span [HEAP_BASE+0x10, HEAP_BASE+0x290) so separate 8-byte reads never fall
# back into the tcache blob.
VICTIM_CHUNK = HEAP_BASE + 0x700
VICTIM_USER = VICTIM_CHUNK + 0x10

# A writable non-heap region standing in for a global target (e.g. near
# __free_hook).  The derived target chunk must stay 16-byte aligned.
TARGET_MAP = {
    "file": "[anon:global]",
    "start": "0x7ffff7000000",
    "end": "0x7ffff7001000",
    "permissions": "rw-p",
    "readable": True,
    "writable": True,
    "executable": False,
}
TARGET_USER = 0x7FFFF7000100
TARGET_CHUNK = TARGET_USER - 0x10


class AttackFakeDebugger(TcacheAwareFakeDebugger):
    """TcacheAwareFakeDebugger plus a write_memory that hits the gdb store."""

    def __init__(
        self, gdb: ArenaFakeGdb, *, fail_write: bool = False, **kw: Any
    ) -> None:
        super().__init__(**kw)
        self.gdb = gdb
        self.fail_write = fail_write
        self.writes: List[tuple] = []

    def _read_placed(self, addr: int, byte_count: int) -> Optional[bytes]:
        data = self.gdb._memory.get(addr)
        if data is not None:
            return data[:byte_count]
        for base, region in self.gdb._memory.items():
            if base <= addr < base + len(region):
                offset = addr - base
                end = offset + byte_count
                if end <= len(region):
                    return region[offset:end]
        return None

    def write_memory(
        self,
        address: str,
        data_hex: str,
        verify: bool = True,
        allow_unsafe: bool = False,
    ) -> Dict[str, Any]:
        self.writes.append((address, data_hex))
        if self.fail_write:
            return {
                "success": False,
                "command": "write_memory",
                "state": "stopped",
                "type": "MemoryWriteError",
                "error": "synthetic write failure",
            }
        addr = int(address, 16)
        new_bytes = bytes.fromhex(data_hex)
        original = self._read_placed(addr, len(new_bytes))
        original_hex = original.hex() if original is not None else None
        self.gdb._memory[addr] = new_bytes
        return {
            "success": True,
            "state": "stopped",
            "parsed": {
                "address": address,
                "byte_count": len(new_bytes),
                "data_hex": data_hex,
                "verify_requested": verify,
                "verified": True,
                "actual_hex": data_hex,
                "original_hex": original_hex,
                "revertible": original_hex is not None,
                "write_applied": True,
                "verification_uncertain": False,
                "mapping_validation": {
                    "checked": True,
                    "mapped": True,
                    "writable": True,
                    "override_used": allow_unsafe,
                },
            },
            "raw_responses": {"before": {}, "write": {}, "verify": {}},
        }


def build_attack(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
    fail_write: bool = False,
):
    gdb = ArenaFakeGdb(live=live)
    resolved_libc = libc if libc is not None else _libc_version("2.35")
    resolved_maps = (
        mappings if mappings is not None else _extend_maps_with_heap() + [TARGET_MAP]
    )
    inspection = ArenaFakeInspection(
        gdb,
        mappings=resolved_maps,
        libc=None if libc is None and mappings is not None else resolved_libc,
    )
    dbg = AttackFakeDebugger(gdb, fail_write=fail_write)
    heap = HeapTools(
        gdb,
        inspection,
        dbg,
        "sess",
        binary_path_provider=lambda: "/bin/chal",
    )
    return heap, gdb, dbg


def place_victim(
    gdb: ArenaFakeGdb,
    victim_user: int,
    fd_raw: int,
    *,
    chunk_size_field: int = 0x21,
) -> None:
    """Place a freed chunk's header + its tcache fd slot (at user offset 0)."""
    chunk_addr = victim_user - 0x10
    header = (0).to_bytes(8, "little") + chunk_size_field.to_bytes(8, "little")
    gdb.place(chunk_addr, header)
    gdb.place(victim_user, fd_raw.to_bytes(8, "little"))


def place_tcache_scene(
    gdb: ArenaFakeGdb,
    *,
    head_user: int = VICTIM_USER,
    victim_user: Optional[int] = None,
    fd_raw: Optional[int] = None,
    count: int = 1,
) -> None:
    """Empty arena + a tcache struct whose bin 0 chain starts at head_user.

    When ``victim_user``/``fd_raw`` are given the victim chunk is placed
    explicitly (for already-poisoned / custom-fd scenes); otherwise a single
    regular chain ending at NULL is staged via ``place_tcache_chain``.
    """
    _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
    gdb.place(
        TCACHE_ADDR,
        make_tcache_bytes(counts=[count] + [0] * 63, entries=[head_user] + [0] * 63),
    )
    if victim_user is not None:
        assert fd_raw is not None
        place_victim(gdb, victim_user, fd_raw)
    else:
        place_tcache_chain(gdb, [head_user], safe_linking=True)


# ---------------------------------------------------------------------------
# Pure helper: encode_safe_linked
# ---------------------------------------------------------------------------
class TestEncodeSafeLinked:
    def test_round_trips_with_decode(self):
        for plain in (0x0, 0x41, 0x41414141, 0x7FFFF70000F0):
            for position in (0x1000, VICTIM_USER, 0x12345678):
                encoded = encode_safe_linked(plain, position)
                decoded = encoded ^ (position >> 12)
                assert decoded == plain

    def test_known_vector(self):
        # PROTECT_PTR: encoded = plain ^ (position >> 12).
        assert encode_safe_linked(0x4141414141414141, 0x1000) == 0x4141414141414140


# ---------------------------------------------------------------------------
# Orchestration guards
# ---------------------------------------------------------------------------
class TestGuards:
    def test_requires_live_inferior(self):
        heap, _, _ = build_attack(live=False)
        with pytest.raises(LiveInferiorError):
            heap.poison_tcache_link(
                victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
            )

    def test_bad_victim_address_raises(self):
        heap, gdb, _ = build_attack()
        place_tcache_scene(gdb)
        with pytest.raises(ValueError):
            heap.poison_tcache_link(
                victim_address="not-a-number", target_address=hex(TARGET_USER)
            )

    def test_bad_address_kind_raises(self):
        heap, gdb, _ = build_attack()
        place_tcache_scene(gdb)
        with pytest.raises(ValueError):
            heap.poison_tcache_link(
                victim_address=hex(VICTIM_USER),
                target_address=hex(TARGET_USER),
                address_kind="bogus",
            )


# ---------------------------------------------------------------------------
# Dry run: compute everything, write nothing
# ---------------------------------------------------------------------------
class TestDryRun:
    def test_dry_run_computes_plan_and_writes_nothing(self):
        heap, gdb, dbg = build_attack()
        place_tcache_scene(gdb)

        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]

        assert parsed["schema"] == ATTACK_TCACHE_POISON_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "dry_run"
        assert parsed["victim_chunk_address"] == VICTIM_CHUNK
        assert parsed["victim_user_address"] == VICTIM_USER
        assert parsed["victim_bin_index"] == 0
        assert parsed["victim_chain_position"] == 0
        assert parsed["victim_size_class"] == 0x20
        assert parsed["safe_linking_enabled"] is True
        assert parsed["target_chunk_address"] == TARGET_CHUNK
        assert (
            parsed["encoded_fd_hex"]
            == encode_safe_linked(TARGET_CHUNK, VICTIM_USER).to_bytes(8, "little").hex()
        )
        assert parsed["original_fd_raw_hex"] == hex(VICTIM_USER >> 12)
        assert parsed["original_fd_decoded"] == 0
        assert parsed["snapshot_id"] is None
        assert parsed["verify"] is None
        assert parsed["revert"] is None
        assert parsed["target_mapping"]["file"] == TARGET_MAP["file"]
        # Plan anchors on the chunk header; address + chunk_offset resolves to
        # the fd slot (== write_target_address), so no offset double-count.
        assert parsed["write_plan"] == {
            "address": hex(VICTIM_CHUNK),
            "chunk_offset": 0x10,
            "byte_count": 8,
            "data_hex": encode_safe_linked(TARGET_CHUNK, VICTIM_USER)
            .to_bytes(8, "little")
            .hex(),
            "write_target_address": hex(VICTIM_USER),
        }
        assert parsed["follow_up"]["target_returned_on_malloc"] == 2
        assert dbg.writes == []


# ---------------------------------------------------------------------------
# Apply: snapshot -> write -> verify
# ---------------------------------------------------------------------------
class TestApply:
    def test_apply_writes_verifies_and_snapshots(self):
        heap, gdb, dbg = build_attack()
        place_tcache_scene(gdb)

        result = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK),
            target_address=hex(TARGET_USER),
            dry_run=False,
        )
        parsed = result["parsed"]
        assert parsed["verdict"] == "applied"

        expected = encode_safe_linked(TARGET_CHUNK, VICTIM_USER)
        assert dbg.writes == [(hex(VICTIM_USER), expected.to_bytes(8, "little").hex())]
        # The store now holds the poison; a later malloc would read it back.
        assert gdb._memory[VICTIM_USER] == expected.to_bytes(8, "little")

        assert parsed["snapshot_id"] is not None
        assert parsed["verify"]["writer_verified"] is True
        assert parsed["verify"]["decoded_after"] == TARGET_CHUNK
        assert parsed["verify"]["semantic_match"] is True
        assert parsed["revert"]["revertible"] is True
        assert (
            parsed["revert"]["original_hex"]
            == (VICTIM_USER >> 12).to_bytes(8, "little").hex()
        )
        assert parsed["revert"]["address"] == hex(VICTIM_USER)

    def test_snapshot_before_false_skips_snapshot(self):
        heap, gdb, dbg = build_attack()
        place_tcache_scene(gdb)
        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK),
            target_address=hex(TARGET_USER),
            dry_run=False,
            snapshot_before=False,
        )["parsed"]
        assert parsed["verdict"] == "applied"
        assert parsed["snapshot_id"] is None

    def test_already_applied_is_noop(self):
        heap, gdb, dbg = build_attack()
        expected = encode_safe_linked(TARGET_CHUNK, VICTIM_USER)
        place_tcache_scene(gdb, victim_user=VICTIM_USER, fd_raw=expected)

        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK),
            target_address=hex(TARGET_USER),
            dry_run=False,
        )["parsed"]
        assert parsed["verdict"] == "already_applied"
        assert parsed["snapshot_id"] is None
        assert dbg.writes == []

    def test_write_failure_reported(self):
        heap, gdb, dbg = build_attack(fail_write=True)
        place_tcache_scene(gdb)
        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK),
            target_address=hex(TARGET_USER),
            dry_run=False,
        )["parsed"]
        assert parsed["verdict"] == "write_failed"
        assert "synthetic write failure" in parsed["reason"]
        # Snapshot still captured before the failed write.
        assert parsed["snapshot_id"] is not None


# ---------------------------------------------------------------------------
# Location / family edge cases
# ---------------------------------------------------------------------------
class TestLocateVictim:
    def test_victim_not_in_tcache(self):
        heap, gdb, _ = build_attack()
        place_tcache_scene(gdb, head_user=VICTIM_USER + 0x100, count=0)

        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]
        assert parsed["verdict"] == "victim_not_in_tcache"
        assert parsed["supported"] is False
        assert parsed["scanned_bins"] == 64

    def test_chain_position_shapes_follow_up(self):
        heap, gdb, _ = build_attack()
        head_user = VICTIM_USER - 0x100
        place_tcache_scene(gdb, head_user=head_user, count=2)
        place_tcache_chain(gdb, [head_user, VICTIM_USER], safe_linking=True)

        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]
        assert parsed["verdict"] == "dry_run"
        assert parsed["victim_chain_position"] == 1
        assert parsed["victim_bin_index"] == 0
        assert parsed["follow_up"]["victim_returned_on_malloc"] == 2
        assert parsed["follow_up"]["target_returned_on_malloc"] == 3


class TestFamilyAndVersion:
    def test_musl_target_returns_unsupported(self):
        heap, _, _ = build_attack(mappings=MUSL_MAPS, libc=None)
        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]
        assert parsed["schema"] == ATTACK_TCACHE_POISON_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert "musl" in parsed["reason"]

    def test_pre_232_uses_plain_target(self):
        heap, gdb, _ = build_attack(libc=_libc_version("2.27"))
        place_tcache_scene(gdb)
        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]
        assert parsed["safe_linking_enabled"] is False
        assert parsed["encoded_fd_hex"] == TARGET_CHUNK.to_bytes(8, "little").hex()

    def test_unknown_safe_linking_refuses_write(self):
        heap, gdb, dbg = build_attack(libc=_libc_version(None))
        place_tcache_scene(gdb)
        parsed = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )["parsed"]
        assert parsed["schema"] == ATTACK_TCACHE_POISON_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unresolvable"
        assert "safe-linking" in parsed["reason"]
        assert dbg.writes == []


# ---------------------------------------------------------------------------
# Advisory target warnings (never fatal)
# ---------------------------------------------------------------------------
class TestWarnings:
    def test_target_outside_any_mapping_warns(self):
        heap, gdb, _ = build_attack(
            mappings=_extend_maps_with_heap(),  # no writable global region
            libc=_libc_version("2.35"),
        )
        place_tcache_scene(gdb)
        result = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(TARGET_USER)
        )
        assert result["parsed"]["verdict"] == "dry_run"
        assert result["parsed"]["target_mapping"] is None
        warnings = result.get("warnings") or []
        assert any("not inside any mapping" in w for w in warnings)

    def test_misaligned_target_chunk_warns(self):
        heap, gdb, _ = build_attack()
        place_tcache_scene(gdb)
        misaligned_user = TARGET_USER + 0x8  # derived chunk unaligned
        result = heap.poison_tcache_link(
            victim_address=hex(VICTIM_CHUNK), target_address=hex(misaligned_user)
        )
        assert result["parsed"]["verdict"] == "dry_run"
        warnings = result.get("warnings") or []
        assert any("not 16-byte aligned" in w for w in warnings)
