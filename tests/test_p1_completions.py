"""P1 completion tests: jemalloc phase2, scudo, attack pack, alloc trace."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError

from tests.test_glibc_arenas import (
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    mapping,
)
from tests.test_jemalloc_analyzer import (
    EXTENT_INFO,
    FIND_EXTENT_FOUND,
    HEAP_TWO_EXTENTS,
    build_jemalloc,
)


def _scudo_maps() -> List[Dict[str, Any]]:
    return [
        mapping("/bin/chal", "0x555555554000", "0x555555556000", writable=False),
        mapping(
            "/usr/lib/x86_64-linux-gnu/libclang_rt.scudo-x86_64.so",
            "0x7ffff7c00000",
            "0x7ffff7e00000",
        ),
        {
            "file": "[anon:scudo]",
            "start": "0x7ffff6800000",
            "end": "0x7ffff6900000",
            "permissions": "rw-p",
            "readable": True,
            "writable": True,
            "executable": False,
        },
    ]


def _build_scudo(gdb: Optional[ArenaFakeGdb] = None, live: bool = True):
    from paomcp.tools.backends.heap import HeapTools

    gdb = gdb or ArenaFakeGdb(live=live)
    inspection = ArenaFakeInspection(gdb, mappings=_scudo_maps(), libc=None)
    debugger = ArenaFakeDebugger()
    heap = HeapTools(
        gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    return heap


def _scudo_header(*, class_id: int = 5, state: int = 2) -> bytes:
    buf = bytearray(16)
    buf[0:8] = (0x123456789ABC).to_bytes(8, "little")
    buf[8:10] = int(class_id).to_bytes(2, "little")
    buf[10] = int(state)
    buf[11] = 0
    buf[12:16] = (0x40).to_bytes(4, "little")
    return bytes(buf)


class TestJemallocPhase2:
    def test_corruption_valid_on_two_extents(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        parsed = heap._analyzer().classify_heap_corruption()["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] in ("valid", "suspicious")
        assert parsed["extent_count"] == 2

    def test_corruption_detects_overlap(self):
        overlap = [
            "Jemalloc heap\n",
            "\n",
            "Allocated Address: 0x7ffff6801000\n",
            "Extent Address: 0x7ffff6800000\n",
            "Size: 0x2000\n",
            "Small class: True\n",
            "State: active\n",
            "\n",
            "Allocated Address: 0x7ffff6801000\n",
            "Extent Address: 0x7ffff6801000\n",
            "Size: 0x2000\n",
            "Small class: True\n",
            "State: active\n",
            "\n",
        ]
        heap, _ = build_jemalloc(commands={"jemalloc heap": overlap})
        parsed = heap._analyzer().classify_heap_corruption()["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert "extent_overlap" in parsed["failed_checks"]

    def test_uaf_clean_when_no_freed_extents(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        # HEAP_TWO_EXTENTS has one active + one inactive -> freed set non-empty,
        # but no pointers into it in empty memory -> clean.
        parsed = heap._analyzer().detect_uaf()["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] in ("clean", "suspicious")

    def test_snapshot_captured_and_listed(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        analyzer = heap._analyzer()
        snap = analyzer.capture_heap_snapshot(name="j-test")["parsed"]
        assert snap["verdict"] == "captured"
        assert snap["summary"]["extent_count"] == 2
        assert snap["snapshot_id"]
        listed = heap.list_heap_snapshots()["parsed"]
        ids = [s["snapshot_id"] for s in listed.get("snapshots", [])]
        assert snap["snapshot_id"] in ids


class TestScudoMinimal:
    def test_requires_live(self):
        heap = _build_scudo(live=False)
        with pytest.raises(LiveInferiorError):
            heap._analyzer().inspect_chunk(address="0x7ffff6800010")

    def test_inspect_allocated(self):
        gdb = ArenaFakeGdb(live=True)
        chunk = 0x7FFFF6801000
        gdb.place(chunk, _scudo_header(class_id=5, state=2))
        heap = _build_scudo(gdb)
        parsed = heap.inspect_chunk(address=hex(chunk + 16))["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] == "allocated"
        assert parsed["header"]["class_id"] == 5

    def test_validate_quarantined_is_invalid(self):
        gdb = ArenaFakeGdb(live=True)
        chunk = 0x7FFFF6802000
        gdb.place(chunk, _scudo_header(class_id=3, state=1))
        heap = _build_scudo(gdb)
        parsed = heap.validate_chunk(address=hex(chunk + 16))["parsed"]
        assert parsed["verdict"] == "invalid"

    def test_list_regions(self):
        heap = _build_scudo()
        parsed = heap.list_arenas()["parsed"]
        assert parsed["verdict"] == "listed"
        assert parsed["region_count"] >= 1

    def test_corruption_valid(self):
        heap = _build_scudo()
        parsed = heap._analyzer().classify_heap_corruption()["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] in ("valid", "suspicious", "corrupted")

    def test_snapshot_captured(self):
        heap = _build_scudo()
        parsed = heap._analyzer().capture_heap_snapshot(name="s-test")["parsed"]
        assert parsed["verdict"] == "captured"

    def test_non_scudo_is_unsupported(self):
        from paomcp.tools.backends.scudo import ScudoAnalyzer

        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        analyzer = ScudoAnalyzer(heap)
        parsed = analyzer.inspect_chunk(address="0x7ffff6801000")["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["supported"] is False


class TestAttackPack:
    def test_fastbin_non_glibc_unsupported(self):
        heap, _ = build_jemalloc(commands={"jemalloc heap": HEAP_TWO_EXTENTS})
        parsed = heap.poison_fastbin_fd(
            victim_address="0x7ffff6800000", target_address="0x7ffff7000100"
        )["parsed"]
        assert parsed["verdict"] == "unsupported"

    def test_fastbin_dry_run_with_stubbed_bins(self):
        from tests.test_attack_helpers import TARGET_MAP, AttackFakeDebugger
        from tests.test_glibc_tcache import _extend_maps_with_heap, _libc_version

        gdb = ArenaFakeGdb(live=True)
        maps = _extend_maps_with_heap() + [TARGET_MAP]
        inspection = ArenaFakeInspection(gdb, mappings=maps, libc=_libc_version("2.35"))
        debugger = AttackFakeDebugger(gdb)
        from paomcp.tools.backends.heap import HeapTools

        heap = HeapTools(
            gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
        )
        victim_chunk = 0x555555559700
        victim_user = victim_chunk + 0x10
        target_user = 0x7FFFF7000100
        heap.bins.inspect_fastbins = lambda **kw: {  # type: ignore[method-assign]
            "success": True,
            "parsed": {
                "verdict": "parsed",
                "fastbins": [
                    {
                        "index": 5,
                        "chain": [
                            {
                                "chunk_address": victim_chunk,
                                "user_address": victim_user,
                                "index": 0,
                                "next_raw": 0,
                            }
                        ],
                    }
                ],
            },
        }
        parsed = heap.poison_fastbin_fd(
            victim_address=hex(victim_chunk), target_address=hex(target_user)
        )["parsed"]
        assert parsed["verdict"] == "dry_run"
        assert parsed["supported"] is True
        assert parsed["write_plan"]["write_target_address"] == hex(victim_user)
        assert debugger.writes == []

    def test_spirit_dry_run_and_bad_size(self):
        from tests.test_attack_helpers import TARGET_MAP, AttackFakeDebugger
        from tests.test_glibc_tcache import _extend_maps_with_heap, _libc_version

        gdb = ArenaFakeGdb(live=True)
        maps = _extend_maps_with_heap() + [TARGET_MAP]
        inspection = ArenaFakeInspection(gdb, mappings=maps, libc=_libc_version("2.35"))
        debugger = AttackFakeDebugger(gdb)
        from paomcp.tools.backends.heap import HeapTools

        heap = HeapTools(
            gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
        )
        fake = 0x7FFFF7000200
        parsed = heap.house_of_spirit_plan(
            fake_chunk_address=hex(fake), fake_size="0x40"
        )["parsed"]
        assert parsed["verdict"] == "dry_run"
        assert parsed["fake_chunk_address"] == fake
        assert len(parsed["playbook"]) == 3
        with pytest.raises(ValueError):
            heap.house_of_spirit_plan(fake_chunk_address=hex(fake), fake_size="0x13")


class TestAllocTrace:
    def _heap_with_trace_fakes(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(gdb, mappings=_scudo_maps(), libc=None)

        class TraceFakeDebugger(ArenaFakeDebugger):
            def __init__(self) -> None:
                self._bps: Dict[int, Dict[str, Any]] = {}
                self._next = 1

            def create_breakpoint(self, location: str, condition=None, temporary=False):
                num = self._next
                self._next += 1
                self._bps[num] = {"number": str(num), "exp": location, "hit_count": 0}
                return {
                    "success": True,
                    "command": "-break-insert",
                    "parsed": {"breakpoint": self._bps[num]},
                }

            def list_breakpoints(self):
                return {
                    "success": True,
                    "parsed": {
                        "count": len(self._bps),
                        "breakpoints": list(self._bps.values()),
                    },
                }

            def delete_breakpoint(self, number: int):
                ok = self._bps.pop(int(number), None) is not None
                return {"success": ok, "parsed": {"number": number, "deleted": ok}}

        debugger = TraceFakeDebugger()
        from paomcp.tools.backends.heap import HeapTools

        heap = HeapTools(
            gdb, inspection, debugger, "sess", binary_path_provider=lambda: "/bin/chal"
        )
        return heap, debugger

    def test_start_list_stop(self):
        heap, debugger = self._heap_with_trace_fakes()
        started = heap.start_alloc_trace()["parsed"]
        assert started["verdict"] == "started"
        tid = started["trace_id"]
        listed = heap.list_alloc_events(trace_id=tid)["parsed"]
        assert listed["verdict"] == "listed"
        assert listed["event_count"] == 0
        debugger._bps[1]["hit_count"] = 3
        listed2 = heap.list_alloc_events(trace_id=tid)["parsed"]
        assert listed2["event_count"] == 3
        stopped = heap.stop_alloc_trace(trace_id=tid)["parsed"]
        assert stopped["verdict"] == "stopped"
        assert stopped["event_count"] == 3

    def test_unknown_trace_raises(self):
        heap, _ = self._heap_with_trace_fakes()
        with pytest.raises(ValueError):
            heap.list_alloc_events(trace_id="nope")
