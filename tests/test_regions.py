"""Tests for the JIT / memory-region tracker.

These build synthetic vmmap fixtures into the fake inspection's
``memory_maps()`` and drive the tracker both directly (``JitRegionTracker``)
and through the ``HeapTools`` facade.
"""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import HeapTools
from paomcp.tools.backends.regions import (
    JIT_REGIONS_SCHEMA,
    MEMORY_REGION_SCHEMA,
    JitRegionTracker,
    _permission_flags,
)

from tests.test_glibc_arenas import (
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def vmap(file: str, start: str, end: str, permissions: str) -> Dict[str, Any]:
    perms = permissions.lower()
    return {
        "file": file,
        "start": start,
        "end": end,
        "permissions": perms,
        "readable": len(perms) > 0 and perms[0] == "r",
        "writable": len(perms) > 1 and perms[1] == "w",
        "executable": len(perms) > 2 and perms[2] == "x",
    }


JIT_MAPS = [
    vmap("/bin/chal", "0x400000", "0x401000", "r-xp"),
    vmap(
        "/lib/x86_64-linux-gnu/libc.so.6",
        "0x7f0000000000",
        "0x7f0000200000",
        "r--p",
    ),
    vmap("", "0x200000", "0x202000", "r-xp"),
    vmap("[anon]", "0x300000", "0x304000", "rwxp"),
    vmap("[vdso]", "0x7fff00000000", "0x7fff00001000", "r-xp"),
    vmap("[heap]", "0x555555600000", "0x555555601000", "rw-p"),
    vmap("[stack]", "0x7ffffffde000", "0x7ffffffff000", "rw-p"),
]

NO_CANDIDATE_MAPS = [
    vmap("/bin/chal", "0x400000", "0x401000", "r-xp"),
    vmap("/lib/x86_64-linux-gnu/libc.so.6", "0x7f0000000000", "0x7f0000200000", "r--p"),
    vmap("[heap]", "0x555555600000", "0x555555601000", "rw-p"),
    vmap("[stack]", "0x7ffffffde000", "0x7ffffffff000", "rw-p"),
]

MALFORMED_MAPS = [
    vmap("", "0x200000", "0x202000", "r-xp"),
    "not-a-dict",
    vmap("/bin/other", "zzz", "0x401000", "r-xp"),
    vmap("/bin/other", "0x400000", "0x401000", "rw"),
]


class BrokenFakeInspection(ArenaFakeInspection):
    """An inspection whose ``memory_maps()`` fails outright."""

    def memory_maps(self) -> Dict[str, Any]:
        return {"success": False, "parsed": {}}


def build_regions(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    inspection: Optional[ArenaFakeInspection] = None,
) -> tuple:
    gdb = ArenaFakeGdb(live=live)
    insp = inspection
    if insp is None:
        insp = ArenaFakeInspection(
            gdb,
            mappings=mappings if mappings is not None else JIT_MAPS,
            libc=None,
        )
    dbg = ArenaFakeDebugger(arena_address=None, struct_size=None)
    heap = HeapTools(gdb, insp, dbg, "sess")
    return heap, gdb


# ---------------------------------------------------------------------------
class TestGuard:
    def test_scan_requires_live_inferior(self):
        heap, _ = build_regions(live=False)
        with pytest.raises(LiveInferiorError):
            JitRegionTracker(heap).scan_jit_regions()

    def test_classify_requires_live_inferior(self):
        heap, _ = build_regions(live=False)
        with pytest.raises(LiveInferiorError):
            JitRegionTracker(heap).classify_memory_region(address=hex(0x300000))

    def test_malformed_address_raises_value_error(self):
        heap, _ = build_regions()
        with pytest.raises(ValueError):
            JitRegionTracker(heap).classify_memory_region(address="not-a-number")


# ---------------------------------------------------------------------------
class TestScanJitRegions:
    def test_lists_candidates_sorted_by_size_desc(self):
        heap, _ = build_regions()

        parsed = JitRegionTracker(heap).scan_jit_regions()["parsed"]
        assert parsed["schema"] == JIT_REGIONS_SCHEMA
        assert parsed["verdict"] == "listed"
        assert parsed["supported"] is True
        assert parsed["component_status"] == {"mappings": True}
        regions = parsed["regions"]
        assert len(regions) == 2
        assert [r["size"] for r in regions] == sorted(
            (r["size"] for r in regions), reverse=True
        )
        # [anon] rwxp (0x4000), anonymous "" r-xp (0x2000). [vdso] is
        # executable but kernel-mapped, never JIT spray space.
        assert regions[0]["permissions"] == "rwxp"
        assert regions[1]["anonymous"] is True
        assert regions[1]["permissions"] == "r-xp"
        assert parsed["summary"]["count"] == 2
        assert parsed["summary"]["total_bytes"] == 0x4000 + 0x2000
        assert parsed["summary"]["largest_region"] == regions[0]

    def test_vdso_is_not_a_jit_candidate(self):
        heap, _ = build_regions()
        tracker = JitRegionTracker(heap)

        regions, _raw, _warnings, _available = tracker._read_maps(None)
        vdso = next(r for r in regions if r["start"] == 0x7FFF00000000)
        assert vdso["executable"] is True
        assert vdso["jit_candidate"] is False

    def test_per_region_flags(self):
        heap, _ = build_regions()

        regions = JitRegionTracker(heap).scan_jit_regions()["parsed"]["regions"]
        by_perm = {r["permissions"]: r for r in regions}
        rwx = by_perm["rwxp"]
        assert rwx["anonymous"] is True
        assert rwx["executable"] is True
        assert rwx["writable"] is True
        assert rwx["jit_candidate"] is True
        assert rwx["rwx"] is True
        assert rwx["start"] == 0x300000
        assert rwx["end"] == 0x304000

    def test_file_backed_and_non_exec_anon_are_not_candidates(self):
        heap, _ = build_regions()
        tracker = JitRegionTracker(heap)

        scan = tracker.scan_jit_regions()["parsed"]
        for region in scan["regions"]:
            assert region["jit_candidate"] is True
        # The file-backed r-xp binary and the anonymous but non-executable
        # [heap]/[stack] must never surface as candidates.
        regions, _raw, _warnings, _available = tracker._read_maps(None)
        file_backed = next(r for r in regions if r["start"] == 0x400000)
        assert file_backed["anonymous"] is False
        assert file_backed["jit_candidate"] is False
        heap_region = next(r for r in regions if r["start"] == 0x555555600000)
        assert heap_region["anonymous"] is True
        assert heap_region["rwx"] is False
        assert heap_region["jit_candidate"] is False

    def test_none_found_when_no_exec_anon(self):
        heap, _ = build_regions(mappings=NO_CANDIDATE_MAPS)

        parsed = JitRegionTracker(heap).scan_jit_regions()["parsed"]
        assert parsed["verdict"] == "none_found"
        assert parsed["supported"] is True
        assert parsed["regions"] == []
        assert parsed["summary"]["count"] == 0
        assert parsed["summary"]["total_bytes"] == 0
        assert parsed["summary"]["largest_region"] is None

    def test_unavailable_when_backend_fails(self):
        heap, gdb = build_regions(inspection=BrokenFakeInspection(gdb=None))
        result = JitRegionTracker(heap).scan_jit_regions()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unavailable"
        assert parsed["supported"] is False
        assert parsed["regions"] == []
        assert "reason" in parsed

    def test_unavailable_when_zero_mappings(self):
        heap, _ = build_regions(mappings=[])

        parsed = JitRegionTracker(heap).scan_jit_regions()["parsed"]
        assert parsed["verdict"] == "unavailable"
        assert parsed["supported"] is False

    def test_skips_malformed_mappings_with_warning(self):
        heap, _ = build_regions(mappings=MALFORMED_MAPS)

        result = JitRegionTracker(heap).scan_jit_regions()
        parsed = result["parsed"]
        assert parsed["verdict"] == "listed"
        assert len(parsed["regions"]) == 1
        assert parsed["regions"][0]["permissions"] == "r-xp"
        assert any("noncanonical" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
class TestClassifyMemoryRegion:
    def test_mapped_returns_containing_region(self):
        heap, _ = build_regions()

        parsed = JitRegionTracker(heap).classify_memory_region(address=hex(0x300000))[
            "parsed"
        ]
        assert parsed["schema"] == MEMORY_REGION_SCHEMA
        assert parsed["verdict"] == "mapped"
        assert parsed["supported"] is True
        assert parsed["address"] == 0x300000
        region = parsed["region"]
        assert region["start"] == 0x300000
        assert region["end"] == 0x304000
        assert region["permissions"] == "rwxp"
        assert region["jit_candidate"] is True
        assert region["rwx"] is True

    def test_unmapped_for_gap_and_end_boundary(self):
        heap, _ = build_regions()
        tracker = JitRegionTracker(heap)

        gap = tracker.classify_memory_region(address=hex(0x500000))["parsed"]
        assert gap["verdict"] == "unmapped"
        assert gap["supported"] is True
        assert gap["region"] is None

        # Half-open [start, end): an address equal to a mapping's end is NOT
        # inside that mapping.
        end_boundary = tracker.classify_memory_region(address=hex(0x7FFF00001000))[
            "parsed"
        ]
        assert end_boundary["verdict"] == "unmapped"
        assert end_boundary["region"] is None

    def test_address_kind_echoed(self):
        heap, _ = build_regions()
        tracker = JitRegionTracker(heap)

        explicit = tracker.classify_memory_region(
            address=hex(0x300000), address_kind="stack"
        )["parsed"]
        assert explicit["address_kind"] == "stack"
        defaulted = tracker.classify_memory_region(address=hex(0x300000))["parsed"]
        assert defaulted["address_kind"] == "user"

    def test_unavailable_when_backend_fails(self):
        heap, _ = build_regions(inspection=BrokenFakeInspection(gdb=None))

        parsed = JitRegionTracker(heap).classify_memory_region(address=hex(0x300000))[
            "parsed"
        ]
        assert parsed["verdict"] == "unavailable"
        assert parsed["supported"] is False
        assert parsed["region"] is None

    def test_mapped_but_unparseable_is_not_unmapped(self):
        # M5: an address that IS inside a mapping whose entry cannot be
        # normalized (undecodable permission string) must not be reported as
        # "unmapped" -- that would change planning.  It gets its own honest
        # "unparseable" verdict.
        maps = [
            vmap("/bin/chal", "0x400000", "0x401000", "r-xp"),
            vmap("", "0x300000", "0x304000", "rw"),  # 2-char perms: undecodable
        ]
        heap, _ = build_regions(mappings=maps)
        tracker = JitRegionTracker(heap)

        result = tracker.classify_memory_region(address=hex(0x301000))
        parsed = result["parsed"]
        assert parsed["verdict"] == "unparseable"
        assert parsed["supported"] is True
        assert parsed["region"] is None
        assert any("noncanonical" in w for w in result["warnings"])

    def test_5_char_and_uppercase_permissions_decode(self):
        # 5-char perms (r-xps) decode r/w/x + private/shared without warning;
        # uppercase permission strings normalize to lowercase.  private/shared
        # are decoder output (region contract carries r/w/x only), so assert
        # both the emitted region and the raw decoder.
        flags, warn = _permission_flags("r-xps")
        assert warn is None
        assert flags == {
            "readable": True,
            "writable": False,
            "executable": True,
            "private": True,
            "shared": False,
        }
        upper_flags, upper_warn = _permission_flags("RWXP")
        assert upper_warn is None
        assert upper_flags["readable"] is True
        assert upper_flags["writable"] is True
        assert upper_flags["executable"] is True
        assert upper_flags["private"] is True

        maps = [
            vmap("/bin/chal", "0x400000", "0x401000", "r-xps"),
            {
                "file": "/lib/libx.so.6",
                "start": "0x7f0000000000",
                "end": "0x7f0000200000",
                "permissions": "RWXP",  # uppercase, as some vmmap impls emit
                "readable": True,
                "writable": True,
                "executable": True,
            },
        ]
        heap, _ = build_regions(mappings=maps)
        tracker = JitRegionTracker(heap)
        regions, _raw, warnings, _available = tracker._read_maps(None)
        assert warnings == []
        by_start = {r["start"]: r for r in regions}

        five = by_start[0x400000]
        assert five["permissions"] == "r-xps"
        assert five["writable"] is False
        assert five["executable"] is True

        upper = by_start[0x7F0000000000]
        assert upper["permissions"] == "rwxp"
        assert upper["writable"] is True
        assert upper["executable"] is True

    def test_non_string_file_field_handled(self):
        # A None file field is anonymous; a numeric file field is treated as a
        # file-like field (never anonymous) without crashing.
        maps = [
            {
                "file": None,
                "start": "0x300000",
                "end": "0x304000",
                "permissions": "rwxp",
                "readable": True,
                "writable": True,
                "executable": True,
            },
            {
                "file": 12345,
                "start": "0x500000",
                "end": "0x501000",
                "permissions": "r-xp",
                "readable": True,
                "writable": False,
                "executable": True,
            },
        ]
        heap, _ = build_regions(mappings=maps)
        tracker = JitRegionTracker(heap)
        regions, _raw, _warnings, _available = tracker._read_maps(None)
        by_start = {r["start"]: r for r in regions}

        assert by_start[0x300000]["anonymous"] is True
        assert by_start[0x500000]["anonymous"] is False


# ---------------------------------------------------------------------------
class TestFacadeRouting:
    def test_facade_scan_jit_regions_routes(self):
        heap, _ = build_regions()

        parsed = heap.scan_jit_regions()["parsed"]
        assert parsed["schema"] == JIT_REGIONS_SCHEMA
        assert parsed["verdict"] == "listed"
        assert parsed["summary"]["count"] == 2

    def test_facade_classify_memory_region_routes(self):
        heap, _ = build_regions()

        parsed = heap.classify_memory_region(address=hex(0x300000))["parsed"]
        assert parsed["verdict"] == "mapped"
        assert parsed["region"]["jit_candidate"] is True


class TestRegionWatch:
    def test_arm_check_unchanged_disarm(self):
        heap, gdb = build_regions()
        gdb.place(0x300000, b"\x90" * 256)
        armed = heap.arm_region_watch(address=hex(0x300000), length=256, label="jit")
        assert armed["parsed"]["verdict"] == "armed"
        watch_id = armed["parsed"]["watch_id"]
        checked = heap.check_region_watches()["parsed"]
        assert checked["watch_count"] == 1
        assert checked["changed_count"] == 0
        assert checked["watches"][0]["verdict"] == "unchanged"
        dropped = heap.disarm_region_watch(watch_id=watch_id)["parsed"]
        assert dropped["removed"] is True
        assert heap.check_region_watches()["parsed"]["watch_count"] == 0

    def test_write_shows_changed_ranges(self):
        heap, gdb = build_regions()
        blob = bytearray(b"\x90" * 256)
        gdb.place(0x300000, bytes(blob))
        heap.arm_region_watch(address=hex(0x300000), length=256)
        # The fake serves exact placements, so emulate a JIT write by
        # replacing the whole blob with two patched ranges.
        blob[10:14] = b"\xcc" * 4
        blob[100:102] = b"\xcc" * 2
        gdb.place(0x300000, bytes(blob))
        report = heap.check_region_watches()["parsed"]["watches"][0]
        assert report["verdict"] == "changed"
        assert report["changed"] is True
        assert report["changed_ranges"] == [[10, 14], [100, 102]]
        assert report["changed_bytes"] == 6

    def test_arm_unreadable_region(self):
        heap, _ = build_regions()
        parsed = heap.arm_region_watch(address=hex(0x9FFF0000), length=64)["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["watch_id"] is None

    def test_arm_validates_bounds(self):
        heap, _ = build_regions()
        with pytest.raises(ValueError, match="length"):
            heap.arm_region_watch(address=hex(0x300000), length=0)
