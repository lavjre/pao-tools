"""Tests for the libmalloc analyzer (malloc_zone introspection, Stage 6.11).

These build a synthetic ``malloc_zone_t`` + ``malloc_introspection_t`` pair in
the fake gdb's memory, seed the ``ArenaFakeDebugger`` with the inferior-call
expressions (``malloc_default_zone`` / ``_malloc_zones`` / ``malloc_size`` /
``malloc_good_size`` / per-zone ``size``), and drive the analyzer both directly
(zone-native surface) and through the ``HeapTools`` facade (protocol routing).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import HeapTools
from paomcp.tools.backends.libmalloc import (
    LibmallocAnalyzer,
    LIBMALLOC_POINTER_SCHEMA,
    LIBMALLOC_ZONE_SCHEMA,
    LIBMALLOC_ZONES_SCHEMA,
)

from tests.test_glibc_arenas import (
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    GLIBC_MAPS,
)
from tests.test_macho_parser import lc_main, load_dylib, make_macho_64

# ---------------------------------------------------------------------------
# libmalloc world constants
# ---------------------------------------------------------------------------
DEFAULT_ZONE = 0x7FF000000000
ZONE2 = 0x7FF000001000
INTROSPECT_ADDR = 0x7FF000010000
NAME_ADDR = 0x7FF000020000
NAME2_ADDR = 0x7FF000021000
REGISTRY_BASE = 0x7FF000030000
SIZE_FN = 0x1000
MALLOC_FN = 0x1008
STATS_FN = 0x2000
ENUMERATOR_FN = 0x2008
PTR = 0x600000001234

LIBMALLOC_MAPS = [
    {
        "file": "/usr/lib/system/libsystem_malloc.dylib",
        "start": "0x7ff000000000",
        "end": "0x7ff001000000",
        "permissions": "r-xp",
        "readable": True,
        "writable": False,
        "executable": True,
    },
    {
        "file": "/usr/lib/libSystem.B.dylib",
        "start": "0x7ff001000000",
        "end": "0x7ff002000000",
        "permissions": "r--p",
        "readable": True,
        "writable": False,
        "executable": False,
    },
]

_ZONE_FIELD_OFFSETS = {
    "size": 0x10,
    "malloc": 0x18,
    "calloc": 0x20,
    "valloc": 0x28,
    "free": 0x30,
    "realloc": 0x38,
    "destroy": 0x40,
    "zone_name": 0x48,
    "introspect": 0x50,
    "batch_malloc": 0x68,
    "batch_free": 0x70,
    "memalign": 0x80,
    "free_definite_size": 0x88,
    "pressure_relief": 0x90,
}

_INTROSPECT_OFFSETS = {
    "enumerator": 0x00,
    "good_size": 0x08,
    "check": 0x10,
    "print": 0x18,
    "log": 0x20,
    "force_lock": 0x28,
    "force_unlock": 0x30,
    "statistics": 0x38,
    "zone_locked": 0x40,
    "enable_disable": 0x48,
    "logger": 0x50,
    "reinit_lock": 0x58,
    "reset_statistics": 0x60,
    "print_task": 0x68,
    "statistics_task": 0x70,
}


def put(buf: bytearray, offset: int, value: Optional[int]) -> None:
    buf[offset : offset + 8] = (value or 0).to_bytes(8, "little")


def place_zone(
    gdb: ArenaFakeGdb,
    address: int,
    *,
    name_ptr: Optional[int] = NAME_ADDR,
    introspect_ptr: Optional[int] = INTROSPECT_ADDR,
) -> None:
    buf = bytearray(0x98)
    for field, offset in _ZONE_FIELD_OFFSETS.items():
        if field == "zone_name":
            put(buf, offset, name_ptr)
        elif field == "introspect":
            put(buf, offset, introspect_ptr)
        elif field in ("size", "malloc"):
            put(buf, offset, SIZE_FN if field == "size" else MALLOC_FN)
        else:
            put(buf, offset, 0)
    gdb.place(address, bytes(buf))


def place_introspect(
    gdb: ArenaFakeGdb,
    address: int = INTROSPECT_ADDR,
    *,
    enumerator: int = ENUMERATOR_FN,
    statistics: int = STATS_FN,
) -> None:
    buf = bytearray(0x78)
    for field, offset in _INTROSPECT_OFFSETS.items():
        if field == "enumerator":
            put(buf, offset, enumerator)
        elif field == "statistics":
            put(buf, offset, statistics)
        else:
            put(buf, offset, 0)
    gdb.place(address, bytes(buf))


def place_registry(
    gdb: ArenaFakeGdb, addrs: List[int], *, base: int = REGISTRY_BASE
) -> None:
    gdb.place(base, b"".join(a.to_bytes(8, "little") for a in addrs))


def default_expression_map(*, size_fn_result: Optional[int] = None) -> Dict[str, str]:
    mapping: Dict[str, str] = {
        "(void*)malloc_default_zone()": (
            f"(void *) 0x{DEFAULT_ZONE:x} <malloc_default_zone()>"
        ),
        "(void*)_malloc_zones": f"(void *) 0x{REGISTRY_BASE:x}",
        "(unsigned long)_num_zones": "2",
    }
    if size_fn_result is not None:
        mapping[
            f"((unsigned long (*)(void *, void *)) 0x{SIZE_FN:x})"
            f"(0x{DEFAULT_ZONE:x}, 0x{PTR:x})"
        ] = str(size_fn_result)
    for requested in (16, 32, 64, 128, 256, 512, 1024, 4096):
        mapping[f"(unsigned long)malloc_good_size({requested})"] = str(requested)
    return mapping


def build_libmalloc(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    expression_map: Optional[Dict[str, str]] = None,
    binary_path_provider=None,
) -> tuple:
    gdb = ArenaFakeGdb(live=live)
    inspection = ArenaFakeInspection(
        gdb, mappings=mappings if mappings is not None else LIBMALLOC_MAPS, libc=None
    )
    dbg = ArenaFakeDebugger(arena_address=None, struct_size=None)
    dbg.expression_map.update(expression_map or {})
    heap = HeapTools(
        gdb,
        inspection,
        dbg,
        "sess",
        binary_path_provider=binary_path_provider,
    )
    return heap, gdb, dbg


def place_standard_scene(gdb: ArenaFakeGdb) -> None:
    gdb.place(NAME_ADDR, b"default malloc zone\x00")
    gdb.place(NAME2_ADDR, b"zone2\x00")
    place_introspect(gdb)
    place_zone(gdb, DEFAULT_ZONE, name_ptr=NAME_ADDR)
    place_zone(gdb, ZONE2, name_ptr=NAME2_ADDR)
    place_registry(gdb, [DEFAULT_ZONE, ZONE2])


def make_macho_binary(tmp_path: Path) -> Path:
    image = make_macho_64(
        commands=load_dylib("/usr/lib/libSystem.B.dylib") + lc_main(entryoff=0x1000)
    )
    path = tmp_path / "chal"
    path.write_bytes(image)
    return path


# ---------------------------------------------------------------------------
class TestGuard:
    def test_requires_live_inferior(self):
        heap, _, _ = build_libmalloc(live=False)
        analyzer = LibmallocAnalyzer(heap)
        with pytest.raises(LiveInferiorError):
            analyzer.inspect_zones()

    def test_bad_pointer_address_raises(self):
        heap, _, _ = build_libmalloc()
        analyzer = LibmallocAnalyzer(heap)
        with pytest.raises(ValueError):
            analyzer.classify_pointer(address="not-a-number")


# ---------------------------------------------------------------------------
class TestZonesDiscovery:
    def test_zones_listed_from_registry_and_default(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zones()["parsed"]
        assert parsed["schema"] == LIBMALLOC_ZONES_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "listed"
        assert parsed["source"] == "registry"
        assert parsed["default_zone"] == DEFAULT_ZONE
        assert parsed["zone_count"] == 2
        zones = parsed["zones"]
        assert zones[0]["name"] == "default malloc zone"
        assert zones[1]["name"] == "zone2"
        assert zones[0]["function_table"]["malloc"] == MALLOC_FN
        assert zones[0]["introspect"]["statistics"] == STATS_FN
        assert zones[0]["introspect"]["enumerator"] == ENUMERATOR_FN
        assert zones[0]["address"] == DEFAULT_ZONE

    def test_default_appended_when_absent_from_registry(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        place_registry(gdb, [ZONE2, 0])  # registry misses the default zone
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zones()["parsed"]
        assert parsed["source"] == "registry+default"
        assert parsed["zone_count"] == 2
        assert parsed["zones"][1]["address"] == DEFAULT_ZONE

    def test_default_zone_fallback_when_registry_symbols_missing(self):
        mapping = default_expression_map()
        del mapping["(void*)_malloc_zones"]
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        result = analyzer.inspect_zones()
        parsed = result["parsed"]
        assert parsed["verdict"] == "listed"
        assert parsed["source"] == "default_zone"
        assert parsed["zone_count"] == 1
        assert parsed["zones"][0]["address"] == DEFAULT_ZONE
        assert any("symbols unavailable" in w for w in result["warnings"])

    def test_no_zone_resolvable_is_zone_unavailable(self):
        heap, _, _ = build_libmalloc()  # no expressions seeded at all
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zones()["parsed"]
        assert parsed["verdict"] == "zone_unavailable"
        assert parsed["supported"] is False

    def test_zone_struct_unreadable(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_registry(gdb, [DEFAULT_ZONE])  # registry present, struct not placed
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zones()["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["supported"] is False


# ---------------------------------------------------------------------------
class TestInspectZone:
    def test_default_zone_deep_with_size_classes(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zone()["parsed"]
        assert parsed["schema"] == LIBMALLOC_ZONE_SCHEMA
        assert parsed["verdict"] == "inspected"
        assert parsed["supported"] is True
        assert parsed["address"] == DEFAULT_ZONE
        assert parsed["zone"]["name"] == "default malloc zone"
        assert parsed["zone"]["introspect"]["statistics"] == STATS_FN
        assert parsed["size_classes"][0] == {"requested": 16, "good_size": 16}
        assert len(parsed["size_classes"]) == 8

    def test_explicit_non_default_zone_omits_probes_with_warning(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        result = analyzer.inspect_zone(zone_address=hex(ZONE2))
        parsed = result["parsed"]
        assert parsed["verdict"] == "inspected"
        assert parsed["address"] == ZONE2
        assert parsed["zone"]["name"] == "zone2"
        assert parsed["size_classes"] == []
        assert any("default zone" in w for w in result["warnings"])

    def test_zone_unreadable(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_registry(gdb, [DEFAULT_ZONE])  # no struct placed
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zone(zone_address=hex(DEFAULT_ZONE))["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["supported"] is False
        assert "0x7ff000000000" in parsed["reason"]

    def test_no_default_zone_unavailable(self):
        heap, _, _ = build_libmalloc()  # no expressions
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.inspect_zone()["parsed"]
        assert parsed["verdict"] == "zone_unavailable"


# ---------------------------------------------------------------------------
class TestClassifyPointer:
    def test_allocated_pointer_with_zone_hint(self):
        mapping = default_expression_map(size_fn_result=128)
        mapping["(unsigned long)malloc_size((void*)0x%x)" % PTR] = "512"
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.classify_pointer(address=hex(PTR))["parsed"]
        assert parsed["schema"] == LIBMALLOC_POINTER_SCHEMA
        assert parsed["verdict"] == "allocated"
        assert parsed["supported"] is True
        assert parsed["size"] == 512
        assert parsed["zone_hint"] == "default malloc zone"

    def test_not_allocated_pointer(self):
        mapping = default_expression_map()
        mapping["(unsigned long)malloc_size((void*)0x%x)" % PTR] = "0"
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.classify_pointer(address=hex(PTR))["parsed"]
        assert parsed["verdict"] == "not_allocated"
        assert parsed["supported"] is True
        assert parsed["size"] == 0

    def test_unresolvable_pointer(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.classify_pointer(address=hex(PTR))["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert parsed["supported"] is False
        assert "malloc_size" in parsed["reason"]


# ---------------------------------------------------------------------------
class TestProtocolRouting:
    def test_facade_list_arenas_routes_to_libmalloc(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)

        result = heap.list_arenas()
        parsed = result["parsed"]
        assert parsed["schema"].endswith("arena-list.v1")
        assert parsed["verdict"] == "listed"
        assert parsed["allocator"] == {
            "family": "libmalloc",
            "version": None,
            "confidence": "likely",
        }
        assert parsed["zones"][0]["name"] == "default malloc zone"

    def test_facade_inspect_arena_routes_to_libmalloc(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)

        parsed = heap.inspect_arena(arena_address=hex(ZONE2))["parsed"]
        assert parsed["verdict"] == "inspected"
        assert parsed["zone"]["name"] == "zone2"

    def test_facade_inspect_chunk_routes_to_libmalloc(self):
        mapping = default_expression_map()
        mapping["(unsigned long)malloc_size((void*)0x%x)" % PTR] = "256"
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)

        parsed = heap.inspect_chunk(address=hex(PTR))["parsed"]
        assert parsed["verdict"] == "allocated"
        assert parsed["size"] == 256

    def test_facade_validate_chunk_valid(self):
        mapping = default_expression_map()
        mapping["(unsigned long)malloc_size((void*)0x%x)" % PTR] = "256"
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)

        parsed = heap.validate_chunk(address=hex(PTR))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["supported"] is True

    def test_facade_validate_chunk_invalid(self):
        mapping = default_expression_map()
        mapping["(unsigned long)malloc_size((void*)0x%x)" % PTR] = "0"
        heap, gdb, _ = build_libmalloc(expression_map=mapping)
        place_standard_scene(gdb)

        parsed = heap.validate_chunk(address=hex(PTR))["parsed"]
        assert parsed["verdict"] == "invalid"
        assert parsed["supported"] is True

    def test_list_arenas_family_gate_unsupported(self):
        # A glibc-classified target must not list malloc zones.
        heap, _, _ = build_libmalloc(mappings=GLIBC_MAPS)
        analyzer = LibmallocAnalyzer(heap)

        parsed = analyzer.list_arenas()["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["supported"] is False
        assert "not classified as libmalloc" in parsed["reason"]


# ---------------------------------------------------------------------------
class TestUnsupportedMethods:
    @pytest.mark.parametrize(
        ("method", "kwargs"),
        [
            ("inspect_fastbins", {}),
            ("inspect_tcache", {}),
            ("inspect_bins", {}),
            ("classify_heap_corruption", {}),
            ("validate_free", {"address": hex(PTR)}),
            ("analyze_heap_technique", {}),
            ("detect_uaf", {}),
            ("capture_heap_snapshot", {}),
        ],
    )
    def test_unsupported_method(self, method: str, kwargs: Dict[str, Any]):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        parsed = getattr(analyzer, method)(**kwargs)["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert parsed["supported"] is False
        assert parsed["allocator"]["family"] == "libmalloc"
        assert parsed["reason"]


# ---------------------------------------------------------------------------
class TestMachoBinaryContext:
    def test_macho_binary_section_present(self, tmp_path: Path):
        binary = make_macho_binary(tmp_path)
        heap, gdb, _ = build_libmalloc(
            expression_map=default_expression_map(),
            binary_path_provider=lambda: str(binary),
        )
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        binary_section = analyzer.inspect_zones()["parsed"]["binary"]
        assert binary_section["is_macho"] is True
        assert binary_section["is_64"] is True
        assert binary_section["filetype_name"] == "MH_EXECUTE"
        assert binary_section["cpu_name"] == "x86_64"
        assert binary_section["entry"] == 0x1000
        assert binary_section["links_libsystem"] is True

    def test_non_macho_binary(self, tmp_path: Path):
        path = tmp_path / "notmacho"
        path.write_bytes(b"this is not a mach-o image" * 4)
        heap, gdb, _ = build_libmalloc(
            expression_map=default_expression_map(),
            binary_path_provider=lambda: str(path),
        )
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        binary_section = analyzer.inspect_zones()["parsed"]["binary"]
        assert binary_section["is_macho"] is False
        assert "reason" in binary_section

    def test_no_binary_provider(self):
        heap, gdb, _ = build_libmalloc(expression_map=default_expression_map())
        place_standard_scene(gdb)
        analyzer = LibmallocAnalyzer(heap)

        binary_section = analyzer.inspect_zones()["parsed"]["binary"]
        assert binary_section["is_macho"] is False
        assert binary_section["reason"] == "no binary path provider wired"
