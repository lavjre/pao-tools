"""Tests for typed heap allocator discovery."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    ALLOCATOR_CAPABILITIES_SCHEMA,
    ALLOCATOR_DETECTION_SCHEMA,
    SAFE_LINKING_MIN_VERSION,
    AllocatorDiscovery,
)


def mapping(file: str, start: str = "0x555555554000", end: str = "0x555555556000"):
    return {
        "file": file,
        "start": start,
        "end": end,
        "permissions": "r-xp",
        "readable": True,
        "writable": False,
        "executable": True,
    }


GLIBC_MAPS = [
    mapping("/bin/chal", "0x555555554000", "0x555555556000"),
    mapping(
        "/lib/x86_64-linux-gnu/libc.so.6",
        "0x7ffff7c00000",
        "0x7ffff7e00000",
    ),
]
MUSL_MAPS = [mapping("/lib/ld-musl-x86_64.so.1", "0x7f0000000000", "0x7f0000200000")]
JEMALLOC_MAPS = [
    mapping("/bin/chal"),
    mapping("/lib/x86_64-linux-gnu/libc.so.6", "0x7ffff7c00000", "0x7ffff7e00000"),
    mapping("/usr/lib/libjemalloc.so.2", "0x7ffff7000000", "0x7ffff7100000"),
]


# ---------------------------------------------------------------------------
# Pure classification core -- no debugger required.
# ---------------------------------------------------------------------------
class TestClassify:
    def test_glibc_confirmed_with_libc_version(self):
        result = AllocatorDiscovery.classify(
            GLIBC_MAPS, {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"}
        )
        assert result["family"] == "glibc"
        assert result["version"] == "2.35"
        assert result["version_source"] == "libcinfo"
        assert result["confidence"] == "confirmed"
        assert result["safe_linking"] == "enabled"
        assert any(item["source"] == "mapping" for item in result["evidence"])

    def test_glibc_likely_without_version(self):
        result = AllocatorDiscovery.classify(GLIBC_MAPS, None)
        assert result["family"] == "glibc"
        assert result["version"] is None
        assert result["confidence"] == "likely"
        # Version unknown => safe-linking cannot be asserted as fact.
        assert result["safe_linking"] == "unknown"

    def test_glibc_version_from_filename_when_no_libcinfo(self):
        maps = [mapping("/lib/x86_64-linux-gnu/libc-2.27.so")]
        result = AllocatorDiscovery.classify(maps, None)
        assert result["family"] == "glibc"
        assert result["version"] == "2.27"
        assert result["version_source"] == "mapping_filename"
        assert result["confidence"] == "confirmed"
        assert result["safe_linking"] == "disabled"  # 2.27 < 2.32

    def test_musl_detected_and_safe_linking_not_applicable(self):
        result = AllocatorDiscovery.classify(MUSL_MAPS, None)
        assert result["family"] == "musl"
        assert result["safe_linking"] == "not_applicable"
        assert result["confidence"] == "likely"

    def test_jemalloc_wins_over_glibc_mapping(self):
        result = AllocatorDiscovery.classify(JEMALLOC_MAPS, None)
        assert result["family"] == "jemalloc"
        assert "glibc" in result["candidate_families"]
        assert "jemalloc" in result["candidate_families"]
        assert result["confidence"] == "ambiguous"
        assert result["safe_linking"] == "not_applicable"

    def test_unknown_when_no_allocator_mapping(self):
        result = AllocatorDiscovery.classify([mapping("/bin/chal")], None)
        assert result["family"] == "unknown"
        assert result["confidence"] == "unknown"
        assert result["evidence"] == []

    def test_libcinfo_object_corroborates_when_no_mapping(self):
        result = AllocatorDiscovery.classify(
            [mapping("/bin/chal")],
            {"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.31"},
        )
        assert result["family"] == "glibc"
        assert result["version"] == "2.31"
        assert result["safe_linking"] == "disabled"
        assert any(item["source"] == "libcinfo" for item in result["evidence"])

    def test_safe_linking_boundary_matches_constant(self):
        below = (SAFE_LINKING_MIN_VERSION[0], SAFE_LINKING_MIN_VERSION[1] - 1)
        assert AllocatorDiscovery._safe_linking_status("glibc", below) == "disabled"
        assert (
            AllocatorDiscovery._safe_linking_status("glibc", SAFE_LINKING_MIN_VERSION)
            == "enabled"
        )
        assert AllocatorDiscovery._safe_linking_status("glibc", None) == "unknown"
        assert (
            AllocatorDiscovery._safe_linking_status("musl", (1, 2)) == "not_applicable"
        )

    def test_ignores_non_mapping_entries(self):
        result = AllocatorDiscovery.classify(
            ["not-a-dict", None, mapping("/lib/libc.so.6")], None  # type: ignore[list-item]
        )
        assert result["family"] == "glibc"


# ---------------------------------------------------------------------------
# Live-target orchestration.
# ---------------------------------------------------------------------------
class FakeGdb:
    def __init__(self, *, live: bool = True, mappings: Optional[List[Dict]] = None):
        self._live = live
        self._mappings = mappings if mappings is not None else GLIBC_MAPS
        self.state = "stopped" if live else "exited"

    def get_state(self):
        return self.state

    def get_inferior_pid(self):
        return 4242 if self._live else None

    def get_last_stop_record(self):
        if not self._live:
            return None
        return {"is_live": True, "identity": "1:1", "inferior_pid": 4242}

    def execute_command(self, command, timeout_sec=60.0):
        if command == "show endian":
            return {
                "success": True,
                "state": self.state,
                "responses": [
                    {
                        "type": "console",
                        "payload": "The target is assumed to be little endian\n",
                    }
                ],
            }
        return {"success": True, "state": self.state, "responses": []}


class FakeInspection:
    timeout_seconds = 60.0

    def __init__(self, gdb, *, mappings=None, libc=None, mappings_ok=True):
        self.gdb = gdb
        self._mappings = mappings if mappings is not None else GLIBC_MAPS
        self._libc = libc
        self._mappings_ok = mappings_ok

    def memory_maps(self):
        return {
            "success": self._mappings_ok,
            "parsed": {"count": len(self._mappings), "mappings": self._mappings},
        }

    def libc_info(self):
        if self._libc is None:
            return {"success": False, "parsed": {}}
        return {"success": True, "parsed": self._libc}

    def checksec(self, binary_path):
        return {
            "success": True,
            "parsed": {"arch": "amd64-64-little", "file": binary_path},
        }


class FakeDebugger:
    def __init__(self, *, pointer_bytes=8):
        self._pointer_bytes = pointer_bytes

    def evaluate_expression(self, expression):
        if expression == "sizeof(void*)":
            return {"success": True, "parsed": {"value": str(self._pointer_bytes)}}
        return {"success": False, "parsed": {}}


def build(*, live=True, mappings=None, libc=None, mappings_ok=True, binary="/bin/chal"):
    gdb = FakeGdb(live=live, mappings=mappings)
    inspection = FakeInspection(
        gdb, mappings=mappings, libc=libc, mappings_ok=mappings_ok
    )
    debugger = FakeDebugger()
    return AllocatorDiscovery(
        gdb,
        inspection,
        debugger,
        "sess",
        binary_path_provider=lambda: binary,
    )


class TestDetectAllocator:
    def test_requires_live_inferior(self):
        discovery = build(live=False)
        with pytest.raises(LiveInferiorError):
            discovery.detect_allocator()

    def test_glibc_full_profile(self):
        discovery = build(
            mappings=GLIBC_MAPS,
            libc={"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"},
        )
        result = discovery.detect_allocator()
        assert result["success"] is True
        parsed = result["parsed"]
        assert parsed["schema"] == ALLOCATOR_DETECTION_SCHEMA
        assert parsed["family"] == "glibc"
        assert parsed["version"] == "2.35"
        assert parsed["safe_linking"] == "enabled"
        profile = parsed["target_profile"]
        assert profile["pointer_bits"] == 64
        assert profile["pointer_bits_source"] == "runtime_sizeof"
        assert profile["byteorder"] == "little"
        assert profile["byteorder_source"] == "gdb_show_endian"
        assert profile["architecture"] == "amd64-64-little"
        assert profile["architecture_source"] == "checksec"
        assert parsed["component_status"] == {"mappings": True, "libcinfo": True}

    def test_unknown_allocator_warns(self):
        discovery = build(mappings=[mapping("/bin/chal")], libc=None)
        result = discovery.detect_allocator()
        assert result["parsed"]["family"] == "unknown"
        assert any("custom allocator" in w for w in result.get("warnings", []))

    def test_pointer_width_falls_back_to_mapping_width(self):
        # 32-bit target: no sizeof, low addresses only.
        maps = [mapping("/lib/libc.so.6", "0x8048000", "0x8050000")]
        discovery = build(mappings=maps, libc=None)
        discovery.debugger = FakeDebugger(pointer_bytes=0)  # sizeof unusable
        result = discovery.detect_allocator()
        profile = result["parsed"]["target_profile"]
        assert profile["pointer_bits"] == 32
        assert profile["pointer_bits_source"] == "mapping_width"


class TestAllocatorCapabilities:
    def test_requires_live_inferior(self):
        discovery = build(live=False)
        with pytest.raises(LiveInferiorError):
            discovery.get_allocator_capabilities()

    def test_glibc_supported_with_full_command_set(self):
        discovery = build(
            libc={"libc": "/lib/x86_64-linux-gnu/libc.so.6", "version": "2.35"}
        )
        commands = ["arena", "arenas", "bins", "heap", "malloc-chunk", "tcachebins"]
        result = discovery.get_allocator_capabilities(debugger_command_names=commands)
        parsed = result["parsed"]
        assert parsed["schema"] == ALLOCATOR_CAPABILITIES_SCHEMA
        assert parsed["detected_family"] == "glibc"
        assert parsed["overall_status"] == "supported"
        glibc = parsed["allocators"]["glibc"]
        assert glibc["detected"] is True
        assert glibc["runtime_commands"]["status"] == "supported"
        assert glibc["runtime_commands"]["missing"] == []
        assert glibc["typed_inspection_api"] == {
            "status": "supported",
            "available": True,
        }

    def test_typed_glibc_remains_supported_when_some_commands_are_missing(self):
        discovery = build()
        result = discovery.get_allocator_capabilities(
            debugger_command_names=["arena", "bins"]
        )
        parsed = result["parsed"]
        assert parsed["allocators"]["glibc"]["runtime_commands"]["status"] == "partial"
        assert (
            parsed["allocators"]["glibc"]["typed_inspection_api"]["available"] is True
        )
        assert parsed["overall_status"] == "supported"

    def test_typed_glibc_remains_supported_without_command_inventory(self):
        discovery = build()
        result = discovery.get_allocator_capabilities(debugger_command_names=[])
        parsed = result["parsed"]
        assert parsed["overall_status"] == "supported"
        assert parsed["command_inventory_size"] == 0
        assert any("unknown" in w for w in result.get("warnings", []))

    def test_musl_family_reports_unsupported_typed_api(self):
        discovery = build(mappings=MUSL_MAPS, libc=None)
        result = discovery.get_allocator_capabilities(
            debugger_command_names=["mallocng"]
        )
        parsed = result["parsed"]
        assert parsed["detected_family"] == "musl"
        musl = parsed["allocators"]["musl"]
        assert musl["runtime_commands"]["status"] == "supported"
        assert musl["typed_inspection_api"]["available"] is False
        assert musl["typed_inspection_api"]["status"] == "unavailable"
        assert parsed["overall_status"] == "partial"

    def test_unknown_allocator_is_unsupported(self):
        discovery = build(mappings=[mapping("/bin/chal")], libc=None)
        result = discovery.get_allocator_capabilities(
            debugger_command_names=["arena", "bins", "heap"]
        )
        assert result["parsed"]["detected_family"] == "unknown"
        assert result["parsed"]["overall_status"] == "unsupported"
