"""Tests for musl mallocng typed inspection (real 1.2.5 layouts)."""

from typing import Any, Dict, List, Optional

from paomcp.tools.backends.musl import MUSL_GROUPS_SCHEMA, MuslAnalyzer

CTX = 0x7F0000001000
AREA = 0x7F0000004000
GROUP = 0x7F0000005000
SECRET = 0x123456789ABCDEF0
AREA_HEAD_OFF = 64
META_SIZE = 40
MEM_OFF = 16
CLASSES = 0x7F0000002000
SIZE_CLASSES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    12,
    15,
    18,
    20,
    25,
    31,
    36,
    42,
    50,
    63,
    72,
    84,
    102,
    127,
    146,
    170,
    204,
    255,
    292,
    340,
    409,
    511,
    584,
    682,
    818,
    1023,
    1169,
    1364,
    1637,
    2047,
    2340,
    2730,
    3276,
    4095,
    4680,
    5460,
    6552,
    8191,
]


def _u64(value: int) -> bytes:
    return value.to_bytes(8, "little")


def _u16(value: int) -> bytes:
    return value.to_bytes(2, "little")


def _meta_bytes(
    *,
    mem: int,
    avail: int = 0,
    freed: int = 0,
    last_idx: int = 3,
    sizeclass: int = 1,
    maplen: int = 0,
) -> bytes:
    bits = last_idx | (1 << 5) | (sizeclass << 6) | (maplen << 12)
    return (
        _u64(0)
        + _u64(0)
        + _u64(mem)
        + avail.to_bytes(4, "little")
        + freed.to_bytes(4, "little")
        + bits.to_bytes(8, "little")
    )


class _FakeGdb:
    def __init__(self) -> None:
        self._memory: Dict[int, bytes] = {}

    def place(self, address: int, data: bytes) -> None:
        self._memory[address] = data

    def get_state(self) -> str:
        return "stopped"

    def get_last_stop_record(self) -> Dict[str, Any]:
        return {"is_live": True, "reason": "breakpoint-hit"}

    def get_inferior_pid(self) -> int:
        return 4242

    def read_memory_bytes(self, address: str, num_bytes: int) -> Dict[str, Any]:
        addr = int(address, 16)
        data = self._memory.get(addr)
        if data is not None:
            take = data[:num_bytes]
        else:
            take = None
            for base, region in self._memory.items():
                if base <= addr < base + len(region) and addr + num_bytes <= base + len(
                    region
                ):
                    take = region[addr - base : addr - base + num_bytes]
                    break
        if take is None:
            return {"success": False, "state": "stopped", "responses": []}
        return {
            "success": True,
            "state": "stopped",
            "responses": [
                {"type": "memory", "payload": {"memory": [{"contents": take.hex()}]}}
            ],
        }


class _FakeDebugger:
    def __init__(self, expressions: Dict[str, str]) -> None:
        self._expressions = expressions

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        if expression not in self._expressions:
            return {"success": False, "parsed": {}}
        return {
            "success": True,
            "parsed": {
                "expression": expression,
                "value": self._expressions[expression],
            },
        }


class _FakeInspection:
    def memory_maps(self) -> Dict[str, Any]:
        return {"success": True, "parsed": {"mappings": []}}


class _FakeDiscovery:
    def __init__(self, family: str = "musl") -> None:
        self._family = family

    def collect_evidence(self, cancellation_check=None):
        return [], None, {"mappings": True, "libcinfo": False}

    def classify(self, mappings, libcinfo):
        return {
            "family": self._family,
            "version": "1.2.5",
            "confidence": "likely",
            "evidence": [
                {
                    "source": "mapping",
                    "family": self._family,
                    "file": "/lib/ld-musl-x86_64.so.1",
                }
            ],
        }

    def target_profile(self, mappings):
        return {"pointer_bits": 64, "byteorder": "little"}


def _expressions() -> Dict[str, str]:
    return {
        "(void*)&__malloc_context": f"(void *) 0x{CTX:x}",
        "(unsigned long)__malloc_context.secret": str(SECRET),
        "(unsigned long)((char*)&((struct malloc_context*)0)->meta_area_head-(char*)0)": str(
            AREA_HEAD_OFF
        ),
        "(unsigned long)sizeof(struct meta)": str(META_SIZE),
        "(unsigned long)((char*)&((struct meta*)0)->mem-(char*)0)": str(MEM_OFF),
        "(void*)&__malloc_size_classes": f"(void *) 0x{CLASSES:x}",
    }


def _build(family: str = "musl"):
    gdb = _FakeGdb()
    analyzer = MuslAnalyzer(
        gdb,
        _FakeInspection(),
        _FakeDebugger(_expressions()),
        "sess",
        discovery=_FakeDiscovery(family),
    )
    return analyzer, gdb


def _place_ctx(gdb: _FakeGdb, *, area_head: int = AREA) -> None:
    blob = bytearray(AREA_HEAD_OFF + 8)
    blob[AREA_HEAD_OFF : AREA_HEAD_OFF + 8] = _u64(area_head)
    gdb.place(CTX, bytes(blob))


def _place_area(gdb: _FakeGdb, metas: List[bytes], *, area: int = AREA) -> None:
    blob = _u64(SECRET) + _u64(0) + len(metas).to_bytes(4, "little") + b"\x00" * 4
    for meta in metas:
        blob += meta
    gdb.place(area, blob)


def _place_group(gdb: _FakeGdb, *, group: int = GROUP, meta: int = 0) -> None:
    gdb.place(group, _u64(meta) + b"\x00" * 8)


def _chunk_addrs(*, group: int = GROUP, stride: int = 32, idx: int = 1):
    # musl records offset from the slot storage, not the group base:
    # *(u16*)(p-2) == (p - (group+16)) / 16 (see enframe in malloc.c).
    storage = group + 16
    user = storage + stride * idx
    offset = (user - storage) // 16
    return user, offset


def _place_chunk(
    gdb: _FakeGdb, *, idx: int = 1, reserved: int = 0, wide: bool = False
) -> int:
    user, offset = _chunk_addrs(idx=idx)
    header = bytearray(8)
    if wide:
        header[0:4] = (0x10000).to_bytes(4, "little")
        header[4] = 1
        header[5] = idx | (reserved << 5)
        header[6:8] = _u16(offset)
    else:
        header[5] = idx | (reserved << 5)
        header[6:8] = _u16(offset)
    gdb.place(user - 8, bytes(header))
    return user


def _standard_layout(gdb: _FakeGdb, *, freed_mask: int = 0):
    """One area, one group (4 slots, sc=1), chunk idx1 in use."""
    meta_addr = AREA + 24
    _place_ctx(gdb)
    _place_area(gdb, [_meta_bytes(mem=GROUP, freed=freed_mask)])
    _place_group(gdb, meta=meta_addr)
    classes = b"".join(_u16(v) for v in SIZE_CLASSES)
    gdb.place(CLASSES, classes)
    return _place_chunk(gdb)


class TestInspectMuslChunk:
    def test_valid_in_use_chunk(self):
        analyzer, gdb = _build()
        user = _standard_layout(gdb)
        result = analyzer.inspect_musl_chunk(address=hex(user))
        assert result["success"] is True
        parsed = result["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["state"] == "in_use"
        assert parsed["slot_index"] == 1
        assert parsed["group_base"] == GROUP
        assert parsed["sizeclass"] == 1
        assert parsed["stride"] == 32
        assert parsed["nominal_size"] == 28
        assert all(c["passed"] for c in parsed["checks"])

    def test_freed_chunk_state(self):
        analyzer, gdb = _build()
        user = _standard_layout(gdb, freed_mask=0b0010)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["state"] == "freed"

    def test_meta_mem_mismatch_is_corrupted(self):
        analyzer, gdb = _build()
        _place_ctx(gdb)
        _place_area(gdb, [_meta_bytes(mem=0xDEAD0000)])
        _place_group(gdb, meta=AREA + 24)
        gdb.place(CLASSES, b"".join(_u16(v) for v in SIZE_CLASSES))
        user = _place_chunk(gdb)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert parsed["corruption"] == "meta->mem mismatch"

    def test_misaligned_pointer_rejected(self):
        analyzer, _ = _build()
        parsed = analyzer.inspect_musl_chunk(address="0x1001")["parsed"]
        assert parsed["verdict"] == "corrupted"

    def test_unsupported_on_glibc(self):
        analyzer, _ = _build(family="glibc")
        parsed = analyzer.inspect_musl_chunk(address="0x1000")["parsed"]
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"

    def test_unresolvable_without_symbols(self):
        gdb = _FakeGdb()
        analyzer = MuslAnalyzer(
            gdb,
            _FakeInspection(),
            _FakeDebugger({}),
            "sess",
            discovery=_FakeDiscovery("musl"),
        )
        parsed = analyzer.inspect_musl_chunk(address="0x1000")["parsed"]
        assert parsed["verdict"] == "unresolvable"


class TestListMuslGroups:
    def test_lists_groups_with_layout(self):
        analyzer, gdb = _build()
        _place_ctx(gdb)
        meta_addr = AREA + 24
        _place_area(
            gdb,
            [
                _meta_bytes(mem=GROUP, last_idx=3, sizeclass=1),
                _meta_bytes(mem=0x6000, last_idx=1, sizeclass=5, freed=0b11),
            ],
        )
        _place_group(gdb, meta=meta_addr)
        gdb.place(CLASSES, b"".join(_u16(v) for v in SIZE_CLASSES))
        parsed = analyzer.list_musl_groups()["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["group_count"] == 2
        assert parsed["area_count"] == 1
        first, second = parsed["groups"]
        assert first["group_base"] == GROUP
        assert first["sizeclass"] == 1
        assert second["freed_mask"] == 0b11

    def test_secret_mismatch_is_corrupted(self):
        analyzer, gdb = _build()
        _place_ctx(gdb)
        blob = _u64(0xBAD) + _u64(0) + (1).to_bytes(4, "little") + b"\x00" * 4
        blob += _meta_bytes(mem=GROUP)
        gdb.place(AREA, blob)
        parsed = analyzer.list_musl_groups()["parsed"]
        assert parsed["verdict"] == "corrupted"

    def test_caps_are_validated(self):
        analyzer, _ = _build()
        import pytest

        with pytest.raises(ValueError, match="max_groups"):
            analyzer.list_musl_groups(max_groups=0)


class TestMuslLayoutCache:
    def test_second_inspect_reuses_cached_layout(self):
        analyzer, gdb = _build()
        user = _standard_layout(gdb)
        first = analyzer.inspect_musl_chunk(address=hex(user))
        assert first["parsed"]["verdict"] == "valid"
        assert analyzer._layout_cache is not None
        # Break every layout expression except the volatile re-reads: a
        # fresh resolve() is now impossible, so valid output proves the hit.
        keep = {
            "(void*)&__malloc_context",
            "(unsigned long)__malloc_context.secret",
            "(void*)&__malloc_size_classes",
        }
        expressions = analyzer.debugger._expressions
        for key in [k for k in expressions if k not in keep]:
            del expressions[key]
        second = analyzer.inspect_musl_chunk(address=hex(user))
        assert second["parsed"]["verdict"] == "valid"
        assert second["parsed"]["state"] == "in_use"
