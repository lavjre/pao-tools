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

# 32-bit-addressable equivalents for the 32-bit fixtures.
CTX32 = 0x08001000
AREA32 = 0x08004000
GROUP32 = 0x08005000
CLASSES32 = 0x08002000
META_SIZE32 = 24
MEM_OFF32 = 8
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


def _uptr(value: int, pointer_bits: int = 64, byteorder: str = "little") -> bytes:
    return value.to_bytes(pointer_bits // 8, byteorder)


def _pack_meta_bits(
    *,
    last_idx: int,
    freeable: int,
    sizeclass: int,
    maplen: int,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> bytes:
    """Pack the struct meta bitfield the way the target compiler would.

    Little-endian targets place ``last_idx`` in the least-significant
    bits; big-endian targets place it in the most-significant bits.
    """
    word_bits = pointer_bits
    if byteorder == "big":
        word = (
            (last_idx << (word_bits - 5))
            | (freeable << (word_bits - 6))
            | (sizeclass << (word_bits - 12))
            | maplen
        )
    else:
        word = last_idx | (freeable << 5) | (sizeclass << 6) | (maplen << 12)
    return word.to_bytes(pointer_bits // 8, byteorder)


def _meta_bytes(
    *,
    mem: int,
    avail: int = 0,
    freed: int = 0,
    last_idx: int = 3,
    sizeclass: int = 1,
    maplen: int = 0,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> bytes:
    size_sz = pointer_bits // 8
    return (
        (0).to_bytes(size_sz, byteorder)
        + (0).to_bytes(size_sz, byteorder)
        + mem.to_bytes(size_sz, byteorder)
        + avail.to_bytes(4, byteorder)
        + freed.to_bytes(4, byteorder)
        + _pack_meta_bits(
            last_idx=last_idx,
            freeable=1,
            sizeclass=sizeclass,
            maplen=maplen,
            pointer_bits=pointer_bits,
            byteorder=byteorder,
        )
    )


def _meta_area_slots_off(pointer_bits: int) -> int:
    size_sz = pointer_bits // 8
    return (12 + size_sz + size_sz - 1) & ~(size_sz - 1)


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
    def __init__(
        self,
        family: str = "musl",
        *,
        pointer_bits: int = 64,
        byteorder: str = "little",
    ) -> None:
        self._family = family
        self._pointer_bits = pointer_bits
        self._byteorder = byteorder

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
        return {"pointer_bits": self._pointer_bits, "byteorder": self._byteorder}


def _expressions(
    *,
    ctx: int = CTX,
    classes: int = CLASSES,
    meta_size: int = META_SIZE,
    mem_off: int = MEM_OFF,
) -> Dict[str, str]:
    return {
        "(void*)&__malloc_context": f"(void *) 0x{ctx:x}",
        "(unsigned long)__malloc_context.secret": str(SECRET),
        "(unsigned long)((char*)&((struct malloc_context*)0)->meta_area_head-(char*)0)": str(
            AREA_HEAD_OFF
        ),
        "(unsigned long)sizeof(struct meta)": str(meta_size),
        "(unsigned long)((char*)&((struct meta*)0)->mem-(char*)0)": str(mem_off),
        "(void*)&__malloc_size_classes": f"(void *) 0x{classes:x}",
    }


def _build(
    family: str = "musl",
    *,
    pointer_bits: int = 64,
    byteorder: str = "little",
    meta_size: int = META_SIZE,
    mem_off: int = MEM_OFF,
    ctx: int = CTX,
    classes: int = CLASSES,
):
    gdb = _FakeGdb()
    analyzer = MuslAnalyzer(
        gdb,
        _FakeInspection(),
        _FakeDebugger(
            _expressions(ctx=ctx, classes=classes, meta_size=meta_size, mem_off=mem_off)
        ),
        "sess",
        discovery=_FakeDiscovery(
            family, pointer_bits=pointer_bits, byteorder=byteorder
        ),
    )
    return analyzer, gdb


def _place_ctx(
    gdb: _FakeGdb,
    *,
    ctx: int = CTX,
    area_head: int = AREA,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> None:
    size_sz = pointer_bits // 8
    blob = bytearray(AREA_HEAD_OFF + size_sz)
    blob[AREA_HEAD_OFF : AREA_HEAD_OFF + size_sz] = _uptr(
        area_head, pointer_bits, byteorder
    )
    gdb.place(ctx, bytes(blob))


def _place_area(
    gdb: _FakeGdb,
    metas: List[bytes],
    *,
    area: int = AREA,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> None:
    size_sz = pointer_bits // 8
    blob = (
        SECRET.to_bytes(8, byteorder)
        + (0).to_bytes(size_sz, byteorder)
        + len(metas).to_bytes(4, byteorder)
    )
    slots_off = _meta_area_slots_off(pointer_bits)
    blob += b"\x00" * (slots_off - len(blob))
    for meta in metas:
        blob += meta
    gdb.place(area, blob)


def _place_group(
    gdb: _FakeGdb,
    *,
    group: int = GROUP,
    meta: int = 0,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> None:
    gdb.place(
        group,
        _uptr(meta, pointer_bits, byteorder) + b"\x00" * (16 - pointer_bits // 8),
    )


def _chunk_addrs(*, group: int = GROUP, stride: int = 32, idx: int = 1):
    # musl records offset from the slot storage, not the group base:
    # *(u16*)(p-2) == (p - (group+16)) / 16 (see enframe in malloc.c).
    storage = group + 16
    user = storage + stride * idx
    offset = (user - storage) // 16
    return user, offset


def _place_chunk(
    gdb: _FakeGdb,
    *,
    group: int = GROUP,
    idx: int = 1,
    reserved: int = 0,
    wide: bool = False,
    stride: int = 32,
    pointer_bits: int = 64,
    byteorder: str = "little",
) -> int:
    user, offset = _chunk_addrs(group=group, idx=idx, stride=stride)
    header = bytearray(8)
    if wide:
        header[0:4] = (0x10000).to_bytes(4, byteorder)
        header[4] = 1
        header[5] = idx | (reserved << 5)
        header[6:8] = offset.to_bytes(2, byteorder)
    else:
        header[5] = idx | (reserved << 5)
        header[6:8] = offset.to_bytes(2, byteorder)
    gdb.place(user - 8, bytes(header))
    return user


def _standard_layout(
    gdb: _FakeGdb,
    *,
    freed_mask: int = 0,
    pointer_bits: int = 64,
    byteorder: str = "little",
    sizeclass: int = 1,
    ctx: int = CTX,
    area: int = AREA,
    group: int = GROUP,
    classes: int = CLASSES,
):
    """One area, one group (4 slots, sc=1), chunk idx1 in use."""
    meta_addr = area + _meta_area_slots_off(pointer_bits)
    _place_ctx(
        gdb,
        ctx=ctx,
        area_head=area,
        pointer_bits=pointer_bits,
        byteorder=byteorder,
    )
    _place_area(
        gdb,
        [
            _meta_bytes(
                mem=group,
                freed=freed_mask,
                sizeclass=sizeclass,
                pointer_bits=pointer_bits,
                byteorder=byteorder,
            )
        ],
        area=area,
        pointer_bits=pointer_bits,
        byteorder=byteorder,
    )
    _place_group(
        gdb,
        group=group,
        meta=meta_addr,
        pointer_bits=pointer_bits,
        byteorder=byteorder,
    )
    class_bytes = b"".join(v.to_bytes(2, byteorder) for v in SIZE_CLASSES)
    gdb.place(classes, class_bytes)
    return _place_chunk(
        gdb,
        group=group,
        pointer_bits=pointer_bits,
        byteorder=byteorder,
    )


def _place_ctx32(gdb: _FakeGdb, byteorder: str = "little") -> None:
    _place_ctx(gdb, ctx=CTX32, area_head=AREA32, pointer_bits=32, byteorder=byteorder)


def _place_group32(gdb: _FakeGdb, meta: int, byteorder: str = "little") -> None:
    _place_group(gdb, group=GROUP32, meta=meta, pointer_bits=32, byteorder=byteorder)


def _place_classes32(gdb: _FakeGdb, byteorder: str = "little") -> None:
    gdb.place(CLASSES32, b"".join(v.to_bytes(2, byteorder) for v in SIZE_CLASSES))


def _place_area32(gdb: _FakeGdb, metas: List[bytes], byteorder: str = "little") -> None:
    _place_area(gdb, metas, area=AREA32, pointer_bits=32, byteorder=byteorder)


def _place_chunk32(
    gdb: _FakeGdb,
    *,
    idx: int = 1,
    stride: int = 32,
    byteorder: str = "little",
) -> int:
    return _place_chunk(
        gdb,
        group=GROUP32,
        idx=idx,
        stride=stride,
        pointer_bits=32,
        byteorder=byteorder,
    )


def _standard_layout32(gdb: _FakeGdb, *, byteorder: str = "little"):
    return _standard_layout(
        gdb,
        pointer_bits=32,
        byteorder=byteorder,
        ctx=CTX32,
        area=AREA32,
        group=GROUP32,
        classes=CLASSES32,
    )


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

    def test_32bit_layout_is_cached(self):
        # Regression: the cache acceptance set used to be (40, 28), so a
        # real 32-bit sizeof(struct meta) == 24 was never cached and every
        # inspection re-resolved.
        analyzer, gdb = _build32()
        user = _standard_layout32(gdb)
        first = analyzer.inspect_musl_chunk(address=hex(user))
        assert first["parsed"]["verdict"] == "valid"
        assert analyzer._layout_cache is not None


def _build32(byteorder: str = "little"):
    return _build(
        pointer_bits=32,
        byteorder=byteorder,
        meta_size=META_SIZE32,
        mem_off=MEM_OFF32,
        ctx=CTX32,
        classes=CLASSES32,
    )


class TestInspectMuslChunk32Bit:
    def test_valid_in_use_chunk_32bit_le(self):
        # Regression for the old 8-byte meta bitfield read / 28-byte meta
        # size assumption: a real 32-bit struct meta is 24 bytes and its
        # bitfield word is 4 bytes wide.
        analyzer, gdb = _build32("little")
        user = _standard_layout32(gdb)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["state"] == "in_use"
        assert parsed["slot_index"] == 1
        assert parsed["group_base"] == GROUP32
        assert parsed["sizeclass"] == 1
        assert parsed["last_idx"] == 3
        assert parsed["stride"] == 32
        assert parsed["nominal_size"] == 28
        assert all(c["passed"] for c in parsed["checks"])

    def test_valid_in_use_chunk_32bit_be(self):
        # Big-endian 32-bit packs last_idx into the top 5 bits of the word;
        # a 64-bit decode would report sizeclass 0 / last_idx 0 here.
        analyzer, gdb = _build32("big")
        user = _standard_layout32(gdb, byteorder="big")
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["state"] == "in_use"
        assert parsed["sizeclass"] == 1
        assert parsed["last_idx"] == 3
        assert parsed["stride"] == 32
        assert parsed["nominal_size"] == 28
        assert all(c["passed"] for c in parsed["checks"])

    def test_32bit_index_out_of_range_detected(self):
        # idx 7 with last_idx 3 must fail even though the old 64-bit decode
        # of a 32-bit BE word would have produced last_idx = 0.
        analyzer, gdb = _build32("big")
        _place_ctx32(gdb, "big")
        _place_area32(
            gdb,
            [
                _meta_bytes(
                    mem=GROUP32,
                    last_idx=3,
                    sizeclass=1,
                    pointer_bits=32,
                    byteorder="big",
                )
            ],
            "big",
        )
        _place_group32(gdb, AREA32 + 16, "big")
        _place_classes32(gdb, "big")
        user = _place_chunk32(gdb, idx=7, byteorder="big")
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert parsed["corruption"] == "slot index out of range"


class TestMuslMmapStride:
    def test_individual_mmap_group_uses_page_stride(self):
        # musl get_stride(): last_idx == 0 and maplen != 0 means the group
        # is its own mapping, so the stride is maplen*4096 - UNIT, not the
        # size-class stride.
        analyzer, gdb = _build()
        _place_ctx(gdb)
        _place_area(
            gdb,
            [_meta_bytes(mem=GROUP, last_idx=0, sizeclass=63, maplen=2)],
        )
        _place_group(gdb, meta=AREA + 24)
        gdb.place(CLASSES, b"".join(_u16(v) for v in SIZE_CLASSES))
        user = _place_chunk(gdb, idx=0, stride=2 * 4096 - 16)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["sizeclass"] == 63
        assert parsed["last_idx"] == 0
        assert parsed["maplen"] == 2
        assert parsed["stride"] == 2 * 4096 - 16
        assert parsed["nominal_size"] == 2 * 4096 - 16 - 4

    def test_mmap_group_with_small_sizeclass_uses_page_stride(self):
        # A single-slot group allocated via mmap can still carry a sizeclass
        # < 48 (alloc_group sets last_idx=0, sizeclass=sc, maplen>0).  The
        # stride must follow get_stride()'s maplen branch there too.
        analyzer, gdb = _build()
        _place_ctx(gdb)
        _place_area(
            gdb,
            [_meta_bytes(mem=GROUP, last_idx=0, sizeclass=20, maplen=1)],
        )
        _place_group(gdb, meta=AREA + 24)
        gdb.place(CLASSES, b"".join(_u16(v) for v in SIZE_CLASSES))
        user = _place_chunk(gdb, idx=0, stride=4096 - 16)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["sizeclass"] == 20
        assert parsed["stride"] == 4096 - 16
        assert parsed["nominal_size"] == 4096 - 16 - 4

    def test_individual_mmap_group_32bit(self):
        analyzer, gdb = _build32()
        _place_ctx32(gdb)
        _place_area32(
            gdb,
            [
                _meta_bytes(
                    mem=GROUP32,
                    last_idx=0,
                    sizeclass=63,
                    maplen=1,
                    pointer_bits=32,
                )
            ],
        )
        _place_group32(gdb, AREA32 + 16)
        _place_classes32(gdb)
        user = _place_chunk32(gdb, idx=0, stride=4096 - 16)
        parsed = analyzer.inspect_musl_chunk(address=hex(user))["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["stride"] == 4096 - 16
        assert parsed["nominal_size"] == 4096 - 16 - 4


class TestListMuslGroups32Bit:
    def test_lists_groups_32bit(self):
        # 32-bit struct meta is 24 bytes and struct meta_area's slots start
        # at offset 16 (uint64 check + 4-byte next + 4-byte nslots).
        analyzer, gdb = _build32()
        _place_ctx32(gdb)
        meta_addr = AREA32 + 16
        _place_area32(
            gdb,
            [
                _meta_bytes(mem=GROUP32, last_idx=3, sizeclass=1, pointer_bits=32),
                _meta_bytes(
                    mem=0x08006000,
                    last_idx=1,
                    sizeclass=5,
                    freed=0b11,
                    pointer_bits=32,
                ),
            ],
        )
        _place_group32(gdb, meta_addr)
        _place_classes32(gdb)
        parsed = analyzer.list_musl_groups()["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["group_count"] == 2
        assert parsed["area_count"] == 1
        first, second = parsed["groups"]
        assert first["meta_address"] == meta_addr
        assert first["group_base"] == GROUP32
        assert first["sizeclass"] == 1
        assert second["sizeclass"] == 5
        assert second["freed_mask"] == 0b11

    def test_lists_groups_32bit_be(self):
        analyzer, gdb = _build32("big")
        _place_ctx32(gdb, "big")
        meta_addr = AREA32 + 16
        _place_area32(
            gdb,
            [
                _meta_bytes(
                    mem=GROUP32,
                    last_idx=3,
                    sizeclass=1,
                    pointer_bits=32,
                    byteorder="big",
                ),
            ],
            "big",
        )
        _place_group32(gdb, meta_addr, "big")
        _place_classes32(gdb, "big")
        parsed = analyzer.list_musl_groups()["parsed"]
        assert parsed["verdict"] == "parsed"
        assert parsed["group_count"] == 1
        assert parsed["groups"][0]["meta_address"] == meta_addr
        assert parsed["groups"][0]["sizeclass"] == 1
