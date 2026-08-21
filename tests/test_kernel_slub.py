"""Tests for the KernelSlab component + HeapTools facade (Stage 6.10).

These build a synthetic BTF blob (real header/record format via the parser
test builders), place a fake ``struct kmem_cache`` + ``struct slab`` in the
fake gdb's memory, and drive the full facade -> component path.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import HeapTools
from paomcp.tools.backends.slub import (
    SLUB_CACHE_SCHEMA,
    SLUB_ENCODING_LEGACY,
    SLUB_ENCODING_SWAB,
    SLUB_FREELIST_SCHEMA,
    SLUB_HEAD_RAW,
    slub_swab,
)

from tests.test_btf_parser import (
    btf_struct_record,
    make_btf_blob,
    make_string_section,
)
from tests.test_glibc_arenas import (
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
)

# Kernel-ish addresses.
CACHE_ADDR = 0xFFFF88800A000000
SLAB_ADDR = 0xFFFF88800A010000
SLAB_MEM = SLAB_ADDR  # slab header sits at the slab page base
NAME_ADDR = 0xFFFF88800A002000

# Test-side kmem_cache layout (arbitrary, but shared by BTF + placements).
KMEM_CACHE_SIZE = 0x2C0
CACHE_OFF = {
    "name": 0x00,
    "object_size": 0x10,
    "size": 0x18,
    "offset": 0x48,
    "oo": 0x78,
    "random": 0xB0,
}

# struct slab: freelist at 0, counters word at byte 8 (inuse:6, objects:19,
# frozen:1), cpu at 0x18.
SLAB_SIZE = 0x40
_COUNTERS_OFF = 0x08
SLAB_OFF = {
    "freelist": 0x00,
    "inuse": (64, 6),
    "objects": (70, 19),
    "frozen": (89, 1),
    "cpu": 0x18,
}

RANDOM = 0xDEADBEEFCAFEBABE
FREEPOINTER_OFF = 0x10
OBJECT_SIZE = 0x40
# Three free objects inside the slab window.
OBJ0 = SLAB_MEM + 0x40
OBJ1 = SLAB_MEM + 0x80
OBJ2 = SLAB_MEM + 0xC0

_KERNEL_MAPS = [
    {
        "file": "[kernel]",
        "start": "0xffffffff80000000",
        "end": "0xffffffffc0000000",
        "permissions": "r-xp",
        "readable": True,
        "writable": False,
        "executable": True,
    },
    {
        "file": "[direct-map]",
        "start": "0xffff888000000000",
        "end": "0xffffc90000000000",
        "permissions": "rw-p",
        "readable": True,
        "writable": True,
        "executable": False,
    },
]


def make_slub_btf(
    *,
    hardened: bool = True,
    include_slab: bool = True,
    include_page: bool = True,
) -> bytes:
    """Build a synthetic vmlinux BTF blob with kmem_cache (+ slab + page)."""
    member_names = [
        "name",
        "object_size",
        "size",
        "offset",
        "oo",
        "random",
        "freelist",
        "inuse",
        "objects",
        "frozen",
        "cpu",
    ]
    strings, off = make_string_section("kmem_cache", "slab", "page", *member_names)
    records = []

    cache_members = [
        (off["name"], 1, CACHE_OFF["name"] * 8),
        (off["object_size"], 1, CACHE_OFF["object_size"] * 8),
        (off["size"], 1, CACHE_OFF["size"] * 8),
        (off["offset"], 1, CACHE_OFF["offset"] * 8),
        (off["oo"], 1, CACHE_OFF["oo"] * 8),
    ]
    if hardened:
        cache_members.append((off["random"], 1, CACHE_OFF["random"] * 8))
    records.append(btf_struct_record(off["kmem_cache"], KMEM_CACHE_SIZE, cache_members))

    if include_slab:
        slab_members = [
            (off["freelist"], 1, SLAB_OFF["freelist"] * 8),
            (off["inuse"], 1, (6 << 24) | (SLAB_OFF["inuse"][0])),
            (off["objects"], 1, (19 << 24) | (SLAB_OFF["objects"][0])),
            (off["frozen"], 1, (1 << 24) | (SLAB_OFF["frozen"][0])),
            (off["cpu"], 1, SLAB_OFF["cpu"] * 8),
        ]
        records.append(
            btf_struct_record(off["slab"], SLAB_SIZE, slab_members, kind_flag=True)
        )
    if include_page:
        page_members = [
            (off["freelist"], 1, 0),
            (off["objects"], 1, (19 << 24) | (SLAB_OFF["objects"][0])),
        ]
        records.append(
            btf_struct_record(off["page"], SLAB_SIZE, page_members, kind_flag=True)
        )
    return make_btf_blob(records, strings)


def place_cache(
    gdb: ArenaFakeGdb,
    *,
    address: int = CACHE_ADDR,
    name_ptr: int = NAME_ADDR,
    object_size: int = OBJECT_SIZE,
    size: int = OBJECT_SIZE + 0x10,
    offset: int = FREEPOINTER_OFF,
    oo_x: int = (0 << 16) | 32,
    random: int = RANDOM,
    byteorder: str = "little",
) -> None:
    buf = bytearray(KMEM_CACHE_SIZE)
    buf[CACHE_OFF["name"] : CACHE_OFF["name"] + 8] = name_ptr.to_bytes(8, byteorder)
    buf[CACHE_OFF["object_size"] : CACHE_OFF["object_size"] + 4] = object_size.to_bytes(
        4, byteorder
    )
    buf[CACHE_OFF["size"] : CACHE_OFF["size"] + 4] = size.to_bytes(4, byteorder)
    buf[CACHE_OFF["offset"] : CACHE_OFF["offset"] + 4] = offset.to_bytes(4, byteorder)
    buf[CACHE_OFF["oo"] : CACHE_OFF["oo"] + 4] = oo_x.to_bytes(4, byteorder)
    buf[CACHE_OFF["random"] : CACHE_OFF["random"] + 8] = random.to_bytes(8, byteorder)
    gdb.place(address, bytes(buf))


def place_slab(
    gdb: ArenaFakeGdb,
    *,
    address: int = SLAB_ADDR,
    freelist_raw: int = 0,
    inuse: int = 1,
    objects: int = 3,
    frozen: int = 0,
    cpu: int = 0,
    byteorder: str = "little",
) -> None:
    buf = bytearray(SLAB_SIZE)
    buf[SLAB_OFF["freelist"] : SLAB_OFF["freelist"] + 8] = freelist_raw.to_bytes(
        8, byteorder
    )
    counters = inuse | (objects << 6) | (frozen << 25)
    buf[_COUNTERS_OFF : _COUNTERS_OFF + 8] = counters.to_bytes(8, byteorder)
    buf[SLAB_OFF["cpu"] : SLAB_OFF["cpu"] + 4] = cpu.to_bytes(4, byteorder)
    gdb.place(address, bytes(buf))


def enc(
    ptr: int, store: int, *, random: int = RANDOM, encoding: str = SLUB_ENCODING_SWAB
) -> int:
    if encoding == SLUB_ENCODING_SWAB:
        return ptr ^ random ^ slub_swab(store, 64)
    if encoding == SLUB_ENCODING_LEGACY:
        return ptr ^ random ^ store
    return ptr


def place_slub_chain(
    gdb: ArenaFakeGdb,
    objects: List[int],
    *,
    random: int = RANDOM,
    encoding: str = SLUB_ENCODING_SWAB,
    offset: int = FREEPOINTER_OFF,
    byteorder: str = "little",
) -> None:
    for i, obj in enumerate(objects):
        nxt = objects[i + 1] if i + 1 < len(objects) else 0
        slot = obj + offset
        gdb.place(
            slot,
            enc(nxt, slot, random=random, encoding=encoding).to_bytes(8, byteorder),
        )


def build_slub(
    *,
    live: bool = True,
    hardened: bool = True,
    include_slab: bool = True,
    include_page: bool = True,
    btf_bytes: Optional[bytes] = None,
    tmp_path: Path,
) -> tuple:
    gdb = ArenaFakeGdb(live=live)
    inspection = ArenaFakeInspection(gdb, mappings=_KERNEL_MAPS, libc=None)
    dbg = ArenaFakeDebugger()
    heap = HeapTools(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    blob = (
        btf_bytes
        if btf_bytes is not None
        else make_slub_btf(
            hardened=hardened, include_slab=include_slab, include_page=include_page
        )
    )
    path = tmp_path / "vmlinux.btf"
    path.write_bytes(blob)
    return heap, gdb, str(path)


def place_scene(
    gdb: ArenaFakeGdb,
    *,
    head: int = OBJ0,
    objects: Optional[List[int]] = None,
    encoding: str = SLUB_ENCODING_SWAB,
    hardened: bool = True,
) -> None:
    gdb.place(NAME_ADDR, b"kmalloc-64\x00")
    place_cache(gdb, random=RANDOM if hardened else 0)
    chain = objects if objects is not None else [OBJ0, OBJ1, OBJ2]
    place_slab(gdb, freelist_raw=head)
    place_slub_chain(gdb, chain, encoding=encoding)


# ---------------------------------------------------------------------------
class TestGuards:
    def test_requires_live_inferior(self, tmp_path: Path):
        heap, _, btf_path = build_slub(live=False, tmp_path=tmp_path)
        with pytest.raises(LiveInferiorError):
            heap.inspect_slub_cache(cache_address=hex(CACHE_ADDR), btf_path=btf_path)

    def test_bad_cache_address_raises(self, tmp_path: Path):
        heap, _, btf_path = build_slub(tmp_path=tmp_path)
        with pytest.raises(ValueError):
            heap.inspect_slub_cache(cache_address="not-a-number", btf_path=btf_path)

    def test_bad_slab_address_raises(self, tmp_path: Path):
        heap, _, btf_path = build_slub(tmp_path=tmp_path)
        with pytest.raises(ValueError):
            heap.decode_slub_freelist(
                cache_address=hex(CACHE_ADDR),
                slab_address="not-a-number",
                btf_path=btf_path,
            )


# ---------------------------------------------------------------------------
class TestBtfFailures:
    def test_btf_missing_file(self, tmp_path: Path):
        heap, _, _ = build_slub(tmp_path=tmp_path)
        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=str(tmp_path / "missing.btf")
        )["parsed"]
        assert parsed["schema"] == SLUB_CACHE_SCHEMA
        assert parsed["verdict"] == "btf_unavailable"
        assert parsed["supported"] is False

    def test_btf_garbage(self, tmp_path: Path):
        heap, _, btf_path = build_slub(
            tmp_path=tmp_path, btf_bytes=b"this is not btf" * 8
        )
        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )["parsed"]
        assert parsed["verdict"] == "btf_unavailable"

    def test_no_slab_or_page_header(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(
            include_slab=False, include_page=False, tmp_path=tmp_path
        )
        place_scene(gdb)
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["verdict"] == "unsupported"
        assert "freelist field" in parsed["reason"]


# ---------------------------------------------------------------------------
class TestInspectCache:
    def test_cache_decoded_hardened(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb)

        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )["parsed"]
        assert parsed["schema"] == SLUB_CACHE_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "decoded"
        assert parsed["allocator"] == {"family": "linux-kernel", "subsystem": "slub"}
        cache = parsed["cache"]
        assert cache["name"] == "kmalloc-64"
        assert cache["object_size"] == OBJECT_SIZE
        assert cache["size"] == OBJECT_SIZE + 0x10
        assert cache["freepointer_offset"] == FREEPOINTER_OFF
        assert cache["random"] == RANDOM
        assert cache["hardened"] is True
        assert cache["order"] == 0
        assert cache["objects_per_slab"] == 32
        assert cache["slab_bytes"] == 0x1000
        assert parsed["btf"]["type_count"] == 3
        assert "random" in parsed["btf"]["resolved_fields"]

    def test_cache_not_hardened(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(hardened=False, tmp_path=tmp_path)
        place_scene(gdb, hardened=False)

        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )["parsed"]
        cache = parsed["cache"]
        assert cache["hardened"] is False
        assert cache["random"] == 0
        assert "random" not in parsed["btf"]["resolved_fields"]

    def test_cache_unreadable(self, tmp_path: Path):
        heap, _, btf_path = build_slub(tmp_path=tmp_path)
        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert parsed["supported"] is False

    def test_cache_name_unreadable_warns(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_cache(gdb, name_ptr=CACHE_ADDR + 0x5000)  # nothing placed there
        place_slab(gdb, freelist_raw=OBJ0)
        place_slub_chain(gdb, [OBJ0, OBJ1, OBJ2])

        result = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )
        parsed = result["parsed"]
        assert parsed["verdict"] == "decoded"
        assert parsed["cache"]["name"] is None
        assert any("cache name string" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
class TestStructOffsetsUnresolved:
    def test_cache_missing_field(self, tmp_path: Path):
        # Build a BTF without the "oo" field by dropping it from the record.
        strings, off = make_string_section(
            "kmem_cache", "name", "object_size", "size", "offset"
        )
        members = [
            (off["name"], 1, 0),
            (off["object_size"], 1, 0x80),
            (off["size"], 1, 0xC0),
            (off["offset"], 1, 0x240),
        ]
        records = [btf_struct_record(off["kmem_cache"], KMEM_CACHE_SIZE, members)]
        heap, gdb, btf_path = build_slub(
            tmp_path=tmp_path, btf_bytes=make_btf_blob(records, strings)
        )
        place_scene(gdb)

        parsed = heap.inspect_slub_cache(
            cache_address=hex(CACHE_ADDR), btf_path=btf_path
        )["parsed"]
        assert parsed["verdict"] == "struct_offsets_unresolved"
        assert "oo" in parsed["missing_fields"]


# ---------------------------------------------------------------------------
class TestDecodeFreelist:
    def test_decode_swab(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb)

        result = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )
        parsed = result["parsed"]
        assert parsed["schema"] == SLUB_FREELIST_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "decoded"
        slab = parsed["slab"]
        assert slab["header_source"] == "struct slab"
        assert slab["objects"] == 3
        assert slab["inuse"] == 1
        assert slab["slab_bytes"] == 0x1000
        freelist = parsed["freelist"]
        assert freelist["hardened"] is True
        assert freelist["encoding"] == SLUB_ENCODING_SWAB
        assert freelist["head_mode"] == SLUB_HEAD_RAW
        assert freelist["object_addresses"] == [OBJ0, OBJ1, OBJ2]
        assert freelist["count"] == 3
        assert freelist["termination"] == "null"
        assert freelist["chain"][0]["decoded_next"] == OBJ1
        assert any("slab_mem_address not given" in w for w in result["warnings"])

    def test_decode_legacy(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb, encoding=SLUB_ENCODING_LEGACY)
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["freelist"]["encoding"] == SLUB_ENCODING_LEGACY
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1, OBJ2]

    def test_decode_not_hardened(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(hardened=False, tmp_path=tmp_path)
        place_scene(gdb, hardened=False, encoding="identity")
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["freelist"]["hardened"] is False
        assert parsed["freelist"]["encoding"] is None
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1, OBJ2]

    def test_decode_raw_head_6x(self, tmp_path: Path):
        # 6.x model: slab->freelist holds a raw decoded head; only slots
        # are obfuscated.
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb, head=OBJ1, objects=[OBJ1, OBJ2])
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["freelist"]["head_mode"] == SLUB_HEAD_RAW
        assert parsed["freelist"]["object_addresses"] == [OBJ1, OBJ2]

    def test_decode_struct_page_fallback(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(
            include_slab=False, include_page=True, tmp_path=tmp_path
        )
        place_scene(gdb)
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["slab"]["header_source"] == "struct page"
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1, OBJ2]

    def test_decode_explicit_slab_mem(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb)
        result = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            slab_mem_address=hex(SLAB_MEM),
            btf_path=btf_path,
        )
        assert result["parsed"]["verdict"] == "decoded"
        assert not any(
            "slab_mem_address not given" in w for w in (result.get("warnings") or [])
        )

    def test_decode_max_objects_truncates(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb)
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
            max_objects=2,
        )["parsed"]
        assert parsed["freelist"]["termination"] == "max_objects"
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1]

    def test_decode_unresolved(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        # Garbage head + garbage slots: nothing decodes inside the window.
        place_scene(gdb, head=SLAB_MEM + 0x8000, objects=[OBJ0])
        gdb.place(OBJ0 + FREEPOINTER_OFF, (0x1234567890ABCDEF).to_bytes(8, "little"))

        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["verdict"] == "unresolved"
        assert parsed["supported"] is True
        assert len(parsed["attempts"]) == 4

    def test_decode_empty_freelist_hardened(self, tmp_path: Path):
        # M6: a hardened cache whose slab->freelist is NULL is an *empty*
        # freelist, not an unresolvable one.  The zero head must be reported
        # as a decoded count-0 chain, symmetric with the unhardened path --
        # never routed through the count>=1 candidate filter to "unresolved".
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb, head=0, objects=[])

        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["verdict"] == "decoded"
        freelist = parsed["freelist"]
        assert freelist["hardened"] is True
        assert freelist["count"] == 0
        assert freelist["object_addresses"] == []
        assert freelist["termination"] == "null"
        assert freelist["head_mode"] == SLUB_HEAD_RAW

    def test_decode_empty_freelist_unhardened(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(hardened=False, tmp_path=tmp_path)
        place_scene(gdb, head=0, objects=[], hardened=False)

        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["verdict"] == "decoded"
        assert parsed["freelist"]["count"] == 0
        assert parsed["freelist"]["termination"] == "null"

    def test_decode_big_endian_target(self, tmp_path: Path):
        # L3: struct reads must honor the target byte order.  Place every
        # struct field + freepointer big-endian (s390x / ppc64 style) and
        # decode with byteorder="big".
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        gdb.place(NAME_ADDR, b"kmalloc-64\x00")
        place_cache(gdb, random=RANDOM, byteorder="big")
        place_slab(gdb, freelist_raw=OBJ0, byteorder="big")
        place_slub_chain(gdb, [OBJ0, OBJ1, OBJ2], byteorder="big")

        result = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
            byteorder="big",
        )
        parsed = result["parsed"]
        assert parsed["verdict"] == "decoded"
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1, OBJ2]
        assert parsed["freelist"]["termination"] == "null"
        # The cache scalars decode the same regardless of endianness.
        assert parsed["slab"]["objects_per_slab"] == 32
        assert parsed["slab"]["order"] == 0

    def test_decode_rejects_bad_byteorder(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_scene(gdb)
        with pytest.raises(ValueError, match="byteorder"):
            heap.decode_slub_freelist(
                cache_address=hex(CACHE_ADDR),
                slab_address=hex(SLAB_ADDR),
                btf_path=btf_path,
                byteorder="middle",
            )

    def test_decode_order_aware_slab_mem_derivation(self, tmp_path: Path):
        # M4: with an order-1 slab (slab_bytes = 2 pages) whose header sits
        # in the *second* page, the omitted-slab_mem derivation must align to
        # the slab's full extent, not just one page, so the window still
        # contains the first-page objects.
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        gdb.place(NAME_ADDR, b"kmalloc-64\x00")
        place_cache(gdb, oo_x=(1 << 16) | 32, random=RANDOM)  # order 1
        slab_header_addr = SLAB_ADDR + 0x1000  # struct slab in 2nd page
        place_slab(gdb, address=slab_header_addr, freelist_raw=OBJ0)
        place_slub_chain(gdb, [OBJ0, OBJ1, OBJ2])

        result = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(slab_header_addr),
            btf_path=btf_path,
        )
        parsed = result["parsed"]
        assert parsed["verdict"] == "decoded"
        assert parsed["freelist"]["object_addresses"] == [OBJ0, OBJ1, OBJ2]
        assert parsed["slab"]["order"] == 1
        assert parsed["slab"]["slab_bytes"] == 0x2000
        assert any("order-aware" in w for w in result.get("warnings") or [])

    def test_decode_unreadable_slab(self, tmp_path: Path):
        heap, gdb, btf_path = build_slub(tmp_path=tmp_path)
        place_cache(gdb)  # no slab placed
        parsed = heap.decode_slub_freelist(
            cache_address=hex(CACHE_ADDR),
            slab_address=hex(SLAB_ADDR),
            btf_path=btf_path,
        )["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert "struct slab" in parsed["reason"]
