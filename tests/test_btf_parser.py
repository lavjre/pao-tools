"""Tests for the pure BTF parser (Stage 6.10).

The parser consumes raw kernel BTF blobs (``/sys/kernel/btf/vmlinux``) and
extracts struct/union layouts.  These tests build small synthetic blobs with
the exact on-disk header/record format and assert the parser walks them.
"""

import struct
from pathlib import Path

import pytest

from paomcp.tools.backends.btf import (
    BTF_KIND_ARRAY,
    BTF_KIND_CONST,
    BTF_KIND_DATASEC,
    BTF_KIND_DECL_TAG,
    BTF_KIND_ENUM,
    BTF_KIND_ENUM64,
    BTF_KIND_FLOAT,
    BTF_KIND_FUNC,
    BTF_KIND_FUNC_PROTO,
    BTF_KIND_FWD,
    BTF_KIND_INT,
    BTF_KIND_PTR,
    BTF_KIND_RESTRICT,
    BTF_KIND_STRUCT,
    BTF_KIND_TYPEDEF,
    BTF_KIND_TYPE_TAG,
    BTF_KIND_UNION,
    BTF_KIND_VAR,
    BTF_KIND_VOLATILE,
    BtfError,
    BtfFile,
    load_btf,
)

_BTF_MAGIC = 0xEB9F


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def make_string_section(*names: str) -> tuple:
    """Build a BTF string section; returns (blob, {name: offset})."""
    blob = b"\x00"  # string 0 is the empty string
    offsets = {}
    for name in names:
        offsets[name] = len(blob)
        blob += name.encode() + b"\x00"
    return blob, offsets


def btf_type_record(name_off: int, info: int, size: int) -> bytes:
    return struct.pack("<III", name_off, info, size)


def btf_struct_record(
    name_off: int,
    size: int,
    members,
    *,
    kind: int = BTF_KIND_STRUCT,
    kind_flag: bool = False,
) -> bytes:
    """members: list of (name_off, type_id, offset_word)."""
    info = (kind << 24) | (len(members) & 0xFFFF)
    if kind_flag:
        info |= 1 << 31
    record = btf_type_record(name_off, info, size)
    for member in members:
        record += struct.pack("<III", member[0], member[1], member[2])
    return record


def btf_non_struct_record(
    kind: int, name_off: int, extra: bytes, *, vlen: int = 0, size: int = 0
) -> bytes:
    info = (kind << 24) | (vlen & 0xFFFF)
    return btf_type_record(name_off, info, size) + extra


def make_btf_blob(records: list, strings: bytes) -> bytes:
    hdr_len = 24
    type_len = sum(len(record) for record in records)
    str_off = type_len
    header = struct.pack(
        "<HBBIIIII",
        _BTF_MAGIC,
        1,  # version
        0,  # flags
        hdr_len,
        0,  # type_off
        type_len,
        str_off,
        len(strings),
    )
    return header + b"".join(records) + strings


def _blob_with_struct(
    *,
    name: str = "kmem_cache",
    size: int = 0x2C0,
    members=(),
    kind: int = BTF_KIND_STRUCT,
    kind_flag: bool = False,
) -> tuple:
    """Build a single-struct BTF blob.

    ``members`` are (name, type_id, bit_offset) triples; names are resolved
    into the string section automatically.
    """
    strings, off = make_string_section(name, *[m[0] for m in members])
    resolved = [(off[m[0]], m[1], m[2]) for m in members]
    records = [
        btf_struct_record(off[name], size, resolved, kind=kind, kind_flag=kind_flag)
    ]
    return make_btf_blob(records, strings), off


# ---------------------------------------------------------------------------
class TestHeader:
    def test_header_fields(self):
        blob, _ = _blob_with_struct(name="cache")
        btf = BtfFile.parse(blob)
        assert btf.header.magic == _BTF_MAGIC
        assert btf.header.version == 1
        assert btf.header.hdr_len == 24
        assert btf.header.type_len == 12  # one memberless struct record
        assert btf.type_count == 1

    def test_bad_magic_raises(self):
        blob, _ = _blob_with_struct()
        bad = b"\x00\x00" + blob[2:]
        with pytest.raises(BtfError, match="magic"):
            BtfFile.parse(bad)

    def test_truncated_header_raises(self):
        with pytest.raises(BtfError, match="header truncated"):
            BtfFile.parse(b"\x00" * 20)

    def test_truncated_type_section_raises(self):
        blob, _ = _blob_with_struct()
        with pytest.raises(BtfError, match="overruns"):
            BtfFile.parse(blob[:-4])

    def test_record_past_section_end_raises(self):
        # Claim the type section is only the 12-byte record header of a
        # vlen=1 struct -- the member parse must overrun type_end.
        strings, off = make_string_section("cache", "a")
        record = btf_struct_record(off["cache"], 0x100, [(off["a"], 1, 0)])
        header = struct.pack("<HBBIIIII", _BTF_MAGIC, 1, 0, 24, 0, 12, 12, len(strings))
        blob = header + record[:12] + strings
        with pytest.raises(BtfError, match="member record"):
            BtfFile.parse(blob)

    def test_unknown_kind_raises(self):
        strings, off = make_string_section("weird")
        record = btf_non_struct_record(20, off["weird"], b"")  # 20 is reserved
        with pytest.raises(BtfError, match="unsupported BTF kind 20"):
            BtfFile.parse(make_btf_blob([record], strings))


# ---------------------------------------------------------------------------
class TestMembers:
    def test_member_byte_offsets(self):
        blob, off = _blob_with_struct(
            members=[("a", 1, 0), ("b", 1, 64), ("c", 1, 192)]
        )
        btf = BtfFile.parse(blob)
        assert [m.byte_offset for m in btf.struct("kmem_cache").members] == [0, 8, 24]
        assert btf.struct("kmem_cache").size_bytes == 0x2C0

    def test_kind_flag_bitfield_member(self):
        # bit_offset 6, bitfield_size 19, packed into the offset word.
        packed = (19 << 24) | 6
        blob, off = _blob_with_struct(
            name="slab", members=[("objects", 1, packed)], kind_flag=True
        )
        member = BtfFile.parse(blob).member("slab", "objects")
        assert member.bit_offset == 6
        assert member.bitfield_size == 19
        assert member.byte_offset == 0
        assert member.is_bitfield

    def test_kind_flag_non_bitfield_has_zero_bitfield_size(self):
        blob, off = _blob_with_struct(
            name="slab", members=[("freelist", 1, 0x40)], kind_flag=True
        )
        member = BtfFile.parse(blob).member("slab", "freelist")
        assert member.byte_offset == 8
        assert member.bitfield_size == 0
        assert not member.is_bitfield

    def test_union_by_name(self):
        blob, off = _blob_with_struct(
            name="page", members=[("freelist", 1, 0)], kind=BTF_KIND_UNION
        )
        btf = BtfFile.parse(blob)
        assert btf.struct("page").kind == BTF_KIND_UNION

    def test_empty_struct(self):
        blob, off = make_string_section("empty")
        record = btf_struct_record(off["empty"], 0, [])
        btf = BtfFile.parse(make_btf_blob([record], blob))
        assert btf.struct("empty").members == []


# ---------------------------------------------------------------------------
class TestLookup:
    def test_struct_and_member_by_name(self):
        blob, off = _blob_with_struct(
            name="cache", members=[("offset", 1, 0x50), ("oo", 1, 0x88)]
        )
        btf = BtfFile.parse(blob)
        assert btf.struct("cache").name == "cache"
        assert btf.member("cache", "offset").byte_offset == 0x0A
        assert btf.member("cache", "oo").byte_offset == 0x11

    def test_missing_struct_returns_none(self):
        blob, _ = _blob_with_struct(name="cache")
        assert BtfFile.parse(blob).struct("nope") is None

    def test_missing_member_returns_none(self):
        blob, _ = _blob_with_struct(name="cache", members=[("offset", 1, 0)])
        assert BtfFile.parse(blob).member("cache", "nope") is None

    def test_last_definition_wins(self):
        strings, off = make_string_section("cache", "a", "b")
        first = btf_struct_record(off["cache"], 0x100, [(off["a"], 1, 0)])
        second = btf_struct_record(off["cache"], 0x200, [(off["b"], 1, 64)])
        btf = BtfFile.parse(make_btf_blob([first, second], strings))
        assert btf.struct("cache").size_bytes == 0x200
        assert btf.struct("cache").members[0].name == "b"

    def test_out_of_range_name_off_yields_empty(self):
        strings, off = make_string_section("cache")
        record = btf_struct_record(0xFFFFFF, 0x100, [])
        btf = BtfFile.parse(make_btf_blob([record], strings))
        assert btf.struct("") is not None  # unnamed struct indexed under ""
        assert btf.type_count == 1

    def test_multi_entry_strings_resolve(self):
        strings, off = make_string_section("alpha", "beta", "gamma")
        records = [
            btf_struct_record(off["alpha"], 0x10, []),
            btf_struct_record(off["gamma"], 0x20, []),
        ]
        btf = BtfFile.parse(make_btf_blob(records, strings))
        assert btf.struct("alpha").size_bytes == 0x10
        assert btf.struct("gamma").size_bytes == 0x20


# ---------------------------------------------------------------------------
class TestKindSkips:
    def test_skips_all_non_struct_kinds(self):
        strings, off = make_string_section("n")
        n = off["n"]
        records = [
            btf_non_struct_record(BTF_KIND_INT, n, b"\x00\x00\x00\x00"),
            btf_non_struct_record(BTF_KIND_PTR, n, b""),
            btf_non_struct_record(BTF_KIND_ARRAY, n, b"\x00" * 12),
            btf_non_struct_record(BTF_KIND_ENUM, n, b"\x00" * 8, vlen=1),
            btf_non_struct_record(BTF_KIND_FWD, n, b""),
            btf_non_struct_record(BTF_KIND_TYPEDEF, n, b""),
            btf_non_struct_record(BTF_KIND_VOLATILE, n, b""),
            btf_non_struct_record(BTF_KIND_CONST, n, b""),
            btf_non_struct_record(BTF_KIND_RESTRICT, n, b""),
            btf_non_struct_record(BTF_KIND_FUNC, n, b""),
            btf_non_struct_record(BTF_KIND_FUNC_PROTO, n, b"\x00" * 8, vlen=1),
            btf_non_struct_record(BTF_KIND_VAR, n, b"\x00\x00\x00\x00"),
            btf_non_struct_record(BTF_KIND_DATASEC, n, b"\x00" * 12, vlen=1),
            btf_non_struct_record(BTF_KIND_FLOAT, n, b""),
            btf_non_struct_record(BTF_KIND_DECL_TAG, n, b"\x00\x00\x00\x00"),
            btf_non_struct_record(BTF_KIND_TYPE_TAG, n, b""),
            btf_non_struct_record(BTF_KIND_ENUM64, n, b"\x00" * 12, vlen=1),
            btf_struct_record(n, 0x10, []),  # STRUCT
            btf_struct_record(n, 0x20, [], kind=BTF_KIND_UNION),
        ]
        btf = BtfFile.parse(make_btf_blob(records, strings))
        assert btf.type_count == 19
        # Every record parsed in order; only struct/union got names indexed.
        assert btf.struct("n") is not None


# ---------------------------------------------------------------------------
class TestLoadBtf:
    def test_load_btf_from_file(self, tmp_path: Path):
        blob, _ = _blob_with_struct(name="cache", members=[("offset", 1, 0x50)])
        path = tmp_path / "vmlinux.btf"
        path.write_bytes(blob)
        btf = load_btf(str(path))
        assert btf.type_count == 1
        assert btf.member("cache", "offset").byte_offset == 0x0A

    def test_load_btf_missing_file_raises_oserror(self, tmp_path: Path):
        with pytest.raises(OSError):
            load_btf(str(tmp_path / "does-not-exist.btf"))
