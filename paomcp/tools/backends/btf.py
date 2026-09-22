"""Pure BPF Type Format (BTF) parser -- stdlib only.

BTF is the type metadata blob the kernel ships under ``/sys/kernel/btf/vmlinux``
(and per-module under ``/sys/kernel/btf/<module>``).  It describes every C type
in the kernel -- including ``struct kmem_cache`` and ``struct slab`` -- which is
what lets the SLUB analyzer resolve field offsets on *any* kernel version
without hard-coding layout tables.

This module is deliberately tiny and read-only: it extracts struct/union
layouts (member names, type ids, byte offsets, bitfield geometry) and ignores
every other kind of record.  The kernel's own parse rules (see
``kernel/bpf/btf.c``) are the reference: string offsets outside the string
section yield the empty name, records must stay inside the type section, and
the format is always little-endian regardless of the target byte order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from struct import unpack_from
from typing import Dict, List, Optional, Set, Tuple

# BTF header (24 bytes, little-endian).
BTF_MAGIC = 0xEB9F
_BTF_HEADER = "<HBBIIIII"
_BTF_HEADER_SIZE = 24

# btf_type header (12 bytes) + btf_member (12 bytes).
_BTF_TYPE = "<III"
_BTF_MEMBER = "<III"
_BTF_TYPE_SIZE = 12
_BTF_MEMBER_SIZE = 12
# The kernel's own BTF_MAX_VLEN (include/uapi/linux/btf.h); a legitimate
# DATASEC can carry up to 0xffff var_secinfos.  The vlen field is 16 bits
# wide, so values beyond this are unrepresentable -- the guard is kept as
# defense-in-depth against a future field-width change, and records whose
# claimed payload overruns the section are rejected by the bounds check.
MAX_BTF_VLEN = 0xFFFF
MAX_BTF_TYPES = 1 << 20
# Anonymous struct/union nesting in kernel BTF is shallow (< 10 levels);
# cap the recursive member search so a malformed blob cannot recurse deep.
MAX_BTF_ANON_DEPTH = 32

# info bit packing.
_BTF_INFO_KIND_MASK = 0x1F << 24
_BTF_INFO_KIND_SHIFT = 24
_BTF_INFO_VLEN_MASK = 0xFFFF
_BTF_KIND_FLAG = 1 << 31
_BTF_MEMBER_BITFIELD_SIZE_SHIFT = 24
_BTF_MEMBER_BIT_OFFSET_MASK = 0xFFFFFF

# BTF kinds 1-19 (the full set as of BTF v1.x).
BTF_KIND_INT = 1
BTF_KIND_PTR = 2
BTF_KIND_ARRAY = 3
BTF_KIND_STRUCT = 4
BTF_KIND_UNION = 5
BTF_KIND_ENUM = 6
BTF_KIND_FWD = 7
BTF_KIND_TYPEDEF = 8
BTF_KIND_VOLATILE = 9
BTF_KIND_CONST = 10
BTF_KIND_RESTRICT = 11
BTF_KIND_FUNC = 12
BTF_KIND_FUNC_PROTO = 13
BTF_KIND_VAR = 14
BTF_KIND_DATASEC = 15
BTF_KIND_FLOAT = 16
BTF_KIND_DECL_TAG = 17
BTF_KIND_TYPE_TAG = 18
BTF_KIND_ENUM64 = 19

# Extra payload bytes past the 12-byte record header for each non-struct kind,
# as (fixed_bytes, per_vlen_bytes).  Struct/union members are handled inline.
_BTF_SKIP: Dict[int, Tuple[int, int]] = {
    BTF_KIND_INT: (4, 0),  # btf_int {u32 encoding}
    BTF_KIND_PTR: (0, 0),
    BTF_KIND_ARRAY: (12, 0),  # btf_array {index_type, type, nelems}
    BTF_KIND_ENUM: (0, 8),  # btf_enum {name_off, val} x vlen
    BTF_KIND_FWD: (0, 0),
    BTF_KIND_TYPEDEF: (0, 0),
    BTF_KIND_VOLATILE: (0, 0),
    BTF_KIND_CONST: (0, 0),
    BTF_KIND_RESTRICT: (0, 0),
    BTF_KIND_FUNC: (0, 0),
    BTF_KIND_FUNC_PROTO: (0, 8),  # btf_param {name_off, type} x vlen
    BTF_KIND_VAR: (4, 0),  # btf_var {u32 linkage}
    BTF_KIND_DATASEC: (0, 12),  # btf_var_secinfo {type, offset, size} x vlen
    BTF_KIND_FLOAT: (0, 0),
    BTF_KIND_DECL_TAG: (4, 0),  # {component_idx u32}
    BTF_KIND_TYPE_TAG: (0, 0),
    BTF_KIND_ENUM64: (0, 12),  # btf_enum64 {name_off, val_lo32, val_hi32} x vlen
}

_STRUCT_LIKE_KINDS = (BTF_KIND_STRUCT, BTF_KIND_UNION)


class BtfError(ValueError):
    """Raised when the BTF blob is malformed (bad magic, truncation, ...)."""


@dataclass(frozen=True)
class BtfHeader:
    """The 24-byte BTF file header."""

    magic: int
    version: int
    flags: int
    hdr_len: int
    type_off: int
    type_len: int
    str_off: int
    str_len: int


@dataclass(frozen=True)
class BtfMember:
    """A struct/union member with resolved geometry."""

    name: str
    type_id: int
    byte_offset: int
    bit_offset: int
    bitfield_size: int

    @property
    def is_bitfield(self) -> bool:
        return self.bitfield_size > 0


@dataclass(frozen=True)
class BtfType:
    """A parsed type record; only struct/union carry members."""

    id: int
    kind: int
    name: str
    size_bytes: int
    vlen: int
    members: List[BtfMember] = field(default_factory=list)

    @property
    def is_struct_like(self) -> bool:
        return self.kind in _STRUCT_LIKE_KINDS


class BtfFile:
    """Parsed BTF blob exposing struct/union layout lookups.

    ``struct(name)`` and ``member(struct_name, member_name)`` return ``None``
    when the type/member is absent -- callers map that to a typed
    ``struct_offsets_unresolved`` verdict instead of an exception.
    """

    __slots__ = ("header", "_types", "_structs_by_name", "_strings")

    def __init__(
        self,
        header: BtfHeader,
        types: List[BtfType],
        structs_by_name: Dict[str, BtfType],
        strings: bytes,
    ) -> None:
        self.header = header
        self._types = types
        self._structs_by_name = structs_by_name
        self._strings = strings

    # ------------------------------------------------------------------
    @property
    def type_count(self) -> int:
        return len(self._types)

    def struct(self, name: str) -> Optional[BtfType]:
        """Return the struct/union with the given name (last definition wins)."""
        return self._structs_by_name.get(name)

    def member(self, struct_name: str, member_name: str) -> Optional[BtfMember]:
        """Return one member by name, or ``None`` when absent.

        Anonymous (empty-name) struct/union members are transparent: their
        members are searched recursively and the bit/byte offsets along the
        nesting path are accumulated.  This is what lets ``struct page``
        resolve ``freelist`` inside its nested anonymous unions on kernels
        that have no ``struct slab``.  Direct members always win over
        nested ones; a cycle in a malformed blob terminates via a
        per-path seen-set plus :data:`MAX_BTF_ANON_DEPTH`.
        """
        struct = self.struct(struct_name)
        if struct is None:
            return None
        return self._member_in(struct, member_name, set(), 0)

    def _member_in(
        self,
        struct: BtfType,
        member_name: str,
        seen: Set[int],
        depth: int,
    ) -> Optional[BtfMember]:
        if struct.id in seen or depth > MAX_BTF_ANON_DEPTH:
            return None
        for member in struct.members:
            if member.name == member_name:
                return member
        path_seen = seen | {struct.id}
        for member in struct.members:
            if member.name:
                continue
            nested = self._type_by_id(member.type_id)
            if nested is None or not nested.is_struct_like:
                continue
            found = self._member_in(nested, member_name, path_seen, depth + 1)
            if found is None:
                continue
            return BtfMember(
                name=found.name,
                type_id=found.type_id,
                byte_offset=member.byte_offset + found.byte_offset,
                bit_offset=member.bit_offset + found.bit_offset,
                bitfield_size=found.bitfield_size,
            )
        return None

    def _type_by_id(self, type_id: int) -> Optional[BtfType]:
        # Type ids are 1-based; the parse list preserves definition order.
        if 0 < type_id <= len(self._types):
            return self._types[type_id - 1]
        return None

    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, data: bytes) -> "BtfFile":
        if len(data) < _BTF_HEADER_SIZE:
            raise BtfError(f"BTF header truncated: {len(data)} bytes")

        magic, version, flags, hdr_len, type_off, type_len, str_off, str_len = (
            unpack_from(_BTF_HEADER, data, 0)
        )
        if magic != BTF_MAGIC:
            raise BtfError(f"bad BTF magic 0x{magic:04x} (expected 0x{BTF_MAGIC:04x})")

        header = BtfHeader(
            magic=magic,
            version=version,
            flags=flags,
            hdr_len=hdr_len,
            type_off=type_off,
            type_len=type_len,
            str_off=str_off,
            str_len=str_len,
        )
        if hdr_len < _BTF_HEADER_SIZE:
            raise BtfError(f"BTF hdr_len {hdr_len} smaller than the base header")

        type_begin = hdr_len + type_off
        type_end = type_begin + type_len
        str_begin = hdr_len + str_off
        str_end = str_begin + str_len
        if type_end > len(data):
            raise BtfError(
                f"BTF type section [0x{type_begin:x}, 0x{type_end:x}) "
                f"overruns the {len(data)}-byte blob"
            )
        if str_end > len(data):
            raise BtfError(
                f"BTF string section [0x{str_begin:x}, 0x{str_end:x}) "
                f"overruns the {len(data)}-byte blob"
            )

        strings = data[str_begin:str_end]
        types: List[BtfType] = []
        structs_by_name: Dict[str, BtfType] = {}

        pos = type_begin
        type_id = 0
        while pos < type_end:
            if pos + _BTF_TYPE_SIZE > type_end:
                raise BtfError(
                    f"BTF type record at offset {pos - type_begin} past the "
                    "type section end"
                )
            name_off, info, size_or_type = unpack_from(_BTF_TYPE, data, pos)
            type_id += 1
            if type_id > MAX_BTF_TYPES:
                raise BtfError("BTF type count exceeds the parse cap")
            kind = (info & _BTF_INFO_KIND_MASK) >> _BTF_INFO_KIND_SHIFT
            vlen = info & _BTF_INFO_VLEN_MASK
            kind_flag = bool(info & _BTF_KIND_FLAG)
            name = cls._resolve_name(strings, name_off)
            if vlen > MAX_BTF_VLEN:
                raise BtfError(
                    f"BTF vlen {vlen} at type id {type_id} exceeds the "
                    f"{MAX_BTF_VLEN} cap"
                )

            if kind in _STRUCT_LIKE_KINDS:
                members = cls._parse_members(
                    data, pos + _BTF_TYPE_SIZE, vlen, kind_flag, type_end, strings
                )
                extra = vlen * _BTF_MEMBER_SIZE
                type_record = BtfType(
                    id=type_id,
                    kind=kind,
                    name=name,
                    size_bytes=size_or_type,
                    vlen=vlen,
                    members=members,
                )
                # Last definition wins (matches kernel BTF semantics); anonymous
                # structs/union index under "".
                structs_by_name[name] = type_record
            else:
                skip = _BTF_SKIP.get(kind)
                if skip is None:
                    raise BtfError(f"unsupported BTF kind {kind} at type id {type_id}")
                fixed, per_vlen = skip
                extra = fixed + per_vlen * vlen
                type_record = BtfType(
                    id=type_id,
                    kind=kind,
                    name=name,
                    size_bytes=size_or_type,
                    vlen=vlen,
                )
            if pos + _BTF_TYPE_SIZE + extra > type_end:
                raise BtfError(
                    f"BTF type id {type_id} (vlen {vlen}) overruns the type section"
                )
            types.append(type_record)
            pos += _BTF_TYPE_SIZE + extra

        return cls(header, types, structs_by_name, strings)

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_name(strings: bytes, name_off: int) -> str:
        """Resolve a name from the string section; out-of-range yields ""."""
        if name_off >= len(strings):
            return ""
        end = strings.find(b"\x00", name_off)
        if end == -1:
            end = len(strings)
        raw = strings[name_off:end]
        return raw.decode("utf-8", errors="replace")

    @classmethod
    def _parse_members(
        cls,
        data: bytes,
        start: int,
        vlen: int,
        kind_flag: bool,
        type_end: int,
        strings: bytes,
    ) -> List[BtfMember]:
        members: List[BtfMember] = []
        pos = start
        for _ in range(vlen):
            if pos + _BTF_MEMBER_SIZE > type_end:
                raise BtfError(
                    f"BTF member record past the type section end (vlen {vlen})"
                )
            m_name_off, m_type, offset_word = unpack_from(_BTF_MEMBER, data, pos)
            if kind_flag:
                bitfield_size = offset_word >> _BTF_MEMBER_BITFIELD_SIZE_SHIFT
                bit_offset = offset_word & _BTF_MEMBER_BIT_OFFSET_MASK
            else:
                bitfield_size = 0
                bit_offset = offset_word
            members.append(
                BtfMember(
                    name=cls._resolve_name(strings, m_name_off),
                    type_id=m_type,
                    byte_offset=bit_offset >> 3,
                    bit_offset=bit_offset,
                    bitfield_size=bitfield_size,
                )
            )
            pos += _BTF_MEMBER_SIZE
        return members


def load_btf(path: str) -> BtfFile:
    """Read a BTF blob from disk and parse it.

    Raises :class:`BtfError` for malformed blobs and :class:`OSError` for I/O
    failures -- the caller decides how to surface either.
    """
    with open(path, "rb") as handle:
        return BtfFile.parse(handle.read())


__all__ = [
    "BtfError",
    "BtfFile",
    "BtfHeader",
    "BtfMember",
    "BtfType",
    "load_btf",
]
