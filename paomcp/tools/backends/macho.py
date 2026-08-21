"""Pure Mach-O parser -- stdlib only.

The repo's binary analysis is ELF-only today (pyelftools +
``StaticBinaryAnalyzer``); this module adds just enough Mach-O awareness for
Apple targets: the file header, the load-command walk, ``__TEXT`` segments, the
``LC_MAIN`` entry point, loaded dylibs, and the build UUID.

Mach-O is the native object/executable format on macOS/iOS.  Unlike ELF it is
*not* self-describing through section tables alone -- the load commands in the
header walk are what name the segments and the entry point.  libsystem_malloc
(the target of the libmalloc analyzer) is always pulled in through
``libSystem.B.dylib`` (or re-exported from it), so the ``dylibs`` list is what
lets the analyzer decide a binary is an Apple-target userspace process.

This module mirrors ``btf.py``: a small, table-driven, read-only parser that
raises a single typed error and never leaks an exception past the loader.
Both little-endian and byte-swapped (big-endian) images are handled by reading
the magic in file byte order.  Universal (fat) binaries are recognized by
:func:`is_macho_magic` but refused by :func:`parse_macho` -- slice extraction
is out of scope.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from struct import unpack_from
from typing import Dict, List, Optional

# Canonical Mach-O magic values.  MH_CIGAM is the byte-swapped form; the
# canonical 32/64-bit values are what ``magic`` normalizes to after byte-order
# detection, whichever endianness the file is stored in.
MH_MAGIC = 0xFEEDFACE
MH_CIGAM = 0xCEFAEDFE
MH_MAGIC_64 = 0xFEEDFACF
MH_CIGAM_64 = 0xCFFAEDFE
FAT_MAGIC = 0xCAFEBABE
FAT_CIGAM = 0xBEBAFECA

_KNOWN_MAGICS = frozenset(
    {MH_MAGIC, MH_CIGAM, MH_MAGIC_64, MH_CIGAM_64, FAT_MAGIC, FAT_CIGAM}
)

# A libsystem_malloc dylib name, anchored the same way the heap family
# detector anchors it (_FAMILY_FILENAME_PATTERNS): a path component that is
# exactly ``libsystem_malloc`` followed by ``.dylib`` (optionally versioned)
# or a ``-<suffix>.dylib`` build-name variant.  The loose ``"libsystem_malloc"
# in name`` substring check it replaces would accept
# ``evil_libsystem_malloc_eater.dylib``.
_LIBSYSTEM_MALLOC_RE = re.compile(
    r"(?:^|/)libsystem_malloc(?:\.dylib(?:\.\d+)?|-[^/]*\.dylib)"
)

_MACHO_HEADER_64 = 32
_MACHO_HEADER_32 = 28

# Load commands we understand (cmd field).
LC_SEGMENT = 0x1
LC_LOAD_DYLIB = 0xC
LC_LOAD_WEAK_DYLIB = 0x18 | 0x80000000
LC_SEGMENT_64 = 0x19
LC_UUID = 0x1B
LC_LAZY_LOAD_DYLIB = 0x20
LC_REEXPORT_DYLIB = 0x1F | 0x80000000
LC_LOAD_UPWARD_DYLIB = 0x23 | 0x80000000
LC_MAIN = 0x28 | 0x80000000

# Every command that pulls a dylib in as a dependency.  LC_ID_DYLIB is the
# dylib's *own* install name, not a dependency, so it is deliberately absent.
_DYLIB_LOAD_COMMANDS = frozenset(
    {
        LC_LOAD_DYLIB,
        LC_LOAD_WEAK_DYLIB,
        LC_LAZY_LOAD_DYLIB,
        LC_REEXPORT_DYLIB,
        LC_LOAD_UPWARD_DYLIB,
    }
)

# segment_command_64 / segment_command fixed header sizes.
_SEGMENT_64_SIZE = 72
_SEGMENT_32_SIZE = 56

CPU_TYPE_NAMES: Dict[int, str] = {
    0x7: "i386",
    0x01000007: "x86_64",
    0xC: "arm",
    0x0100000C: "arm64",
    0x0200000C: "arm64_32",
    0x12: "powerpc",
    0x01000012: "powerpc64",
}

MH_FILETYPE_NAMES: Dict[int, str] = {
    0x1: "MH_OBJECT",
    0x2: "MH_EXECUTE",
    0x3: "MH_FIXUP",
    0x4: "MH_CORE",
    0x5: "MH_PRELOAD",
    0x6: "MH_DYLIB",
    0x7: "MH_DYLINKER",
    0x8: "MH_BUNDLE",
    0xA: "MH_DSYM",
}


class MachOError(ValueError):
    """Raised when the Mach-O image is malformed (bad magic, truncation, ...)."""


class FatMachOError(MachOError):
    """Raised for a universal (fat) Mach-O image.

    A fat image *is* Mach-O -- callers that distinguish it from a non-Mach-O
    blob can catch this subclass and report ``is_macho=True`` while noting the
    slice is not parsed.
    """


@dataclass(frozen=True)
class MachOSegment:
    """One ``LC_SEGMENT(_64)`` entry: name, VM range, and initial protection."""

    name: str
    vmaddr: int
    vmsize: int
    initprot: int


@dataclass(frozen=True)
class MachOFile:
    """Parsed Mach-O image (64-bit emphasis).

    ``entry`` is the ``LC_MAIN`` ``entryoff`` -- a file (VM) offset into the
    ``__TEXT`` segment, not a loaded address -- and ``None`` for dylibs and
    bundles, which have no entry point command.
    """

    magic: int
    is_64: bool
    cputype: int
    cpusubtype: int
    filetype: int
    ncmds: int
    sizeofcmds: int
    flags: int
    entry: Optional[int] = None
    entry_stacksize: Optional[int] = None
    uuid: Optional[str] = None
    segments: List[MachOSegment] = field(default_factory=list)
    dylibs: List[str] = field(default_factory=list)

    @property
    def cpu_name(self) -> str:
        return CPU_TYPE_NAMES.get(self.cputype, f"0x{self.cputype:x}")

    @property
    def filetype_name(self) -> str:
        return MH_FILETYPE_NAMES.get(self.filetype, f"0x{self.filetype:x}")

    @property
    def links_libsystem(self) -> bool:
        return any(name.endswith("libSystem.B.dylib") for name in self.dylibs)

    @property
    def links_libsystem_malloc(self) -> bool:
        # libSystem.B.dylib re-exports libsystem_malloc (it is a re-export
        # stub), so an Apple-target userspace binary that links libSystem.B
        # transitively links the allocator even when libsystem_malloc itself
        # never appears in the load-command list.  Direct links are matched
        # with the anchored _LIBSYSTEM_MALLOC_RE, never a substring.
        return self.links_libsystem or any(
            _LIBSYSTEM_MALLOC_RE.search(name) for name in self.dylibs
        )


def is_macho_magic(data: bytes) -> bool:
    """True when the 4-byte prefix is any Mach-O magic (incl. byte-swapped/fat)."""
    if len(data) < 4:
        return False
    raw = data[:4]
    return (
        int.from_bytes(raw, "big") in _KNOWN_MAGICS
        or int.from_bytes(raw, "little") in _KNOWN_MAGICS
    )


def parse_macho(data: bytes) -> MachOFile:
    if len(data) < 4:
        raise MachOError(f"Mach-O header truncated: {len(data)} bytes")

    raw = data[:4]
    be = int.from_bytes(raw, "big")
    le = int.from_bytes(raw, "little")
    if le in (MH_MAGIC, MH_MAGIC_64):
        endian, magic = "<", le
    elif be in (MH_MAGIC, MH_MAGIC_64):
        endian, magic = ">", be
    elif le in (FAT_MAGIC, FAT_CIGAM):
        raise FatMachOError("universal (fat) Mach-O: slice parsing not supported")
    else:
        raise MachOError(f"not a Mach-O image: magic 0x{le:08x}")

    is_64 = magic == MH_MAGIC_64
    header_size = _MACHO_HEADER_64 if is_64 else _MACHO_HEADER_32
    if len(data) < header_size:
        raise MachOError(f"Mach-O header truncated: {len(data)} bytes")

    if is_64:
        magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags, _reserved = (
            unpack_from(endian + "8I", data, 0)
        )
    else:
        magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags = unpack_from(
            endian + "7I", data, 0
        )

    segments: List[MachOSegment] = []
    dylibs: List[str] = []
    entry: Optional[int] = None
    entry_stacksize: Optional[int] = None
    uuid: Optional[str] = None

    pos = header_size
    for _ in range(ncmds):
        if pos + 8 > len(data):
            raise MachOError(f"load command table overruns the file at 0x{pos:x}")
        cmd, cmdsize = unpack_from(endian + "II", data, pos)
        if cmdsize < 8:
            raise MachOError(f"load command {cmd:#x} has invalid cmdsize {cmdsize}")
        if pos + cmdsize > len(data):
            raise MachOError(
                f"load command {cmd:#x} (size {cmdsize}) past the file end"
            )
        if cmd in (LC_SEGMENT, LC_SEGMENT_64):
            segments.append(
                _parse_segment(endian, data, pos, cmdsize, cmd == LC_SEGMENT_64)
            )
        elif cmd == LC_MAIN:
            if cmdsize < 24:
                raise MachOError("LC_MAIN command truncated")
            entry, entry_stacksize = unpack_from(endian + "QQ", data, pos + 8)
        elif cmd in _DYLIB_LOAD_COMMANDS:
            dylibs.append(_parse_dylib_name(endian, data, pos, cmdsize))
        elif cmd == LC_UUID:
            if cmdsize < 24:
                raise MachOError("LC_UUID command truncated")
            uuid = _format_uuid(data[pos + 8 : pos + 24])
        pos += cmdsize

    return MachOFile(
        magic=magic,
        is_64=is_64,
        cputype=cputype,
        cpusubtype=cpusubtype,
        filetype=filetype,
        ncmds=ncmds,
        sizeofcmds=sizeofcmds,
        flags=flags,
        entry=entry,
        entry_stacksize=entry_stacksize,
        uuid=uuid,
        segments=segments,
        dylibs=dylibs,
    )


def load_macho(path: str) -> MachOFile:
    """Read a Mach-O image from disk and parse it.

    Raises :class:`MachOError` for malformed images and :class:`OSError` for
    I/O failures -- the caller decides how to surface either.
    """
    with open(path, "rb") as handle:
        return parse_macho(handle.read())


# ---------------------------------------------------------------------------
def _decode_cstring(data: bytes, start: int, end: int) -> str:
    """Decode a NUL-terminated string strictly confined to ``[start, end)``."""
    if start >= end:
        raise MachOError("string offset outside its load command")
    term = data.find(b"\x00", start, end)
    if term == -1:
        raise MachOError("unterminated string past its load command")
    return data[start:term].decode("utf-8", errors="replace")


def _parse_segment(
    endian: str, data: bytes, pos: int, cmdsize: int, is_64: bool
) -> MachOSegment:
    fixed = _SEGMENT_64_SIZE if is_64 else _SEGMENT_32_SIZE
    if cmdsize < fixed:
        raise MachOError(f"segment command truncated: cmdsize {cmdsize} < {fixed}")
    raw_name = data[pos + 8 : pos + 24]
    name = raw_name.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    if is_64:
        vmaddr, vmsize, _fileoff, _filesize, _maxprot, initprot, _nsects, _flags = (
            unpack_from(endian + "QQQQIIII", data, pos + 0x18)
        )
    else:
        vmaddr, vmsize, _fileoff, _filesize, _maxprot, initprot, _nsects, _flags = (
            unpack_from(endian + "IIIIIIII", data, pos + 0x18)
        )
    return MachOSegment(name=name, vmaddr=vmaddr, vmsize=vmsize, initprot=initprot)


def _parse_dylib_name(endian: str, data: bytes, pos: int, cmdsize: int) -> str:
    if cmdsize < 24:
        raise MachOError("dylib load command truncated")
    name_off = unpack_from(endian + "I", data, pos + 8)[0]
    return _decode_cstring(data, pos + name_off, pos + cmdsize)


def _format_uuid(raw: bytes) -> str:
    hexed = raw.hex().upper()
    return (
        f"{hexed[0:8]}-{hexed[8:12]}-{hexed[12:16]}-" f"{hexed[16:20]}-{hexed[20:32]}"
    )


__all__ = [
    "FAT_MAGIC",
    "MH_CIGAM",
    "MH_CIGAM_64",
    "MH_MAGIC",
    "MH_MAGIC_64",
    "MH_FILETYPE_NAMES",
    "FatMachOError",
    "MachOError",
    "MachOFile",
    "MachOSegment",
    "is_macho_magic",
    "load_macho",
    "parse_macho",
]
