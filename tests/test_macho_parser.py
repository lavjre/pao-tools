"""Tests for the pure Mach-O parser (Stage 6.11).

These build synthetic Mach-O images with the exact on-disk header/load-command
format and assert the parser walks them -- the same approach ``test_btf_parser``
uses for BTF.  Default builders emit little-endian 64-bit images; byte-swapped
and 32-bit cases are crafted explicitly.
"""

import struct
from pathlib import Path

import pytest

from paomcp.tools.backends.macho import (
    MH_MAGIC,
    MH_MAGIC_64,
    FatMachOError,
    MachOError,
    MachOFile,
    MachOSegment,
    is_macho_magic,
    load_macho,
    parse_macho,
)

# Load command numbers (the parser's module-level names, spelled locally).
LC_SEGMENT = 0x1
LC_LOAD_DYLIB = 0xC
LC_LOAD_WEAK_DYLIB = 0x18 | 0x80000000
LC_SEGMENT_64 = 0x19
LC_UUID = 0x1B
LC_LAZY_LOAD_DYLIB = 0x20
LC_REEXPORT_DYLIB = 0x1F | 0x80000000
LC_MAIN = 0x80000028
LC_BUILD_VERSION = 0x32


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def make_load_command(cmd: int, payload: bytes, *, cmdsize: int | None = None) -> bytes:
    size = cmdsize if cmdsize is not None else 8 + len(payload)
    return struct.pack("<II", cmd, size) + payload


def _count_commands(blob: bytes) -> int:
    n, pos = 0, 0
    while pos < len(blob):
        n += 1
        _, size = struct.unpack_from("<II", blob, pos)
        pos += size
    return n


def make_macho_64(
    commands: bytes = b"",
    *,
    cputype: int = 0x01000007,
    cpusubtype: int = 3,
    filetype: int = 2,
    flags: int = 0,
) -> bytes:
    ncmds = _count_commands(commands)
    header = struct.pack(
        "<8I",
        MH_MAGIC_64,
        cputype,
        cpusubtype,
        filetype,
        ncmds,
        len(commands),
        flags,
        0,
    )
    return header + commands


def seg64(
    *,
    name: str = "__TEXT",
    vmaddr: int = 0x100000000,
    vmsize: int = 0x4000,
    initprot: int = 5,
) -> bytes:
    payload = name.encode().ljust(16, b"\x00")
    payload += struct.pack("<QQQQIIII", vmaddr, vmsize, 0, vmsize, 7, initprot, 0, 0)
    return make_load_command(LC_SEGMENT_64, payload)


def seg32(*, name: str = "__TEXT", vmaddr: int = 0x1000, initprot: int = 5) -> bytes:
    payload = name.encode().ljust(16, b"\x00")
    payload += struct.pack("<IIIIIIII", vmaddr, 0x4000, 0, 0x4000, 7, initprot, 0, 0)
    return make_load_command(LC_SEGMENT, payload)


def lc_main(*, entryoff: int = 0x1000, stacksize: int = 0) -> bytes:
    return make_load_command(LC_MAIN, struct.pack("<QQ", entryoff, stacksize))


def load_dylib(name: str, *, cmd: int = LC_LOAD_DYLIB) -> bytes:
    struct_dylib = struct.pack("<IIII", 24, 0, 0x1000, 0x1000)  # name lc_str=24
    return make_load_command(cmd, struct_dylib + name.encode() + b"\x00")


def uuid_cmd(uuid_bytes: bytes = bytes(range(16))) -> bytes:
    return make_load_command(LC_UUID, uuid_bytes)


def build_version_cmd(*, platform: int = 7, tools=()) -> bytes:
    payload = struct.pack("<IIII", platform, 0x000F0000, 0x000F0000, len(tools))
    payload += b"".join(struct.pack("<II", t, v) for t, v in tools)
    return make_load_command(LC_BUILD_VERSION, payload)


# ---------------------------------------------------------------------------
class TestIsMachoMagic:
    def test_64_magic(self):
        assert is_macho_magic(make_macho_64())

    def test_32_magic(self):
        assert is_macho_magic(b"\xce\xfa\xed\xfe" + b"\x00" * 24)

    def test_byte_swapped_64_magic(self):
        # MH_CIGAM_64 stored little-endian: FE ED FA CF.
        assert is_macho_magic(b"\xfe\xed\xfa\xcf" + b"\x00" * 28)

    def test_fat_magic(self):
        assert is_macho_magic(b"\xca\xfe\xba\xbe" + b"\x00" * 8)

    def test_short_input(self):
        assert not is_macho_magic(b"\xce\xfa")

    def test_not_macho(self):
        assert not is_macho_magic(b"\x7fELF" + b"\x00" * 8)
        assert not is_macho_magic(b"")


# ---------------------------------------------------------------------------
class TestParseHeader:
    def test_64_bit_fields(self):
        image = make_macho_64(cputype=0x01000007, filetype=2, flags=0x00200085)
        macho = parse_macho(image)
        assert macho.magic == MH_MAGIC_64
        assert macho.is_64
        assert macho.cputype == 0x01000007
        assert macho.cpusubtype == 3
        assert macho.filetype == 2
        assert macho.flags == 0x00200085
        assert macho.ncmds == 0
        assert macho.cpu_name == "x86_64"
        assert macho.filetype_name == "MH_EXECUTE"

    def test_32_bit_header(self):
        image = struct.pack("<7I", MH_MAGIC, 0x7, 3, 2, 0, 0, 0)
        macho = parse_macho(image)
        assert not macho.is_64
        assert macho.cputype == 0x7
        assert macho.cpu_name == "i386"
        assert macho.filetype_name == "MH_EXECUTE"

    def test_byte_swapped_header(self):
        # Big-endian stored MH_MAGIC_64, arm64.
        image = struct.pack(">8I", MH_MAGIC_64, 0x0100000C, 0, 2, 0, 0, 0, 0)
        macho = parse_macho(image)
        assert macho.is_64
        assert macho.magic == MH_MAGIC_64
        assert macho.cputype == 0x0100000C
        assert macho.cpu_name == "arm64"

    def test_empty_defaults(self):
        macho = parse_macho(make_macho_64())
        assert macho.entry is None
        assert macho.entry_stacksize is None
        assert macho.uuid is None
        assert macho.segments == []
        assert macho.dylibs == []


# ---------------------------------------------------------------------------
class TestEntry:
    def test_lc_main_entry(self):
        macho = parse_macho(
            make_macho_64(commands=lc_main(entryoff=0x1234, stacksize=0x8000))
        )
        assert macho.entry == 0x1234
        assert macho.entry_stacksize == 0x8000

    def test_no_entry_for_dylib(self):
        image = make_macho_64(
            filetype=6, commands=load_dylib("/usr/lib/libSystem.B.dylib")
        )
        macho = parse_macho(image)
        assert macho.filetype_name == "MH_DYLIB"
        assert macho.entry is None


# ---------------------------------------------------------------------------
class TestDylibs:
    def test_load_dylib_names(self):
        commands = b"".join(
            [
                load_dylib("/usr/lib/libSystem.B.dylib"),
                load_dylib("/usr/lib/libobjc.A.dylib"),
            ]
        )
        macho = parse_macho(make_macho_64(commands=commands))
        assert macho.dylibs == [
            "/usr/lib/libSystem.B.dylib",
            "/usr/lib/libobjc.A.dylib",
        ]
        assert macho.links_libsystem
        # libSystem.B.dylib re-exports libsystem_malloc, so a transitive
        # linker still counts as linking the allocator.
        assert macho.links_libsystem_malloc

    def test_libsystem_malloc_direct(self):
        macho = parse_macho(
            make_macho_64(commands=load_dylib("/usr/lib/system/libsystem_malloc.dylib"))
        )
        assert macho.links_libsystem_malloc

    def test_reexport_dylib_included(self):
        macho = parse_macho(
            make_macho_64(
                commands=load_dylib("/usr/lib/libSystem.B.dylib", cmd=LC_REEXPORT_DYLIB)
            )
        )
        assert macho.dylibs == ["/usr/lib/libSystem.B.dylib"]

    def test_dylib_name_overrun_raises(self):
        # name_off points exactly at the end of the command -> no room for a string.
        command = make_load_command(
            LC_LOAD_DYLIB, struct.pack("<IIII", 24, 0, 0, 0), cmdsize=24
        )
        with pytest.raises(MachOError, match="outside"):
            parse_macho(make_macho_64(commands=command))

    def test_dylib_truncated_raises(self):
        command = make_load_command(LC_LOAD_DYLIB, b"\x00" * 8, cmdsize=16)
        with pytest.raises(MachOError, match="truncated"):
            parse_macho(make_macho_64(commands=command))


# ---------------------------------------------------------------------------
class TestSegments:
    def test_segment_64(self):
        macho = parse_macho(
            make_macho_64(
                commands=seg64(
                    name="__TEXT", vmaddr=0x100000000, vmsize=0x4000, initprot=5
                )
            )
        )
        assert macho.segments == [MachOSegment("__TEXT", 0x100000000, 0x4000, 5)]

    def test_segment_32(self):
        image = make_macho_64(commands=seg32(name="__TEXT", vmaddr=0x1000, initprot=5))
        macho = parse_macho(image)
        assert macho.segments == [MachOSegment("__TEXT", 0x1000, 0x4000, 5)]

    def test_segment_truncated_raises(self):
        command = make_load_command(LC_SEGMENT_64, b"\x00" * 8, cmdsize=16)
        with pytest.raises(MachOError, match="truncated"):
            parse_macho(make_macho_64(commands=command))


# ---------------------------------------------------------------------------
class TestUuid:
    def test_uuid_format(self):
        raw = bytes(range(16))
        macho = parse_macho(make_macho_64(commands=uuid_cmd(raw)))
        assert macho.uuid == "00010203-0405-0607-0809-0A0B0C0D0E0F"

    def test_uuid_absent(self):
        assert parse_macho(make_macho_64()).uuid is None


# ---------------------------------------------------------------------------
class TestUnknownCommands:
    def test_unknown_command_skipped(self):
        commands = build_version_cmd(platform=7, tools=[(0x3, 0x10000)]) + lc_main()
        macho = parse_macho(make_macho_64(commands=commands))
        assert macho.ncmds == 2
        assert macho.entry == 0x1000
        assert macho.segments == []
        assert macho.dylibs == []

    def test_mixed_command_walk(self):
        commands = b"".join(
            [
                make_load_command(0x80000033, b"\x00" * 32),
                seg64(name="__DATA", vmaddr=0x100004000, initprot=3),
                load_dylib("/usr/lib/libSystem.B.dylib"),
                uuid_cmd(bytes(range(16))),
                lc_main(entryoff=0x2000),
            ]
        )
        macho = parse_macho(make_macho_64(commands=commands))
        assert macho.ncmds == 5
        assert macho.segments[0].name == "__DATA"
        assert macho.dylibs == ["/usr/lib/libSystem.B.dylib"]
        assert macho.uuid is not None
        assert macho.entry == 0x2000


# ---------------------------------------------------------------------------
class TestErrors:
    def test_bad_magic(self):
        with pytest.raises(MachOError, match="not a Mach-O"):
            parse_macho(b"\x7fELF" + b"\x00" * 40)

    def test_fat_binary_raises(self):
        # A fat image is Mach-O; the parser refuses the slice with the
        # dedicated FatMachOError so callers can distinguish it from a blob.
        with pytest.raises(FatMachOError, match="fat"):
            parse_macho(b"\xca\xfe\xba\xbe" + b"\x00" * 64)

    def test_byte_swapped_fat_raises_fat_macho_error(self):
        with pytest.raises(FatMachOError, match="fat"):
            parse_macho(b"\xbe\xba\xfe\xca" + b"\x00" * 64)

    def test_truncated_header_raises(self):
        with pytest.raises(MachOError, match="header truncated"):
            parse_macho(b"\xcf\xfa\xed\xfe" + b"\x00" * 20)

    def test_short_input_raises(self):
        with pytest.raises(MachOError, match="header truncated"):
            parse_macho(b"\xce\xfa")

    def test_command_table_overruns(self):
        header = struct.pack("<8I", MH_MAGIC_64, 0x01000007, 3, 2, 1, 0, 0, 0)
        with pytest.raises(MachOError, match="overruns"):
            parse_macho(header)

    def test_command_past_file_end(self):
        command = make_load_command(LC_SEGMENT_64, b"", cmdsize=72)  # claims 72 bytes
        header = struct.pack("<8I", MH_MAGIC_64, 0x01000007, 3, 2, 1, 72, 0, 0)
        with pytest.raises(MachOError, match="past the file end"):
            parse_macho(header + command)

    def test_invalid_cmdsize(self):
        command = make_load_command(LC_SEGMENT_64, b"", cmdsize=4)
        header = struct.pack("<8I", MH_MAGIC_64, 0x01000007, 3, 2, 1, 4, 0, 0)
        with pytest.raises(MachOError, match="invalid cmdsize 4"):
            parse_macho(header + command)


# ---------------------------------------------------------------------------
class TestLoadMacho:
    def test_load_macho_from_file(self, tmp_path: Path):
        image = make_macho_64(commands=lc_main(entryoff=0x1234))
        path = tmp_path / "chal"
        path.write_bytes(image)
        macho = load_macho(str(path))
        assert macho.entry == 0x1234
        assert macho.is_64

    def test_load_macho_missing_file_raises_oserror(self, tmp_path: Path):
        with pytest.raises(OSError):
            load_macho(str(tmp_path / "does-not-exist"))


# ---------------------------------------------------------------------------
class TestLinksLibsystem:
    def _macho(self, *names):
        commands = b"".join(load_dylib(name) for name in names)
        return parse_macho(make_macho_64(commands=commands))

    def test_libsystem_b_implies_libmalloc(self):
        # libSystem.B.dylib re-exports libsystem_malloc, so linking it means
        # the binary transitively links the allocator even when the allocator
        # dylib never appears in the load-command list.
        macho = self._macho("/usr/lib/libSystem.B.dylib")
        assert macho.links_libsystem is True
        assert macho.links_libsystem_malloc is True

    def test_direct_libsystem_malloc(self):
        macho = self._macho("/usr/lib/system/libsystem_malloc.dylib")
        assert macho.links_libsystem_malloc is True

    def test_versioned_libsystem_malloc(self):
        macho = self._macho("/usr/lib/system/libsystem_malloc.dylib.1")
        assert macho.links_libsystem_malloc is True

    def test_suffixed_libsystem_malloc(self):
        macho = self._macho("/usr/lib/system/libsystem_malloc-2.0.dylib")
        assert macho.links_libsystem_malloc is True

    def test_no_false_positive_on_lookalike_name(self):
        # The old `"libsystem_malloc" in name` substring check would accept
        # this; the anchored regex must not.
        macho = self._macho("/usr/lib/evil_libsystem_malloc_eater.dylib")
        assert macho.links_libsystem is False
        assert macho.links_libsystem_malloc is False

    def test_unrelated_dylib(self):
        macho = self._macho("/usr/lib/libz.dylib")
        assert macho.links_libsystem is False
        assert macho.links_libsystem_malloc is False


# ---------------------------------------------------------------------------
class TestAnalyzeMachoEnvelope:
    def test_fat_image_is_fat_macho_verdict(self, tmp_path: Path):
        from paomcp.tools.macho import _macho_envelope

        path = tmp_path / "fat"
        path.write_bytes(b"\xca\xfe\xba\xbe" + b"\x00" * 64)
        parsed = _macho_envelope(str(path))["parsed"]
        assert parsed["verdict"] == "fat_macho"
        assert parsed["is_macho"] is True
        assert "slice" in parsed["reason"]

    def test_non_macho_is_not_macho_verdict(self, tmp_path: Path):
        from paomcp.tools.macho import _macho_envelope

        path = tmp_path / "notmacho"
        path.write_bytes(b"\x7fELF" + b"\x00" * 40)
        parsed = _macho_envelope(str(path))["parsed"]
        assert parsed["verdict"] == "not_macho"
        assert parsed["is_macho"] is False
