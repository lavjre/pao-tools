import os
import shutil
import struct
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationCancelledError, OperationRegistry
from paomcp.tools.backends.static_analysis import StaticBinaryAnalyzer
from paomcp.tools.static_analysis import _resolve_workspace_file, _stream_all_strings


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _string_table(values):
    data = bytearray(b"\x00")
    offsets = {}
    for value in values:
        offsets[value] = len(data)
        data.extend(value.encode("ascii") + b"\x00")
    return bytes(data), offsets


def _build_fixture(
    path: Path,
    *,
    needed: str = "libc.so.6",
    marker: str = "fixture-one",
    build_seed: int = 0,
) -> None:
    """Build a deterministic ELF64 fixture without requiring a host compiler."""
    base = 0x123456789ABC0000
    text_offset = 0x1000
    text_address = base + text_offset
    text = b"\x90\x90\xc3"
    rodata = b"hello-static\x00" + marker.encode("ascii") + b"\x00"
    dynstr, dyn_offsets = _string_table(
        [needed, "sample.so", "puts", "main", "/opt/sample/lib"]
    )
    dynsym = b"".join(
        [
            struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0),
            struct.pack("<IBBHQQ", dyn_offsets["puts"], 0x12, 0, 0, 0, 0),
            struct.pack(
                "<IBBHQQ",
                dyn_offsets["main"],
                0x12,
                0,
                1,
                text_address,
                len(text),
            ),
        ]
    )
    relocation = struct.pack("<QQq", text_address + 1, (1 << 32) | 4, -4)
    dynamic = b"".join(
        [
            struct.pack("<QQ", 1, dyn_offsets[needed]),
            struct.pack("<QQ", 14, dyn_offsets["sample.so"]),
            struct.pack("<QQ", 29, dyn_offsets["/opt/sample/lib"]),
            struct.pack("<QQ", 0, 0),
        ]
    )
    build_id = bytes((build_seed + index) & 0xFF for index in range(20))
    note = struct.pack("<III", 4, len(build_id), 3) + b"GNU\x00" + build_id
    debug_name = b"sample.debug\x00"
    debug_link = debug_name + b"\x00" * ((_align(len(debug_name), 4) - len(debug_name)))
    debug_link += struct.pack("<I", 0xAABBCCDD)

    section_names = [
        ".text",
        ".rodata",
        ".dynstr",
        ".dynsym",
        ".rela.plt",
        ".dynamic",
        ".note.gnu.build-id",
        ".gnu_debuglink",
        ".debug_info",
        ".debug_abbrev",
        ".shstrtab",
    ]
    shstrtab, sh_name_offsets = _string_table(section_names)
    contents = [
        (".text", text, 16),
        (".rodata", rodata, 1),
        (".dynstr", dynstr, 1),
        (".dynsym", dynsym, 8),
        (".rela.plt", relocation, 8),
        (".dynamic", dynamic, 8),
        (".note.gnu.build-id", note, 4),
        (".gnu_debuglink", debug_link, 4),
        (".debug_info", b"", 1),
        (".debug_abbrev", b"", 1),
        (".shstrtab", shstrtab, 1),
    ]

    payload = bytearray(b"\x00" * text_offset)
    locations = {}
    for name, data, alignment in contents:
        offset = _align(len(payload), alignment)
        payload.extend(b"\x00" * (offset - len(payload)))
        locations[name] = (offset, len(data), alignment)
        payload.extend(data)
    section_header_offset = _align(len(payload), 8)
    payload.extend(b"\x00" * (section_header_offset - len(payload)))

    def section_header(
        name,
        section_type,
        flags,
        *,
        link=0,
        info=0,
        entry_size=0,
    ):
        offset, size, alignment = locations[name]
        address = base + offset if flags & 0x2 else 0
        return struct.pack(
            "<IIQQQQIIQQ",
            sh_name_offsets[name],
            section_type,
            flags,
            address,
            offset,
            size,
            link,
            info,
            alignment,
            entry_size,
        )

    section_headers = [b"\x00" * 64]
    section_headers.extend(
        [
            section_header(".text", 1, 0x6),
            section_header(".rodata", 1, 0x2),
            section_header(".dynstr", 3, 0x2),
            section_header(".dynsym", 11, 0x2, link=3, info=1, entry_size=24),
            section_header(".rela.plt", 4, 0x2, link=4, info=1, entry_size=24),
            section_header(".dynamic", 6, 0x3, link=3, entry_size=16),
            section_header(".note.gnu.build-id", 7, 0x2),
            section_header(".gnu_debuglink", 1, 0),
            section_header(".debug_info", 1, 0),
            section_header(".debug_abbrev", 1, 0),
            section_header(".shstrtab", 3, 0),
        ]
    )
    payload.extend(b"".join(section_headers))
    final_size = len(payload)

    ident = b"\x7fELF" + bytes([2, 1, 1, 0, 0]) + b"\x00" * 7
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        ident,
        3,
        62,
        1,
        text_address,
        64,
        section_header_offset,
        0,
        64,
        56,
        2,
        64,
        len(section_headers),
        11,
    )
    dynamic_offset, dynamic_size, _ = locations[".dynamic"]
    program_headers = b"".join(
        [
            struct.pack(
                "<IIQQQQQQ",
                1,
                5,
                0,
                base,
                base,
                final_size,
                final_size,
                0x1000,
            ),
            struct.pack(
                "<IIQQQQQQ",
                2,
                6,
                dynamic_offset,
                base + dynamic_offset,
                base + dynamic_offset,
                dynamic_size,
                dynamic_size,
                8,
            ),
        ]
    )
    payload[:64] = header
    payload[64 : 64 + len(program_headers)] = program_headers
    path.write_bytes(payload)


def _strip_sections_and_use_segment_metadata(path: Path) -> None:
    """Turn the fixture into a sectionless ELF with dynamic/note segments."""
    payload = bytearray(path.read_bytes())
    base = 0x123456789ABC0000
    dynstr = b"\x00libc.so.6\x00sample.so\x00puts\x00main\x00/opt/sample/lib\x00"
    dynstr_offset = payload.find(dynstr)
    assert dynstr_offset >= 0

    second_program_header = 64 + 56
    dynamic_offset = struct.unpack_from("<Q", payload, second_program_header + 8)[0]
    needed_offset = dynstr.index(b"libc.so.6")
    dynamic = b"".join(
        [
            struct.pack("<QQ", 5, base + dynstr_offset),
            struct.pack("<QQ", 10, len(dynstr)),
            struct.pack("<QQ", 1, needed_offset),
            struct.pack("<QQ", 0, 0),
        ]
    )
    payload[dynamic_offset : dynamic_offset + len(dynamic)] = dynamic

    build_id = bytes(range(20))
    note_offset = payload.find(b"GNU\x00" + build_id) - 12
    assert note_offset >= 0
    third_program_header = 64 + 2 * 56
    payload[third_program_header : third_program_header + 56] = struct.pack(
        "<IIQQQQQQ",
        4,
        4,
        note_offset,
        base + note_offset,
        base + note_offset,
        36,
        36,
        4,
    )

    struct.pack_into("<H", payload, 56, 3)  # e_phnum
    struct.pack_into("<Q", payload, 40, 0)  # e_shoff
    struct.pack_into("<H", payload, 60, 0)  # e_shnum
    struct.pack_into("<H", payload, 62, 0)  # e_shstrndx
    path.write_bytes(payload)


def test_static_analysis_returns_complete_typed_elf_model(tmp_path):
    binary = tmp_path / "sample.elf"
    _build_fixture(binary)
    analyzer = StaticBinaryAnalyzer(max_cache_entries=2)

    result = analyzer.analyze(str(binary))
    parsed = result["parsed"]

    assert result["success"] is True
    assert result["fingerprint"]["sha256"] == parsed["fingerprint"]["sha256"]
    assert parsed["elf_header"]["class_bits"] == 64
    assert parsed["elf_header"]["entry"] == "0x123456789abc1000"
    assert parsed["program_headers"]["count"] == 2
    assert parsed["sections"]["entries"][1]["name"] == ".text"
    assert parsed["sections"]["entries"][1]["address"].startswith("0x")
    assert [record["name"] for record in parsed["imports"]] == ["puts"]
    assert [record["name"] for record in parsed["exports"]] == ["main"]
    assert parsed["relocations"]["entries"][0]["symbol"] == "puts"
    assert parsed["relocations"]["entries"][0]["offset"] == ("0x123456789abc1001")
    assert parsed["dynamic"]["needed"] == ["libc.so.6"]
    assert parsed["dynamic"]["runpath"] == ["/opt/sample/lib"]
    assert parsed["dynamic"]["soname"] == "sample.so"
    assert parsed["build_id"] == "000102030405060708090a0b0c0d0e0f10111213"
    assert parsed["debug_link"] == {
        "file": "sample.debug",
        "crc32": "0xaabbccdd",
    }
    assert parsed["dwarf"]["present"] is True
    assert parsed["strings"]["limited_by_caller"] is False
    assert any(item["value"] == "hello-static" for item in parsed["strings"]["entries"])


def test_analysis_cache_is_content_keyed_and_return_values_are_isolated(tmp_path):
    binary = tmp_path / "sample.elf"
    _build_fixture(binary)
    analyzer = StaticBinaryAnalyzer(max_cache_entries=2)

    first = analyzer.analyze(str(binary), include_strings=False)
    first["parsed"]["sections"]["entries"][1]["name"] = "corrupted-by-caller"
    second = analyzer.analyze(str(binary), include_strings=False)
    refreshed = analyzer.analyze(str(binary), include_strings=False, refresh=True)

    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is True
    assert second["parsed"]["sections"]["entries"][1]["name"] == ".text"
    assert refreshed["cache"]["hit"] is False
    assert refreshed["cache"]["refreshed"] is True


def test_string_limit_is_explicit_and_full_results_are_default(tmp_path):
    binary = tmp_path / "sample.elf"
    _build_fixture(binary, marker="another-printable-marker")
    analyzer = StaticBinaryAnalyzer(max_cache_entries=2)

    complete = analyzer.extract_strings(str(binary), min_length=5)
    limited = analyzer.extract_strings(str(binary), min_length=5, limit=1)

    assert complete["strings"]["returned_count"] == complete["strings"]["total_count"]
    assert complete["strings"]["limited_by_caller"] is False
    assert limited["strings"]["returned_count"] == 1
    assert limited["strings"]["total_count"] == complete["strings"]["total_count"]
    assert limited["strings"]["limited_by_caller"] is True
    assert limited["cache"]["hit"] is True
    with pytest.raises(ValueError, match="limit"):
        analyzer.extract_strings(str(binary), limit=0)


def test_limited_string_scan_stays_partial_until_a_complete_result_is_cached(tmp_path):
    binary = tmp_path / "sample.elf"
    _build_fixture(binary, marker="bounded-string-scan")
    analyzer = StaticBinaryAnalyzer(max_cache_entries=2)

    first_limited = analyzer.extract_strings(str(binary), min_length=5, limit=1)
    complete = analyzer.extract_strings(str(binary), min_length=5)
    cached_limited = analyzer.extract_strings(str(binary), min_length=5, limit=1)

    assert first_limited["cache"]["hit"] is False
    assert first_limited["strings"]["returned_count"] == 1
    assert complete["cache"]["hit"] is False
    assert cached_limited["cache"]["hit"] is True
    assert (
        cached_limited["strings"]["total_count"] == complete["strings"]["total_count"]
    )


def test_static_cache_refuses_entries_larger_than_its_byte_budget(tmp_path):
    binary = tmp_path / "sample.elf"
    _build_fixture(binary)
    analyzer = StaticBinaryAnalyzer(max_cache_entries=2, max_cache_bytes=2)

    first = analyzer.analyze(str(binary), include_strings=False)
    second = analyzer.analyze(str(binary), include_strings=False)

    assert first["cache"]["max_bytes"] == 2
    assert first["cache"]["hit"] is False
    assert second["cache"]["hit"] is False


def test_compare_binaries_reports_semantic_changes_without_limits(tmp_path):
    left = tmp_path / "left.elf"
    right = tmp_path / "right.elf"
    _build_fixture(left, needed="libc.so.6", marker="left", build_seed=0)
    _build_fixture(right, needed="libm.so.6", marker="right", build_seed=1)
    analyzer = StaticBinaryAnalyzer(max_cache_entries=4)

    result = analyzer.compare(str(left), str(right))

    assert result["identical"] is False
    needed = result["differences"]["needed_libraries"]
    assert needed["removed"] == ["libc.so.6"]
    assert needed["added"] == ["libm.so.6"]
    assert result["differences"]["build_id"]["changed"] is True
    assert result["differences"]["sections"]["changed"]


def test_compare_identical_copies_hides_filesystem_identity(tmp_path):
    left = tmp_path / "left.elf"
    right = tmp_path / "right.elf"
    _build_fixture(left)
    shutil.copy2(left, right)

    result = StaticBinaryAnalyzer(max_cache_entries=2).compare(str(left), str(right))

    assert result["identical"] is True
    assert result["differences"]["fingerprint"] == {}
    assert set(result["left"]["fingerprint"]) == {
        "sha256",
        "size_bytes",
        "modified_ns",
    }


def test_static_analysis_rejects_non_elf_and_workspace_escape(tmp_path):
    not_elf = tmp_path / "not-elf.bin"
    not_elf.write_bytes(b"plain text")
    analyzer = StaticBinaryAnalyzer(max_cache_entries=1)

    with pytest.raises(ValueError, match="not an ELF"):
        analyzer.analyze(str(not_elf))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.elf"
    _build_fixture(outside)
    with pytest.raises(ValueError):
        _resolve_workspace_file(
            os.path.join("..", "outside.elf"), str(workspace), "binary_path"
        )


def test_sectionless_elf_uses_dynamic_and_note_segments(tmp_path):
    binary = tmp_path / "sectionless.elf"
    _build_fixture(binary)
    _strip_sections_and_use_segment_metadata(binary)

    parsed = StaticBinaryAnalyzer(max_cache_entries=1).analyze(
        str(binary), include_strings=False
    )["parsed"]

    assert parsed["sections"]["count"] == 0
    assert parsed["program_headers"]["count"] == 3
    assert parsed["dynamic"]["needed"] == ["libc.so.6"]
    assert parsed["build_id"] == "000102030405060708090a0b0c0d0e0f10111213"


def test_unlimited_strings_stream_to_lossless_nested_artifact(tmp_path):
    binary = tmp_path / "large-strings.elf"
    _build_fixture(binary)
    with binary.open("ab") as target:
        target.write(b"A" * (2 * 1024 * 1024))
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        inline_response_bytes=128,
        quota_bytes=0,
        min_free_bytes=0,
        cleanup_interval_seconds=0,
    )
    services = SimpleNamespace(
        runtime_paths=SimpleNamespace(runtime_root=str(tmp_path)),
        artifact_store=store,
        operation_registry=OperationRegistry(),
    )

    result = _stream_all_strings(
        services,
        str(binary),
        min_length=4,
        refresh=False,
        tool="extract_binary_strings",
    )

    entries = result["strings"]["entries"]
    assert entries["delivery"] == "artifact"
    assert entries["format"] == "ndjson"
    assert result["strings"]["returned_count"] == result["strings"]["total_count"]
    payload = store.read(entries["artifact_id"], length=store.max_read_bytes)
    assert payload["length"] > 2 * 1024 * 1024
    assert "A" * 1024 in payload["content"]


def test_streamed_string_scan_honors_cancellation_and_removes_partial_file(tmp_path):
    binary = tmp_path / "sample.elf"
    output = tmp_path / "strings.jsonl"
    _build_fixture(binary)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(OperationCancelledError):
        StaticBinaryAnalyzer(max_cache_entries=1).stream_strings_to_file(
            str(binary),
            str(output),
            cancellation_event=cancelled,
        )
    assert not output.exists()


def test_core_static_analysis_honors_cancellation_before_parsing(tmp_path):
    binary = tmp_path / "cancel-core.elf"
    _build_fixture(binary)
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(OperationCancelledError):
        StaticBinaryAnalyzer(max_cache_entries=1).analyze(
            str(binary),
            include_strings=False,
            cancellation_event=cancelled,
        )
