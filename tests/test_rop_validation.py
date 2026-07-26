"""Tests for Stage 7.4 typed chain validation and export."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

from paomcp.tools.backends.rop import (
    CHAIN_EXPORT_FORMATS,
    CHAIN_EXPORT_SCHEMA,
    CHAIN_VALIDATION_SCHEMA,
    ChainBuilder,
    ChainValidator,
    ChainValidatorError,
    GadgetIndex,
)


def _record(address: int, instructions: List[str]) -> Dict[str, Any]:
    return {
        "address": address,
        "bytes": b"\x90" * len(instructions),
        "instructions": instructions,
        "arch": "x86_64",
    }


def _loader(
    records: Sequence[Mapping[str, Any]],
) -> Callable[..., Sequence[Mapping[str, Any]]]:
    def _load(binary_path: str, arch: Optional[str], image_base: Optional[int]):
        return list(records)

    return _load


def _tempbin() -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    handle.write(b"BIN")
    handle.close()
    return handle.name


def _standard_chain() -> Sequence[Dict[str, Any]]:
    binary = _tempbin()
    try:
        builder = ChainBuilder(
            gadgets=GadgetIndex(
                loader=_loader(
                    [
                        _record(0x400100, ["pop rdi", "ret"]),
                        _record(0x400800, ["ret"]),
                    ]
                )
            )
        )
        return builder.build_rop_chain(
            binary_path=binary,
            kind="ret2libc",
            target_address=0x400500,
            target_symbol="system",
            args=[0x400600],
        )["parsed"]["entries"]
    finally:
        os.unlink(binary)


class TestValidateChain:
    def test_no_mappings_leaves_exec_check_unknown(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.validate_rop_chain(chain=chain)
        parsed = response["parsed"]
        assert parsed["schema"] == CHAIN_VALIDATION_SCHEMA
        assert parsed["verdict"] == "unknown"
        checks = {c["name"]: c for c in parsed["checks"]}
        assert checks["exec_mapping"]["verdict"] == "unknown"
        assert checks["bad_bytes"]["verdict"] == "pass"

    def test_pass_when_every_gadget_lands_in_executable_mapping(self) -> None:
        chain = _standard_chain()
        mappings = [
            {
                "start": 0x400000,
                "end": 0x401000,
                "permissions": "r-xp",
                "file": "/bin/target",
            }
        ]
        validator = ChainValidator()
        response = validator.validate_rop_chain(
            chain=chain, executable_mappings=mappings
        )
        parsed = response["parsed"]
        checks = {c["name"]: c for c in parsed["checks"]}
        assert checks["exec_mapping"]["verdict"] == "pass"
        assert checks["exec_mapping"]["issues"] == []

    def test_fail_when_address_is_outside_every_mapping(self) -> None:
        chain = _standard_chain()
        mappings = [
            {
                "start": 0x600000,
                "end": 0x601000,
                "permissions": "r-xp",
                "file": "/bin/target",
            }
        ]
        validator = ChainValidator()
        response = validator.validate_rop_chain(
            chain=chain, executable_mappings=mappings
        )
        parsed = response["parsed"]
        assert parsed["verdict"] == "fail"
        exec_check = next(c for c in parsed["checks"] if c["name"] == "exec_mapping")
        assert exec_check["verdict"] == "fail"
        assert any(
            "outside every provided mapping" in issue["reason"]
            for issue in exec_check["issues"]
        )

    def test_fail_when_address_is_in_non_executable_mapping(self) -> None:
        chain = _standard_chain()
        mappings = [
            {
                "start": 0x400000,
                "end": 0x401000,
                "permissions": "rw-p",  # no x
                "file": "/bin/data",
            }
        ]
        validator = ChainValidator()
        response = validator.validate_rop_chain(
            chain=chain, executable_mappings=mappings
        )
        parsed = response["parsed"]
        assert parsed["verdict"] == "fail"
        exec_check = next(c for c in parsed["checks"] if c["name"] == "exec_mapping")
        assert exec_check["verdict"] == "fail"
        assert any(
            "non-executable" in issue["reason"] for issue in exec_check["issues"]
        )

    def test_bad_byte_fails_encoded_pointer(self) -> None:
        # 0x400a00 encoded little-endian contains 0x0a
        binary = _tempbin()
        try:
            builder = ChainBuilder(
                gadgets=GadgetIndex(
                    loader=_loader([_record(0x400A00, ["pop rdi", "ret"])])
                )
            )
            chain = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x400500,
                args=[0x400600],
                align_stack=False,
            )["parsed"]["entries"]
        finally:
            os.unlink(binary)
        validator = ChainValidator()
        response = validator.validate_rop_chain(chain=chain, bad_bytes=[0x0A])
        parsed = response["parsed"]
        bb = next(c for c in parsed["checks"] if c["name"] == "bad_bytes")
        assert bb["verdict"] == "fail"
        assert any(i["slot_index"] == 0 for i in bb["issues"])

    def test_register_clobber_is_warn_not_fail(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.validate_rop_chain(chain=chain)
        clobber = next(
            c for c in response["parsed"]["checks"] if c["name"] == "register_clobber"
        )
        assert clobber["verdict"] in ("pass", "warn")

    def test_overall_pass_requires_all_checks_pass(self) -> None:
        chain = _standard_chain()
        mappings = [
            {
                "start": 0x400000,
                "end": 0x401000,
                "permissions": "r-xp",
                "file": "/bin/target",
            }
        ]
        validator = ChainValidator()
        response = validator.validate_rop_chain(
            chain=chain, executable_mappings=mappings, bad_bytes=[0xFF]
        )
        parsed = response["parsed"]
        # register_clobber may be warn; that does not fail the chain
        assert parsed["verdict"] in ("pass", "warn")

    def test_bad_bytes_field_is_normalized(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.validate_rop_chain(chain=chain, bad_bytes=["0x0a", 0x0D])
        bb = next(c for c in response["parsed"]["checks"] if c["name"] == "bad_bytes")
        assert bb["bad_bytes"] == [0x0A, 0x0D]


class TestExportChain:
    def test_raw_export_returns_hex_bytes(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.export_chain(chain=chain, format="raw")
        parsed = response["parsed"]
        assert parsed["schema"] == CHAIN_EXPORT_SCHEMA
        assert parsed["format"] == "raw"
        assert parsed["raw_bytes_length"] == len(chain) * 8
        # hex text and raw_bytes must agree
        assert parsed["text"] == parsed["raw_bytes"]

    def test_python_export_uses_struct_pack(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.export_chain(
            chain=chain, format="python", variable_name="payload"
        )
        text = response["parsed"]["text"]
        assert "import struct" in text
        assert 'payload = b""' in text
        assert "struct.pack" in text
        # each entry gets a comment with its purpose
        assert "# load rdi" in text
        assert "# call system" in text

    def test_pwntools_export_uses_p64_and_sets_context(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        response = validator.export_chain(chain=chain, format="pwntools")
        text = response["parsed"]["text"]
        assert "from pwn import" in text
        assert "context.arch = 'amd64'" in text
        assert "p64(" in text
        # every non-raw entry is packed
        assert text.count("p64(") == len(chain)

    def test_unsupported_format_raises(self) -> None:
        chain = _standard_chain()
        validator = ChainValidator()
        with pytest.raises(ChainValidatorError, match="unsupported export format"):
            validator.export_chain(chain=chain, format="binary-blob")

    def test_export_formats_constant_shape(self) -> None:
        assert set(CHAIN_EXPORT_FORMATS) == {"raw", "python", "pwntools"}
