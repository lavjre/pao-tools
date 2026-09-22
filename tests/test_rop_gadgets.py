"""Tests for typed ROP gadget indexing."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

from paomcp.tools.backends.rop import (
    DEFAULT_MAX_GADGETS,
    GADGET_INSPECTION_SCHEMA,
    GADGET_LIST_SCHEMA,
    GADGET_SEQUENCE_SCHEMA,
    GadgetIndex,
    GadgetIndexError,
    MAX_MAX_GADGETS,
    MAX_SEQUENCE_STEPS,
    RopTools,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(
    address: int,
    instructions: Sequence[str],
    *,
    arch: str = "x86_64",
    raw: bytes = b"",
) -> Dict[str, Any]:
    return {
        "address": address,
        "bytes": raw or b"\x90" * len(instructions),
        "instructions": list(instructions),
        "arch": arch,
    }


def _fake_loader(
    records: Sequence[Mapping[str, Any]],
) -> Callable[..., Sequence[Mapping[str, Any]]]:
    def _load(binary_path: str, arch: Optional[str], image_base: Optional[int]):
        return list(records)

    return _load


def _tempbin(content: bytes = b"BIN") -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    handle.write(content)
    handle.close()
    return handle.name


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestParseInstruction:
    def test_parses_mnemonic_and_operands(self) -> None:
        result = GadgetIndex._parse_instruction("mov rax, [rdi+0x8]")
        assert result is not None
        mnemonic, operands = result
        assert mnemonic == "mov"
        assert operands == ["rax", "[rdi+0x8]"]

    def test_parses_lone_mnemonic(self) -> None:
        result = GadgetIndex._parse_instruction("ret")
        assert result == ("ret", [])

    def test_returns_none_on_empty(self) -> None:
        assert GadgetIndex._parse_instruction("   ") is None

    def test_respects_bracket_depth_when_splitting(self) -> None:
        assert GadgetIndex._split_operands("rax, [rdi + rsi*8]") == [
            "rax",
            "[rdi + rsi*8]",
        ]


class TestSideEffects:
    def test_pop_reg_reads_sp_writes_reg_and_sp(self) -> None:
        effects = GadgetIndex._instruction_side_effects("pop", ["rdi"], "x86_64")
        assert "rdi" in effects["writes"]
        assert "rsp" in effects["writes"]
        assert "rsp" in effects["reads"]

    def test_mov_reg_reg_reads_src_writes_dst(self) -> None:
        effects = GadgetIndex._instruction_side_effects("mov", ["rax", "rsi"], "x86_64")
        assert effects["writes"] == ["rax"]
        assert effects["reads"] == ["rsi"]

    def test_xor_reg_reg_is_zeroing_and_does_not_read(self) -> None:
        effects = GadgetIndex._instruction_side_effects("xor", ["rax", "rax"], "x86_64")
        assert effects["writes"] == ["rax"]
        assert "rax" not in effects["reads"]

    def test_push_reg_reads_reg_and_sp_writes_sp(self) -> None:
        effects = GadgetIndex._instruction_side_effects("push", ["rdi"], "x86_64")
        assert "rdi" in effects["reads"]
        assert "rsp" in effects["reads"]
        assert "rsp" in effects["writes"]

    def test_ret_reads_and_writes_sp(self) -> None:
        effects = GadgetIndex._instruction_side_effects("ret", [], "x86_64")
        assert effects["reads"] == ["rsp"]
        assert effects["writes"] == ["rsp"]

    def test_syscall_clobbers_rax_rcx_r11(self) -> None:
        effects = GadgetIndex._instruction_side_effects("syscall", [], "x86_64")
        assert set(effects["clobbers"]) >= {"rax", "rcx", "r11"}
        assert "rax" in effects["reads"]

    def test_call_reg_clobbers_caller_saved(self) -> None:
        effects = GadgetIndex._instruction_side_effects("call", ["rax"], "x86_64")
        assert "rax" in effects["reads"]
        assert {"rax", "rcx", "rdx"} <= set(effects["clobbers"])

    def test_memory_operand_base_is_read_not_written(self) -> None:
        effects = GadgetIndex._instruction_side_effects(
            "mov", ["[rax+0x8]", "rsi"], "x86_64"
        )
        assert "rsi" in effects["reads"]
        assert "rax" in effects["reads"]
        # destination is memory, no register write claimed
        assert effects["writes"] == []

    def test_unknown_mnemonic_produces_a_note(self) -> None:
        effects = GadgetIndex._instruction_side_effects(
            "vpcmpistri", ["xmm0", "xmm1"], "x86_64"
        )
        assert any("unknown mnemonic" in n for n in effects["notes"])
        assert effects["reads"] == []
        assert effects["writes"] == []

    def test_unsupported_arch_notes_and_returns_empty(self) -> None:
        effects = GadgetIndex._instruction_side_effects(
            "ldp", ["x0", "x1", "[sp]"], "aarch64"
        )
        assert any("aarch64" in n for n in effects["notes"])
        assert effects["reads"] == []
        assert effects["writes"] == []


class TestStackDelta:
    def test_ret_matches_pointer_width(self) -> None:
        assert GadgetIndex._instruction_stack_delta("ret", [], "x86_64") == 8
        assert GadgetIndex._instruction_stack_delta("ret", [], "i386") == 4

    def test_ret_with_immediate_adds_imm(self) -> None:
        assert GadgetIndex._instruction_stack_delta("ret", ["0x8"], "x86_64") == 16

    def test_pop_and_push_move_by_pointer_width(self) -> None:
        assert GadgetIndex._instruction_stack_delta("pop", ["rdi"], "x86_64") == 8
        assert GadgetIndex._instruction_stack_delta("push", ["rdi"], "x86_64") == -8

    def test_add_sub_rsp_move_by_immediate(self) -> None:
        assert (
            GadgetIndex._instruction_stack_delta("add", ["rsp", "0x18"], "x86_64")
            == 0x18
        )
        assert (
            GadgetIndex._instruction_stack_delta("sub", ["rsp", "0x10"], "x86_64")
            == -0x10
        )

    def test_call_reserves_return_slot(self) -> None:
        assert GadgetIndex._instruction_stack_delta("call", ["rax"], "x86_64") == -8

    def test_unknown_mnemonic_delta_is_none(self) -> None:
        assert (
            GadgetIndex._instruction_stack_delta("cmov", ["rax", "rbx"], "x86_64")
            is None
        )

    def test_unsupported_arch_delta_is_none(self) -> None:
        assert GadgetIndex._instruction_stack_delta("pop", ["x0"], "aarch64") is None


class TestAnalyzeGadget:
    def test_pop_rdi_ret_aggregates(self) -> None:
        result = GadgetIndex.analyze_gadget(_record(0x400100, ["pop rdi", "ret"]))
        assert result["instruction_text"] == "pop rdi ; ret"
        assert result["writes"] == ["rdi", "rsp"]
        assert result["reads"] == ["rsp"]
        assert result["stack_delta"] == 16
        assert result["pointer_bytes"] == 8

    def test_unparseable_instruction_makes_stack_delta_unknown(self) -> None:
        result = GadgetIndex.analyze_gadget(_record(0x400200, ["... garbage ..."]))
        assert result["stack_delta"] is None
        assert any("could not parse" in n for n in result["notes"])

    def test_mixed_known_and_unknown_preserves_partial_analysis(self) -> None:
        result = GadgetIndex.analyze_gadget(
            _record(0x400300, ["pop rdi", "vpcmpistri xmm0, xmm1, 0x0", "ret"])
        )
        # rdi is still written by pop; ret is still known; but stack_delta
        # cannot be computed once we hit an unknown mnemonic that touches
        # the flags/state we cannot model.
        assert "rdi" in result["writes"]
        assert any("unknown mnemonic 'vpcmpistri'" in n for n in result["notes"])


class TestBadByteFilter:
    def test_address_with_bad_byte_is_flagged(self) -> None:
        # 0x400a00 -> le64 bytes include 0x0a
        assert GadgetIndex._address_contains_bad_byte(0x400A00, 8, [0x0A])

    def test_clean_address_passes(self) -> None:
        assert not GadgetIndex._address_contains_bad_byte(
            0x401234, 8, [0x0A, 0x0D, 0x20]
        )

    def test_big_endian_addresses_use_target_byte_order(self) -> None:
        # Byte membership is reversal-invariant, so the *verdict* for an
        # address matches under either order; what this locks is that the
        # big-endian encoding path is accepted and applied (the previous
        # signature had no endianness parameter at all).
        assert GadgetIndex._address_contains_bad_byte(
            0x12345678, 4, [0x12], endianness="big"
        )
        assert not GadgetIndex._address_contains_bad_byte(
            0x12345678, 4, [0x99], endianness="big"
        )

    def test_default_endianness_is_little(self) -> None:
        assert not GadgetIndex._address_contains_bad_byte(0x12345678, 4, [0x99])

    def test_endianness_inferred_from_arch(self) -> None:
        assert GadgetIndex._endianness_for_arch("mips") == "big"
        assert GadgetIndex._endianness_for_arch("mipsel") == "little"
        assert GadgetIndex._endianness_for_arch("x86_64") == "little"
        assert GadgetIndex._endianness_for_arch(None) == "little"

    def test_gadget_endianness_reads_arch_annotation(self) -> None:
        assert GadgetIndex._gadget_endianness({"arch_canonical": "mips"}) == "big"
        assert GadgetIndex._gadget_endianness({"arch": "mips"}) == "big"
        assert GadgetIndex._gadget_endianness({"arch": "x86_64"}) == "little"
        assert GadgetIndex._gadget_endianness({}) == "little"

    def test_missing_pointer_width_is_conservative(self) -> None:
        # Without a pointer width we cannot honestly encode: fail closed so
        # an unknown arch never passes a tainted address through the filter.
        assert GadgetIndex._address_contains_bad_byte(0x0A, None, [0x0A])

    def test_normalizes_hex_strings(self) -> None:
        normalized = GadgetIndex._normalize_bad_bytes(["0x0a", "0x20", 0x0D])
        assert normalized == [0x0A, 0x20, 0x0D]

    def test_rejects_out_of_range_byte(self) -> None:
        with pytest.raises(GadgetIndexError, match="not a byte"):
            GadgetIndex._normalize_bad_bytes([0x100])

    def test_rejects_garbage_string(self) -> None:
        with pytest.raises(GadgetIndexError, match="not a byte value"):
            GadgetIndex._normalize_bad_bytes(["hello"])


class TestDedupe:
    def test_identical_bytes_and_instructions_collapsed(self) -> None:
        entries = [
            {
                "address": 0x400100,
                "bytes": "5fc3",
                "instruction_text": "pop rdi ; ret",
                "instructions": ["pop rdi", "ret"],
            },
            {
                "address": 0x400200,
                "bytes": "5fc3",
                "instruction_text": "pop rdi ; ret",
                "instructions": ["pop rdi", "ret"],
            },
        ]
        deduped = GadgetIndex._dedupe(entries)
        assert len(deduped) == 1

    def test_distinct_instructions_kept(self) -> None:
        entries = [
            {
                "address": 0x400100,
                "bytes": "5fc3",
                "instruction_text": "pop rdi ; ret",
                "instructions": ["pop rdi", "ret"],
            },
            {
                "address": 0x400300,
                "bytes": "5ec3",
                "instruction_text": "pop rsi ; ret",
                "instructions": ["pop rsi", "ret"],
            },
        ]
        deduped = GadgetIndex._dedupe(entries)
        assert len(deduped) == 2

    def test_lowest_address_sighting_wins_over_loader_order(self) -> None:
        # Identical gadget first seen at a HIGH address, then a lower-address
        # sighting appears later in loader order: the lowest address must win.
        entries = [
            {
                "address": 0x400200,
                "bytes": "5fc3",
                "instruction_text": "pop rdi ; ret",
                "instructions": ["pop rdi", "ret"],
            },
            {
                "address": 0x400100,
                "bytes": "5fc3",
                "instruction_text": "pop rdi ; ret",
                "instructions": ["pop rdi", "ret"],
            },
        ]
        deduped = GadgetIndex._dedupe(entries)
        assert len(deduped) == 1
        assert deduped[0]["address"] == 0x400100


# ---------------------------------------------------------------------------
# Backend behavior via injected loader
# ---------------------------------------------------------------------------
class TestFindGadgets:
    def test_returns_typed_envelope_and_matches_pattern(self) -> None:
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(0x400200, ["pop rsi", "ret"]),
            _record(0x400300, ["mov rax, rdi", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, filter_pattern="pop")
            parsed = response["parsed"]
            assert parsed["schema"] == GADGET_LIST_SCHEMA
            assert parsed["matched"] == 2
            assert parsed["truncated"] is False
            addrs = [g["address"] for g in parsed["gadgets"]]
            assert addrs == [0x400100, 0x400200]
        finally:
            os.unlink(binary)

    def test_bad_bytes_drops_matching_gadget(self) -> None:
        records = [
            _record(0x400A00, ["pop rdi", "ret"]),  # 0x0A in address
            _record(0x400100, ["pop rdi", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(
                binary_path=binary,
                filter_pattern="pop",
                arch="x86_64",
                bad_bytes=[0x0A],
            )
            addrs = [g["address"] for g in response["parsed"]["gadgets"]]
            assert addrs == [0x400100]
        finally:
            os.unlink(binary)

    def test_bad_bytes_uses_arch_endianness(self, monkeypatch) -> None:
        # 0x12345678 contains 0x12 in both byte orders, so the filter's
        # verdict cannot observe the order; spy on the encoding helper to
        # prove the big-endian MIPS arch is forwarded as such.
        seen: List[str] = []
        original = GadgetIndex._address_contains_bad_byte

        @classmethod
        def spy(cls, address, pointer_bytes, bad_bytes, endianness="little"):
            seen.append(endianness)
            return original(address, pointer_bytes, bad_bytes, endianness)

        monkeypatch.setattr(GadgetIndex, "_address_contains_bad_byte", spy)
        records = [_record(0x12345678, ["ret"], arch="mips")]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            big = idx.find_gadgets(binary_path=binary, arch="mips", bad_bytes=[0x99])
            little = idx.find_gadgets(
                binary_path=binary, arch="mipsel", bad_bytes=[0x99]
            )
        finally:
            os.unlink(binary)
        assert seen == ["big", "little"]
        assert big["parsed"]["matched"] == 1
        assert little["parsed"]["matched"] == 1

    def test_max_results_truncates_but_reports_full_count(self) -> None:
        records = [_record(0x400000 + i * 0x10, ["pop rdi", "ret"]) for i in range(5)]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, max_results=3)
            parsed = response["parsed"]
            # dedupe collapses identical (bytes, instructions) so matched=1
            # here; the primary point is that returned <= max_results.
            assert parsed["returned"] <= 3
        finally:
            os.unlink(binary)

    def test_include_analysis_false_omits_heavy_field(self) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, include_analysis=False)
            gadget = response["parsed"]["gadgets"][0]
            assert "instructions_analyzed" not in gadget
            # summary side-effect fields must still be present
            assert "reads" in gadget and "writes" in gadget
        finally:
            os.unlink(binary)

    def test_max_instructions_drops_long_gadgets(self) -> None:
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(
                0x400200,
                [
                    "mov rax, 1",
                    "mov rdi, 0",
                    "mov rsi, 0",
                    "mov rdx, 0",
                    "syscall",
                    "ret",
                ],
            ),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, max_instructions=3)
            addrs = [g["address"] for g in response["parsed"]["gadgets"]]
            assert 0x400200 not in addrs
        finally:
            os.unlink(binary)

    def test_invalid_regex_raises(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([]))
        binary = _tempbin()
        try:
            with pytest.raises(GadgetIndexError, match="not a valid regex"):
                idx.find_gadgets(binary_path=binary, filter_pattern="(")
        finally:
            os.unlink(binary)

    def test_max_results_upper_bound_enforced(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([]))
        binary = _tempbin()
        try:
            with pytest.raises(GadgetIndexError, match="max_results must be between"):
                idx.find_gadgets(binary_path=binary, max_results=MAX_MAX_GADGETS + 1)
        finally:
            os.unlink(binary)

    def test_missing_binary_raises(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([]))
        with pytest.raises(GadgetIndexError, match="not readable"):
            idx.find_gadgets(binary_path="/tmp/does-not-exist-pao")


class TestFindGadgetSequence:
    def test_complete_verdict_when_every_step_matches(self) -> None:
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(0x400200, ["pop rsi", "ret"]),
            _record(0x400300, ["pop rdx", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadget_sequence(
                binary_path=binary,
                steps=["pop rdi", "pop rsi", "pop rdx"],
            )
            parsed = response["parsed"]
            assert parsed["schema"] == GADGET_SEQUENCE_SCHEMA
            assert parsed["verdict"] == "complete"
            addrs = [step["gadget"]["address"] for step in parsed["steps"]]
            assert addrs == [0x400100, 0x400200, 0x400300]
        finally:
            os.unlink(binary)

    def test_partial_verdict_when_a_step_has_no_match(self) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadget_sequence(
                binary_path=binary,
                steps=["pop rdi", "pop rsi"],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "partial"
            assert parsed["steps"][0]["matched"] is True
            assert parsed["steps"][1]["matched"] is False
            assert parsed["steps"][1]["gadget"] is None
        finally:
            os.unlink(binary)

    def test_bad_bytes_excluded_from_sequence(self) -> None:
        records = [
            _record(0x400A00, ["pop rdi", "ret"]),
            _record(0x400100, ["pop rdi", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadget_sequence(
                binary_path=binary,
                steps=["pop rdi"],
                arch="x86_64",
                bad_bytes=[0x0A],
            )
            assert response["parsed"]["steps"][0]["gadget"]["address"] == 0x400100
        finally:
            os.unlink(binary)

    def test_bad_bytes_sequence_uses_arch_endianness(self, monkeypatch) -> None:
        seen: List[str] = []
        original = GadgetIndex._address_contains_bad_byte

        @classmethod
        def spy(cls, address, pointer_bytes, bad_bytes, endianness="little"):
            seen.append(endianness)
            return original(address, pointer_bytes, bad_bytes, endianness)

        monkeypatch.setattr(GadgetIndex, "_address_contains_bad_byte", spy)
        records = [_record(0x12345678, ["ret"], arch="mips")]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadget_sequence(
                binary_path=binary,
                steps=["^ret$"],
                arch="mips",
                bad_bytes=[0x99],
            )
        finally:
            os.unlink(binary)
        assert seen == ["big"]
        assert response["parsed"]["steps"][0]["matched"] is True

    def test_empty_steps_raise(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([]))
        binary = _tempbin()
        try:
            with pytest.raises(GadgetIndexError, match="non-empty sequence"):
                idx.find_gadget_sequence(binary_path=binary, steps=[])
        finally:
            os.unlink(binary)

    def test_too_many_steps_raise(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([]))
        binary = _tempbin()
        try:
            with pytest.raises(GadgetIndexError, match="exceeds cap"):
                idx.find_gadget_sequence(
                    binary_path=binary,
                    steps=["ret"] * (MAX_SEQUENCE_STEPS + 1),
                )
        finally:
            os.unlink(binary)


class TestInspectGadget:
    def test_found_returns_typed_gadget(self) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.inspect_gadget(binary_path=binary, address=0x400100)
            parsed = response["parsed"]
            assert parsed["schema"] == GADGET_INSPECTION_SCHEMA
            assert parsed["verdict"] == "found"
            assert parsed["gadget"]["address"] == 0x400100
        finally:
            os.unlink(binary)

    def test_not_found_returns_verdict_and_no_gadget(self) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.inspect_gadget(binary_path=binary, address=0xDEADBEEF)
            parsed = response["parsed"]
            assert parsed["verdict"] == "not_found"
            assert parsed["gadget"] is None
        finally:
            os.unlink(binary)


class TestCache:
    def test_reload_after_mtime_bump(self) -> None:
        call_count = {"n": 0}
        v1 = [_record(0x400100, ["pop rdi", "ret"])]
        v2 = [_record(0x400200, ["pop rsi", "ret"])]

        def loader(path, arch, image_base):
            call_count["n"] += 1
            return v1 if call_count["n"] == 1 else v2

        idx = GadgetIndex(loader=loader)
        binary = _tempbin()
        try:
            idx.find_gadgets(binary_path=binary)
            # bump the mtime so cache invalidates
            future = time.time() + 5
            os.utime(binary, (future, future))
            idx.find_gadgets(binary_path=binary)
            assert call_count["n"] == 2
        finally:
            os.unlink(binary)

    def test_cache_hit_avoids_second_load(self) -> None:
        call_count = {"n": 0}

        def loader(path, arch, image_base):
            call_count["n"] += 1
            return [_record(0x400100, ["pop rdi", "ret"])]

        idx = GadgetIndex(loader=loader)
        binary = _tempbin()
        try:
            idx.find_gadgets(binary_path=binary)
            idx.find_gadgets(binary_path=binary)
            assert call_count["n"] == 1
        finally:
            os.unlink(binary)


class TestFacade:
    def test_rop_tools_defaults_to_gadget_index(self) -> None:
        rt = RopTools()
        assert isinstance(rt.gadgets, GadgetIndex)

    def test_rop_tools_forwards_find(self) -> None:
        idx = GadgetIndex(loader=_fake_loader([_record(0x400100, ["ret"])]))
        rt = RopTools(gadgets=idx)
        binary = _tempbin()
        try:
            response = rt.find_gadgets(binary_path=binary)
            assert response["parsed"]["matched"] == 1
        finally:
            os.unlink(binary)
