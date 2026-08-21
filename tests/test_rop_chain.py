"""Tests for Stage 7.3 typed ROP chain construction and SROP frame builder."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

from paomcp.tools.backends.rop import (
    CHAIN_CONSTRAINTS_SCHEMA,
    ROP_CHAIN_SCHEMA,
    SIGRETURN_FRAME_SCHEMA,
    ChainBuilder,
    ChainBuilderError,
    GadgetIndex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _record(address: int, instructions: List[str]) -> Dict[str, Any]:
    return {
        "address": address,
        "bytes": b"\x90" * len(instructions),
        "instructions": instructions,
        "arch": "x86_64",
    }


def _fake_loader(
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


def _rich_gadget_set() -> Sequence[Mapping[str, Any]]:
    return [
        _record(0x400100, ["pop rdi", "ret"]),
        _record(0x400200, ["pop rsi", "ret"]),
        _record(0x400300, ["pop rdx", "ret"]),
        _record(0x400400, ["pop r10", "ret"]),
        _record(0x400500, ["pop r8", "ret"]),
        _record(0x400600, ["pop r9", "ret"]),
        _record(0x400700, ["pop rax", "ret"]),
        _record(0x400800, ["ret"]),
        _record(0x400900, ["syscall", "ret"]),
    ]


def _builder_over(records: Sequence[Mapping[str, Any]]) -> ChainBuilder:
    idx = GadgetIndex(loader=_fake_loader(records))
    return ChainBuilder(gadgets=idx)


# ---------------------------------------------------------------------------
# ret2libc
# ---------------------------------------------------------------------------
class TestRet2LibcChain:
    def test_single_arg_chain_lays_out_pop_rdi_align_target(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="system",
                args=[0x7FFF_1234_9000],
            )
            parsed = response["parsed"]
            assert parsed["schema"] == ROP_CHAIN_SCHEMA
            assert parsed["kind"] == "ret2libc"
            assert parsed["verdict"] == "built"
            assert parsed["missing_gadgets"] == []
            entries = parsed["entries"]
            kinds = [e["kind"] for e in entries]
            # pop rdi ; ret --> imm(arg) --> align ret --> target
            assert kinds == ["gadget", "immediate", "gadget", "target"]
            assert entries[0]["address"] == 0x400100
            assert entries[1]["value"] == 0x7FFF_1234_9000
            assert entries[2]["address"] == 0x400800  # bare ret
            assert entries[3]["address"] == 0x7FFF_1234_5000
            assert entries[3]["symbol"] == "system"
            # raw bytes = 4 slots × 8 bytes
            assert parsed["raw_bytes_length"] == 32
            # ALIGNMENT, not just ret-existence: the chain is entered via
            # ``ret`` with rsp at 8 (mod 16) on the first slot, and the target
            # is entered by the final pop, so a 16-byte-aligned target entry
            # requires the chain length (in bytes) to be a multiple of 16.
            assert parsed["raw_bytes_length"] % 16 == 0
            assert bytes.fromhex(parsed["raw_bytes"])[:8] == (0x400100).to_bytes(
                8, "little"
            )
        finally:
            os.unlink(binary)

    def test_three_arg_chain_uses_rdi_rsi_rdx(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="execve",
                args=[0x7FFF_1234_9000, 0, 0],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "built"
            registers = [a["register"] for a in parsed["args"]]
            assert registers == ["rdi", "rsi", "rdx"]
            # 3 pops + 3 imms + align + target = 8 slots × 8 bytes
            assert parsed["raw_bytes_length"] == 8 * 8
            # Alignment invariant (see test_single_arg): target entry is on
            # the correct 16-byte SysV phase iff the chain length is a
            # multiple of 16 bytes.
            assert parsed["raw_bytes_length"] % 16 == 0
        finally:
            os.unlink(binary)

    def test_missing_pop_reg_reports_missing_gadgets(self) -> None:
        binary = _tempbin()
        # No pop rsi gadget in this fixture
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(0x400800, ["ret"]),
        ]
        try:
            builder = _builder_over(records)
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="execve",
                args=[0x1000, 0x2000],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "missing_gadgets"
            assert "pop rsi ; ret" in parsed["missing_gadgets"]
        finally:
            os.unlink(binary)

    def test_align_stack_false_omits_align_ret(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x1000,
                args=[0x2000],
                align_stack=False,
            )
            parsed = response["parsed"]
            kinds = [e["kind"] for e in parsed["entries"]]
            # pop rdi ; ret --> imm --> target
            assert kinds == ["gadget", "immediate", "target"]
            # 3 slots (24 bytes): the natural odd pop count, i.e. the
            # deliberately-misaligned phase for an aligned delivery point.
            assert parsed["raw_bytes_length"] == 24
            assert parsed["raw_bytes_length"] % 16 == 8
        finally:
            os.unlink(binary)

    def test_bare_ret_missing_records_alignment_assumption(self) -> None:
        binary = _tempbin()
        records = [_record(0x400100, ["pop rdi", "ret"])]  # no bare ret gadget
        try:
            builder = _builder_over(records)
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x1000,
                args=[0x2000],
            )
            parsed = response["parsed"]
            assumptions = " ".join(parsed["assumptions"])
            assert "no bare 'ret' gadget available" in assumptions
        finally:
            os.unlink(binary)

    def test_alignment_phase_for_every_arg_count(self) -> None:
        # M2: the align decision must be parity-computed, and every emitted
        # chain must land the target on the correct 16-byte SysV phase.  A
        # pure pop-args chain pops 2*n + 1 slots (always odd), so with
        # align_stack=True a bare ret is appended and the chain length is a
        # multiple of 16 bytes; with align_stack=False the ret is omitted and
        # the length is 8 (mod 16) -- the misaligned phase a caller must
        # deliberately choose only for an already-aligned delivery.
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            # 0..3 args: 0 uses no pop gadget, 1..3 need rdi/rsi/rdx which the
            # rich fixture provides (the 4th SysV arg would need rcx).
            for arg_count in range(4):
                args = [0x7FFF_1234_9000 + i for i in range(arg_count)]
                aligned = builder.build_rop_chain(
                    binary_path=binary,
                    kind="ret2libc",
                    target_address=0x7FFF_1234_5000,
                    target_symbol="system",
                    args=args,
                    align_stack=True,
                )["parsed"]
                assert aligned["verdict"] == "built"
                assert aligned["raw_bytes_length"] % 16 == 0, (
                    f"{arg_count} args with align_stack=True must leave the "
                    f"target on the aligned phase, got length "
                    f"{aligned['raw_bytes_length']}"
                )
                unaligned = builder.build_rop_chain(
                    binary_path=binary,
                    kind="ret2libc",
                    target_address=0x7FFF_1234_5000,
                    target_symbol="system",
                    args=args,
                    align_stack=False,
                )["parsed"]
                kinds = [e["kind"] for e in unaligned["entries"]]
                assert kinds.count("gadget") == arg_count  # no align ret
                assert unaligned["raw_bytes_length"] % 16 == 8, (
                    f"{arg_count} args with align_stack=False keeps the "
                    f"natural odd pop count (length 8 mod 16)"
                )
        finally:
            os.unlink(binary)

    def test_zero_arg_chain_gets_alignment_ret(self) -> None:
        # A bare call (no register args) still pops one slot for the target,
        # so the 1-slot chain is misaligned and the alignment ret is needed.
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            parsed = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="one_gadget",
                args=[],
            )["parsed"]
            kinds = [e["kind"] for e in parsed["entries"]]
            assert kinds == ["gadget", "target"]
            assert parsed["raw_bytes_length"] == 16
            assert parsed["raw_bytes_length"] % 16 == 0
        finally:
            os.unlink(binary)

    def test_missing_target_address_raises(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            with pytest.raises(ChainBuilderError, match="target_address"):
                builder.build_rop_chain(
                    binary_path=binary,
                    kind="ret2libc",
                    target_symbol="system",
                    args=[0x1000],
                )
        finally:
            os.unlink(binary)

    def test_too_many_args_raises(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            with pytest.raises(ChainBuilderError, match="up to 6 args"):
                builder.build_rop_chain(
                    binary_path=binary,
                    kind="ret2libc",
                    target_address=0x1000,
                    args=[0] * 7,
                )
        finally:
            os.unlink(binary)


# ---------------------------------------------------------------------------
# ret2syscall
# ---------------------------------------------------------------------------
class TestRet2SyscallChain:
    def test_execve_layout_uses_r10_not_rcx(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2syscall",
                syscall_number=59,  # execve
                args=[0x7FFF_1234_9000, 0, 0],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "built"
            arg_registers = [a["register"] for a in parsed["args"]]
            assert arg_registers == ["rdi", "rsi", "rdx"]
            # Syscall chain: pop rax, imm, pop rdi, imm, pop rsi, imm, pop rdx, imm, syscall
            expected_kinds = [
                "gadget",
                "immediate",
                "gadget",
                "immediate",
                "gadget",
                "immediate",
                "gadget",
                "immediate",
                "gadget",
            ]
            assert [e["kind"] for e in parsed["entries"]] == expected_kinds
            # 5 gadgets + 4 immediates = 9 slots × 8 bytes
            assert parsed["raw_bytes_length"] == 9 * 8
        finally:
            os.unlink(binary)

    def test_missing_syscall_ret_reports_missing(self) -> None:
        binary = _tempbin()
        # rich set minus syscall gadget
        records = [
            r for r in _rich_gadget_set() if "syscall" not in r["instructions"][0]
        ]
        try:
            builder = _builder_over(records)
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2syscall",
                syscall_number=59,
                args=[0, 0, 0],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "missing_gadgets"
            assert "syscall ; ret" in parsed["missing_gadgets"]
        finally:
            os.unlink(binary)

    def test_missing_syscall_number_raises(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            with pytest.raises(ChainBuilderError, match="syscall_number"):
                builder.build_rop_chain(
                    binary_path=binary,
                    kind="ret2syscall",
                    args=[0, 0, 0],
                )
        finally:
            os.unlink(binary)

    def test_uses_r10_for_fourth_arg(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2syscall",
                syscall_number=9,  # mmap
                args=[0, 0x1000, 7, 0x22, 0xFFFF_FFFF, 0],
            )
            arg_registers = [a["register"] for a in response["parsed"]["args"]]
            assert arg_registers == ["rdi", "rsi", "rdx", "r10", "r8", "r9"]
        finally:
            os.unlink(binary)


# ---------------------------------------------------------------------------
# Chain kind + arch dispatch
# ---------------------------------------------------------------------------
class TestChainDispatch:
    def test_unsupported_kind_raises(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            with pytest.raises(ChainBuilderError, match="unsupported chain kind"):
                builder.build_rop_chain(
                    binary_path=binary, kind="ret2csu", target_address=0x1000
                )
        finally:
            os.unlink(binary)

    def test_non_x86_64_arch_returns_unsupported_verdict(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            response = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                arch="aarch64",
                target_address=0x1000,
                args=[0],
            )
            parsed = response["parsed"]
            assert parsed["supported"] is False
            assert parsed["verdict"] == "unsupported_arch"
        finally:
            os.unlink(binary)


# ---------------------------------------------------------------------------
# Sigreturn frame
# ---------------------------------------------------------------------------
class TestSigreturnFrame:
    def test_amd64_frame_is_248_bytes_with_expected_registers(self) -> None:
        builder = ChainBuilder()
        response = builder.inspect_sigreturn_frame(
            arch="x86_64",
            registers={"rip": 0xDEADBEEF, "rdi": 0x1000, "rax": 59},
        )
        parsed = response["parsed"]
        assert parsed["schema"] == SIGRETURN_FRAME_SCHEMA
        assert parsed["arch"] == "x86_64"
        assert parsed["frame_size"] == 248
        assert parsed["syscall_number"] == 15  # rt_sigreturn on x86_64
        assert parsed["syscall_register"] == "rax"
        assert parsed["register_offsets"]["rip"] == 168
        assert parsed["register_offsets"]["rdi"] == 104
        assert parsed["registers"] == {"rip": 0xDEADBEEF, "rdi": 0x1000, "rax": 59}
        raw = bytes.fromhex(parsed["frame_bytes"])
        assert len(raw) == 248
        # rip is little-endian at offset 168
        assert int.from_bytes(raw[168:176], "little") == 0xDEADBEEF

    def test_i386_frame_uses_eax_and_syscall_173(self) -> None:
        builder = ChainBuilder()
        response = builder.inspect_sigreturn_frame(arch="i386")
        parsed = response["parsed"]
        assert parsed["arch"] == "i386"
        assert parsed["syscall_register"] == "eax"
        assert parsed["syscall_number"] == 173

    def test_unknown_register_raises(self) -> None:
        builder = ChainBuilder()
        with pytest.raises(ChainBuilderError, match="not part of"):
            builder.inspect_sigreturn_frame(
                arch="x86_64", registers={"totally_bogus_register": 1}
            )

    def test_unsupported_arch_returns_verdict_not_raises(self) -> None:
        builder = ChainBuilder()
        response = builder.inspect_sigreturn_frame(arch="aarch64")
        parsed = response["parsed"]
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported_arch"


# ---------------------------------------------------------------------------
# explain_chain_constraints
# ---------------------------------------------------------------------------
class TestExplainChain:
    def test_walks_slots_and_flags_overwrites(self) -> None:
        binary = _tempbin()
        try:
            builder = _builder_over(_rich_gadget_set())
            built = builder.build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="execve",
                args=[0x1000, 0x2000, 0x3000],
            )["parsed"]
            response = builder.explain_chain_constraints(chain=built["entries"])
            parsed = response["parsed"]
            assert parsed["schema"] == CHAIN_CONSTRAINTS_SCHEMA
            # 3 pop-gadget slots, 3 immediate slots, 1 align-ret, 1 target = 8
            assert parsed["slot_count"] == 8
            gadget_slots = [s for s in parsed["slots"] if s["kind"] == "gadget"]
            assert {"rdi", "rsi", "rdx", "rsp"} <= {
                reg for s in gadget_slots for reg in s.get("writes", [])
            }
            # register_state_after_chain must remember every load
            assert "rdi" in parsed["register_state_after_chain"]
        finally:
            os.unlink(binary)

    def test_empty_chain_yields_empty_slots(self) -> None:
        builder = ChainBuilder()
        response = builder.explain_chain_constraints(chain=[])
        assert response["parsed"]["slot_count"] == 0
        assert response["parsed"]["slots"] == []
