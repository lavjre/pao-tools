"""Tests for ChainConfirmer against a scripted fake GDB controller."""

from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from paomcp.server import create_mcp
from paomcp.tools.backends.chain_confirm import ChainConfirmer
from paomcp.tools.backends.rop import ChainBuilder, GadgetIndex


@pytest.mark.asyncio
async def test_confirm_chain_tool_is_published() -> None:
    tools = await create_mcp().list_tools(run_middleware=False)
    by_name = {tool.name: tool for tool in tools}
    assert "confirm_chain" in by_name
    properties = by_name["confirm_chain"].parameters["properties"]
    assert properties["arch"]["default"] == "x86_64"
    assert properties["max_steps"]["default"] == 200
    assert properties["restore"]["default"] is True


_WRITE_SLOT = re.compile(
    r"set \{(unsigned long long|unsigned int)\}(0x[0-9a-fA-F]+|\d+) = (0x[0-9a-fA-F]+|\d+)"
)
_WRITE_QWORD = _WRITE_SLOT
_SET_REG = re.compile(r"set \$(\w+) = (0x[0-9a-fA-F]+|\d+)")


def _parse(value: str) -> int:
    return int(value, 0)


class FakeGdb:
    """Minimal GdbController stand-in with a scripted step machine.

    ``program`` is a list of per-step dicts; each ``stepi()`` applies the
    next entry (optionally setting ``rip``/``rsp``) and emits a ``*stopped``
    notify with the entry's ``reason`` / ``signal``.  Past the end of the
    program the machine advances ``rip`` by one and reports
    ``end-stepping-range`` forever (a loop), so budget-exhaustion tests work
    without scripting.
    """

    def __init__(
        self,
        *,
        state: str = "stopped",
        rip: int = 0x401000,
        rsp: int = 0x7FFFFFFDE00,
        program: Optional[List[Dict[str, Any]]] = None,
        write_ok: bool = True,
        step_ok: bool = True,
        pc_name: str = "rip",
        sp_name: str = "rsp",
        registers: Optional[Dict[str, int]] = None,
    ) -> None:
        self._state = state
        self.pc_name = pc_name
        self.sp_name = sp_name
        self.regs: Dict[str, int] = (
            dict(registers) if registers else {"rip": rip, "rsp": rsp}
        )
        self.mem: Dict[int, int] = {}
        self.write_widths: List[int] = []
        self.program = program or []
        self.write_ok = write_ok
        self.step_ok = step_ok
        self.step_count = 0
        self.commands: List[str] = []

    # -- GdbController surface -----------------------------------------
    def get_state(self) -> str:
        return self._state

    def is_destroyed(self) -> bool:
        return False

    def execute_command(self, command: str) -> Dict[str, Any]:
        self.commands.append(command)
        match = _WRITE_SLOT.match(command)
        if match:
            width = 4 if match.group(1) == "unsigned int" else 8
            address = _parse(match.group(2))
            value = _parse(match.group(3))
            self.write_widths.append(width)
            if not self.write_ok:
                return {
                    "success": False,
                    "responses": [
                        {
                            "type": "result",
                            "message": "error",
                            "payload": {"msg": "Cannot access memory"},
                        }
                    ],
                }
            self.mem[address] = value
            return {"success": True, "responses": []}
        match = _SET_REG.match(command)
        if match:
            self.regs[match.group(1)] = _parse(match.group(2))
            return {"success": True, "responses": []}
        return {"success": True, "responses": []}

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        if expression.startswith("*("):
            # Scratch readability probe: readable unless the test says otherwise.
            if getattr(self, "probe_ok", True):
                return {
                    "success": True,
                    "responses": [
                        {
                            "type": "result",
                            "message": "done",
                            "payload": {"value": "0x0"},
                        }
                    ],
                }
            return {"success": False, "responses": []}
        name = expression.lstrip("$")
        if name in self.regs:
            return {
                "success": True,
                "responses": [
                    {
                        "type": "result",
                        "message": "done",
                        "payload": {"value": f"0x{self.regs[name]:x}"},
                    }
                ],
            }
        return {"success": False, "responses": []}

    def stepi(self) -> Dict[str, Any]:
        if not self.step_ok:
            return {
                "success": False,
                "responses": [
                    {
                        "type": "result",
                        "message": "error",
                        "payload": {"msg": "Cannot step because of a signal"},
                    }
                ],
            }
        if self.step_count < len(self.program):
            step = self.program[self.step_count]
        else:
            step = {
                self.pc_name: self.regs[self.pc_name] + 1,
                self.sp_name: self.regs[self.sp_name] + 8,
                "reason": "end-stepping-range",
            }
        self.step_count += 1
        if self.pc_name in step:
            self.regs[self.pc_name] = step[self.pc_name]
        if self.sp_name in step:
            self.regs[self.sp_name] = step[self.sp_name]
        reason = step.get("reason", "end-stepping-range")
        payload: Dict[str, Any] = {"reason": reason}
        if reason == "signal-received":
            payload["signal-name"] = step.get("signal", "SIGSEGV")
        return {
            "success": True,
            "state": "stopped",
            "responses": [{"type": "notify", "message": "stopped", "payload": payload}],
        }


@pytest.fixture
def confirmer() -> ChainConfirmer:
    return ChainConfirmer(gdb=FakeGdb())


def _verdict(result: Dict[str, Any]) -> str:
    return result["parsed"]["verdict"]


def _record(address: int, instructions: Sequence[str]) -> Dict[str, Any]:
    return {
        "address": address,
        "bytes": b"\x90" * len(instructions),
        "instructions": list(instructions),
        "arch": "x86_64",
    }


def _fake_loader(
    records: Sequence[Mapping[str, Any]],
) -> Any:
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
        _record(0x400400, ["pop rcx", "ret"]),
        _record(0x400500, ["pop r8", "ret"]),
        _record(0x400600, ["pop r9", "ret"]),
        _record(0x400800, ["ret"]),
    ]


def _builder_over(records: Sequence[Mapping[str, Any]]) -> ChainBuilder:
    return ChainBuilder(gadgets=GadgetIndex(loader=_fake_loader(records)))


def _entry_value(entry: Mapping[str, Any]) -> int:
    if entry["kind"] in ("gadget", "target"):
        return int(entry["address"])
    return int(entry["value"])


def _scripted_program(
    entries: Sequence[Mapping[str, Any]], buffer_address: int
) -> List[Dict[str, Any]]:
    """One step per chain slot consumed after the entry slot.

    The confirmer arms ``$sp`` at ``buffer + 8``, so the step that consumes
    slot ``k`` lands ``rsp`` at ``buffer + 8*(k+1)``; the final step pops
    the target slot with exactly the rsp the target sees on entry.
    """
    return [
        {
            "rip": _entry_value(entry),
            "rsp": buffer_address + 8 * (index + 1),
            "reason": "end-stepping-range",
        }
        for index, entry in enumerate(entries[1:], start=1)
    ]


class TestReachTarget:
    def test_chain_flows_to_default_target(self) -> None:
        gdb = FakeGdb(
            program=[
                {"rip": 0x401234, "rsp": 0x7FFFFFFDE08, "reason": "end-stepping-range"},
                {"rip": 0x401567, "rsp": 0x7FFFFFFDE18, "reason": "end-stepping-range"},
            ]
        )
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234, 0x401567])
        parsed = result["parsed"]
        assert parsed["verdict"] == "reached_target"
        assert parsed["target_reached"] is True
        assert parsed["steps_executed"] == 2
        # Chain written at snapshot rsp: slot 0 at rsp, slot 1 at rsp+8, ...
        assert gdb.mem[0x7FFFFFFDE00] == 0x401000
        assert gdb.mem[0x7FFFFFFDE08] == 0x401234
        assert gdb.mem[0x7FFFFFFDE10] == 0x401567
        # SysV AMD64 entry phase: 0x...DE18 % 16 == 8 -> aligned.
        assert parsed["stack_aligned"] is True

    def test_explicit_target_address(self) -> None:
        gdb = FakeGdb(
            program=[
                {"rip": 0x401234, "rsp": 0x7FFFFFFDE10, "reason": "end-stepping-range"}
            ]
        )
        result = ChainConfirmer(gdb).confirm(
            chain=[0x401000, 0x401234], target_address=0x401234
        )
        assert result["parsed"]["verdict"] == "reached_target"
        assert result["parsed"]["steps_executed"] == 1

    def test_one_gadget_chain_reaches_without_stepping(self) -> None:
        # chain[0] IS the target: reported immediately, no stepping.
        gdb = FakeGdb(program=[])
        result = ChainConfirmer(gdb).confirm(chain=[0x401567])
        parsed = result["parsed"]
        assert parsed["verdict"] == "reached_target"
        assert parsed["steps_executed"] == 0
        assert parsed["trace"] == []
        assert gdb.step_count == 0

    def test_misaligned_stack_reported_not_fatal(self) -> None:
        gdb = FakeGdb(
            rsp=0x7FFFFFFDE08,
            program=[
                {"rip": 0x401567, "rsp": 0x7FFFFFFDE0F, "reason": "end-stepping-range"}
            ],
        )
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401567])
        parsed = result["parsed"]
        assert parsed["verdict"] == "reached_target"
        assert parsed["stack_aligned"] is False
        assert parsed["alignment_expected"] == "16-byte"


class TestBuilderAlignmentParity:
    """Confirmer alignment must agree with the builder's SysV parity model.

    A real saved return-address slot sits at ``rsp % 16 == 8``.  The
    builder appends a bare ``ret`` so a pop-args chain's total pop count is
    even, which keeps the target's entry rsp at 8 (mod 16).  The confirmer
    must report that aligned construction as aligned and flag one missing
    the alignment ret.
    """

    BUFFER = 0x7FFFFFFFDE08  # saved return-address slot: 8 (mod 16)

    @pytest.mark.parametrize("arg_count", [0, 1, 3, 6])
    def test_built_chain_alignment_matches_builder_parity(self, arg_count: int) -> None:
        binary = _tempbin()
        try:
            parsed = _builder_over(_rich_gadget_set()).build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="system",
                args=[0xDEAD0000 + index for index in range(arg_count)],
            )["parsed"]
        finally:
            os.unlink(binary)
        assert parsed["verdict"] == "built"
        entries = parsed["entries"]
        # Builder's documented parity: target entry is on the correct
        # 16-byte phase iff the chain length is a multiple of 16 bytes.
        assert parsed["raw_bytes_length"] % 16 == 0
        gdb = FakeGdb(rsp=self.BUFFER, program=_scripted_program(entries, self.BUFFER))
        result = ChainConfirmer(gdb).confirm(
            chain=[_entry_value(entry) for entry in entries],
            stack_address=self.BUFFER,
        )
        assert result["parsed"]["verdict"] == "reached_target"
        assert result["parsed"]["stack_aligned"] is True

    def test_chain_missing_alignment_ret_is_reported_misaligned(self) -> None:
        binary = _tempbin()
        try:
            parsed = _builder_over(_rich_gadget_set()).build_rop_chain(
                binary_path=binary,
                kind="ret2libc",
                target_address=0x7FFF_1234_5000,
                target_symbol="system",
                args=[0xDEAD0001],
                align_stack=False,
            )["parsed"]
        finally:
            os.unlink(binary)
        entries = parsed["entries"]
        assert [entry["kind"] for entry in entries] == [
            "gadget",
            "immediate",
            "target",
        ]
        # Odd pop count -> 24 bytes -> 8 (mod 16): the target entry rsp is
        # NOT on the SysV phase, so the confirmer must say so.
        assert parsed["raw_bytes_length"] % 16 == 8
        gdb = FakeGdb(rsp=self.BUFFER, program=_scripted_program(entries, self.BUFFER))
        result = ChainConfirmer(gdb).confirm(
            chain=[_entry_value(entry) for entry in entries],
            stack_address=self.BUFFER,
        )
        assert result["parsed"]["verdict"] == "reached_target"
        assert result["parsed"]["stack_aligned"] is False

    def test_one_gadget_alignment_uses_armed_entry_rsp(self) -> None:
        # No pivot ret is modeled: the armed $sp (buffer + 8) is the entry
        # rsp, so an ABI-correct saved-RA slot (8 mod 16) leaves the target
        # misaligned and needs a leading ``ret``.
        gdb = FakeGdb(program=[])
        result = ChainConfirmer(gdb).confirm(
            chain=[0x401567], stack_address=self.BUFFER
        )
        assert result["parsed"]["verdict"] == "reached_target"
        assert result["parsed"]["steps_executed"] == 0
        assert result["parsed"]["stack_aligned"] is False


class TestCrash:
    def test_signal_stops_with_crash_detail(self) -> None:
        gdb = FakeGdb(
            program=[
                {
                    "rip": 0xDEADBEEF,
                    "rsp": 0x7FFFFFFDE08,
                    "reason": "signal-received",
                    "signal": "SIGSEGV",
                }
            ]
        )
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234])
        parsed = result["parsed"]
        assert parsed["verdict"] == "crashed"
        assert parsed["crash"] == {"signal": "SIGSEGV", "step": 1}
        assert parsed["target_reached"] is False


class TestExhaustion:
    def test_budget_exhausted_with_trace(self) -> None:
        gdb = FakeGdb(program=[])
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234], max_steps=5)
        parsed = result["parsed"]
        assert parsed["verdict"] == "steps_exhausted"
        assert parsed["steps_executed"] == 5
        assert len(parsed["trace"]) == 5


class TestRestore:
    def test_registers_restored_by_default(self) -> None:
        gdb = FakeGdb(
            rip=0x401000,
            rsp=0x7FFFFFFDE00,
            program=[{"rip": 0x401567, "rsp": 0x7FFFFFFDE10}],
        )
        ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401567])
        assert gdb.regs["rip"] == 0x401000
        assert gdb.regs["rsp"] == 0x7FFFFFFDE00

    def test_restore_false_leaves_registers(self) -> None:
        gdb = FakeGdb(program=[{"rip": 0x401567, "rsp": 0x7FFFFFFDE10}])
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401567], restore=False)
        assert result["parsed"]["restored"] is False
        assert gdb.regs["rip"] == 0x401567


class TestHarnessFailures:
    def test_not_stopped_is_invalid_setup(self) -> None:
        gdb = FakeGdb(state="running")
        result = ChainConfirmer(gdb).confirm(chain=[0x401000])
        assert _verdict(result) == "invalid_setup"
        assert "stopped" in result["parsed"]["message"]

    def test_write_failure_reports_slot(self) -> None:
        gdb = FakeGdb(write_ok=False)
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234])
        parsed = result["parsed"]
        assert parsed["verdict"] == "write_failed"
        assert parsed["failed_slot"] == 0
        assert "slot 0" in parsed["message"]

    def test_step_failure_surfaces_gdb_message(self) -> None:
        gdb = FakeGdb(step_ok=False)
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234])
        assert result["parsed"]["verdict"] == "step_failed"
        assert "Cannot step" in result["parsed"]["message"]


class TestArguments:
    def test_empty_chain_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ChainConfirmer(FakeGdb()).confirm(chain=[])

    def test_unsupported_arch_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported arch"):
            ChainConfirmer(FakeGdb()).confirm(chain=[0x401000], arch="mips")

    def test_chain_slot_cap_enforced(self) -> None:
        with pytest.raises(ValueError, match="cap"):
            ChainConfirmer(FakeGdb()).confirm(chain=[0] * 2000)

    def test_max_steps_bounds_enforced(self) -> None:
        with pytest.raises(ValueError, match="max_steps"):
            ChainConfirmer(FakeGdb()).confirm(chain=[0x401000], max_steps=0)

    def test_stack_address_used_for_writes(self) -> None:
        gdb = FakeGdb()
        ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234], stack_address=0x600000)
        assert gdb.mem[0x600000] == 0x401000
        assert gdb.mem[0x600008] == 0x401234

    def test_i386_uses_4byte_slots_and_esp(self) -> None:
        gdb = FakeGdb(
            pc_name="eip",
            sp_name="esp",
            registers={"eip": 0x401000, "esp": 0x7FFFFFFDE00},
        )
        result = ChainConfirmer(gdb).confirm(
            chain=[0x401000, 0x401234],
            stack_address=0x600000,
            arch="i386",
            max_steps=2,
        )
        assert result["parsed"]["alignment_expected"] == "none"
        assert gdb.mem[0x600000] == 0x401000
        assert gdb.mem[0x600004] == 0x401234
        assert set(gdb.write_widths) == {4}
        assert any(cmd.startswith("set $eip") for cmd in gdb.commands)
        assert any(cmd.startswith("set $esp") for cmd in gdb.commands)

    def test_sp_starts_past_entry_slot(self) -> None:
        gdb = FakeGdb()
        ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234])
        esp_sets = [
            _parse(cmd.split(" = ")[1])
            for cmd in gdb.commands
            if cmd.startswith("set $rsp = ")
        ]
        assert esp_sets, "expected an rsp assignment"
        # Armed past chain[0] (ret pops chain[1]); restored to snapshot after.
        assert esp_sets[0] == 0x7FFFFFFDE00 + 8
        assert esp_sets[-1] == 0x7FFFFFFDE00

    def test_unreadable_scratch_fails_fast(self) -> None:
        gdb = FakeGdb()
        gdb.probe_ok = False
        result = ChainConfirmer(gdb).confirm(chain=[0x401000, 0x401234])
        assert result["parsed"]["verdict"] == "write_failed"
        assert gdb.mem == {}
