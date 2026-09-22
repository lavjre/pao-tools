"""Resident-GDB chain confirmation (dynamic counterpart to chain construction).

``validate_rop_chain`` is purely static.  ``confirm_chain`` is the dynamic
half: it dry-runs a chain inside the *live* debugger session by writing the
chain values onto a scratch stack region, pointing ``$pc``/``$sp`` at it,
single-stepping until the chain either reaches its target address, crashes,
or exhausts the step budget -- then restoring ``$pc``/``$sp`` so the session
is left exactly where it was.  Because the loop runs in the resident
:class:`~paomcp.tools.backends.gdb.GdbController`, a confirm pass never pays
process startup cost; re-confirming after tweaking one gadget is just another
in-memory step loop.

The execution model mirrors a real overflow payload: ``chain[0]`` is loaded
into ``$pc`` and ``$sp`` points at ``chain[1]`` (``buffer + ptr``), so the
first gadget executes normally and its closing ``ret`` pops ``chain[1]``
exactly like the overwritten return address would on the actual stack.
A single-element chain whose entry *is* the target (a one-gadget) is
reported as ``reached_target`` immediately.

The scratch region's *contents* are clobbered by design (the chain is
written there); only ``$pc``/``$sp`` are restored. Prefer passing an
explicit ``stack_address`` over letting the live stack serve as scratch.

Verdicts:

* ``reached_target``  -- control transferred to ``target_address`` (defaults
  to the last chain value) without a signal; stack alignment is reported
  against the SysV AMD64 entry contract, i.e. ``stack_aligned`` is true
  when ``rsp % 16 == 8`` at the moment the target is entered.
* ``crashed``         -- the inferior received a signal while stepping.
* ``step_failed``     -- GDB could not execute a single step (e.g. an
  invalid instruction or an unreadable instruction fetch).
* ``steps_exhausted`` -- the step budget ran out without reaching the target
  or crashing (loop gadget, or a chain longer than the budget).
* ``exited``          -- the inferior exited while stepping.
* ``write_failed``    -- a chain value could not be written to the scratch
  region (address outside a writable mapping).
* ``setup_failed``    -- ``$pc``/``$sp`` could not be pointed at the chain.
* ``invalid_setup``   -- the session has no stopped inferior to drive.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import build_backend_response, parse_int
from paomcp.tools.backends.rop_schemas import CHAIN_CONFIRM_SCHEMA

MAX_CHAIN_SLOTS = 1024
MAX_CONFIRM_STEPS = 512
DEFAULT_CONFIRM_STEPS = 200
TRACE_LIMIT = 512

# Per-arch (program-counter, stack-pointer, pointer-width, alignment).
# SysV AMD64 requires rsp % 16 == 8 at function entry -- the caller's
# ``call`` pushed the 8-byte return address, so the entry rsp phase is
# 8 (mod 16).  i386 has no hard requirement, so ``alignment`` is ``None``
# there and the field reports it.
_ARCH_REGISTERS: Dict[str, Tuple[str, str, int, Optional[int]]] = {
    "x86_64": ("rip", "rsp", 8, 16),
    "i386": ("eip", "esp", 4, None),
}


def _first_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first ``^done`` payload from a raw MI result."""
    for response in result.get("responses", []):
        if response.get("type") != "result":
            continue
        payload = response.get("payload")
        if isinstance(payload, dict):
            return payload
    return {}


def _stopped_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ``*stopped`` notify payload (step reason/signal) if present."""
    for response in result.get("responses", []):
        if response.get("type") == "notify" and response.get("message") == "stopped":
            payload = response.get("payload")
            if isinstance(payload, dict):
                return payload
    return {}


def _error_message(result: Dict[str, Any]) -> str:
    """Collect the first ``msg`` from any result payload, else ``error``."""
    for response in result.get("responses", []):
        if response.get("type") != "result":
            continue
        payload = response.get("payload")
        if isinstance(payload, dict):
            message = payload.get("msg")
            if message:
                return str(message)
    return str(result.get("error") or "unknown error")


class ChainConfirmer:
    """Dry-run a ROP chain against the live GDB inferior of a session."""

    def __init__(self, gdb: Any) -> None:
        self.gdb = gdb

    # ------------------------------------------------------------------
    # Small GDB primitives
    # ------------------------------------------------------------------
    def _eval_int(self, expression: str) -> Optional[int]:
        """Evaluate a GDB expression to an integer, or ``None``."""
        result = self.gdb.evaluate_expression(expression)
        if not result.get("success"):
            return None
        return parse_int(_first_payload(result).get("value"))

    def _set_reg(self, name: str, value: int) -> bool:
        result = self.gdb.execute_command(f"set {name} = {value}")
        return bool(result.get("success"))

    def _write_slot(
        self, address: int, value: int, pointer_bytes: int
    ) -> Optional[str]:
        """Write one pointer-width value; return an error string or ``None``.

        Width-aware: 4-byte targets get ``unsigned int`` so an i386 write
        cannot spill into the neighboring slot the way a blind qword would.
        """
        cast = "unsigned int" if pointer_bytes == 4 else "unsigned long long"
        result = self.gdb.execute_command(f"set {{{cast}}}{address} = {value}")
        if result.get("success"):
            return None
        return _error_message(result)

    def _write_qword(self, address: int, value: int) -> Optional[str]:
        """Write one 8-byte value; kept for backward compatibility."""
        return self._write_slot(address, value, 8)

    def _probe_readable(self, address: int) -> Optional[str]:
        """One cheap dereference probe; error string or ``None`` when readable."""
        if self._eval_int(f"*(unsigned char *){address:#x}") is not None:
            return None
        return (
            f"scratch base {address:#x} is not readable; pick a writable "
            "stack_address inside a mapped region"
        )

    # ------------------------------------------------------------------
    # Public operation
    # ------------------------------------------------------------------
    def confirm(
        self,
        *,
        chain: Sequence[Any],
        stack_address: Optional[Any] = None,
        target_address: Optional[Any] = None,
        max_steps: int = DEFAULT_CONFIRM_STEPS,
        arch: str = "x86_64",
        restore: bool = True,
    ) -> Dict[str, Any]:
        """Run ``chain`` against the live inferior; see module docstring."""
        # --- argument validation -------------------------------------
        if not isinstance(chain, (list, tuple)) or not chain:
            raise ValueError("chain must be a non-empty sequence of values")
        if len(chain) > MAX_CHAIN_SLOTS:
            raise ValueError(f"chain exceeds the {MAX_CHAIN_SLOTS}-slot cap")
        if max_steps < 1 or max_steps > MAX_CONFIRM_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_CONFIRM_STEPS}")
        values: List[int] = []
        for entry in chain:
            value = parse_int(entry)
            if value is None:
                raise ValueError(f"chain entry is not an integer: {entry!r}")
            values.append(value)
        registers = _ARCH_REGISTERS.get(arch)
        if registers is None:
            raise ValueError(
                f"unsupported arch for confirm_chain: {arch!r} "
                "(x86_64 and i386 only)"
            )
        pc, sp, pointer_bytes, alignment = registers
        target = parse_int(target_address) if target_address is not None else values[-1]
        if target is None:
            raise ValueError("target_address must be an integer")

        def _response(
            verdict: str,
            *,
            message: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            parsed: Dict[str, Any] = {
                "schema": CHAIN_CONFIRM_SCHEMA,
                "arch": arch,
                "verdict": verdict,
                "chain_length": len(values),
                "entry_address": values[0],
                "target_address": target,
                "steps_executed": 0,
                "max_steps": max_steps,
                "crash": None,
                "target_reached": verdict == "reached_target",
                "stack_aligned": None,
                "alignment_expected": (
                    f"{alignment}-byte" if alignment is not None else "none"
                ),
                "restored": False,
                "trace": [],
            }
            if message is not None:
                parsed["message"] = message
            if extra:
                parsed.update(extra)
            return build_backend_response(
                "confirm_chain",
                parsed=parsed,
                raw_responses=[],
                state=self.gdb.get_state(),
                # ``success`` here means "the confirm loop ran to a verdict".
                # A crash / exit / exhausted budget is a *finding*, not a
                # harness failure; only the setup-and-run failures are false.
                success=verdict
                not in (
                    "invalid_setup",
                    "write_failed",
                    "setup_failed",
                    "step_failed",
                ),
                warnings=None,
            )

        # --- session state gate ---------------------------------------
        state = self.gdb.get_state()
        if state != "stopped":
            return _response(
                "invalid_setup",
                message=(
                    f"confirm_chain needs a stopped inferior; session is " f"'{state}'"
                ),
            )

        # --- snapshot + scratch region --------------------------------
        snapshot_pc = self._eval_int(f"${pc}")
        snapshot_sp = self._eval_int(f"${sp}")
        if snapshot_pc is None or snapshot_sp is None:
            return _response(
                "invalid_setup",
                message="could not read the inferior's pc/sp registers",
            )
        if stack_address is not None:
            buffer_address = parse_int(stack_address)
            if buffer_address is None:
                raise ValueError("stack_address must be an integer")
        else:
            buffer_address = snapshot_sp
        end_address = buffer_address + len(values) * pointer_bytes

        # --- writability probe before touching the scratch region -------
        probe_error = self._probe_readable(buffer_address)
        if probe_error is not None:
            return _response(
                "write_failed",
                message=probe_error,
                extra={
                    "buffer_address": buffer_address,
                    "scratch_end": end_address,
                },
            )

        # --- write chain onto the scratch region -----------------------
        for index, value in enumerate(values):
            error = self._write_slot(
                buffer_address + index * pointer_bytes, value, pointer_bytes
            )
            if error is not None:
                return _response(
                    "write_failed",
                    message=(
                        f"could not write chain slot {index} at "
                        f"0x{buffer_address + index * pointer_bytes:x}: {error}"
                    ),
                    extra={
                        "buffer_address": buffer_address,
                        "scratch_end": end_address,
                        "failed_slot": index,
                    },
                )

        # --- arm pc/sp ------------------------------------------------
        # $sp starts past chain[0]: the first gadget's closing ret pops
        # chain[1], mirroring the overwritten return address slot.
        if not self._set_reg(f"${pc}", values[0]) or not self._set_reg(
            f"${sp}", buffer_address + pointer_bytes
        ):
            return _response(
                "setup_failed",
                message="could not point $pc/$sp at the chain",
                extra={"buffer_address": buffer_address, "scratch_end": end_address},
            )

        # --- single-step loop ------------------------------------------
        trace: List[Dict[str, Any]] = []
        steps = 0
        verdict: str = "steps_exhausted"
        crash: Optional[Dict[str, Any]] = None
        step_error: Optional[str] = None
        # A chain whose *entry* is the target (a one-gadget) needs no
        # stepping: control is already at the destination.
        entry_rip = self._eval_int(f"${pc}")
        if entry_rip == target:
            verdict = "reached_target"
        else:
            for step_index in range(max_steps):
                steps = step_index + 1
                step_result = self.gdb.stepi()
                if not step_result.get("success"):
                    verdict = "step_failed"
                    step_error = _error_message(step_result)
                    break
                stopped = _stopped_payload(step_result)
                reason = stopped.get("reason") or "unknown"
                rip = self._eval_int(f"${pc}")
                rsp = self._eval_int(f"${sp}")
                trace.append({"step": steps, "rip": rip, "rsp": rsp, "reason": reason})
                if reason == "signal-received":
                    verdict = "crashed"
                    crash = {"signal": stopped.get("signal-name"), "step": steps}
                    break
                if reason in (
                    "exited",
                    "exited-normally",
                    "exited-signalled",
                    "thread-group-exited",
                ):
                    verdict = "exited"
                    break
                if rip == target:
                    verdict = "reached_target"
                    break
            else:
                verdict = "steps_exhausted"

        # --- restore pc/sp (attempted whenever asked) -------------------
        # A crashed/exited inferior may refuse the writes; the boolean says
        # what happened. Stack *contents* at the scratch region are not
        # restored by design (see module docstring).
        restored = False
        if restore:
            restored = bool(
                self._set_reg(f"${pc}", snapshot_pc)
                and self._set_reg(f"${sp}", snapshot_sp)
            )

        # --- assemble the final payload ---------------------------------
        stack_aligned: Optional[bool] = None
        if verdict == "reached_target" and alignment is not None:
            # SysV AMD64: rsp % 16 == 8 when control reaches a function
            # entry.  ``entry_phase`` is that expected remainder (8 for
            # x86_64, where the pointer width matches the pushed return
            # address).
            entry_phase = pointer_bytes % alignment
            if trace:
                # The stepi that landed on the target already executed the
                # transferring ``ret``, so the traced rsp *is* the target's
                # entry rsp.
                final_rsp = trace[-1].get("rsp")
                if isinstance(final_rsp, int):
                    stack_aligned = final_rsp % alignment == entry_phase
            else:
                # Entry *is* the target (a one-gadget): $sp was armed to
                # buffer + pointer_bytes, and that armed value is the
                # target's entry rsp.
                stack_aligned = (
                    buffer_address + pointer_bytes
                ) % alignment == entry_phase
        return _response(
            verdict,
            extra={
                "buffer_address": buffer_address,
                "scratch_end": end_address,
                "steps_executed": steps,
                "crash": crash,
                "target_reached": verdict == "reached_target",
                "stack_aligned": stack_aligned,
                "restored": restored,
                "scratch_clobbered": True,
                "scratch_note": (
                    "chain bytes were written over the scratch region; "
                    "$pc/$sp restoration does not undo that"
                ),
                "trace": trace[-TRACE_LIMIT:],
            },
            message=step_error,
        )


__all__ = ["ChainConfirmer", "CHAIN_CONFIRM_SCHEMA"]
