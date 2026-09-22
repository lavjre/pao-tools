"""Typed crash-stop collection, triage, and core-file generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from elftools.elf.sections import NoteSection

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorError,
    LiveInferiorGuard,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

from paomcp.tools.backends.crash_compare import CrashCompareMixin
from paomcp.tools.backends.crash_core import (
    CoreDumpGenerationError,
    CrashCoreMixin,
)
from paomcp.tools.backends.crash_cyclic import CrashCyclicMixin
from paomcp.tools.backends.crash_schemas import (
    PARSER_VERSION,
    CRASH_SNAPSHOT_VERSION,
    DEFAULT_FATAL_SIGNALS,
    SAFE_SIGNAL_PATTERN,
    MAX_RUN_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    DEFAULT_CORE_TIMEOUT_SECONDS,
    DEFAULT_MAX_CORE_DUMP_BYTES,
    MAX_CONFIGURED_CORE_DUMP_BYTES,
    MAX_STACK_BYTES,
    MAX_CYCLIC_LENGTH,
    MAX_CYCLIC_STACK_SCAN_BYTES,
    MAX_CYCLIC_SCAN_MATCHES,
    MAX_CYCLIC_VALUE_CHARS,
    MAX_CYCLIC_ALPHABET_CHARS,
    MAX_ELF_CORE_NOTE_ENTRIES,
    MAX_CRASH_HISTORY_ENTRIES,
    MAX_CRASH_HISTORY_BYTES,
    CRASH_CLASSIFICATION_SCHEMA,
    CRASH_FINGERPRINT_SCHEMA,
    CRASH_COMPARISON_SCHEMA,
)


@dataclass(frozen=True)
class _CrashHistoryEntry:
    """One immutable serialized crash snapshot and its compact descriptor."""

    crash_id: str
    captured_at: str
    snapshot_json: Optional[bytes]
    descriptor_json: bytes

    @property
    def retained_bytes(self) -> int:
        return len(self.descriptor_json) + (
            len(self.snapshot_json) if self.snapshot_json is not None else 0
        )


class CrashRecordError(RuntimeError):
    """Raised when no current crash record matches the requested identity."""


class CrashTriage(CrashCyclicMixin, CrashCoreMixin, CrashCompareMixin):
    """Provide conservative typed crash analysis over GDB/MI evidence."""

    def __init__(
        self,
        gdb: GdbController,
        debugger: DebuggerOperations,
        inspection: PwndbgInspection,
        session_id: str,
        session_generation: Optional[str] = None,
    ) -> None:
        self.gdb = gdb
        self.debugger = debugger
        self.inspection = inspection
        self.session_id = session_id
        # A rebuilt controller resets its MI epoch/stop sequence. Include an
        # instance generation so a stale crash id can never collide with a new
        # debugger created under the same caller-controlled session id.
        self.session_generation = session_generation or uuid.uuid4().hex
        if not 0 < DEFAULT_RUN_TIMEOUT_SECONDS <= MAX_RUN_TIMEOUT_SECONDS:
            raise ValueError(
                "PAO_CRASH_RUN_TIMEOUT_SECONDS must be greater than zero and at "
                f"most {MAX_RUN_TIMEOUT_SECONDS}"
            )
        if not 0 < DEFAULT_CORE_TIMEOUT_SECONDS <= MAX_RUN_TIMEOUT_SECONDS:
            raise ValueError(
                "PAO_CORE_DUMP_TIMEOUT_SECONDS must be greater than zero and at "
                f"most {MAX_RUN_TIMEOUT_SECONDS}"
            )
        if not 1 <= DEFAULT_MAX_CORE_DUMP_BYTES <= MAX_CONFIGURED_CORE_DUMP_BYTES:
            raise ValueError(
                "PAO_MAX_CORE_DUMP_BYTES must be between 1 and "
                f"{MAX_CONFIGURED_CORE_DUMP_BYTES}"
            )
        self.default_run_timeout_seconds = DEFAULT_RUN_TIMEOUT_SECONDS
        self.default_core_timeout_seconds = DEFAULT_CORE_TIMEOUT_SECONDS
        self.max_core_dump_bytes = DEFAULT_MAX_CORE_DUMP_BYTES
        self._latest_snapshot: Optional[Dict[str, Any]] = None
        self._history: "OrderedDict[str, _CrashHistoryEntry]" = OrderedDict()
        self._history_bytes = 0
        self._history_lock = threading.Lock()

    @staticmethod
    def _timeout(value: Optional[float], default: float, *, name: str) -> float:
        resolved = default if value is None else float(value)
        if not 0 < resolved <= MAX_RUN_TIMEOUT_SECONDS:
            raise ValueError(
                f"{name} must be greater than zero and at most "
                f"{MAX_RUN_TIMEOUT_SECONDS} seconds"
            )
        return resolved

    @staticmethod
    def _fatal_signals(values: Optional[Sequence[str]]) -> List[str]:
        if values is None:
            return sorted(DEFAULT_FATAL_SIGNALS)
        if not values or len(values) > 32:
            raise ValueError("fatal_signals must contain between 1 and 32 values")
        normalized = []
        for value in values:
            signal = str(value).strip().upper()
            if not SAFE_SIGNAL_PATTERN.fullmatch(signal):
                raise ValueError(f"Invalid signal name: '{value}'")
            if signal not in normalized:
                normalized.append(signal)
        return normalized

    def _effective_fatal_signals(
        self,
        expected_crash_id: Optional[str],
        values: Optional[Sequence[str]],
    ) -> List[str]:
        """Reuse the policy captured for this exact crash when none is supplied."""
        if values is not None:
            return self._fatal_signals(values)
        cached = self._latest_snapshot
        snapshot = (
            cached.get("parsed", {}).get("crash_snapshot", {})
            if isinstance(cached, dict)
            else {}
        )
        if (
            expected_crash_id
            and isinstance(snapshot, dict)
            and snapshot.get("crash_id") == expected_crash_id
            and isinstance(snapshot.get("fatal_signals"), list)
        ):
            return self._fatal_signals(snapshot["fatal_signals"])
        return self._fatal_signals(None)

    def _invalidate_gdb(self, reason: str) -> None:
        """Mark the debugger unusable without depending on a concrete adapter."""
        marker = getattr(self.gdb, "mark_destroyed", None)
        if callable(marker):
            marker(reason)
        else:
            # Compatibility for small test doubles and older controller adapters.
            setattr(self.gdb, "_destroyed", True)

    def _crash_id(self, record: Dict[str, Any]) -> str:
        identity = str(record.get("identity", "unknown"))
        return f"{self.session_id}:{self.session_generation}:{identity}"

    @staticmethod
    def _stop_payload(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        payload = record.get("payload")
        return dict(payload) if isinstance(payload, dict) else {}

    @classmethod
    def _classify_stop(
        cls, record: Optional[Dict[str, Any]], fatal_signals: Sequence[str]
    ) -> Dict[str, Any]:
        payload = cls._stop_payload(record)
        reason = str(payload.get("reason", "unknown"))
        signal_name_value = payload.get("signal-name")
        signal_name = (
            str(signal_name_value).upper() if signal_name_value is not None else None
        )
        signal_stop = reason == "signal-received"
        fatal = bool(signal_stop and signal_name in set(fatal_signals))
        return {
            "reason": reason,
            "kind": (
                "crash"
                if fatal
                else (
                    "signal"
                    if signal_stop
                    else ("exit" if reason.startswith("exited") else "non_crash_stop")
                )
            ),
            "crashed": fatal,
            "signal": (
                {
                    "name": signal_name,
                    "meaning": payload.get("signal-meaning"),
                    "fatal": fatal,
                }
                if signal_name is not None
                else None
            ),
            "thread_id": payload.get("thread-id"),
            "stopped_threads": payload.get("stopped-threads"),
            "core": payload.get("core"),
            "frame": payload.get("frame"),
            "exit_code": payload.get("exit-code"),
            "source": "gdb-mi",
        }

    def get_last_stop(
        self, fatal_signals: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Return the latest durable MI stop identity and classification."""
        record = self.gdb.get_last_stop_record()
        crash_id = self._crash_id(record) if record is not None else None
        signals = self._effective_fatal_signals(crash_id, fatal_signals)
        stop = self._classify_stop(record, signals)
        return {
            "success": record is not None,
            "command": "get_last_stop",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_id": crash_id,
                "record": record,
                "stop": stop,
                "fatal_signals": signals,
            },
            "raw_responses": [],
            **(
                {}
                if record is not None
                else {
                    "error": "No GDB stop notification has been observed",
                    "type": "CrashRecordError",
                }
            ),
        }

    @staticmethod
    def _arguments(values: Optional[Sequence[str]]) -> List[str]:
        if values is None:
            return []
        if len(values) > 256:
            raise ValueError("arguments cannot contain more than 256 values")
        normalized: List[str] = []
        total = 0
        for value in values:
            argument = str(value)
            if any(character in argument for character in ("\x00", "\r", "\n")):
                raise ValueError("arguments cannot contain NUL or newline characters")
            if len(argument.encode("utf-8")) > 4096:
                raise ValueError("each argument must be at most 4096 UTF-8 bytes")
            total += len(argument.encode("utf-8"))
            normalized.append(argument)
        if total > 65536:
            raise ValueError("combined arguments must be at most 65536 UTF-8 bytes")
        return normalized

    def _set_arguments(self, arguments: Sequence[str]) -> Dict[str, Any]:
        command = "-exec-arguments"
        if arguments:
            command += " " + " ".join(
                self.gdb.quote_mi_argument(value) for value in arguments
            )
        return self.gdb.execute_mi_command(command)

    def run_until_crash(
        self,
        *,
        arguments: Optional[Sequence[str]] = None,
        mode: str = "auto",
        timeout_seconds: Optional[float] = None,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Run, continue, or wait until GDB reaches one observable terminal stop."""
        if mode not in {"auto", "run", "continue", "wait"}:
            raise ValueError("mode must be one of: auto, run, continue, wait")
        resolved_arguments = self._arguments(arguments)
        signals = self._fatal_signals(fatal_signals)
        timeout = self._timeout(
            timeout_seconds, self.default_run_timeout_seconds, name="timeout_seconds"
        )
        started = time.monotonic()
        deadline = started + timeout
        state_before = self.gdb.get_state()
        current_record = self.gdb.get_last_stop_record()
        initial_stop_identity = (
            current_record.get("identity") if current_record is not None else None
        )
        current_stop = self._classify_stop(current_record, signals)

        resolved_mode = mode
        if mode == "auto":
            if state_before == "running":
                resolved_mode = "wait"
            elif state_before == "stopped" and current_stop["crashed"]:
                resolved_mode = "wait"
            elif state_before == "stopped" and self.gdb.get_inferior_pid() is not None:
                resolved_mode = "continue"
            else:
                resolved_mode = "run"

        launch_result: Optional[Dict[str, Any]] = None
        if cancellation_check is not None:
            cancellation_check()
        if resolved_mode == "run":
            arguments_result = self._set_arguments(resolved_arguments)
            if not arguments_result.get("success"):
                return {
                    "success": False,
                    "command": "run_until_crash",
                    "state": self.gdb.get_state(),
                    "parser_version": PARSER_VERSION,
                    "parsed": {"mode": resolved_mode, "crashed": False},
                    "raw_responses": {"arguments": arguments_result},
                    "error": "GDB rejected the typed argument vector",
                    "type": "DebuggerRunError",
                }
            launch_result = self.gdb.run()
        elif resolved_mode == "continue":
            if state_before != "stopped" or self.gdb.get_inferior_pid() is None:
                raise ValueError("continue mode requires a stopped live inferior")
            launch_result = self.gdb.continue_execution()
        elif resolved_mode == "wait" and state_before not in {"running", "stopped"}:
            raise ValueError("wait mode requires a running or stopped inferior")

        event_responses: List[Dict[str, Any]] = []
        if launch_result is not None:
            event_responses.extend(launch_result.get("responses", []))
            if not launch_result.get("success"):
                return {
                    "success": False,
                    "command": "run_until_crash",
                    "state": self.gdb.get_state(),
                    "parser_version": PARSER_VERSION,
                    "parsed": {"mode": resolved_mode, "crashed": False},
                    "raw_responses": {
                        "launch": launch_result,
                        "events": event_responses,
                    },
                    "error": "GDB could not start or continue the inferior",
                    "type": "DebuggerRunError",
                }

        # GDB/MI can return from ``-exec-run`` before its asynchronous
        # ``*running`` notification is observed.  In that window the controller
        # still reports the previous ``stopped`` state even though no new stop
        # record exists.  Waiting on state alone would therefore manufacture an
        # ``unknown`` stop before the inferior has reached the crash.  Commands
        # that start/resume execution must observe a *new* immutable stop record
        # (or process exit) before they are complete.
        requires_new_stop = resolved_mode in {"run", "continue"} or (
            resolved_mode == "wait" and state_before == "running"
        )

        def reached_terminal_stop() -> bool:
            state = self.gdb.get_state()
            if state == "exited":
                return True
            if state != "stopped":
                return False
            latest = self.gdb.get_last_stop_record()
            if not requires_new_stop:
                return True
            if latest is None:
                return False
            return latest.get("identity") != initial_stop_identity

        def raise_if_cancelled_after_sync() -> None:
            """Propagate cancellation only after the worker owns a terminal state."""
            if cancellation_check is None:
                return
            try:
                cancellation_check()
            except BaseException:
                cancel_deadline = time.monotonic() + 1.0
                while (
                    not reached_terminal_stop() and time.monotonic() < cancel_deadline
                ):
                    drained = self.gdb.drain_async(
                        timeout_sec=min(
                            0.1, max(0.0, cancel_deadline - time.monotonic())
                        ),
                        max_rounds=100,
                    )
                    if drained.get("backend_error"):
                        break
                    responses = drained.get("responses", [])
                    event_responses.extend(responses)
                    if not responses:
                        time.sleep(0.01)
                if not reached_terminal_stop():
                    killer = getattr(self.gdb, "kill_process_group", None)
                    if callable(killer):
                        killer()
                    self._invalidate_gdb(
                        "cancellation did not reach a synchronized terminal stop"
                    )
                raise

        timed_out = False
        backend_failure: Optional[Dict[str, Any]] = None
        while not reached_terminal_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            drained = self.gdb.drain_async(
                timeout_sec=min(0.25, remaining), max_rounds=100
            )
            if drained.get("backend_error"):
                # A dead debugger can no longer produce a terminal stop; stop
                # polling and surface the destroyed backend instead of burning
                # the remaining timeout budget.
                backend_failure = drained
                break
            new_responses = drained.get("responses", [])
            event_responses.extend(new_responses)
            if not new_responses:
                time.sleep(min(0.01, max(0.0, remaining)))
            # Check cancellation AFTER draining. If its stop notification is late,
            # the worker performs a bounded drain before propagating cancellation.
            raise_if_cancelled_after_sync()

        if backend_failure is not None:
            self._invalidate_gdb("debugger backend failed while waiting for a crash")
            return {
                "success": False,
                "command": "run_until_crash",
                "state": self.gdb.get_state(),
                "timeout_seconds": timeout,
                "parser_version": PARSER_VERSION,
                "parsed": {
                    "mode": resolved_mode,
                    "crashed": False,
                    "backend_error": True,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                },
                "raw_responses": {"launch": launch_result, "events": event_responses},
                "error": backend_failure.get(
                    "error", "Debugger backend failed while waiting for a crash"
                ),
                "type": backend_failure.get("type", "DebuggerBackendError"),
            }

        timeout_interrupt = None
        if timed_out:
            timeout_interrupt = self.gdb.interrupt(timeout_sec=1.0)
            event_responses.extend(timeout_interrupt.get("responses", []))
            timeout_synchronized = reached_terminal_stop()
            if not timeout_synchronized:
                killer = getattr(self.gdb, "kill_process_group", None)
                if callable(killer):
                    killer()
                self._invalidate_gdb(
                    "run timeout did not reach a synchronized terminal stop"
                )
            return {
                "success": False,
                "command": "run_until_crash",
                "state": self.gdb.get_state(),
                "timeout_seconds": timeout,
                "timed_out": True,
                "parser_version": PARSER_VERSION,
                "parsed": {
                    "mode": resolved_mode,
                    "crashed": False,
                    "interrupt_synchronized": timeout_synchronized,
                    "session_destroyed": not timeout_synchronized,
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                },
                "raw_responses": {
                    "launch": launch_result,
                    "events": event_responses,
                    "timeout_interrupt": timeout_interrupt,
                },
                "error": f"Inferior did not reach a terminal stop within {timeout}s",
                "type": "TimeoutError",
            }

        if cancellation_check is not None:
            cancellation_check()
        record = self.gdb.get_last_stop_record()
        stop = self._classify_stop(record, signals)
        crash_id = self._crash_id(record) if record is not None else None
        response: Dict[str, Any] = {
            "success": True,
            "command": "run_until_crash",
            "state": self.gdb.get_state(),
            "timeout_seconds": timeout,
            "parser_version": PARSER_VERSION,
            "parsed": {
                "mode": resolved_mode,
                "arguments": resolved_arguments,
                "crash_id": crash_id,
                "crashed": stop["crashed"],
                "stop": stop,
                "record": record,
                "fatal_signals": signals,
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
            "raw_responses": {"launch": launch_result, "events": event_responses},
        }
        record_is_live = LiveInferiorGuard.is_live(record, self.gdb.get_inferior_pid())
        if stop["crashed"] and crash_id is not None and record_is_live:
            initial_snapshot = self.capture_crash_snapshot(
                crash_id,
                stack_bytes=256,
                disassembly_bytes=64,
                include_mappings=True,
                fatal_signals=fatal_signals,
                cancellation_check=cancellation_check,
            )
            response["parsed"]["initial_snapshot"] = initial_snapshot.get(
                "parsed", {}
            ).get("crash_snapshot")
            response["parsed"]["snapshot_complete"] = bool(
                initial_snapshot.get("complete")
            )
            response["raw_responses"]["initial_snapshot"] = initial_snapshot.get(
                "raw_responses", {}
            )
            if initial_snapshot.get("warnings"):
                response.setdefault("warnings", []).extend(initial_snapshot["warnings"])
        if record is None or not record_is_live:
            # A terminal exit (including ``continue`` -> exit) may leave the last
            # crash record from a *previous* inferior in place.  We never treat a
            # non-live record as a fresh crash, and we surface why.
            if stop["crashed"] and crash_id is not None:
                response["parsed"]["crashed"] = False
                response["parsed"]["stop"] = {**stop, "crashed": False}
                response.setdefault("warnings", []).append(
                    "The most recent stop record is not attached to a live "
                    "inferior; it was not treated as a current crash"
                )
            else:
                response.setdefault("warnings", []).append(
                    "Inferior exited without a retained live stopped notification"
                )
        elif not stop["crashed"]:
            response.setdefault("warnings", []).append(
                "Execution stopped, but the stop is not classified as a crash; "
                "the tool did not auto-resume past it"
            )
        return response

    def _require_crash_record(
        self,
        expected_crash_id: str,
        fatal_signals: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if not expected_crash_id:
            raise ValueError("expected_crash_id cannot be empty")
        record = self.gdb.get_last_stop_record()
        if record is None:
            raise CrashRecordError("No GDB stop record is available")
        actual = self._crash_id(record)
        if actual != expected_crash_id:
            raise CrashRecordError(
                f"Crash id is stale: expected current '{actual}', got "
                f"'{expected_crash_id}'"
            )
        signals = self._effective_fatal_signals(expected_crash_id, fatal_signals)
        stop = self._classify_stop(record, signals)
        if not stop["crashed"]:
            raise CrashRecordError(
                f"Stop '{expected_crash_id}' is not a fatal signal crash"
            )
        return record

    @staticmethod
    def _integer(value: Any) -> Optional[int]:
        return parse_int(value)

    @staticmethod
    def _register_map(snapshot: Dict[str, Any]) -> Dict[str, Any]:
        return {
            str(item.get("name", "")).lower(): item.get("value")
            for item in snapshot.get("registers", [])
            if isinstance(item, dict) and item.get("name")
        }

    @classmethod
    def _register_alias(
        cls, register_map: Dict[str, Any], aliases: Sequence[str]
    ) -> Dict[str, Any]:
        for name in aliases:
            if name in register_map:
                return {
                    "name": name,
                    "value": register_map[name],
                    "integer": cls._integer(register_map[name]),
                }
        return {"name": None, "value": None, "integer": None}

    @staticmethod
    def _mapping_for(
        value: Optional[int], mappings: Sequence[Any]
    ) -> Optional[Dict[str, Any]]:
        return mapping_for(value, mappings)

    def _fault_address(self) -> Dict[str, Any]:
        attempts: Dict[str, Any] = {}
        for expression in (
            "$_siginfo._sifields._sigfault.si_addr",
            "$_siginfo.si_addr",
        ):
            result = self.debugger.evaluate_expression(expression)
            attempts[expression] = result
            if result.get("success"):
                value = result.get("parsed", {}).get("value")
                integer = self._integer(value)
                if integer is not None:
                    return {
                        "expression": expression,
                        "value": value,
                        "integer": integer,
                        "source": "gdb-mi",
                        "attempts": attempts,
                        "si_code": self._siginfo_code(attempts),
                    }
        return {
            "expression": None,
            "value": None,
            "integer": None,
            "source": "gdb-mi",
            "attempts": attempts,
            "si_code": self._siginfo_code(attempts),
        }

    def _siginfo_code(self, attempts: Dict[str, Any]) -> Optional[int]:
        """Best-effort ``$_siginfo.si_code`` (1=MAPERR, 2=ACCERR on SIGSEGV)."""
        result = self.debugger.evaluate_expression("$_siginfo.si_code")
        attempts["$_siginfo.si_code"] = result
        if not result.get("success"):
            return None
        return self._integer(result.get("parsed", {}).get("value"))

    # si_code values for SIGSEGV/SIGBUS faults (Linux asm-generic/siginfo.h).
    _SI_MAPERR = 1
    _SI_ACCERR = 2

    # Mnemonics whose operands are read-only: a memory destination is never
    # written, so ``cmp [mem], reg`` faults on a read, not a write.
    _READ_ONLY_MEMORY_MNEMONICS = frozenset({"cmp", "test", "cmn", "tst"})

    @classmethod
    def _memory_direction(cls, instruction: Any) -> Optional[str]:
        """Read/write direction of one faulting instruction (Intel syntax).

        Returns ``"read"`` / ``"write"`` only for unambiguous shapes;
        anything else (AT&T syntax, unknown mnemonic, missing operands)
        is ``None`` so callers report ``unknown`` instead of guessing.
        """
        if not isinstance(instruction, str):
            return None
        text = instruction.strip().lower()
        if not text or "%" in text:
            return None
        mnemonic = cls._instruction_mnemonic(text)
        if mnemonic is None:
            return None
        parts = text[len(mnemonic) :].strip().split(",", 1)
        operands = [part.strip() for part in parts if part.strip()]
        if mnemonic in {"push", "pushf", "pushfq"}:
            return "write"
        if mnemonic in {"pop", "popf", "popfq", "ret", "leave"}:
            return "read"
        if mnemonic in {"call", "jmp"} and any("[" in op for op in operands):
            return "read"
        if mnemonic in cls._READ_ONLY_MEMORY_MNEMONICS:
            return "read" if any("[" in op for op in operands) else None
        if len(operands) == 2:
            destination, source = operands
            if destination.startswith("[") or re.match(
                r"^(byte|word|dword|qword)\s+ptr\s*\[", destination
            ):
                return "write"
            if "[" in source:
                return "read"
        return None

    @classmethod
    def _fault_access(
        cls,
        fault_integer: Optional[int],
        pc_integer: Optional[int],
        instruction: Any,
        si_code: Optional[int],
    ) -> Dict[str, Any]:
        """Classify a memory fault as execute/read/write/unknown + evidence."""
        if (
            fault_integer is not None
            and pc_integer is not None
            and fault_integer == pc_integer
        ):
            return {"access": "execute", "si_code": si_code, "mnemonic": None}
        direction = cls._memory_direction(instruction)
        mnemonic = cls._instruction_mnemonic(instruction)
        if direction is not None:
            return {"access": direction, "si_code": si_code, "mnemonic": mnemonic}
        return {"access": "unknown", "si_code": si_code, "mnemonic": mnemonic}

    @staticmethod
    def _memory_contents(result: Dict[str, Any]) -> Optional[str]:
        return DebuggerOperations._memory_contents(result)

    @staticmethod
    def _json_bytes(value: Dict[str, Any]) -> bytes:
        """Serialize internal evidence canonically so retained history is immutable."""
        return json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")

    @staticmethod
    def _elf_build_id(path: str) -> Optional[str]:
        """Read a GNU ELF build id without treating its absence as an error."""
        try:
            with open(path, "rb") as stream:
                elf = ELFFile(stream)
                for section in elf.iter_sections():
                    if not isinstance(section, NoteSection):
                        continue
                    for note in section.iter_notes():
                        if str(note["n_type"]) == "NT_GNU_BUILD_ID":
                            value = note["n_desc"]
                            if isinstance(value, bytes):
                                return value.hex()
                            return re.sub(r"[^0-9a-fA-F]", "", str(value)).lower()
                for segment in elf.iter_segments():
                    if not isinstance(segment, NoteSegment):
                        continue
                    for note in segment.iter_notes():
                        if str(note["n_type"]) == "NT_GNU_BUILD_ID":
                            value = note["n_desc"]
                            if isinstance(value, bytes):
                                return value.hex()
                            return re.sub(r"[^0-9a-fA-F]", "", str(value)).lower()
        except (ELFError, OSError, TypeError, ValueError):
            return None
        return None

    @classmethod
    def _module_identity(
        cls, mapping: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Return a capture-time identity for a file-backed mapping.

        A basename alone is unsafe for deduplication because two unrelated targets
        can both be named ``target``. Special and anonymous mappings are explicitly
        unstable and are never promoted to module-relative identities.
        """
        if not isinstance(mapping, dict):
            return None
        value = mapping.get("file")
        if value is None:
            return None
        raw_path = str(value).strip()
        if not raw_path:
            return None
        normalized = raw_path.replace("\\", "/").rstrip("/")
        if not normalized:
            return None
        if normalized.startswith("[") and normalized.endswith("]"):
            return {
                "kind": "special_mapping",
                "name": normalized,
                "stable": False,
            }
        if normalized.lower() in {"anonymous", "anon"}:
            return {"kind": "anonymous_mapping", "stable": False}
        if normalized.endswith(" (deleted)"):
            return {
                "kind": "deleted_file",
                "path": normalized,
                "stable": False,
            }

        if (
            os.name == "nt"
            and normalized.startswith("/")
            and not normalized.startswith("//")
        ):
            # A POSIX-style absolute path observed from an inferior (e.g. a
            # Linux container target) is not a Windows host path; resolving
            # it would prepend the current drive letter and fabricate an
            # identity (``/workspace/target`` -> ``D:/Workspace/target``).
            # It is also not *verifiable* from this host, so a failed stat
            # below must not demote stability (see below).
            host_path = normalized
            verifiable = False
        else:
            host_path = os.path.realpath(raw_path)
            verifiable = True
        resolved = host_path.replace("\\", "/").rstrip("/")
        identity: Dict[str, Any] = {
            "kind": "file",
            "path": resolved,
            "name": resolved.rsplit("/", 1)[-1],
            "stable": True,
        }
        try:
            stat_result = os.stat(host_path)
        except OSError:
            if not verifiable:
                # A container path observed from a foreign host cannot be
                # stated from here; keep the verbatim identity without
                # claiming verification either way.
                identity["identity_source"] = "resolved_path"
                return identity
            # Host path that cannot be stated: a different build at the same
            # path would dedup together, so demote stability instead of
            # claiming it.
            identity["stable"] = False
            identity["identity_source"] = "resolved_path_unverified"
            return identity

        build_id = cls._elf_build_id(host_path)
        if build_id:
            identity["build_id"] = build_id
            identity["identity_source"] = "resolved_path+elf_build_id"
        else:
            identity["file_metadata"] = {
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "size": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
            }
            identity["identity_source"] = "resolved_path+file_metadata"
        return identity

    @classmethod
    def _target_identity(cls, crash_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Select the primary executable's identity from captured mappings."""
        debugger_snapshot = crash_snapshot.get("debugger", {})
        mappings = (
            debugger_snapshot.get("mappings", [])
            if isinstance(debugger_snapshot, dict)
            else []
        )
        if not isinstance(mappings, list):
            return {"kind": "unknown", "stable": False}
        for mapping in mappings:
            if not isinstance(mapping, dict) or not mapping.get("executable"):
                continue
            identity = cls._module_identity(mapping)
            if identity and identity.get("kind") == "file":
                return identity
        return {"kind": "unknown", "stable": False}

    @classmethod
    def _normalized_address(
        cls, value: Any, mapping: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Return an ASLR-stable address token without embedding a raw address."""
        integer = cls._integer(value)
        if integer is None:
            return {"kind": "unknown"}
        if isinstance(mapping, dict):
            start = cls._integer(mapping.get("start"))
            identity = cls._module_identity(mapping)
            if (
                start is not None
                and start <= integer
                and identity is not None
                and identity.get("kind") == "file"
                and identity.get("stable") is True
            ):
                return {
                    "kind": "module_offset",
                    "module": identity.get("name"),
                    "module_identity": identity,
                    "offset": integer - start,
                }
            if identity is not None:
                return {
                    "kind": str(identity.get("kind") or "unstable_mapping"),
                    "mapping": identity.get("name") or identity.get("path"),
                    "stable": False,
                }
        if integer < 4096:
            return {"kind": "near_null", "offset": integer}
        # An absolute unmapped address is deliberately omitted: including it would
        # turn ASLR or allocator placement into a false crash-family difference.
        return {"kind": "unmapped"}

    @classmethod
    def _faulting_instruction(
        cls,
        debugger_snapshot: Dict[str, Any],
        program_counter: Dict[str, Any],
        program_counter_mapping: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Select the instruction whose typed MI address exactly equals the PC."""
        pc_integer = program_counter.get("integer")
        instructions = debugger_snapshot.get("instructions", [])
        if not isinstance(instructions, list) or pc_integer is None:
            return {
                "status": "unknown",
                "address": None,
                "address_integer": None,
                "function": None,
                "offset": None,
                "instruction": None,
                "location": {"kind": "unknown"},
                "source": "gdb-mi",
            }
        for item in instructions:
            if not isinstance(item, dict):
                continue
            address_integer = cls._integer(item.get("address"))
            if address_integer != pc_integer:
                continue
            return {
                "status": "observed",
                "address": item.get("address"),
                "address_integer": address_integer,
                "function": item.get("function"),
                "offset": item.get("offset"),
                "instruction": item.get("instruction"),
                "location": cls._normalized_address(
                    address_integer, program_counter_mapping
                ),
                "source": "gdb-mi",
            }
        return {
            "status": "unknown",
            "address": None,
            "address_integer": None,
            "function": None,
            "offset": None,
            "instruction": None,
            "location": cls._normalized_address(
                program_counter.get("integer"), program_counter_mapping
            ),
            "source": "gdb-mi",
        }

    @classmethod
    def _classification_for_snapshot(
        cls, crash_snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a conservative primitive hypothesis from typed snapshot evidence."""
        analysis = crash_snapshot.get("analysis", {})
        stop = crash_snapshot.get("stop", {})
        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(stop, dict):
            stop = {}
        signal = stop.get("signal") or {}
        signal_name = signal.get("name") if isinstance(signal, dict) else None
        category = str(analysis.get("category") or "unknown")
        fault_mapping = analysis.get("fault_mapping")
        mapping_name = (
            str((fault_mapping or {}).get("file", "")).lower()
            if isinstance(fault_mapping, dict)
            else ""
        )
        risk = analysis.get("risk_indicators")
        if not isinstance(risk, dict):
            risk = {}
        candidates: List[Dict[str, Any]] = []

        def candidate(
            kind: str,
            confidence: str,
            summary: str,
            evidence: Sequence[Dict[str, Any]],
        ) -> None:
            candidates.append(
                {
                    "kind": kind,
                    "assessment": "possible",
                    "confidence": confidence,
                    "summary": summary,
                    "evidence": list(evidence),
                }
            )

        if category == "invalid_instruction_pointer":
            candidate(
                "invalid_control_flow_target",
                "medium",
                "The stopped program counter is not backed by executable memory.",
                (
                    {
                        "field": "analysis.category",
                        "value": category,
                        "source": "local",
                    },
                    {
                        "field": "analysis.fault_access",
                        "value": analysis.get("fault_access"),
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "null_page_access":
            candidate(
                "bad_pointer_dereference",
                "medium",
                "The observed fault address is in the near-null page.",
                (
                    {
                        "field": "analysis.fault_address",
                        "value": analysis.get("fault_address"),
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "memory_access_violation":
            candidate(
                "invalid_memory_access",
                "low",
                "A memory-fault signal was observed, but pointer origin is unknown.",
                (
                    {
                        "field": "stop.signal.name",
                        "value": signal_name,
                        "source": "gdb-mi",
                    },
                    {
                        "field": "analysis.fault_access",
                        "value": analysis.get("fault_access"),
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "abort":
            candidate(
                "abort_or_runtime_guard",
                "low",
                "SIGABRT can indicate an explicit abort or a runtime integrity guard.",
                (
                    {
                        "field": "stop.signal.name",
                        "value": signal_name,
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "illegal_instruction":
            candidate(
                "illegal_instruction_execution",
                "medium",
                "The processor rejected the instruction at the stopped PC.",
                (
                    {
                        "field": "analysis.faulting_instruction",
                        "value": analysis.get("faulting_instruction"),
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "arithmetic_exception":
            candidate(
                "arithmetic_fault",
                "medium",
                "An arithmetic exception was reported by GDB.",
                (
                    {
                        "field": "stop.signal.name",
                        "value": signal_name,
                        "source": "gdb-mi",
                    },
                ),
            )
        elif category == "bad_system_call":
            candidate(
                "system_call_policy_violation",
                "medium",
                "The process received SIGSYS while executing a system call.",
                (
                    {
                        "field": "stop.signal.name",
                        "value": signal_name,
                        "source": "gdb-mi",
                    },
                ),
            )

        if "[stack]" in mapping_name:
            candidate(
                "stack_memory_fault",
                "low",
                "The observed fault address lies in a stack mapping; overflow is not proven.",
                (
                    {
                        "field": "analysis.fault_mapping.file",
                        "value": (fault_mapping or {}).get("file"),
                        "source": "pwndbg-cli",
                    },
                ),
            )
        if "[heap]" in mapping_name:
            candidate(
                "heap_memory_fault",
                "low",
                "The observed fault address lies in a heap mapping; corruption is not proven.",
                (
                    {
                        "field": "analysis.fault_mapping.file",
                        "value": (fault_mapping or {}).get("file"),
                        "source": "pwndbg-cli",
                    },
                ),
            )
        if risk.get("stack_executable"):
            candidate(
                "executable_stack_exposure",
                "low",
                "The current stack mapping is executable; this is a mitigation signal, not a primitive.",
                (
                    {
                        "field": "analysis.risk_indicators.stack_executable",
                        "value": True,
                        "source": "pwndbg-cli",
                    },
                ),
            )

        primary = (
            candidates[0]
            if candidates
            else {
                "kind": "unknown",
                "assessment": "unknown",
                "confidence": "unknown",
                "summary": "Typed evidence is insufficient to infer a crash primitive.",
                "evidence": [],
            }
        )
        return {
            "schema": CRASH_CLASSIFICATION_SCHEMA,
            "crash_id": crash_snapshot.get("crash_id"),
            "category": category,
            "primary": primary,
            "candidates": candidates,
            "exploitability": "unknown",
            "limitations": [
                "Classification is heuristic and does not prove data-flow control.",
                "Stack overflow and heap corruption require allocator or taint evidence.",
            ],
            "source": "local",
        }

    @staticmethod
    def _instruction_mnemonic(value: Any) -> Optional[str]:
        if value is None:
            return None
        tokens = str(value).strip().lower().split()
        prefixes = {"lock", "rep", "repe", "repz", "repne", "repnz"}
        while tokens and tokens[0] in prefixes:
            tokens.pop(0)
        return tokens[0] if tokens else None

    @classmethod
    def _fingerprint_for_snapshot(
        cls,
        crash_snapshot: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a versioned fingerprint from ASLR-normalized stable fields."""
        analysis = crash_snapshot.get("analysis", {})
        debugger_snapshot = crash_snapshot.get("debugger", {})
        stop = crash_snapshot.get("stop", {})
        if not isinstance(analysis, dict):
            analysis = {}
        if not isinstance(debugger_snapshot, dict):
            debugger_snapshot = {}
        if not isinstance(stop, dict):
            stop = {}
        signal = stop.get("signal") or {}
        signal_name = signal.get("name") if isinstance(signal, dict) else None
        pc = analysis.get("program_counter") or {}
        fault = analysis.get("fault_address") or {}
        pc_location = cls._normalized_address(
            pc.get("integer") if isinstance(pc, dict) else None,
            analysis.get("program_counter_mapping"),
        )
        fault_location = cls._normalized_address(
            fault.get("integer") if isinstance(fault, dict) else None,
            analysis.get("fault_mapping"),
        )
        faulting = analysis.get("faulting_instruction") or {}
        frames = debugger_snapshot.get("frames", [])
        mappings = debugger_snapshot.get("mappings", [])
        normalized_frames: List[Dict[str, Any]] = []
        if isinstance(frames, list):
            for frame in frames[:5]:
                if not isinstance(frame, dict):
                    continue
                address = cls._integer(frame.get("address"))
                mapping = (
                    cls._mapping_for(address, mappings)
                    if isinstance(mappings, list)
                    else None
                )
                location = cls._normalized_address(address, mapping)
                function = frame.get("function")
                if location.get("kind") == "unknown" and not function:
                    continue
                normalized_frames.append(
                    {
                        "level": frame.get("level"),
                        "function": function,
                        "location": location,
                    }
                )
        primary = classification.get("primary") or {}
        target_identity = crash_snapshot.get("target_identity")
        if not isinstance(target_identity, dict):
            target_identity = cls._target_identity(crash_snapshot)
        instruction_canonical: Dict[str, Any] = {
            "mnemonic": cls._instruction_mnemonic(
                faulting.get("instruction") if isinstance(faulting, dict) else None
            ),
            "function": (
                faulting.get("function") if isinstance(faulting, dict) else None
            ),
            "location": (
                faulting.get("location")
                if isinstance(faulting, dict)
                else {"kind": "unknown"}
            ),
        }
        canonical: Dict[str, Any] = {
            "target_identity": target_identity,
            "signal": signal_name,
            "category": analysis.get("category"),
            "fault_access": analysis.get("fault_access"),
            "primitive": primary.get("kind") if isinstance(primary, dict) else None,
            "program_counter": pc_location,
            "fault_address": fault_location,
            "instruction": instruction_canonical,
            "frames": normalized_frames,
        }
        stable_pc = (
            pc_location.get("kind") == "module_offset"
            and isinstance(pc_location.get("module_identity"), dict)
            and pc_location["module_identity"].get("stable") is True
        )
        stable_target = target_identity.get("stable") is True
        confidence = (
            "high"
            if (
                signal_name
                and stable_target
                and stable_pc
                and instruction_canonical["mnemonic"]
            )
            else (
                "medium"
                if signal_name and stable_target and (stable_pc or normalized_frames)
                else "low"
            )
        )
        digest = hashlib.sha256(cls._json_bytes(canonical)).hexdigest()
        return {
            "schema": CRASH_FINGERPRINT_SCHEMA,
            "algorithm": "sha256",
            "digest": digest,
            "confidence": confidence,
            "normalization": (
                "target-identity-bound module offsets; raw unmapped and unstable "
                "mapping addresses omitted"
            ),
            "canonical": canonical,
            "source": "local",
        }

    @classmethod
    def _descriptor_for_snapshot(cls, crash_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        classification = crash_snapshot.get("classification")
        if not isinstance(classification, dict):
            classification = cls._classification_for_snapshot(crash_snapshot)
        signature = crash_snapshot.get("signature")
        if not isinstance(signature, dict):
            signature = cls._fingerprint_for_snapshot(crash_snapshot, classification)
        return {
            "crash_id": crash_snapshot.get("crash_id"),
            "captured_at": crash_snapshot.get("captured_at"),
            "capture_options": crash_snapshot.get("capture_options"),
            "target_identity": crash_snapshot.get("target_identity"),
            "stop": crash_snapshot.get("stop"),
            "classification": classification,
            "signature": signature,
        }

    def _remember_snapshot(self, crash_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Retain immutable evidence under count and aggregate-byte limits."""
        crash_id = str(crash_snapshot.get("crash_id") or "")
        if not crash_id:
            raise CrashRecordError("Crash snapshot has no crash_id")
        serialized_snapshot = self._json_bytes(crash_snapshot)
        snapshot_json: Optional[bytes] = serialized_snapshot
        descriptor = self._descriptor_for_snapshot(crash_snapshot)
        descriptor["snapshot_size_bytes"] = len(serialized_snapshot)
        retain_snapshot = len(serialized_snapshot) < MAX_CRASH_HISTORY_BYTES
        descriptor["snapshot_retained"] = retain_snapshot
        descriptor_json = self._json_bytes(descriptor)
        if len(descriptor_json) > MAX_CRASH_HISTORY_BYTES:
            raise CrashRecordError(
                "Crash descriptor exceeds the bounded history budget"
            )
        if len(serialized_snapshot) + len(descriptor_json) > MAX_CRASH_HISTORY_BYTES:
            snapshot_json = None
            descriptor["snapshot_retained"] = False
            descriptor_json = self._json_bytes(descriptor)
        entry = _CrashHistoryEntry(
            crash_id=crash_id,
            captured_at=str(crash_snapshot.get("captured_at") or ""),
            snapshot_json=snapshot_json,
            descriptor_json=descriptor_json,
        )
        with self._history_lock:
            previous = self._history.pop(crash_id, None)
            if previous is not None:
                self._history_bytes -= previous.retained_bytes
            while self._history and (
                len(self._history) >= MAX_CRASH_HISTORY_ENTRIES
                or self._history_bytes + entry.retained_bytes > MAX_CRASH_HISTORY_BYTES
            ):
                _, removed = self._history.popitem(last=False)
                self._history_bytes -= removed.retained_bytes
            self._history[crash_id] = entry
            self._history_bytes += entry.retained_bytes
        return json.loads(descriptor_json)

    def _history_descriptor(self, crash_id: str) -> Optional[Dict[str, Any]]:
        with self._history_lock:
            entry = self._history.get(crash_id)
            payload = entry.descriptor_json if entry is not None else None
        return json.loads(payload) if payload is not None else None

    def _history_snapshot(self, crash_id: str) -> Optional[Dict[str, Any]]:
        with self._history_lock:
            entry = self._history.get(crash_id)
            payload = entry.snapshot_json if entry is not None else None
        if payload is not None:
            return json.loads(payload)
        cached = self._latest_snapshot
        snapshot = (
            cached.get("parsed", {}).get("crash_snapshot", {})
            if isinstance(cached, dict)
            else {}
        )
        if isinstance(snapshot, dict) and snapshot.get("crash_id") == crash_id:
            # Return a serialized copy: no caller can mutate the live cache.
            return json.loads(self._json_bytes(snapshot))
        return None

    def capture_crash_snapshot(
        self,
        expected_crash_id: str,
        *,
        stack_bytes: int = 256,
        disassembly_bytes: int = 64,
        include_mappings: bool = True,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture one canonical crash snapshot from typed MI components."""
        if not 1 <= stack_bytes <= MAX_STACK_BYTES:
            raise ValueError(f"stack_bytes must be between 1 and {MAX_STACK_BYTES}")
        signals = self._effective_fatal_signals(expected_crash_id, fatal_signals)
        record = self._require_crash_record(expected_crash_id, signals)
        try:
            LiveInferiorGuard.require_live(
                record,
                self.gdb.get_inferior_pid(),
                detail=(
                    "Crash record is no longer attached to a stopped live inferior"
                ),
            )
        except LiveInferiorError as exc:
            raise CrashRecordError(str(exc)) from exc
        if cancellation_check is not None:
            cancellation_check()

        deadline = self.debugger.snapshot_deadline()
        mapping_result = self.inspection.memory_maps() if include_mappings else None
        debugger_result = self.debugger.capture_snapshot(
            disassembly_bytes=disassembly_bytes,
            mapping_result=mapping_result,
            deadline=deadline,
            cancellation_check=cancellation_check,
        )
        snapshot = debugger_result.get("parsed", {}).get("snapshot", {})
        register_map = self._register_map(snapshot)
        pc = self._register_alias(register_map, ("rip", "eip", "pc"))
        sp = self._register_alias(register_map, ("rsp", "esp", "sp"))
        bp = self._register_alias(register_map, ("rbp", "ebp", "fp", "x29"))
        mappings = snapshot.get("mappings", []) if isinstance(snapshot, dict) else []
        stack_result: Optional[Dict[str, Any]] = None
        stack_hex = None
        if sp["integer"] is not None:
            stack_result = self.gdb.read_memory_bytes(str(sp["value"]), stack_bytes)
            stack_hex = self._memory_contents(stack_result)
        stack_success = stack_hex is not None
        stop = self._classify_stop(record, signals)
        signal_name = str((stop.get("signal") or {}).get("name") or "")
        memory_fault_signal = signal_name in {"SIGSEGV", "SIGBUS"}
        fault = (
            self._fault_address()
            if memory_fault_signal
            else {
                "expression": None,
                "value": None,
                "integer": None,
                "source": "gdb-mi",
                "attempts": {},
            }
        )
        fault_mapping = self._mapping_for(fault["integer"], mappings)
        pc_mapping = self._mapping_for(pc["integer"], mappings)
        sp_mapping = self._mapping_for(sp["integer"], mappings)
        # Placeholder: refined below once the faulting instruction is known.
        fault_access_detail: Dict[str, Any] = {
            "access": (
                "execute"
                if fault["integer"] is not None and fault["integer"] == pc["integer"]
                else "unknown"
            ),
            "si_code": fault.get("si_code"),
            "mnemonic": None,
        }
        fault_access = fault_access_detail["access"]
        category = {
            "SIGSEGV": "memory_access_violation",
            "SIGBUS": "memory_access_violation",
            "SIGABRT": "abort",
            "SIGILL": "illegal_instruction",
            "SIGFPE": "arithmetic_exception",
            "SIGSYS": "bad_system_call",
        }.get(signal_name, "signal_crash")
        if (
            memory_fault_signal
            and fault["integer"] is not None
            and fault["integer"] < 4096
        ):
            category = "null_page_access"
        elif (
            memory_fault_signal
            and fault_access == "execute"
            and not (pc_mapping and pc_mapping.get("executable"))
        ):
            category = "invalid_instruction_pointer"
        faulting_instruction = self._faulting_instruction(snapshot, pc, pc_mapping)
        if fault_access == "unknown" and memory_fault_signal:
            refined = self._fault_access(
                fault["integer"],
                pc["integer"],
                (
                    faulting_instruction.get("instruction")
                    if isinstance(faulting_instruction, dict)
                    else None
                ),
                fault.get("si_code"),
            )
            fault_access_detail = refined
            fault_access = refined["access"]

        component_status = {
            "stop_record": True,
            "debugger_snapshot": bool(debugger_result.get("success")),
            "mappings": (
                bool(mapping_result and mapping_result.get("success"))
                if include_mappings
                else True
            ),
            "fault_address": (
                fault["integer"] is not None if memory_fault_signal else True
            ),
            "stack_memory": stack_success,
        }
        crash_snapshot = {
            "snapshot_version": CRASH_SNAPSHOT_VERSION,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "crash_id": expected_crash_id,
            "session_id": self.session_id,
            "inferior_pid": self.gdb.get_inferior_pid(),
            "fatal_signals": signals,
            "capture_options": {
                "stack_bytes": stack_bytes,
                "disassembly_bytes": disassembly_bytes,
                "include_mappings": include_mappings,
            },
            "stop_record": record,
            "stop": stop,
            "analysis": {
                "category": category,
                "fault_address": {
                    key: value for key, value in fault.items() if key != "attempts"
                },
                "fault_access": fault_access,
                "fault_access_detail": fault_access_detail,
                "program_counter": pc,
                "stack_pointer": sp,
                "frame_pointer": bp,
                "faulting_instruction": faulting_instruction,
                "fault_mapping": fault_mapping,
                "program_counter_mapping": pc_mapping,
                "stack_pointer_mapping": sp_mapping,
                "risk_indicators": {
                    "near_null_fault": bool(
                        fault["integer"] is not None and fault["integer"] < 4096
                    ),
                    "instruction_pointer_unmapped": bool(
                        pc["integer"] is not None and pc_mapping is None
                    ),
                    "stack_executable": bool(
                        sp_mapping and sp_mapping.get("executable")
                    ),
                },
                "exploitability": "unknown",
                "exploitability_note": (
                    "Automated triage does not claim exploitability; validate data "
                    "flow, control, mitigations, and reproducibility separately."
                ),
            },
            "debugger": snapshot,
            "stack": {
                "address": sp.get("value"),
                "requested_bytes": stack_bytes,
                "captured_bytes": len(stack_hex) // 2 if stack_hex else 0,
                "data_hex": stack_hex,
                "source": "gdb-mi",
            },
            "component_status": component_status,
            "sources": {
                "stop": "gdb-mi",
                "debugger": "gdb-mi",
                "mappings": (
                    "pwndbg-cli"
                    if include_mappings
                    and mapping_result is not None
                    and mapping_result.get("success")
                    else ("unavailable" if include_mappings else None)
                ),
                "fault_address": "gdb-mi",
                "stack": "gdb-mi",
            },
        }
        crash_snapshot["target_identity"] = self._target_identity(crash_snapshot)
        classification = self._classification_for_snapshot(crash_snapshot)
        signature = self._fingerprint_for_snapshot(crash_snapshot, classification)
        crash_snapshot["classification"] = classification
        crash_snapshot["signature"] = signature
        history_descriptor = self._remember_snapshot(crash_snapshot)
        complete = all(component_status.values())
        usable = bool(debugger_result.get("usable") and component_status["stop_record"])
        result: Dict[str, Any] = {
            "success": usable,
            "usable": usable,
            "complete": complete,
            "command": "capture_crash_snapshot",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_snapshot": crash_snapshot,
                "component_status": component_status,
                "usable": usable,
                "complete": complete,
                "classification": classification,
                "signature": signature,
                "history": {
                    "retained": True,
                    "snapshot_retained": history_descriptor.get(
                        "snapshot_retained", False
                    ),
                },
            },
            "raw_responses": {
                "debugger": debugger_result.get("raw_responses", {}),
                "mappings": (
                    mapping_result.get("raw_responses", []) if mapping_result else []
                ),
                "fault_address": fault["attempts"],
                "stack": stack_result.get("responses", []) if stack_result else [],
            },
        }
        failed = [name for name, succeeded in component_status.items() if not succeeded]
        if failed:
            result["warnings"] = ["Failed crash components: " + ", ".join(failed)]
        if not usable:
            result["error"] = "No usable typed debugger snapshot could be captured"
            result["type"] = "CrashSnapshotError"
        self._latest_snapshot = result
        return result

    def analyze_crash(
        self,
        expected_crash_id: str,
        *,
        stack_bytes: Optional[int] = 256,
        disassembly_bytes: Optional[int] = 64,
        include_mappings: Optional[bool] = True,
        refresh: bool = False,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return typed crash analysis, reusing the exact captured record by default."""
        cached = self._latest_snapshot
        cached_id = (
            cached.get("parsed", {}).get("crash_snapshot", {}).get("crash_id")
            if isinstance(cached, dict)
            else None
        )
        cached_snapshot = (
            cached.get("parsed", {}).get("crash_snapshot", {})
            if isinstance(cached, dict)
            else {}
        )
        cached_signals = (
            cached_snapshot.get("fatal_signals")
            if isinstance(cached_snapshot, dict)
            else None
        )
        cached_options = (
            cached_snapshot.get("capture_options")
            if isinstance(cached_snapshot, dict)
            else None
        )
        cached_options_for_crash = (
            cached_options
            if cached_id == expected_crash_id and isinstance(cached_options, dict)
            else {}
        )
        resolved_stack_bytes = int(
            cached_options_for_crash.get("stack_bytes", 256)
            if stack_bytes is None
            else stack_bytes
        )
        resolved_disassembly_bytes = int(
            cached_options_for_crash.get("disassembly_bytes", 64)
            if disassembly_bytes is None
            else disassembly_bytes
        )
        resolved_include_mappings = bool(
            cached_options_for_crash.get("include_mappings", True)
            if include_mappings is None
            else include_mappings
        )
        requested_options = {
            "stack_bytes": resolved_stack_bytes,
            "disassembly_bytes": resolved_disassembly_bytes,
            "include_mappings": resolved_include_mappings,
        }
        if fatal_signals is not None:
            effective_signals = self._fatal_signals(fatal_signals)
        elif cached_id == expected_crash_id and isinstance(cached_signals, list):
            effective_signals = self._fatal_signals(cached_signals)
        else:
            effective_signals = self._fatal_signals(None)
        cache_matches_policy = (
            cached_id == expected_crash_id
            and isinstance(cached_signals, list)
            and self._fatal_signals(cached_signals) == effective_signals
            and cached_options == requested_options
        )
        snapshot_result = cached
        reused_cached_snapshot = not refresh and cache_matches_policy
        if not reused_cached_snapshot:
            snapshot_result = self.capture_crash_snapshot(
                expected_crash_id,
                stack_bytes=resolved_stack_bytes,
                disassembly_bytes=resolved_disassembly_bytes,
                include_mappings=resolved_include_mappings,
                fatal_signals=effective_signals,
                cancellation_check=cancellation_check,
            )
        assert snapshot_result is not None
        crash_snapshot = snapshot_result.get("parsed", {}).get("crash_snapshot", {})
        classification = (
            crash_snapshot.get("classification", {})
            if isinstance(crash_snapshot, dict)
            else {}
        )
        signature = (
            crash_snapshot.get("signature", {})
            if isinstance(crash_snapshot, dict)
            else {}
        )
        return {
            "success": bool(snapshot_result.get("success")),
            "usable": bool(snapshot_result.get("usable")),
            "complete": bool(snapshot_result.get("complete")),
            "command": "analyze_crash",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_id": expected_crash_id,
                "analysis": crash_snapshot.get("analysis", {}),
                "stop": crash_snapshot.get("stop", {}),
                "crash_snapshot": crash_snapshot,
                "classification": classification,
                "signature": signature,
                "capture_options": requested_options,
                "component_status": snapshot_result.get("parsed", {}).get(
                    "component_status", {}
                ),
                "cached": reused_cached_snapshot,
            },
            "raw_responses": snapshot_result.get("raw_responses", {}),
            **(
                {
                    "warnings": snapshot_result.get("warnings", []),
                    "error": snapshot_result.get(
                        "error", "Crash analysis has no usable snapshot"
                    ),
                    "type": snapshot_result.get("type", "CrashAnalysisError"),
                }
                if not snapshot_result.get("success")
                else (
                    {"warnings": snapshot_result.get("warnings", [])}
                    if snapshot_result.get("warnings")
                    else {}
                )
            ),
        }

    def classify_crash(
        self,
        expected_crash_id: str,
        *,
        stack_bytes: Optional[int] = 256,
        disassembly_bytes: Optional[int] = 64,
        include_mappings: Optional[bool] = True,
        refresh: bool = False,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return a versioned conservative primitive classification."""
        analysis_result = self.analyze_crash(
            expected_crash_id,
            stack_bytes=stack_bytes,
            disassembly_bytes=disassembly_bytes,
            include_mappings=include_mappings,
            refresh=refresh,
            fatal_signals=fatal_signals,
            cancellation_check=cancellation_check,
        )
        parsed = analysis_result.get("parsed", {})
        classification = parsed.get("classification", {})
        signature = parsed.get("signature", {})
        result: Dict[str, Any] = {
            "success": bool(analysis_result.get("success")),
            "usable": bool(analysis_result.get("usable")),
            "complete": bool(analysis_result.get("complete")),
            "command": "classify_crash",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_id": expected_crash_id,
                "classification": classification,
                "signature": signature,
                "capture_options": parsed.get("capture_options", {}),
                "component_status": parsed.get("component_status", {}),
                "cached": parsed.get("cached", False),
            },
            "raw_responses": analysis_result.get("raw_responses", {}),
        }
        if analysis_result.get("warnings"):
            result["warnings"] = analysis_result["warnings"]
        if not analysis_result.get("success"):
            result["error"] = analysis_result.get(
                "error", "Crash classification has no usable snapshot"
            )
            result["type"] = analysis_result.get("type", "CrashClassificationError")
        return result
