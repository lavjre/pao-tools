"""Typed crash-stop collection, triage, and core-file generation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from elftools.elf.sections import NoteSection
from pwnlib.util.cyclic import de_bruijn  # type: ignore[import-untyped]

from paomcp.artifacts import (
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    WriterNotQuiescentError,
)
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorError,
    LiveInferiorGuard,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

PARSER_VERSION = "1"
CRASH_SNAPSHOT_VERSION = "1"
DEFAULT_FATAL_SIGNALS = frozenset(
    {"SIGSEGV", "SIGABRT", "SIGBUS", "SIGILL", "SIGFPE", "SIGSYS"}
)
SAFE_SIGNAL_PATTERN = re.compile(r"^SIG[A-Z0-9]{1,31}$")
MAX_RUN_TIMEOUT_SECONDS = 60 * 60
DEFAULT_RUN_TIMEOUT_SECONDS = float(
    os.environ.get("PAO_CRASH_RUN_TIMEOUT_SECONDS", "60")
)
DEFAULT_CORE_TIMEOUT_SECONDS = float(
    os.environ.get("PAO_CORE_DUMP_TIMEOUT_SECONDS", "120")
)
DEFAULT_MAX_CORE_DUMP_BYTES = int(
    os.environ.get("PAO_MAX_CORE_DUMP_BYTES", str(1024 * 1024 * 1024))
)
MAX_CONFIGURED_CORE_DUMP_BYTES = 64 * 1024**3
MAX_STACK_BYTES = 1024 * 1024
MAX_CYCLIC_LENGTH = 64 * 1024 * 1024
MAX_CYCLIC_STACK_SCAN_BYTES = 1024 * 1024
MAX_CYCLIC_SCAN_MATCHES = 256
MAX_CYCLIC_VALUE_CHARS = 132
MAX_CYCLIC_ALPHABET_CHARS = 256
MAX_ELF_CORE_NOTE_ENTRIES = 65536
MAX_CRASH_HISTORY_ENTRIES = 32
MAX_CRASH_HISTORY_BYTES = 64 * 1024 * 1024
CRASH_CLASSIFICATION_SCHEMA = "pao.crash-classification.v1"
CRASH_FINGERPRINT_SCHEMA = "pao.crash-fingerprint.v1"
CRASH_COMPARISON_SCHEMA = "pao.crash-comparison.v1"


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


class CoreDumpGenerationError(RuntimeError):
    """Raised when GDB cannot produce a complete, validated ELF core file."""


class CrashTriage:
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
        while not reached_terminal_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            drained = self.gdb.drain_async(
                timeout_sec=min(0.25, remaining), max_rounds=100
            )
            new_responses = drained.get("responses", [])
            event_responses.extend(new_responses)
            if not new_responses:
                time.sleep(min(0.01, max(0.0, remaining)))
            # Check cancellation AFTER draining. If its stop notification is late,
            # the worker performs a bounded drain before propagating cancellation.
            raise_if_cancelled_after_sync()

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
                    }
        return {
            "expression": None,
            "value": None,
            "integer": None,
            "source": "gdb-mi",
            "attempts": attempts,
        }

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

        host_path = os.path.realpath(raw_path)
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
            identity["identity_source"] = "resolved_path"
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
        fault_access = (
            "execute"
            if fault["integer"] is not None and fault["integer"] == pc["integer"]
            else "unknown"
        )
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

    @staticmethod
    def _comparable_value(value: Any) -> bool:
        if value is None or value == "" or value == []:
            return False
        if isinstance(value, dict) and value.get("kind") in {"unknown", "unmapped"}:
            return False
        return True

    @classmethod
    def _comparison_fields(
        cls, left: Dict[str, Any], right: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        left_canonical = left.get("signature", {}).get("canonical", {})
        right_canonical = right.get("signature", {}).get("canonical", {})
        left_instruction = left_canonical.get("instruction", {})
        right_instruction = right_canonical.get("instruction", {})
        left_frames = left_canonical.get("frames", [])
        right_frames = right_canonical.get("frames", [])
        fields = (
            (
                "target_identity",
                left_canonical.get("target_identity"),
                right_canonical.get("target_identity"),
            ),
            ("signal", left_canonical.get("signal"), right_canonical.get("signal")),
            (
                "category",
                left_canonical.get("category"),
                right_canonical.get("category"),
            ),
            (
                "primitive",
                left_canonical.get("primitive"),
                right_canonical.get("primitive"),
            ),
            (
                "program_counter",
                left_canonical.get("program_counter"),
                right_canonical.get("program_counter"),
            ),
            (
                "fault_address",
                left_canonical.get("fault_address"),
                right_canonical.get("fault_address"),
            ),
            (
                "instruction.mnemonic",
                (
                    left_instruction.get("mnemonic")
                    if isinstance(left_instruction, dict)
                    else None
                ),
                (
                    right_instruction.get("mnemonic")
                    if isinstance(right_instruction, dict)
                    else None
                ),
            ),
            (
                "instruction.function",
                (
                    left_instruction.get("function")
                    if isinstance(left_instruction, dict)
                    else None
                ),
                (
                    right_instruction.get("function")
                    if isinstance(right_instruction, dict)
                    else None
                ),
            ),
            (
                "top_frame",
                (
                    left_frames[0]
                    if isinstance(left_frames, list) and left_frames
                    else None
                ),
                (
                    right_frames[0]
                    if isinstance(right_frames, list) and right_frames
                    else None
                ),
            ),
        )
        evidence: List[Dict[str, Any]] = []
        for field, left_value, right_value in fields:
            comparable = cls._comparable_value(left_value) and cls._comparable_value(
                right_value
            )
            evidence.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                    "status": (
                        "match"
                        if comparable and left_value == right_value
                        else ("different" if comparable else "unknown")
                    ),
                    "comparable": comparable,
                }
            )
        return evidence

    def compare_crashes(
        self,
        left_crash_id: str,
        right_crash_id: str,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Compare two retained crash descriptors using ASLR-stable fields."""
        if not left_crash_id or not right_crash_id:
            raise ValueError("left_crash_id and right_crash_id cannot be empty")
        if cancellation_check is not None:
            cancellation_check()
        left = self._history_descriptor(left_crash_id)
        right = self._history_descriptor(right_crash_id)
        missing = [
            crash_id
            for crash_id, descriptor in (
                (left_crash_id, left),
                (right_crash_id, right),
            )
            if descriptor is None
        ]
        if missing:
            raise CrashRecordError(
                "Crash descriptor is no longer retained: " + ", ".join(missing)
            )
        assert left is not None and right is not None
        evidence = self._comparison_fields(left, right)
        if cancellation_check is not None:
            cancellation_check()
        left_signature = left.get("signature", {})
        right_signature = right.get("signature", {})
        digest_equal = left_signature.get("digest") and left_signature.get(
            "digest"
        ) == right_signature.get("digest")
        statuses = {item["field"]: item["status"] for item in evidence}
        pc_values = {
            item["field"]: (item.get("left"), item.get("right")) for item in evidence
        }.get("program_counter", (None, None))
        stable_pc_difference = statuses.get("program_counter") == "different" and all(
            isinstance(value, dict) and value.get("kind") == "module_offset"
            for value in pc_values
        )
        target_identity_difference = statuses.get("target_identity") == "different"
        match_count = sum(item["status"] == "match" for item in evidence)
        differing = [
            item["field"] for item in evidence if item["status"] == "different"
        ]
        if (
            target_identity_difference
            or statuses.get("signal") == "different"
            or stable_pc_difference
        ):
            relationship = "different"
            confidence = "high"
            reasons = [
                "A stable crash-family field differs: "
                + (
                    "target identity"
                    if target_identity_difference
                    else (
                        "signal"
                        if statuses.get("signal") == "different"
                        else "module-relative program counter"
                    )
                )
            ]
        elif digest_equal and left_signature.get("confidence") in {"high", "medium"}:
            relationship = "same"
            confidence = str(left_signature.get("confidence"))
            reasons = ["ASLR-normalized crash signatures match"]
        elif (
            statuses.get("signal") == "match"
            and statuses.get("program_counter") == "match"
            and match_count >= 3
        ):
            relationship = "same"
            confidence = "medium"
            reasons = ["Signal and module-relative program counter match"]
        elif len(differing) >= 2 and statuses.get("signal") != "unknown":
            relationship = "different"
            confidence = "medium"
            reasons = ["Multiple comparable crash-family fields differ"]
        else:
            relationship = "unknown"
            confidence = "low"
            reasons = [
                "Retained evidence lacks enough stable matching or conflicting fields"
            ]
        return {
            "success": True,
            "complete": relationship != "unknown",
            "command": "compare_crashes",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "schema": CRASH_COMPARISON_SCHEMA,
                "left_crash_id": left_crash_id,
                "right_crash_id": right_crash_id,
                "relationship": relationship,
                "same_crash_family": (
                    True
                    if relationship == "same"
                    else (False if relationship == "different" else None)
                ),
                "confidence": confidence,
                "reasons": reasons,
                "field_evidence": evidence,
                "left_signature": left_signature,
                "right_signature": right_signature,
                "source": "local",
            },
            "raw_responses": [],
        }

    @staticmethod
    def _cyclic_parameters(
        *,
        n: int,
        endian: str,
        max_length: int,
        alphabet: str,
    ) -> tuple[bytes, int, int]:
        if not 2 <= n <= 8:
            raise ValueError("n must be between 2 and 8")
        if endian not in {"little", "big"}:
            raise ValueError("endian must be 'little' or 'big'")
        if not 1 <= max_length <= MAX_CYCLIC_LENGTH:
            raise ValueError(f"max_length must be between 1 and {MAX_CYCLIC_LENGTH}")
        if not isinstance(alphabet, str):
            raise ValueError("alphabet must be a string")
        if not 2 <= len(alphabet) <= MAX_CYCLIC_ALPHABET_CHARS:
            raise ValueError("alphabet must contain 2-256 unique ASCII characters")
        alphabet_bytes = alphabet.encode("ascii", errors="strict")
        if not 2 <= len(alphabet_bytes) <= 256 or len(set(alphabet_bytes)) != len(
            alphabet_bytes
        ):
            raise ValueError("alphabet must contain 2-256 unique ASCII characters")
        pattern_space = len(alphabet_bytes) ** n
        searched_length = min(max_length, pattern_space)
        return alphabet_bytes, pattern_space, searched_length

    @staticmethod
    def _build_cyclic_pattern(
        alphabet_bytes: bytes,
        n: int,
        searched_length: int,
        cancellation_check: Optional[Callable[[], None]],
    ) -> bytes:
        """Generate a bounded pattern with cooperative cancellation points."""
        if cancellation_check is not None:
            cancellation_check()
        pattern = bytearray()
        for index, item in enumerate(de_bruijn(alphabet_bytes, n)):
            if index >= searched_length:
                break
            pattern.append(int(item))
            if cancellation_check is not None and not index % (64 * 1024):
                cancellation_check()
        if cancellation_check is not None:
            cancellation_check()
        return bytes(pattern)

    @staticmethod
    def _decode_cyclic_value(
        value: str,
        *,
        endian: str,
        width: Optional[int],
    ) -> tuple[bytes, str]:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if not 1 <= len(value) <= MAX_CYCLIC_VALUE_CHARS:
            raise ValueError(
                f"value must contain between 1 and {MAX_CYCLIC_VALUE_CHARS} characters"
            )
        raw_value = value.strip()
        if not raw_value:
            raise ValueError("value cannot be empty")
        input_kind = "text"
        if raw_value.lower().startswith("hex:"):
            hex_value = raw_value[4:]
            if len(hex_value) % 2:
                hex_value = "0" + hex_value
            query = bytes.fromhex(hex_value)
            input_kind = "hex-bytes"
        elif raw_value.lower().startswith("text:"):
            query = raw_value[5:].encode("utf-8")
        elif raw_value.lower().startswith("0x"):
            integer = int(raw_value, 16)
            inferred_width = max(1, (integer.bit_length() + 7) // 8)
            resolved_width = inferred_width if width is None else width
            if not 1 <= resolved_width <= 64:
                raise ValueError("width must be between 1 and 64")
            if integer >= 1 << (resolved_width * 8):
                raise ValueError(
                    f"integer value does not fit in the requested {resolved_width} bytes"
                )
            integer_byteorder: Literal["little", "big"] = (
                "little" if endian == "little" else "big"
            )
            query = integer.to_bytes(
                resolved_width, byteorder=integer_byteorder, signed=False
            )
            input_kind = "integer"
        else:
            query = raw_value.encode("utf-8")
        if not query or len(query) > 64:
            raise ValueError("decoded value must contain between 1 and 64 bytes")
        return query, input_kind

    @staticmethod
    def _search_cyclic_query(
        pattern: bytes,
        query: bytes,
        *,
        input_kind: str,
        n: int,
        endian: str,
        cancellation_check: Optional[Callable[[], None]],
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        if cancellation_check is not None:
            cancellation_check()
        if len(query) < n:
            # A de Bruijn sequence only guarantees uniqueness for n-byte windows.
            # Enumerate bounded candidates for shorter evidence and never present
            # the first hit as a unique/exact offset.
            start = 0
            while len(matches) <= MAX_CYCLIC_SCAN_MATCHES:
                offset = pattern.find(query, start)
                if offset < 0:
                    break
                matches.append(
                    {
                        "kind": "partial_candidate",
                        "offset": offset,
                        "query_hex": query.hex(),
                        "byte_count": len(query),
                    }
                )
                start = offset + 1
                if cancellation_check is not None and not len(matches) % 32:
                    cancellation_check()
            return matches

        exact_offset = pattern.find(query)
        if exact_offset >= 0:
            matches.append(
                {
                    "kind": "exact",
                    "offset": exact_offset,
                    "query_hex": query.hex(),
                    "byte_count": len(query),
                }
            )
        if len(query) != n:
            windows: List[tuple[str, bytes]] = []
            if len(query) > n:
                # Label by byte position (unambiguous regardless of endian).
                leading = query[:n]
                trailing = query[-n:]
                windows.extend([("leading_n", leading), ("trailing_n", trailing)])
                # For integer input the caller usually wants the low/high-order
                # window.  Which byte position that maps to depends on endian:
                # little-endian puts the least-significant byte first, big-endian
                # puts it last.  Only add these labels for integer input so text
                # or raw-byte queries are not mislabeled as having an "order".
                if input_kind == "integer":
                    if endian == "little":
                        windows.extend(
                            [("low_order_n", leading), ("high_order_n", trailing)]
                        )
                    else:
                        windows.extend(
                            [("low_order_n", trailing), ("high_order_n", leading)]
                        )
            seen = set()
            for kind, candidate in windows:
                if cancellation_check is not None:
                    cancellation_check()
                offset = pattern.find(candidate)
                identity = (kind, offset, candidate)
                if offset >= 0 and identity not in seen:
                    seen.add(identity)
                    matches.append(
                        {
                            "kind": kind,
                            "offset": offset,
                            "query_hex": candidate.hex(),
                            "byte_count": len(candidate),
                        }
                    )
        return matches

    @staticmethod
    def find_cyclic_offset(
        value: str,
        *,
        n: int = 4,
        endian: str = "little",
        width: Optional[int] = None,
        max_length: int = 1024 * 1024,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Find exact and n-byte cyclic-pattern offsets without silent truncation."""
        alphabet_bytes, pattern_space, searched_length = CrashTriage._cyclic_parameters(
            n=n,
            endian=endian,
            max_length=max_length,
            alphabet=alphabet,
        )
        query, input_kind = CrashTriage._decode_cyclic_value(
            value, endian=endian, width=width
        )
        pattern = CrashTriage._build_cyclic_pattern(
            alphabet_bytes, n, searched_length, cancellation_check
        )
        matches = CrashTriage._search_cyclic_query(
            pattern,
            query,
            input_kind=input_kind,
            n=n,
            endian=endian,
            cancellation_check=cancellation_check,
        )
        candidates_truncated = len(matches) > MAX_CYCLIC_SCAN_MATCHES
        if candidates_truncated:
            matches = matches[:MAX_CYCLIC_SCAN_MATCHES]
        ambiguous = len(query) < n and bool(matches)
        primary = matches[0] if matches and not ambiguous else None
        return {
            "success": True,
            "command": "find_cyclic_offset",
            "state": "local",
            "parser_version": PARSER_VERSION,
            "parsed": {
                "found": bool(matches),
                "offset": primary.get("offset") if primary else None,
                "match_kind": (
                    "ambiguous"
                    if ambiguous
                    else (primary.get("kind") if primary else None)
                ),
                "resolution": (
                    "ambiguous"
                    if ambiguous
                    else ("unique_in_searched_pattern" if primary else "not_found")
                ),
                "ambiguous": ambiguous,
                "candidate_count": len(matches),
                "candidates_truncated": candidates_truncated,
                "matches": matches,
                "input": {
                    "value": value,
                    "kind": input_kind,
                    "query_hex": query.hex(),
                    "endian": endian,
                    "width": len(query),
                },
                "pattern": {
                    "n": n,
                    "alphabet": alphabet,
                    "space": pattern_space,
                    "searched_length": searched_length,
                    "search_complete": searched_length == pattern_space,
                },
                "source": "local",
            },
            "raw_responses": [],
        }

    def find_cyclic_offsets_for_crash(
        self,
        expected_crash_id: str,
        *,
        value: Optional[str] = None,
        n: int = 4,
        endian: str = "little",
        width: Optional[int] = None,
        max_length: int = 1024 * 1024,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        scan_registers: bool = True,
        scan_stack: bool = True,
        max_stack_scan_bytes: int = 4096,
        max_matches: int = 64,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan bounded retained register/stack evidence for cyclic values."""
        if not scan_registers and not scan_stack and value is None:
            raise ValueError(
                "Provide value or enable at least one crash evidence scan source"
            )
        if not 0 <= max_stack_scan_bytes <= MAX_CYCLIC_STACK_SCAN_BYTES:
            raise ValueError(
                "max_stack_scan_bytes must be between 0 and "
                f"{MAX_CYCLIC_STACK_SCAN_BYTES}"
            )
        if not 1 <= max_matches <= MAX_CYCLIC_SCAN_MATCHES:
            raise ValueError(
                f"max_matches must be between 1 and {MAX_CYCLIC_SCAN_MATCHES}"
            )
        snapshot = self._history_snapshot(expected_crash_id)
        if snapshot is None:
            # The exact current live crash can still be captured lazily. Historical
            # crashes must already have an immutable retained snapshot.
            record = self.gdb.get_last_stop_record()
            if (
                record is None
                or self._crash_id(record) != expected_crash_id
                or not LiveInferiorGuard.is_live(record, self.gdb.get_inferior_pid())
            ):
                raise CrashRecordError(
                    f"Crash snapshot is no longer retained: {expected_crash_id}"
                )
            captured = self.capture_crash_snapshot(
                expected_crash_id,
                cancellation_check=cancellation_check,
            )
            snapshot = captured.get("parsed", {}).get("crash_snapshot")
        if not isinstance(snapshot, dict):
            raise CrashRecordError(
                f"Crash snapshot is unavailable: {expected_crash_id}"
            )
        alphabet_bytes, pattern_space, searched_length = self._cyclic_parameters(
            n=n,
            endian=endian,
            max_length=max_length,
            alphabet=alphabet,
        )
        pattern = self._build_cyclic_pattern(
            alphabet_bytes, n, searched_length, cancellation_check
        )
        matches: List[Dict[str, Any]] = []
        explicit_result = None
        if value is not None:
            query, input_kind = self._decode_cyclic_value(
                value, endian=endian, width=width
            )
            explicit_matches = self._search_cyclic_query(
                pattern,
                query,
                input_kind=input_kind,
                n=n,
                endian=endian,
                cancellation_check=cancellation_check,
            )
            explicit_candidates_truncated = (
                len(explicit_matches) > MAX_CYCLIC_SCAN_MATCHES
            )
            if explicit_candidates_truncated:
                explicit_matches = explicit_matches[:MAX_CYCLIC_SCAN_MATCHES]
            explicit_ambiguous = len(query) < n and bool(explicit_matches)
            explicit_result = {
                "value": value,
                "kind": input_kind,
                "query_hex": query.hex(),
                "resolution": (
                    "ambiguous"
                    if explicit_ambiguous
                    else (
                        "unique_in_searched_pattern"
                        if explicit_matches
                        else "not_found"
                    )
                ),
                "ambiguous": explicit_ambiguous,
                "candidate_count": len(explicit_matches),
                "candidates_truncated": explicit_candidates_truncated,
                "matches": explicit_matches,
            }
            matches.extend(
                {**match, "evidence": {"kind": "explicit_value"}}
                for match in explicit_matches[:max_matches]
            )

        debugger_snapshot = snapshot.get("debugger", {})
        if not isinstance(debugger_snapshot, dict):
            debugger_snapshot = {}
        if scan_registers and len(matches) < max_matches:
            registers = debugger_snapshot.get("registers", [])
            if isinstance(registers, list):
                for register in registers[:512]:
                    if cancellation_check is not None:
                        cancellation_check()
                    if not isinstance(register, dict) or register.get("value") is None:
                        continue
                    integer = self._integer(register.get("value"))
                    if integer is None or integer < 0:
                        continue
                    register_value = f"0x{integer:x}"
                    # ``width`` describes the caller-supplied value. Applying it
                    # to every architectural register would make one wide register
                    # abort the whole scan when the requested width is smaller.
                    register_width = max(1, (integer.bit_length() + 7) // 8)
                    query, input_kind = self._decode_cyclic_value(
                        register_value,
                        endian=endian,
                        width=register_width,
                    )
                    register_matches = self._search_cyclic_query(
                        pattern,
                        query,
                        input_kind=input_kind,
                        n=n,
                        endian=endian,
                        cancellation_check=cancellation_check,
                    )
                    for match in register_matches:
                        matches.append(
                            {
                                **match,
                                "evidence": {
                                    "kind": "register",
                                    "name": register.get("name"),
                                    "value": register.get("value"),
                                    "source": "gdb-mi",
                                },
                            }
                        )
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break

        stack = snapshot.get("stack", {})
        stack_scanned_bytes = 0
        if (
            scan_stack
            and len(matches) < max_matches
            and isinstance(stack, dict)
            and isinstance(stack.get("data_hex"), str)
            and max_stack_scan_bytes
        ):
            try:
                stack_bytes = bytes.fromhex(stack["data_hex"])[:max_stack_scan_bytes]
            except ValueError:
                stack_bytes = b""
            stack_scanned_bytes = len(stack_bytes)
            candidate_offsets: Dict[bytes, List[int]] = {}
            alphabet_set = set(alphabet_bytes)
            for stack_offset in range(max(0, len(stack_bytes) - n + 1)):
                if cancellation_check is not None and not stack_offset % (64 * 1024):
                    cancellation_check()
                candidate = stack_bytes[stack_offset : stack_offset + n]
                if all(byte in alphabet_set for byte in candidate):
                    candidate_offsets.setdefault(candidate, []).append(stack_offset)
            found_pattern_offsets: Dict[bytes, int] = {}
            if candidate_offsets:
                for pattern_offset in range(max(0, len(pattern) - n + 1)):
                    if cancellation_check is not None and not pattern_offset % (
                        64 * 1024
                    ):
                        cancellation_check()
                    candidate = pattern[pattern_offset : pattern_offset + n]
                    if (
                        candidate in candidate_offsets
                        and candidate not in found_pattern_offsets
                    ):
                        found_pattern_offsets[candidate] = pattern_offset
                        if len(found_pattern_offsets) == len(candidate_offsets):
                            break
            stack_base = self._integer(stack.get("address"))
            for candidate, offsets in candidate_offsets.items():
                found_offset = found_pattern_offsets.get(candidate)
                if found_offset is None:
                    continue
                for stack_offset in offsets:
                    matches.append(
                        {
                            "kind": "stack_window",
                            "offset": found_offset,
                            "query_hex": candidate.hex(),
                            "byte_count": len(candidate),
                            "evidence": {
                                "kind": "stack",
                                "stack_offset": stack_offset,
                                "address": (
                                    f"0x{stack_base + stack_offset:x}"
                                    if stack_base is not None
                                    else None
                                ),
                                "source": "gdb-mi",
                            },
                        }
                    )
                    if len(matches) >= max_matches:
                        break
                if len(matches) >= max_matches:
                    break
        if cancellation_check is not None:
            cancellation_check()
        unambiguous_matches = [
            match for match in matches if match.get("kind") != "partial_candidate"
        ]
        primary = unambiguous_matches[0] if unambiguous_matches else None
        ambiguous = bool(matches) and primary is None
        return {
            "success": True,
            "command": "find_cyclic_offset",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "found": bool(matches),
                "offset": primary.get("offset") if primary else None,
                "match_kind": (
                    primary.get("kind")
                    if primary
                    else ("ambiguous" if ambiguous else None)
                ),
                "resolution": (
                    "unique_in_searched_pattern"
                    if primary
                    else ("ambiguous" if ambiguous else "not_found")
                ),
                "ambiguous": ambiguous,
                "matches": matches,
                "input": explicit_result,
                "crash_scan": {
                    "crash_id": expected_crash_id,
                    "registers_scanned": scan_registers,
                    "stack_scanned": scan_stack,
                    "stack_scanned_bytes": stack_scanned_bytes,
                    "max_matches": max_matches,
                    "truncated": len(matches) >= max_matches,
                },
                "pattern": {
                    "n": n,
                    "alphabet": alphabet,
                    "space": pattern_space,
                    "searched_length": searched_length,
                    "search_complete": searched_length == pattern_space,
                },
                "source": "local-over-typed-crash-snapshot",
            },
            "raw_responses": [],
        }

    def estimate_core_size(self) -> Dict[str, Any]:
        """Estimate an upper bound from readable mappings before core generation."""
        mapping_result = self.inspection.memory_maps()
        mappings = mapping_result.get("parsed", {}).get("mappings", [])
        total = 1024 * 1024
        counted = 0
        saturated = False
        for item in mappings if isinstance(mappings, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                start = int(str(item.get("start")), 16)
                end = int(str(item.get("end")), 16)
            except (TypeError, ValueError):
                continue
            mapping_size = end - start
            if mapping_size <= 0:
                continue
            counted += 1
            total = min(MAX_CONFIGURED_CORE_DUMP_BYTES, total + mapping_size)
            saturated = total == MAX_CONFIGURED_CORE_DUMP_BYTES
        return {
            "success": bool(mapping_result.get("success") and counted),
            "estimated_upper_bound_bytes": total,
            "mapping_count": counted,
            "estimate_saturated": saturated,
            "source": "pwndbg-cli",
            "raw_responses": mapping_result.get("raw_responses", []),
        }

    @staticmethod
    def _validate_elf_core(path: str) -> Dict[str, Any]:
        file_size = os.path.getsize(path)
        with open(path, "rb") as source:
            header = source.read(64)
        if len(header) < 24 or header[:4] != b"\x7fELF":
            raise CoreDumpGenerationError("Generated file is not an ELF file")
        elf_class = header[4]
        if elf_class not in {1, 2}:
            raise CoreDumpGenerationError("Generated ELF has an unsupported class")
        data_encoding = header[5]
        if data_encoding not in {1, 2}:
            raise CoreDumpGenerationError("Generated ELF has an unsupported byte order")
        if header[6] != 1:
            raise CoreDumpGenerationError(
                "Generated ELF has an unsupported ident version"
            )
        byteorder: Literal["little", "big"] = "little" if data_encoding == 1 else "big"
        elf_type = int.from_bytes(header[16:18], byteorder=byteorder)
        if elf_type != 4:
            raise CoreDumpGenerationError(
                f"Generated ELF type is {elf_type}, expected ET_CORE (4)"
            )
        if int.from_bytes(header[20:24], byteorder=byteorder) != 1:
            raise CoreDumpGenerationError("Generated ELF has an unsupported version")
        if elf_class == 2:  # ELF64
            ehsize = 64
            expected_phentsize = 56
            if len(header) < 64:
                raise CoreDumpGenerationError("Generated ELF64 header is truncated")
            e_phoff = int.from_bytes(header[32:40], byteorder=byteorder)
            e_ehsize = int.from_bytes(header[52:54], byteorder=byteorder)
            e_phentsize = int.from_bytes(header[54:56], byteorder=byteorder)
            e_phnum = int.from_bytes(header[56:58], byteorder=byteorder)
        else:  # ELF32
            ehsize = 52
            expected_phentsize = 32
            with open(path, "rb") as source:
                header32 = source.read(52)
            if len(header32) < 52:
                raise CoreDumpGenerationError("Generated ELF32 header is truncated")
            e_phoff = int.from_bytes(header32[28:32], byteorder=byteorder)
            e_ehsize = int.from_bytes(header32[40:42], byteorder=byteorder)
            e_phentsize = int.from_bytes(header32[42:44], byteorder=byteorder)
            e_phnum = int.from_bytes(header32[44:46], byteorder=byteorder)
        if e_ehsize != ehsize:
            raise CoreDumpGenerationError(
                f"Generated ELF header size is {e_ehsize}, expected {ehsize}"
            )
        if e_phentsize != expected_phentsize:
            raise CoreDumpGenerationError(
                "Generated core has an invalid program-header entry size"
            )
        if e_phnum < 1 or e_phoff < ehsize:
            raise CoreDumpGenerationError(
                "Generated core has no program-header table; the file is truncated"
            )
        required = e_phoff + e_phnum * e_phentsize
        if file_size < required:
            raise CoreDumpGenerationError(
                f"Generated core is {file_size} bytes but its program-header table "
                f"requires at least {required}; the file is truncated"
            )
        with open(path, "rb") as source:
            source.seek(e_phoff)
            program_headers = source.read(e_phnum * e_phentsize)
        types: List[int] = []
        load_segments = 0
        material_load_segments = 0
        note_segments = 0
        recognized_core_notes = 0
        note_entries = 0
        for index in range(e_phnum):
            entry = program_headers[index * e_phentsize : (index + 1) * e_phentsize]
            segment_type = int.from_bytes(entry[0:4], byteorder=byteorder)
            if elf_class == 2:
                segment_offset = int.from_bytes(entry[8:16], byteorder=byteorder)
                file_bytes = int.from_bytes(entry[32:40], byteorder=byteorder)
                memory_bytes = int.from_bytes(entry[40:48], byteorder=byteorder)
            else:
                segment_offset = int.from_bytes(entry[4:8], byteorder=byteorder)
                file_bytes = int.from_bytes(entry[16:20], byteorder=byteorder)
                memory_bytes = int.from_bytes(entry[20:24], byteorder=byteorder)
            if segment_offset > file_size or file_bytes > file_size - segment_offset:
                raise CoreDumpGenerationError(
                    f"Generated core program header {index} points outside the file"
                )
            if segment_type == 1:
                load_segments += 1
                if file_bytes > memory_bytes:
                    raise CoreDumpGenerationError(
                        f"Generated core PT_LOAD {index} has filesz greater than memsz"
                    )
                if file_bytes > 0 and memory_bytes > 0:
                    material_load_segments += 1
            elif segment_type == 4:
                note_segments += 1
                if file_bytes < 12:
                    raise CoreDumpGenerationError(
                        f"Generated core PT_NOTE {index} has no complete note header"
                    )
                note_end = segment_offset + file_bytes
                cursor = segment_offset
                with open(path, "rb") as source:
                    while cursor + 12 <= note_end:
                        note_entries += 1
                        if note_entries > MAX_ELF_CORE_NOTE_ENTRIES:
                            raise CoreDumpGenerationError(
                                "Generated core contains too many ELF note entries"
                            )
                        source.seek(cursor)
                        note_header = source.read(12)
                        if len(note_header) != 12:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} is truncated"
                            )
                        name_size = int.from_bytes(
                            note_header[0:4], byteorder=byteorder
                        )
                        description_size = int.from_bytes(
                            note_header[4:8], byteorder=byteorder
                        )
                        note_type = int.from_bytes(
                            note_header[8:12], byteorder=byteorder
                        )
                        if name_size == 0:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} contains an empty note name"
                            )
                        padded_name_size = (name_size + 3) & ~3
                        padded_description_size = (description_size + 3) & ~3
                        next_cursor = (
                            cursor + 12 + padded_name_size + padded_description_size
                        )
                        if next_cursor <= cursor or next_cursor > note_end:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} contains a malformed note"
                            )
                        # ELF core note names are tiny. Avoid allocating an
                        # attacker-controlled namesz while still advancing over
                        # a structurally valid, unrecognized note.
                        note_name = b""
                        if name_size <= 4096:
                            source.seek(cursor + 12)
                            note_name = source.read(name_size)
                            if len(note_name) != name_size:
                                raise CoreDumpGenerationError(
                                    f"Generated core PT_NOTE {index} is truncated"
                                )
                        if (
                            note_name.rstrip(b"\x00") == b"CORE"
                            and note_type == 1  # NT_PRSTATUS
                            and description_size > 0
                        ):
                            recognized_core_notes += 1
                        cursor = next_cursor
                    trailing = note_end - cursor
                    if trailing:
                        source.seek(cursor)
                        padding = source.read(trailing)
                        if trailing > 3 or padding != b"\x00" * trailing:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} has malformed trailing data"
                            )
            types.append(segment_type)
        if not note_segments or not load_segments:
            raise CoreDumpGenerationError(
                "Generated core must contain both PT_NOTE and PT_LOAD segments"
            )
        if not recognized_core_notes:
            raise CoreDumpGenerationError(
                "Generated core has no CORE/NT_PRSTATUS process-status note"
            )
        if not material_load_segments:
            raise CoreDumpGenerationError(
                "Generated core has no non-empty PT_LOAD memory segment"
            )
        return {
            "elf_class": 32 if elf_class == 1 else 64,
            "byte_order": byteorder,
            "elf_type": "ET_CORE",
            "program_header_count": e_phnum,
            "program_header_types": types,
            "note_segment_count": note_segments,
            "load_segment_count": load_segments,
            "material_load_segment_count": material_load_segments,
            "recognized_core_note_count": recognized_core_notes,
            "note_entry_count": note_entries,
            "size_bytes": file_size,
        }

    def generate_core_file(
        self,
        expected_crash_id: str,
        output_path: str,
        *,
        max_size_bytes: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        min_free_bytes: int = 0,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Use GDB CLI to generate one monitored core file at a trusted path."""
        record = self._require_crash_record(expected_crash_id, fatal_signals)
        try:
            LiveInferiorGuard.require_live(
                record,
                self.gdb.get_inferior_pid(),
                detail="Core capture requires the exact current stopped live crash",
            )
        except LiveInferiorError as exc:
            raise CrashRecordError(str(exc)) from exc
        cap = (
            self.max_core_dump_bytes if max_size_bytes is None else int(max_size_bytes)
        )
        if not 1 <= cap <= self.max_core_dump_bytes:
            raise ValueError(
                f"max_size_bytes must be between 1 and {self.max_core_dump_bytes}"
            )
        timeout = self._timeout(
            timeout_seconds, self.default_core_timeout_seconds, name="timeout_seconds"
        )
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        if cancellation_check is not None:
            cancellation_check()

        # ``generate-core-file`` is one of GDB's raw-filename commands: the
        # complete remainder of the command line is the filename, including
        # spaces and quote characters.  Adding shell-style quotes makes those
        # quotes part of the actual filename.  Only command delimiters need to
        # be rejected here; the artifact store supplies this internal path and
        # GDB receives no shell interpolation.
        if any(character in output_path for character in ("\x00", "\r", "\n")):
            raise ValueError("Core output path contains a forbidden control character")
        output_path = os.path.abspath(output_path)
        command = "generate-core-file " + output_path
        monitor_done = threading.Event()
        violation: List[BaseException] = []
        last_reported = -1

        def signal_core_writer() -> None:
            """Interrupt a blocking GDB command without racing its MI reader."""
            signal_interrupt = getattr(self.gdb, "signal_interrupt", None)
            if callable(signal_interrupt):
                interrupted = signal_interrupt()
            else:
                # A legacy adapter without signal-only support cannot safely
                # consume MI from this monitor thread. Terminate the writer
                # boundary instead and force the registry to rebuild it.
                killer = getattr(self.gdb, "kill_process_group", None)
                interrupted = (
                    killer()
                    if callable(killer)
                    else {
                        "success": False,
                        "error": "No signal-only or process-kill capability",
                    }
                )
                self._invalidate_gdb("core monitor lacked signal-only interruption")
            if not interrupted.get("success", False):
                violation.append(
                    CoreDumpGenerationError(
                        "Could not signal the active GDB core writer: "
                        + str(interrupted.get("error", "unknown interrupt failure"))
                    )
                )

        def ensure_writer_quiescent() -> None:
            """Guarantee gdb has stopped writing before staging can be unlinked.

            After signalling, a cooperative SIGINT may not stop gdb mid-write of
            a large core: on POSIX the file would remain open and, once staging
            is unlinked, keep consuming disk through the deleted descriptor with
            the size monitor already gone.  We wait a bounded time for the gdb
            process to exit and, if it has not, SIGKILL its process group so the
            descriptor is closed and the inode released before cleanup runs.

            If neither the wait nor the kill can *confirm* the process is dead,
            we raise :class:`WriterNotQuiescentError` so the artifact store
            quarantines (rather than unlinks) the staging file — an unlink here
            would recreate the very deleted-fd disk leak we are preventing.
            """
            waiter = getattr(self.gdb, "wait_for_writer_quiescence", None)
            if callable(waiter):
                try:
                    if waiter(timeout_sec=2.0):
                        return
                except BaseException as exc:
                    raise WriterNotQuiescentError(
                        "Could not verify that gdb stopped writing the core file; "
                        "the staging file is quarantined"
                    ) from exc
            killer = getattr(self.gdb, "kill_process_group", None)
            if not callable(killer):
                raise WriterNotQuiescentError(
                    "The debugger adapter cannot confirm or force writer quiescence; "
                    "the staging file is quarantined"
                )
            try:
                outcome = killer()
            except BaseException as exc:
                raise WriterNotQuiescentError(
                    "The debugger writer could not be terminated; the staging file "
                    "is quarantined"
                ) from exc
            if not isinstance(outcome, dict):
                raise WriterNotQuiescentError(
                    "The debugger returned no verifiable writer-termination result; "
                    "the staging file is quarantined"
                )
            # kill_process_group only reports success once it has *confirmed* the
            # process is dead (poll() returned a code). Treat a non-success/
            # unconfirmed result as a quarantine condition.
            confirmed = outcome.get("success") and outcome.get(
                "confirmed_dead", outcome.get("success")
            )
            if not confirmed:
                raise WriterNotQuiescentError(
                    "Could not confirm gdb stopped writing the core file; the "
                    "staging file is quarantined instead of unlinked to avoid a "
                    "deleted-descriptor disk leak"
                )

        def monitor() -> None:
            nonlocal last_reported
            while not monitor_done.wait(0.05):
                try:
                    if cancellation_check is not None:
                        cancellation_check()
                    size = os.path.getsize(output_path)
                except FileNotFoundError:
                    size = 0
                except Exception as exc:
                    violation.append(exc)
                    signal_core_writer()
                    return
                if progress_callback is not None and size != last_reported:
                    try:
                        last_reported = size
                        progress_callback(size, cap, "Generating core file")
                    except Exception as exc:
                        violation.append(exc)
                if size > cap:
                    violation.append(
                        ArtifactSizeLimitError(
                            f"Core staging file exceeded its {cap}-byte hard limit"
                        )
                    )
                elif (
                    shutil.disk_usage(os.path.dirname(output_path)).free
                    < min_free_bytes
                ):
                    violation.append(
                        ArtifactStorageFullError(
                            "Core generation crossed the configured free-space reserve"
                        )
                    )
                if violation:
                    # Only signal here; the authoritative quiescence guarantee
                    # (and its quarantine-on-failure) runs once in the finally
                    # block on the command-owning thread, where an exception can
                    # actually propagate.
                    signal_core_writer()
                    return

        watcher = threading.Thread(
            target=monitor, name="core-size-monitor", daemon=True
        )
        watcher_started = False
        started = time.monotonic()
        timeout_interrupt: Optional[Dict[str, Any]] = None
        # The command is about to touch the output file. Until it cleanly
        # returns and the file is validated, we must assume gdb may still hold
        # the descriptor open, so every abnormal exit escalates to quiescence
        # before the caller is allowed to unlink staging.
        command_completed_cleanly = False
        fsize_state: Dict[str, Any] = {"applied": False}
        apply_limit = getattr(self.gdb, "apply_fsize_limit", None)
        restore_limit = getattr(self.gdb, "restore_fsize_limit", None)
        fsize_restored: Optional[bool] = None
        quiescence_error: Optional[BaseException] = None
        primary_exc: Optional[BaseException] = None
        restore_failure: Optional[BaseException] = None
        writer_started = False
        try:
            # Scope a hard RLIMIT_FSIZE onto gdb's OWN pid for the duration of the
            # core write only. The inferior was forked earlier and is unaffected,
            # so this cannot alter the debugged program's behavior; it only makes
            # the kernel abort an oversized core with zero overshoot.
            if not callable(apply_limit):
                raise CoreDumpGenerationError(
                    "Core capture requires kernel RLIMIT_FSIZE support"
                )
            fsize_state = apply_limit(cap)
            if not fsize_state.get("applied"):
                detail = fsize_state.get("error", "the kernel limit was not applied")
                raise CoreDumpGenerationError(
                    "Core capture refused because its kernel file-size limit "
                    f"could not be enforced: {detail}"
                )
            watcher.start()
            watcher_started = True
            writer_started = True
            result = self.gdb.execute_command(command, timeout_sec=timeout)
            if result.get("timed_out"):
                # The original write call has relinquished the MI response
                # stream, so it is now safe to signal and drain synchronously.
                timeout_interrupt = self.gdb.interrupt(timeout_sec=1.0)
            else:
                command_completed_cleanly = not result.get("timed_out")
        except BaseException as exc:
            primary_exc = exc
        finally:
            monitor_done.set()
            if watcher_started:
                watcher.join(timeout=2)
            # P1: confirm writer is gone BEFORE restoring the soft limit so the
            # cap stays in place while GDB may still hold the core descriptor.
            if writer_started and (
                (not command_completed_cleanly) or violation or primary_exc is not None
            ):
                try:
                    ensure_writer_quiescent()
                except WriterNotQuiescentError as exc:
                    quiescence_error = exc
                    self._invalidate_gdb("core writer could not be confirmed quiescent")
            # Restore soft limit only when writer is confirmed gone. If
            # quiescence failed, keep the cap in place — the file may still
            # be growing and will be quarantined; removing the cap now could
            # let it fill the disk before quarantine takes effect.
            if fsize_state.get("applied") and quiescence_error is None:
                if callable(restore_limit):
                    try:
                        fsize_restored = bool(restore_limit(fsize_state))
                    except BaseException as exc:
                        fsize_restored = False
                        restore_failure = exc
                else:
                    fsize_restored = False
                    restore_failure = CoreDumpGenerationError(
                        "The debugger adapter cannot restore its file-size limit"
                    )
                if not fsize_restored:
                    # A live session retaining the temporary cap is unsafe for
                    # later inferiors. Invalidate it, while preserving any earlier
                    # command/quiescence exception as the primary failure.
                    self._invalidate_gdb("failed to restore the gdb file-size limit")
                    if restore_failure is None:
                        restore_failure = CoreDumpGenerationError(
                            "The debugger adapter reported that the file-size limit "
                            "was not restored"
                        )
        # Quiescence failure (potential disk leak) takes precedence over every
        # other error, including the primary exception from execute_command.
        if quiescence_error is not None:
            raise quiescence_error
        if primary_exc is not None:
            raise primary_exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if watcher.is_alive():
            raise CoreDumpGenerationError(
                "Core monitor did not terminate; staging cleanup requires manual "
                "runtime validation"
            )
        if violation:
            raise violation[0]
        if cancellation_check is not None:
            cancellation_check()
        if result.get("timed_out"):
            interrupt_success = bool(
                timeout_interrupt and timeout_interrupt.get("success")
            )
            raise CoreDumpGenerationError(
                "GDB core generation timed out and was interrupted; partial output "
                f"was not committed (interrupt success={interrupt_success})"
            )
        if not result.get("success"):
            raise CoreDumpGenerationError(self.debugger._error_message(result, command))
        if not os.path.isfile(output_path):
            raise CoreDumpGenerationError(
                "GDB reported success but created no core file"
            )
        size_bytes = os.path.getsize(output_path)
        if size_bytes > cap:
            raise ArtifactSizeLimitError(
                f"Core file is {size_bytes} bytes, exceeding the {cap}-byte limit"
            )
        validated = self._validate_elf_core(output_path)
        if restore_failure is not None:
            raise CoreDumpGenerationError(
                "Failed to restore the debugger file-size limit: " f"{restore_failure}"
            )
        if progress_callback is not None:
            progress_callback(size_bytes, cap, "Core file generated")
        response: Dict[str, Any] = {
            "success": True,
            "command": command,
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_id": expected_crash_id,
                "size_bytes": size_bytes,
                "max_size_bytes": cap,
                "elapsed_ms": elapsed_ms,
                "limit_enforcement": {
                    "kernel_rlimit_applied": True,
                    "kernel_rlimit_bytes": fsize_state.get("applied_soft"),
                    "kernel_rlimit_restored": fsize_restored,
                    "fallback": None,
                },
                "validation": validated,
                "source": "gdb-cli",
            },
            "raw_output": self.debugger._raw_output(result),
            "raw_responses": result.get("responses", []),
        }
        if fsize_restored is False:
            response["warnings"] = [
                "The debugger file-size limit could not be restored; the session "
                "was invalidated and must be rebuilt"
            ]
        return response
