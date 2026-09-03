"""
GDB Controller for Pao Tools Server

This module provides a synchronous interface to GDB/pwndbg via pygdbmi.
Unlike pwndbg-gui, this is designed for discrete tool invocations with
immediate responses suitable for LLM interaction.
"""

import logging
import os
import shutil
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Tuple, Union

from pygdbmi import gdbcontroller
from pygdbmi.IoManager import IoManager
from pygdbmi.constants import GdbTimeoutError

from paomcp.tools.backends.foundation import validate_attach_pid

logger = logging.getLogger(__name__)

DEFAULT_STOP_RECORD_LIMIT = 64
DEFAULT_GDB_CLOSE_TIMEOUT_SEC = 2.0

# On POSIX the gdb process is placed in its own session so the whole process
# group can be signalled at once; Windows has no session/process-group model, so
# group-kill degrades to best-effort termination of the gdb process alone.
_POSIX = os.name != "nt"


def _session_popen_kwargs() -> Dict[str, Any]:
    """Return Popen kwargs that isolate gdb into its own session/group.

    ``start_new_session=True`` is the thread-safe replacement for a ``setsid``
    ``preexec_fn``: CPython performs the ``setsid`` in the child via a path that
    is safe in a multithreaded parent, whereas an arbitrary ``preexec_fn`` may
    deadlock the child before ``exec``.  No resource limit is applied at spawn —
    doing so would be inherited by the debugged inferior across fork/exec and
    change its behavior (e.g. spurious ``SIGXFSZ``).  The core-size cap is
    instead applied to gdb's *own* pid, scoped to the core-dump command only.
    """
    if _POSIX:
        return {"start_new_session": True}
    return {}


class _SessionIsolatedController(gdbcontroller.GdbController):
    """pygdbmi controller that spawns gdb in its own session/process group.

    pygdbmi's own ``Popen`` call forwards no process-creation kwargs, so the
    only clean seam for ``start_new_session`` is to override the single spawn
    method.  The body mirrors upstream exactly except for the injected kwargs,
    keeping us forward-compatible with the pinned pygdbmi version.

    The second deviation is **lazy spawning**.  Upstream ``__init__`` Popen's
    gdb unconditionally, which means constructing a session -- e.g. the default
    session ``create_services`` builds just to read an id -- starts a full gdb
    subprocess, and a missing ``pwndbg`` on ``PATH`` fails the entire server.
    Here the executable is still resolved at construction (cheap and fail-fast),
    but the process itself is deferred to the first ``write`` /
    ``get_gdb_response``.
    """

    def __init__(
        self,
        command: Optional[List[str]] = None,
        time_to_check_for_additional_output_sec: float = gdbcontroller.DEFAULT_TIME_TO_CHECK_FOR_ADDITIONAL_OUTPUT_SEC,
    ) -> None:
        # Mirrors upstream __init__ except the trailing
        # spawn_new_gdb_subprocess() -- deferred to first use.
        if command is None:
            command = gdbcontroller.DEFAULT_GDB_LAUNCH_COMMAND
        if not any(("--interpreter=mi" in c) for c in command):
            gdbcontroller.logger.warning(
                "Adding `--interpreter=mi3` (or similar) is recommended "
                "to get structured output. See "
                "https://sourceware.org/gdb/onlinedocs/gdb/Mode-Options.html"
                "#Mode-Options."
            )
        self.abs_gdb_path: Optional[str] = None
        self.command = list(command)
        self.time_to_check_for_additional_output_sec = float(
            time_to_check_for_additional_output_sec
        )
        self.gdb_process: Optional[subprocess.Popen[Any]] = None
        self._allow_overwrite_timeout_times = (
            self.time_to_check_for_additional_output_sec > 0
        )
        gdb_path = command[0]
        if not gdb_path:
            raise ValueError("a valid path to gdb must be specified")
        abs_gdb_path = shutil.which(gdb_path)
        if abs_gdb_path is None:
            raise ValueError(
                'gdb executable could not be resolved from "%s"' % gdb_path
            )
        self.abs_gdb_path = abs_gdb_path
        # No spawn here: io_manager exists only once the process does.

    def _ensure_spawned(self) -> None:
        if self.gdb_process is None:
            self.spawn_new_gdb_subprocess()

    def write(
        self,
        mi_cmd_to_write: Union[str, List[str]],
        timeout_sec: float = gdbcontroller.DEFAULT_GDB_TIMEOUT_SEC,
        raise_error_on_timeout: bool = True,
        read_response: bool = True,
    ) -> List[Dict[str, Any]]:
        self._ensure_spawned()
        return super().write(
            mi_cmd_to_write,
            timeout_sec,
            raise_error_on_timeout,
            read_response,
        )

    def get_gdb_response(
        self,
        timeout_sec: float = gdbcontroller.DEFAULT_GDB_TIMEOUT_SEC,
        raise_error_on_timeout: bool = True,
    ) -> List[Dict[str, Any]]:
        self._ensure_spawned()
        return super().get_gdb_response(timeout_sec, raise_error_on_timeout)

    @staticmethod
    def _process_running(process: subprocess.Popen[Any]) -> bool:
        try:
            return process.poll() is None
        except OSError:
            return True

    def _stop_existing_gdb_subprocess(
        self, timeout_sec: float = DEFAULT_GDB_CLOSE_TIMEOUT_SEC
    ) -> None:
        """Replace pygdbmi's unbounded ``exit`` with bounded terminate/kill."""
        process = self.gdb_process
        if process is None:
            return
        timeout = max(0.0, float(timeout_sec))

        if self._process_running(process):
            try:
                process.terminate()
            except OSError:
                logger.debug("Failed to terminate prior gdb process", exc_info=True)
            try:
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, TimeoutError):
                logger.warning("Prior GDB did not exit before respawn timeout")
            except OSError:
                logger.debug("Failed while waiting for prior gdb", exc_info=True)

        if self._process_running(process):
            try:
                process.kill()
            except OSError:
                logger.debug("Failed to kill prior gdb process", exc_info=True)
            try:
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, TimeoutError):
                pass
            except OSError:
                logger.debug("Failed while reaping prior gdb", exc_info=True)

        if self._process_running(process):
            raise RuntimeError(
                "Prior GDB process remained alive after bounded shutdown"
            )

        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    logger.debug(
                        "Failed to close prior gdb %s pipe",
                        stream_name,
                        exc_info=True,
                    )
        self.gdb_process = None
        if hasattr(self, "io_manager"):
            setattr(self, "io_manager", None)

    def spawn_new_gdb_subprocess(self) -> int:
        if self.gdb_process:
            self._stop_existing_gdb_subprocess()
        popen_kwargs: Dict[str, Any] = {
            "shell": False,
            "stdout": subprocess.PIPE,
            "stdin": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
        }
        popen_kwargs.update(_session_popen_kwargs())
        self.gdb_process = subprocess.Popen(self.command, **popen_kwargs)
        try:
            assert self.gdb_process.stdin is not None
            assert self.gdb_process.stdout is not None
            self.io_manager = IoManager(
                self.gdb_process.stdin,
                self.gdb_process.stdout,
                self.gdb_process.stderr,
                self.time_to_check_for_additional_output_sec,
            )
        except BaseException:
            # Once Popen succeeds this object owns the child. Never let a pipe
            # or IoManager initialization error abandon a live GDB process.
            try:
                self._stop_existing_gdb_subprocess()
            except Exception:
                logger.exception("Failed to clean up partially spawned GDB")
            raise
        assert self.gdb_process is not None
        return self.gdb_process.pid


@dataclass(frozen=True)
class _StopRecord:
    """One immutable snapshot of a GDB/MI ``stopped`` notification."""

    stop_sequence: int
    captured_at: str
    inferior_epoch: int
    inferior_pid: Optional[int]
    payload: Mapping[str, Any]
    identity: str


def _freeze_stop_value(value: Any) -> Any:
    """Freeze the JSON-like values emitted by pygdbmi."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_stop_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_stop_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_stop_value(item) for item in value)
    return value


def _copy_stop_value(value: Any) -> Any:
    """Return a mutable copy of a frozen stop-record value."""
    if isinstance(value, Mapping):
        return {key: _copy_stop_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_copy_stop_value(item) for item in value]
    return value


class GdbController:
    """Manages GDB instance and command execution via Machine Interface"""

    def __init__(self, pwndbg_executable: str = "pwndbg"):
        """
        Initialize GDB controller

        Args:
            pwndbg_executable: Path to the pwndbg executable.
        """
        self.controller = _SessionIsolatedController(
            command=[pwndbg_executable, "--interpreter=mi3", "--quiet"]
        )
        self._initialized = False
        self._inferior_pid: Optional[int] = None
        self._inferior_epoch = 0
        self._state = "idle"  # idle, running, stopped, exited
        # Set once the gdb process is force-killed (e.g. core-size enforcement):
        # the controller can no longer serve MI, so the session must be treated
        # as destroyed and rebuilt by the registry rather than silently reused.
        self._destroyed = False
        self._stop_sequence = 0
        self._stop_records: Deque[_StopRecord] = deque(maxlen=DEFAULT_STOP_RECORD_LIMIT)

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize GDB with pwndbg from ~/.gdbinit

        Returns:
            Dictionary with initialization status and messages
        """
        if self._initialized:
            return {"status": "already_initialized", "messages": []}

        results = []

        # Load .gdbinit if it exists
        # gdbinit = Path.home() / ".gdbinit"
        # if gdbinit.exists():
        #     logger.info(f"Loading .gdbinit from {gdbinit}")
        #     response = self.execute_command(f"source {gdbinit}")
        #     results.append(response)
        # else:
        #     logger.warning("No .gdbinit found")

        # Core GDB settings via MI (-gdb-set) for reliable, fast non-interactive behavior
        for setting in [
            "-gdb-set mi-async on",
            "-gdb-set pagination off",
            "-gdb-set confirm off",
            "-gdb-set detach-on-fork off",
            "-gdb-set follow-fork-mode parent",
            "-gdb-set follow-exec-mode same",
        ]:
            results.append(self.execute_mi_command(setting))

        # Ensure pwndbg is active (optional; comment out if too slow in your env)
        # pwndbg_check = self.execute_command("pwndbg")
        # results.append(pwndbg_check)

        self._initialized = True
        return {"status": "initialized", "messages": results}

    @staticmethod
    def quote_mi_argument(value: str) -> str:
        """Encode one GDB/MI C-string argument without changing its expression."""
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("MI argument contains a forbidden control character")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def execute_mi_command(
        self, command: str, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        """Execute a GDB/MI command and return raw MI responses."""
        logger.debug(f"Executing MI command: {command}")
        try:
            responses = self.controller.write(command, timeout_sec=timeout_sec)
        except (GdbTimeoutError, TimeoutError) as exc:
            return {
                "command": command,
                "responses": [],
                "success": False,
                "state": self._state,
                "error": str(exc) or f"GDB command timed out after {timeout_sec}s",
                "type": "TimeoutError",
                "timed_out": True,
                "timeout_seconds": timeout_sec,
            }
        # Update internal state from notify messages and detect errors
        error_found = False
        for response in responses:
            if response.get("type") == "notify":
                self._handle_notify(response)
            elif (
                response.get("type") == "result" and response.get("message") == "error"
            ):
                error_found = True
        return {
            "command": command,
            "responses": responses,
            "success": not error_found,
            "state": self._state,
        }

    def execute_command(self, command: str, timeout_sec: float = 7.0) -> Dict[str, Any]:
        """Execute a classic GDB command (non-MI) and return raw responses."""
        logger.debug(f"Executing command: {command}")
        try:
            responses = self.controller.write(command, timeout_sec=timeout_sec)
        except (GdbTimeoutError, TimeoutError) as exc:
            return {
                "command": command,
                "responses": [],
                "success": False,
                "state": self._state,
                "error": str(exc) or f"GDB command timed out after {timeout_sec}s",
                "type": "TimeoutError",
                "timed_out": True,
                "timeout_seconds": timeout_sec,
            }
        error_found = False
        for response in responses:
            if response.get("type") == "notify":
                self._handle_notify(response)
            elif (
                response.get("type") == "result" and response.get("message") == "error"
            ):
                error_found = True
        return {
            "command": command,
            "responses": responses,
            "success": not error_found,
            "state": self._state,
        }

    def _handle_notify(self, response: Dict[str, Any]):
        """Handle GDB notification messages to track state"""
        self._ensure_stop_tracking()
        message = response.get("message", "")

        if message == "running":
            self._state = "running"
            logger.debug("Inferior state: RUNNING")

        elif message == "stopped":
            self._state = "stopped"
            payload = response.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            self._record_stop(payload)
            # Extract stop reason if available
            reason = payload.get("reason", "unknown")
            logger.debug(f"Inferior state: STOPPED (reason: {reason})")

        elif message == "thread-group-exited":
            self._state = "exited"
            self._inferior_pid = None
            self._inferior_epoch += 1
            # The prior stop record is intentionally retained so get_last_stop
            # can still report the final stop, but the epoch bump makes it
            # non-live: callers must gate crash handling on ``is_live`` and never
            # treat a post-exit record as a fresh crash. Records are only cleared
            # when a *new* inferior starts (thread-group-started / run / load).
            logger.debug("Inferior state: EXITED")

        elif message == "thread-group-started":
            # This happens when attaching to a process
            payload = response.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            self._clear_stop_records()
            self._inferior_pid = self._parse_inferior_pid(payload.get("pid"))
            self._inferior_epoch += 1
            logger.debug(f"Thread group started, PID: {self._inferior_pid}")

    def _ensure_stop_tracking(self) -> None:
        """Initialize stop tracking for legacy tests constructing via ``__new__``."""
        if not hasattr(self, "_state"):
            self._state = "idle"
        if not hasattr(self, "_inferior_pid"):
            self._inferior_pid = None
        if not hasattr(self, "_inferior_epoch"):
            self._inferior_epoch = 0
        if not hasattr(self, "_stop_sequence"):
            self._stop_sequence = 0
        if not hasattr(self, "_stop_records"):
            self._stop_records = deque(maxlen=DEFAULT_STOP_RECORD_LIMIT)

    def _clear_stop_records(self) -> None:
        """Forget records from a prior inferior without resetting the sequence."""
        self._ensure_stop_tracking()
        self._stop_records.clear()

    @staticmethod
    def _parse_inferior_pid(value: Any) -> Optional[int]:
        """Normalize pygdbmi's string PID while tolerating absent metadata."""
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _record_stop(self, payload: Mapping[str, Any]) -> None:
        """Store one bounded, immutable stop notification record."""
        self._ensure_stop_tracking()
        self._stop_sequence += 1
        record = _StopRecord(
            stop_sequence=self._stop_sequence,
            captured_at=datetime.now(timezone.utc).isoformat(),
            inferior_epoch=self._inferior_epoch,
            inferior_pid=self._inferior_pid,
            payload=_freeze_stop_value(payload),
            identity=f"{self._inferior_epoch}:{self._stop_sequence}",
        )
        self._stop_records.append(record)

    def _copy_stop_record(
        self, record: _StopRecord, latest_identity: Optional[str]
    ) -> Dict[str, Any]:
        """Render a record with liveness derived from current controller state."""
        state = getattr(self, "_state", "idle")
        epoch = getattr(self, "_inferior_epoch", 0)
        return {
            "stop_sequence": record.stop_sequence,
            "captured_at": record.captured_at,
            "inferior_epoch": record.inferior_epoch,
            "inferior_pid": record.inferior_pid,
            "payload": _copy_stop_value(record.payload),
            "identity": record.identity,
            "controller_state": state,
            "is_live": (
                state == "stopped"
                and record.inferior_epoch == epoch
                and record.identity == latest_identity
            ),
        }

    def get_last_stop_record(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the newest observed stop record, if any."""
        self._ensure_stop_tracking()
        if not self._stop_records:
            return None
        latest = self._stop_records[-1]
        return self._copy_stop_record(latest, latest.identity)

    def get_stop_sequence(self) -> int:
        """Return the monotonic sequence assigned to the newest observed stop."""
        self._ensure_stop_tracking()
        return self._stop_sequence

    def get_stop_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return copies of bounded stop history in oldest-to-newest order."""
        self._ensure_stop_tracking()
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        records = list(self._stop_records)
        if limit is not None:
            records = records[-limit:] if limit else []
        latest_identity = records[-1].identity if records else None
        return [self._copy_stop_record(record, latest_identity) for record in records]

    def _get_async_responses(self, timeout_sec: float) -> List[Dict[str, Any]]:
        getter = getattr(self.controller, "get_gdb_response", None)
        if not callable(getter):
            return []
        try:
            responses = getter(timeout_sec=timeout_sec, raise_error_on_timeout=False)
        except TypeError:
            try:
                responses = getter(timeout_sec=timeout_sec)
            except Exception:
                return []
        except Exception:
            return []
        return responses if isinstance(responses, list) else []

    def drain_async(
        self, timeout_sec: float = 0.0, max_rounds: int = 10
    ) -> Dict[str, Any]:
        """Drain pending async MI notifications from GDB.

        Args:
            timeout_sec: Max time to wait for the first batch.
            max_rounds: Max additional drain loops to clear buffered output.
        """
        responses: List[Dict[str, Any]] = []
        error_found = False

        start = time.monotonic()
        first = True
        for _ in range(max_rounds):
            if first:
                remaining = max(0.0, timeout_sec - (time.monotonic() - start))
                batch = self._get_async_responses(timeout_sec=remaining)
                first = False
            else:
                batch = self._get_async_responses(timeout_sec=0.0)

            if not batch:
                break

            for response in batch:
                if response.get("type") == "notify":
                    self._handle_notify(response)
                elif (
                    response.get("type") == "result"
                    and response.get("message") == "error"
                ):
                    error_found = True
            responses.extend(batch)

        return {
            "responses": responses,
            "success": not error_found,
            "state": self._state,
        }

    def signal_interrupt(self) -> Dict[str, Any]:
        """Signal GDB without concurrently draining its MI response stream.

        This is the thread-safe half of cancellation for a command currently
        blocked inside ``controller.write``.  The command-owning thread remains
        responsible for consuming the resulting MI responses.
        """
        try:
            interrupt_fn = getattr(self.controller, "interrupt_gdb", None)
            if not callable(interrupt_fn):
                return {
                    "responses": [],
                    "success": False,
                    "state": self._state,
                    "error": "The pygdbmi controller has no signal-only interrupt",
                }
            interrupt_fn()
            return {"responses": [], "success": True, "state": self._state}
        except Exception as exc:
            return {
                "responses": [],
                "success": False,
                "state": self._state,
                "error": str(exc),
            }

    def interrupt(self, timeout_sec: float = 1.0) -> Dict[str, Any]:
        """Interrupt the inferior and drain async notifications."""
        responses: List[Dict[str, Any]] = []
        error_found = False
        signalled = self.signal_interrupt()
        responses.extend(signalled.get("responses", []))
        if not signalled.get("success", False):
            # Retain the existing MI fallback for callers that own the response
            # stream.  Background monitors must use ``signal_interrupt`` only.
            result = self.execute_mi_command("-exec-interrupt")
            responses.extend(result.get("responses", []))
            if not result.get("success", True):
                error_found = True

        drained = self.drain_async(timeout_sec=timeout_sec)
        responses.extend(drained.get("responses", []))
        success = (not error_found) and drained.get("success", True)
        return {"responses": responses, "success": success, "state": self._state}

    def get_context(self, context_type: str) -> Dict[str, Any]:
        """Get a specific pwndbg context; return raw responses"""
        if self._state != "stopped":
            return {
                "command": f"context {context_type}",
                "responses": [],
                "success": False,
                "state": self._state,
                "error": f"Cannot get context while inferior is {self._state}",
            }
        return self.execute_command(f"context {context_type}")

    def set_file(self, filepath: str) -> Dict[str, Any]:
        """Load an executable file for debugging using MI command"""
        result = self.execute_mi_command(
            f"-file-exec-and-symbols {self.quote_mi_argument(filepath)}"
        )
        # Change working directory for relative paths during debugging
        dirname = os.path.dirname(filepath) or "."
        self.execute_mi_command(f"-environment-cd {self.quote_mi_argument(dirname)}")
        if result["success"]:
            self._clear_stop_records()
            self._state = "stopped"
        # Ensure returned state reflects any updates
        result["state"] = self._state
        return result

    def attach(self, pid: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Attach to an existing process using MI command (-target-attach)"""
        try:
            pid = validate_attach_pid(pid)
        except ValueError as exc:
            failure: Dict[str, Any] = {
                "success": False,
                "command": f"-target-attach {pid}",
                "state": self._state,
                "pid": self._inferior_pid,
                "error": str(exc),
                "type": "ValueError",
            }
            return failure, []
        result = self.execute_mi_command(f"-target-attach {pid}")
        if result["success"]:
            self._inferior_pid = pid
            self._state = "stopped"
        result["state"] = self._state
        result["pid"] = self._inferior_pid

        context = []
        context.append(self.get_context("backtrace"))
        context.append(self.get_context("heap"))

        return result, context

    def get_registers_mi(self) -> Dict[str, Any]:
        """Get register values using MI (hex)."""
        return self.execute_mi_command("-data-list-register-values x")

    def get_register_names_mi(self) -> Dict[str, Any]:
        """Get register names in GDB's target-specific number order."""
        return self.execute_mi_command("-data-list-register-names")

    def get_register_values_mi(
        self, register_numbers: Optional[List[int]] = None, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        """Get selected or all register values in hexadecimal format."""
        suffix = ""
        if register_numbers:
            suffix = " " + " ".join(str(number) for number in register_numbers)
        return self.execute_mi_command(
            f"-data-list-register-values x{suffix}", timeout_sec=timeout_sec
        )

    def get_backtrace_mi(self) -> Dict[str, Any]:
        """Get backtrace/frames using MI."""
        return self.execute_mi_command("-stack-list-frames")

    def get_disassembly_mi(self, length: int = 64) -> Dict[str, Any]:
        """Disassemble around $pc using MI where supported."""
        # Mode 1 = mixed source/asm if available; fall back handled by GDB
        return self.execute_mi_command(f"-data-disassemble -s $pc -e $pc+{length} -- 1")

    def get_quick_context(self) -> Dict[str, Any]:
        """Collect a lightweight, MI-based context snapshot (fast)."""
        if self._state != "stopped":
            return {
                "command": "quick-context",
                "responses": [],
                "success": False,
                "state": self._state,
                "error": f"Cannot get context while inferior is {self._state}",
            }
        # Return a structured payload with MI results for key views
        return {
            "success": True,
            "state": self._state,
            "contexts": {
                "regs": self.get_registers_mi(),
                "backtrace": self.get_backtrace_mi(),
                "disasm": self.get_disassembly_mi(),
            },
        }

    def run(self, args: str = "", start: bool = False) -> Dict[str, Any]:
        """Run the loaded program using MI command"""
        if args:
            set_args_result = self.execute_mi_command(f"-exec-arguments {args}")
            if not set_args_result["success"]:
                return set_args_result

        run_command = "-exec-run --start" if start else "-exec-run"
        self._clear_stop_records()
        result = self.execute_mi_command(run_command)
        return result

    def continue_execution(self) -> Dict[str, Any]:
        """Continue execution using MI command"""
        return self.execute_mi_command("-exec-continue")

    def finish(self) -> Dict[str, Any]:
        """Finish current function using MI command (-exec-finish)"""
        return self.execute_mi_command("-exec-finish")

    def next(self) -> Dict[str, Any]:
        """Step over using MI command"""
        return self.execute_mi_command("-exec-next")

    def step(self) -> Dict[str, Any]:
        """Step into using MI command"""
        return self.execute_mi_command("-exec-step")

    def nexti(self) -> Dict[str, Any]:
        """Step one instruction using MI command"""
        return self.execute_mi_command("-exec-next-instruction")

    def stepi(self) -> Dict[str, Any]:
        """Step into one instruction using MI command"""
        return self.execute_mi_command("-exec-step-instruction")

    def jump(self, locspec: str) -> Dict[str, Any]:
        """Jump to a specific location using MI command (-exec-jump)"""
        return self.execute_mi_command(f"-exec-jump {self.quote_mi_argument(locspec)}")

    def return_from_function(self) -> Dict[str, Any]:
        """Force return from current function using MI command (-exec-return)"""
        return self.execute_mi_command("-exec-return")

    def until(self, locspec: Optional[str] = None) -> Dict[str, Any]:
        """Run until a specific location or next line using MI command (-exec-until)"""
        mi_cmd = (
            "-exec-until"
            if not locspec
            else f"-exec-until {self.quote_mi_argument(locspec)}"
        )
        return self.execute_mi_command(mi_cmd)

    def set_breakpoint(
        self,
        location: str,
        condition: Optional[str] = None,
        temporary: bool = False,
    ) -> Dict[str, Any]:
        """Set a persistent or temporary breakpoint using GDB/MI."""
        parts = ["-break-insert"]
        if temporary:
            parts.append("-t")
        if condition:
            escaped_condition = condition.replace("\\", "\\\\").replace('"', '\\"')
            parts.extend(["-c", f'"{escaped_condition}"'])
        parts.append("--")
        parts.append(self.quote_mi_argument(location))
        mi_command = " ".join(parts)
        return self.execute_mi_command(mi_command)

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        """Evaluate an expression using MI command"""
        return self.execute_mi_command(
            f"-data-evaluate-expression {self.quote_mi_argument(expression)}"
        )

    def select_thread(self, thread_id: int) -> Dict[str, Any]:
        """Select a GDB thread by global thread id."""
        return self.execute_mi_command(f"-thread-select {thread_id}")

    def list_threads_mi(self, timeout_sec: float = 7.0) -> Dict[str, Any]:
        """List threads using the stable GDB/MI thread schema."""
        return self.execute_mi_command("-thread-info", timeout_sec=timeout_sec)

    def list_thread_groups_mi(
        self, recurse: int = 1, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        """List inferiors/thread groups and their child groups using GDB/MI."""
        return self.execute_mi_command(
            f"-list-thread-groups --recurse {recurse}", timeout_sec=timeout_sec
        )

    def list_stack_frames_mi(
        self,
        low: int = 0,
        high: Optional[int] = None,
        timeout_sec: float = 7.0,
    ) -> Dict[str, Any]:
        """List stack frames, optionally bounded by inclusive frame levels."""
        suffix = "" if high is None else f" {low} {high}"
        return self.execute_mi_command(
            f"-stack-list-frames{suffix}", timeout_sec=timeout_sec
        )

    def select_frame(self, level: int, timeout_sec: float = 7.0) -> Dict[str, Any]:
        """Select a stack frame by zero-based level."""
        return self.execute_mi_command(
            f"-stack-select-frame {level}", timeout_sec=timeout_sec
        )

    def get_selected_frame_mi(self, timeout_sec: float = 7.0) -> Dict[str, Any]:
        """Return the currently selected frame through GDB/MI."""
        return self.execute_mi_command("-stack-info-frame", timeout_sec=timeout_sec)

    def set_watchpoint(self, expression: str, access: str = "write") -> Dict[str, Any]:
        """Set a write, read, or access hardware watchpoint."""
        option = {"write": "", "read": "-r ", "access": "-a "}[access]
        return self.execute_mi_command(
            f"-break-watch {option}{self.quote_mi_argument(expression)}"
        )

    def write_memory_bytes(self, address: str, data_hex: str) -> Dict[str, Any]:
        """Write raw bytes expressed as an even-length hexadecimal string."""
        return self.execute_mi_command(
            f"-data-write-memory-bytes {self.quote_mi_argument(address)} {data_hex}"
        )

    def disassemble_range(
        self, start: str, byte_count: int, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        """Disassemble a byte range without source interleaving."""
        return self.execute_mi_command(
            f"-data-disassemble -s {self.quote_mi_argument(start)} "
            f"-e {self.quote_mi_argument(f'{start}+{byte_count}')} -- 0",
            timeout_sec=timeout_sec,
        )

    def read_memory_mi(
        self, address: str, word_format: str, word_size: int, nr_rows: int, nr_cols: int
    ) -> Dict[str, Any]:
        """Read memory using MI command"""
        mi_command = (
            f"-data-read-memory {self.quote_mi_argument(address)} "
            f"{word_format} {word_size} {nr_rows} {nr_cols}"
        )
        return self.execute_mi_command(mi_command)

    def read_memory_bytes(self, address: str, num_bytes: int) -> Dict[str, Any]:
        """Read raw memory bytes using MI (fast path)."""
        mi_command = (
            f"-data-read-memory-bytes {self.quote_mi_argument(address)} {num_bytes}"
        )
        return self.execute_mi_command(mi_command)

    def list_breakpoints(self) -> Dict[str, Any]:
        """List all breakpoints using MI command"""
        return self.execute_mi_command("-break-list")

    def delete_breakpoint(self, number: int) -> Dict[str, Any]:
        """Delete a breakpoint using MI command"""
        return self.execute_mi_command(f"-break-delete {number}")

    def enable_breakpoint(self, number: int) -> Dict[str, Any]:
        """Enable a breakpoint using MI command"""
        return self.execute_mi_command(f"-break-enable {number}")

    def disable_breakpoint(self, number: int) -> Dict[str, Any]:
        """Disable a breakpoint using MI command"""
        return self.execute_mi_command(f"-break-disable {number}")

    def get_state(self) -> str:
        """Get current inferior state"""
        return self._state

    def get_inferior_pid(self) -> Optional[int]:
        """Get currently tracked inferior PID, if known."""
        return self._inferior_pid

    def get_inferior_epoch(self) -> int:
        """Return a monotonic counter for observed inferior start/exit events."""
        return self._inferior_epoch

    def is_destroyed(self) -> bool:
        """Return whether the gdb process can no longer serve MI requests."""
        if getattr(self, "_destroyed", False):
            return True
        if self.gdb_process_running() is False:
            self._state = "exited"
            self.mark_destroyed("gdb process exited unexpectedly")
            return True
        return False

    def mark_destroyed(self, reason: str) -> None:
        """Invalidate this controller so its owning session cannot be reused."""
        self._destroyed = True
        self._destroyed_reason = str(reason)

    def get_destroyed_reason(self) -> Optional[str]:
        """Return the reason this controller was invalidated, when available."""
        reason = getattr(self, "_destroyed_reason", None)
        return str(reason) if reason is not None else None

    def get_gdb_process_pid(self) -> Optional[int]:
        """Return the OS pid of the gdb process itself (not the inferior)."""
        controller = getattr(self, "controller", None)
        process = getattr(controller, "gdb_process", None) if controller else None
        pid = getattr(process, "pid", None)
        try:
            return int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return None

    def gdb_process_running(self) -> Optional[bool]:
        """Return whether the gdb process is still alive, or None if unknown."""
        controller = getattr(self, "controller", None)
        process = getattr(controller, "gdb_process", None) if controller else None
        if process is None or not hasattr(process, "poll"):
            return None
        return process.poll() is None

    def apply_fsize_limit(self, cap_bytes: int) -> Dict[str, Any]:
        """Scope an RLIMIT_FSIZE cap onto the gdb process itself, returning prior.

        This targets gdb's *own* pid via ``resource.prlimit`` — never the inferior,
        which was already forked before this call and keeps its own limits.  The
        kernel raises ``SIGXFSZ`` in gdb the instant a written file (the core
        dump) would exceed ``cap_bytes``, giving zero-overshoot enforcement.  The
        caller must invoke :meth:`restore_fsize_limit` with the returned previous
        limit under the same session lock once the core command completes.

        On non-POSIX or when ``resource.prlimit`` is unavailable this is a no-op
        and returns ``{"applied": False}``. Core capture treats that result as a
        fail-closed refusal; its polling monitor is only a secondary detector.
        """
        pid = self.get_gdb_process_pid()
        if (
            not _POSIX
            or pid is None
            or cap_bytes <= 0
            or self.gdb_process_running() is not True
        ):
            return {"applied": False}
        try:
            import resource

            prlimit = getattr(resource, "prlimit")
            rlimit_fsize = getattr(resource, "RLIMIT_FSIZE")
            rlim_infinity = getattr(resource, "RLIM_INFINITY")
        except (ImportError, AttributeError):
            return {"applied": False}
        try:
            soft_prev, hard_prev = prlimit(pid, rlimit_fsize)
            cap = int(cap_bytes)
            # Clamp against both existing soft and hard limits so we never
            # raise a policy the operator already set lower than the request.
            if soft_prev != rlim_infinity:
                cap = min(cap, soft_prev)
            if hard_prev != rlim_infinity:
                cap = min(cap, hard_prev)
            if (
                self.get_gdb_process_pid() != pid
                or self.gdb_process_running() is not True
            ):
                return {"applied": False}
            # Only lower the soft limit; never touch the hard limit so it can
            # always be restored without elevated privileges.
            prlimit(pid, rlimit_fsize, (cap, hard_prev))
        except (OSError, ValueError, PermissionError) as exc:
            return {"applied": False, "error": str(exc)}
        return {
            "applied": True,
            "pid": pid,
            "previous_soft": soft_prev,
            "previous_hard": hard_prev,
            "applied_soft": cap,
        }

    def restore_fsize_limit(self, previous: Dict[str, Any]) -> bool:
        """Restore a prior RLIMIT_FSIZE captured by :meth:`apply_fsize_limit`."""
        if not previous.get("applied"):
            return False
        pid = self.get_gdb_process_pid()
        try:
            previous_pid = int(previous["pid"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not _POSIX
            or pid is None
            or pid != previous_pid
            or self.gdb_process_running() is not True
        ):
            return False
        try:
            import resource

            prlimit = getattr(resource, "prlimit")
            rlimit_fsize = getattr(resource, "RLIMIT_FSIZE")
        except (ImportError, AttributeError):
            return False
        try:
            # Only restore the soft limit; hard limit was never changed.
            _, hard_current = prlimit(pid, rlimit_fsize)
            if (
                self.get_gdb_process_pid() != previous_pid
                or self.gdb_process_running() is not True
            ):
                return False
            prlimit(pid, rlimit_fsize, (previous["previous_soft"], hard_current))
        except (OSError, ValueError, PermissionError):
            return False
        return True

    def kill_process_group(self) -> Dict[str, Any]:
        """Forcibly terminate the gdb process (and its group on POSIX).

        This is the writer-quiescence guarantee: after a core-dump command has
        timed out or violated a size/disk limit, a cooperative SIGINT may not
        stop gdb before its open core file descriptor keeps growing on disk.
        SIGKILL-ing the whole session guarantees the descriptor is closed — and
        therefore the on-disk inode is released — before any staging cleanup
        unlinks the path.
        """
        pid = self.get_gdb_process_pid()
        if pid is None:
            return {"success": False, "error": "No gdb process pid is available"}
        killed_group = False
        if _POSIX:
            try:
                killpg = getattr(os, "killpg")
                getpgid = getattr(os, "getpgid")
                sigkill = int(getattr(signal, "SIGKILL", signal.SIGTERM))
                killpg(getpgid(pid), sigkill)
                killed_group = True
            except OSError:
                killed_group = False
        if not killed_group:
            controller = getattr(self, "controller", None)
            process = getattr(controller, "gdb_process", None) if controller else None
            if process is not None:
                try:
                    process.kill()
                except OSError as exc:
                    return {"success": False, "error": str(exc), "pid": pid}
        controller = getattr(self, "controller", None)
        process = getattr(controller, "gdb_process", None) if controller else None
        reaped = False
        if process is not None and hasattr(process, "wait"):
            try:
                process.wait(timeout=5)
                reaped = True
            except Exception:  # noqa: BLE001 - best-effort reap of a killed child
                reaped = False
        # Confirm death rather than assuming it: poll() must report a return code.
        still_running = self.gdb_process_running()
        confirmed_dead = still_running is False or (still_running is None and reaped)
        if confirmed_dead:
            self._state = "exited"
            # The controller can no longer serve MI; flag the session destroyed
            # so callers surface a typed error and the registry can rebuild it.
            self.mark_destroyed("gdb process group was terminated")
        return {
            "success": confirmed_dead,
            "pid": pid,
            "killed_group": killed_group,
            "confirmed_dead": confirmed_dead,
            **(
                {}
                if confirmed_dead
                else {"error": "gdb process still alive after kill"}
            ),
        }

    def wait_for_writer_quiescence(
        self, *, timeout_sec: float = 2.0, poll_interval: float = 0.02
    ) -> bool:
        """Wait until the gdb process is confirmed exited; return True once it is.

        Callers use this after signalling gdb to confirm the process is truly
        gone before treating an open output file as safe to unlink.  An *unknown*
        liveness (no observable process handle) is treated as NOT quiescent, so
        the caller escalates to :meth:`kill_process_group` rather than unlinking
        a file that a still-live writer may hold open.  Returns False if gdb is
        still alive (or liveness cannot be confirmed) when the timeout elapses.
        """
        deadline = time.monotonic() + max(0.0, timeout_sec)
        while True:
            running = self.gdb_process_running()
            if running is False:
                # GDB exited on its own (e.g. SIGXFSZ); mark destroyed so the
                # session registry knows to rebuild it on the next call.
                self._state = "exited"
                self.mark_destroyed("gdb process exited while core writer was active")
                return True
            # running is True (alive) or None (unknown): keep waiting, then fail
            # closed so the caller escalates to a hard kill.
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.0, poll_interval))

    def close(self, timeout_sec: float = DEFAULT_GDB_CLOSE_TIMEOUT_SEC) -> bool:
        """Stop GDB within a bounded interval and release its pipe handles.

        ``pygdbmi.GdbController.exit`` calls ``Popen.wait`` without a timeout,
        so a wedged debugger can otherwise block registry cleanup forever.  A
        failed hard-kill deliberately retains the process handle: the registry
        can then keep the session as a capacity-consuming tombstone and retry
        cleanup without risking reuse of a still-live debugger.
        """
        controller = getattr(self, "controller", None)
        process = getattr(controller, "gdb_process", None) if controller else None
        self.mark_destroyed("gdb controller close requested")
        if controller is None or process is None:
            self._state = "exited"
            self._initialized = False
            return True

        if self.gdb_process_running() is True:
            try:
                process.terminate()
            except OSError:
                logger.debug("Failed to terminate gdb gracefully", exc_info=True)
            try:
                process.wait(timeout=max(0.0, float(timeout_sec)))
            except (subprocess.TimeoutExpired, TimeoutError):
                logger.warning("GDB did not exit before the graceful close timeout")
            except OSError:
                logger.debug("Failed while waiting for gdb to exit", exc_info=True)

        if self.gdb_process_running() is not False:
            killed = self.kill_process_group()
            if not killed.get("success", False):
                self.mark_destroyed("gdb process remained alive after bounded close")
                return False

        if self.gdb_process_running() is not False:
            self.mark_destroyed("gdb process liveness remained unknown after close")
            return False

        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    logger.debug(
                        "Failed to close gdb %s pipe", stream_name, exc_info=True
                    )
        controller.gdb_process = None
        if hasattr(controller, "io_manager"):
            controller.io_manager = None
        self._state = "exited"
        self._initialized = False
        self.mark_destroyed("gdb controller closed")
        return True
