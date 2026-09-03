"""Minimal ``pwndbg-lldb`` controller for the ``lldb-cli`` backend.

``pwndbg-lldb`` is an lldb command-language REPL: it reads one command per
line from stdin and prints text output (banner, echoes, results) followed by
a prompt.  This controller mirrors the subset of the :class:`GdbController`
surface that the debug-session lifecycle and the non-pwndbg tools actually
consume: spawn/teardown, set-file/run/attach, stepping, memory/register
reads, breakpoints, and state tracking.

Unlike GDB there is no Machine Interface, so ``execute_mi_command`` runs a
*token translator*: the handful of MI commands the higher layers issue are
mapped onto lldb commands and re-shaped into the MI response records those
layers parse (``-data-evaluate-expression``, ``-data-read-memory-bytes``,
``-data-list-register-names``/``-values``, stepping, breakpoints, threads,
frames, disassembly).  Any other MI token is answered with a typed
``GdbOnlyBackendError`` verdict -- in practice the tool layer gates pwndbg-only
operations before they reach this controller.

The rich pwndbg surface (typed heap parsers, elf runtime, crash triage) is
intentionally *not* implemented here; those tools report a clear
``requires the gdb/pwndbg backend`` error on lldb sessions.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Mapping, Optional, Tuple

from paomcp.tools.backends.backend import BackendUnavailableError
from paomcp.tools.backends.foundation import parse_int, validate_attach_pid
from paomcp.tools.backends.gdb import (
    DEFAULT_STOP_RECORD_LIMIT,
    _session_popen_kwargs,
)

logger = logging.getLogger(__name__)

DEFAULT_LLDB_CLOSE_TIMEOUT_SEC = 2.0

# Matches "(lldb) " / "(pwndbg-lldb) " style prompts.
_PROMPT_PATTERN = re.compile(r"^\([^()\r\n]*\)\s*$")
# lldb command echo prefix, e.g. "(lldb) process launch".
_ECHO_PATTERN = re.compile(r"^\([^()\r\n]*\)\s+(.+)\s*$")
# "Process 123 stopped" / "Process 123 exited" / "Process 123 launched"
_PROCESS_EVENT_PATTERN = re.compile(
    r"Process (\d+)\s+(stopped|exited|launched|started|running)"
)
# "* thread #1, stop reason = signal SIGSEGV" / "stop reason = breakpoint 1"
_STOP_REASON_PATTERN = re.compile(r"stop reason\s*=\s*(.+)")
# lldb expression result, e.g. "(unsigned long) 140737488346112"
_EXPRESSION_RESULT_PATTERN = re.compile(r"^\(([^)]*)\)\s+(\S.*)$")
# memory read hex dump line: "0x00007fff...: 48 89 e5 c3 ..."
_MEMORY_DUMP_PATTERN = re.compile(r"^0x[0-9a-fA-F]+:\s+(.*)$")
# register read line: "rip = 0x0000000000401000"
_REGISTER_LINE_PATTERN = re.compile(
    r"^([a-zA-Z_][\w]*(?:\.[\w]+)*)\s*=\s*(0x[0-9a-fA-F]+|\d+)\s*$"
)
# disassembly line: "0x401000 <+0>:  pushq  %rbp"
_DISASM_LINE_PATTERN = re.compile(r"^0x([0-9a-fA-F]+)\s+<\+\d+>:\s+(.+)$")
# backtrace frame line: "  frame #0: 0x0000000000401000 a.out`main + 4 at main.c:3"
_FRAME_LINE_PATTERN = re.compile(
    r"^frame #(\d+):\s+0x([0-9a-fA-F]+)\s+.*?`(.*?)(?:\s+\+\s+\d+)?(?: at .*)?$"
)

_KNOWN_MI_PREFIXES = (
    "-data-",
    "-break-",
    "-exec-",
    "-target-",
    "-thread-",
    "-stack-",
    "-list-",
    "-file-",
    "-gdb-",
    "-inferior-",
    "-add-",
    "-remove-",
    "-interpreter-",
)


def _is_error_line(text: str) -> bool:
    """Heuristic for an lldb error line."""
    lowered = text.strip().lower()
    return (
        lowered.startswith("error:")
        or "error: " in lowered[:80]
        or lowered.startswith("usage: ")
    )


@dataclass(frozen=True)
class _StopRecord:
    """One immutable snapshot of an lldb stop event."""

    stop_sequence: int
    captured_at: str
    inferior_epoch: int
    inferior_pid: Optional[int]
    payload: Mapping[str, Any]
    identity: str


def _freeze_stop_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _freeze_stop_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_freeze_stop_value(item) for item in value)
    return value


class LldbController:
    """Drives a ``pwndbg-lldb`` subprocess over stdin/stdout pipes."""

    def __init__(self, executable: str = "pwndbg-lldb", timeout_sec: float = 7.0):
        self.executable = executable
        self.default_timeout = timeout_sec
        self._process: Optional[subprocess.Popen[Any]] = None
        self._reader: Optional[threading.Thread] = None
        self._lines: Deque[str] = deque()
        self._lines_lock = threading.RLock()
        self._consumed = 0
        self._initialized = False
        self._destroyed = False
        self._destroyed_reason: Optional[str] = None
        self._state = "idle"
        self._inferior_pid: Optional[int] = None
        self._inferior_epoch = 0
        self._stop_sequence = 0
        self._stop_records: Deque[_StopRecord] = deque(maxlen=DEFAULT_STOP_RECORD_LIMIT)
        self._cmd_lock = threading.RLock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> Dict[str, Any]:
        """Spawn pwndbg-lldb and confirm it reached its prompt."""
        if self._initialized:
            return {"status": "already_initialized", "messages": []}
        popen_kwargs: Dict[str, Any] = {
            "shell": False,
            "stdout": subprocess.PIPE,
            "stdin": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "bufsize": 0,
        }
        popen_kwargs.update(_session_popen_kwargs())
        try:
            self._process = subprocess.Popen([self.executable], **popen_kwargs)
        except OSError as exc:
            self.mark_destroyed(f"failed to spawn {self.executable}: {exc}")
            raise BackendUnavailableError(
                f"Could not launch {self.executable}: {exc}"
            ) from exc
        assert self._process.stdout is not None
        self._reader = threading.Thread(
            target=self._read_loop,
            name=f"pao-lldb-reader-{self._process.pid}",
            daemon=True,
        )
        self._reader.start()

        banner = self._wait_for_prompt(timeout_sec=max(3.0, self.default_timeout))
        from paomcp.tools.backends.backend import LLDB_FAILURE_MARKERS

        if banner is None:
            # No prompt before timeout or process death.  A dying launcher
            # (e.g. the bundled vendor lldb failing to load host libc) still
            # prints its failure on stderr/stdout -- grab the leftover lines
            # so the failure-marker scan can classify it.
            leftover = self._drain_new_lines()
            combined = "\n".join(leftover).encode("utf-8", "replace")
            if any(marker in combined for marker in LLDB_FAILURE_MARKERS):
                self.close()
                raise BackendUnavailableError(
                    f"{self.executable} is present but its bundled lldb cannot "
                    "load host libraries; the lldb backend is unavailable"
                )
            if self._process is not None:
                try:
                    self._process.kill()
                except OSError:
                    pass
            self.mark_destroyed("pwndbg-lldb never reached its prompt")
            raise BackendUnavailableError(
                f"{self.executable} did not reach its prompt; the lldb backend "
                "is unavailable on this host"
            )
        combined = "\n".join(banner).encode("utf-8", "replace")
        if any(marker in combined for marker in LLDB_FAILURE_MARKERS):
            self.close()
            raise BackendUnavailableError(
                f"{self.executable} is present but its bundled lldb cannot "
                "load host libraries; the lldb backend is unavailable"
            )
        self._state = "idle"
        self._initialized = True
        logger.info("pwndbg-lldb backend initialized (pid %s)", self._process.pid)
        return {"status": "initialized", "messages": []}

    def _read_loop(self) -> None:
        stream = getattr(self._process, "stdout", None)
        if stream is None:
            return
        for raw_line in stream:
            text = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            with self._lines_lock:
                self._lines.append(text)

    def _drain_new_lines(self) -> List[str]:
        """Return and consume all lines the reader has appended so far."""
        with self._lines_lock:
            lines = list(self._lines)[self._consumed :]
            self._consumed = len(self._lines)
        return lines

    def _rewind(self, count: int) -> None:
        """Un-consume the trailing ``count`` lines (keep them for next command)."""
        if count <= 0:
            return
        with self._lines_lock:
            self._consumed = max(0, self._consumed - count)

    def _process_running(self) -> bool:
        process = self._process
        if process is None:
            return False
        try:
            return process.poll() is None
        except OSError:
            return True

    def _wait_for_prompt(self, timeout_sec: float) -> Optional[List[str]]:
        """Read until a prompt line appears; return the lines read (no echo).

        Returns ``None`` when the process died or the timeout elapsed before a
        prompt was reached; the caller decides whether leftover output is a
        usable (failure) banner or a hang.
        """
        collected: List[str] = []
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            batch = self._drain_new_lines()
            if batch:
                prompt_index = None
                for index, line in enumerate(batch):
                    if _PROMPT_PATTERN.match(line):
                        prompt_index = index
                        break
                if prompt_index is not None:
                    # Lines after the prompt belong to the next command.
                    self._rewind(len(batch) - (prompt_index + 1))
                    collected.extend(batch[:prompt_index])
                    return collected
                collected.extend(batch)
            if not self._process_running():
                return None
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def _scan_async(self, text: str) -> None:
        """Update state/pid/stop-records from lldb async markers in text."""
        for match in _PROCESS_EVENT_PATTERN.finditer(text):
            pid_text, event = match.group(1), match.group(2)
            pid = parse_int(pid_text)
            if event == "running":
                self._state = "running"
            elif event in ("launched", "started"):
                self._state = "stopped"
                self._inferior_pid = pid
                self._inferior_epoch += 1
                self._stop_records.clear()
            elif event == "stopped":
                self._state = "stopped"
                self._inferior_pid = pid
                self._record_stop({"reason": self._last_stop_reason(text)})
            elif event == "exited":
                self._state = "exited"
                self._inferior_pid = None
                self._inferior_epoch += 1

    @staticmethod
    def _last_stop_reason(text: str) -> str:
        matches = list(_STOP_REASON_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
        return "unknown"

    def _record_stop(self, payload: Mapping[str, Any]) -> None:
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
        return {
            "stop_sequence": record.stop_sequence,
            "captured_at": record.captured_at,
            "inferior_epoch": record.inferior_epoch,
            "inferior_pid": record.inferior_pid,
            "payload": dict(record.payload),
            "identity": record.identity,
            "controller_state": self._state,
            "is_live": (
                self._state == "stopped"
                and record.inferior_epoch == self._inferior_epoch
                and record.identity == latest_identity
            ),
        }

    def get_last_stop_record(self) -> Optional[Dict[str, Any]]:
        if not self._stop_records:
            return None
        latest = self._stop_records[-1]
        return self._copy_stop_record(latest, latest.identity)

    def get_stop_sequence(self) -> int:
        return self._stop_sequence

    def get_stop_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        records = list(self._stop_records)
        if limit is not None:
            records = records[-limit:] if limit else []
        latest_identity = records[-1].identity if records else None
        return [self._copy_stop_record(record, latest_identity) for record in records]

    def _execute(self, cmd: str, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """Write one command and read its console output up to the prompt."""
        timeout = self.default_timeout if timeout_sec is None else timeout_sec
        with self._cmd_lock:
            process = self._process
            if process is None or process.stdin is None:
                return {
                    "command": cmd,
                    "responses": [],
                    "success": False,
                    "state": self._state,
                    "error": "lldb backend is not initialized",
                    "type": "GdbOnlyBackendError",
                }
            try:
                process.stdin.write((cmd + "\n").encode("utf-8"))
                process.stdin.flush()
            except OSError as exc:
                self.mark_destroyed(f"failed to write to lldb stdin: {exc}")
                return {
                    "command": cmd,
                    "responses": [],
                    "success": False,
                    "state": self._state,
                    "error": str(exc),
                    "type": "BackendIOError",
                }

            collected: List[str] = []
            prompt_seen = False
            timed_out = False
            deadline = time.monotonic() + max(0.0, float(timeout))
            while True:
                batch = self._drain_new_lines()
                if batch:
                    prompt_index = None
                    for index, line in enumerate(batch):
                        if _PROMPT_PATTERN.match(line):
                            prompt_seen = True
                            prompt_index = index
                            break
                    if prompt_index is not None:
                        # Lines after the prompt belong to the next command.
                        self._rewind(len(batch) - (prompt_index + 1))
                        collected.extend(batch[:prompt_index])
                    else:
                        collected.extend(batch)
                    if prompt_seen:
                        break
                if not self._process_running():
                    self._state = "exited"
                    self._inferior_pid = None
                    self._inferior_epoch += 1
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.005)

        # Strip the command echo line ("(lldb) cmd").
        filtered: List[str] = []
        for line in collected:
            stripped = line.strip()
            if stripped == cmd:
                continue
            echo = _ECHO_PATTERN.match(line)
            if echo and echo.group(1).strip() == cmd:
                continue
            filtered.append(line)
        # Scan the joined block once so "Process N stopped" and its trailing
        # "stop reason = ..." line (which lldb prints on a separate line) are
        # attributed to the same stop record.
        self._scan_async("\n".join(filtered))
        error_found = any(_is_error_line(line) for line in filtered) or timed_out
        return {
            "command": cmd,
            "responses": [{"type": "console", "payload": line} for line in filtered],
            "success": not error_found and self._process_running(),
            "state": self._state,
        }

    @staticmethod
    def _console_lines(result: Dict[str, Any]) -> List[str]:
        return [
            str(response.get("payload", ""))
            for response in result.get("responses", [])
            if response.get("type") == "console"
        ]

    # ------------------------------------------------------------------
    # GdbController-compatible surface
    # ------------------------------------------------------------------
    @staticmethod
    def quote_mi_argument(value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("MI argument contains a forbidden control character")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    def execute_command(self, command: str, timeout_sec: float = 7.0) -> Dict[str, Any]:
        return self._execute(command, timeout_sec=timeout_sec)

    def execute_mi_command(
        self, command: str, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        """Translate a known MI token to lldb; otherwise a gated verdict."""
        translated = _translate_mi_command(command, controller=self)
        if translated is None:
            return {
                "command": command,
                "responses": [],
                "success": False,
                "state": self.get_state(),
                "error": (
                    f"'{command}' requires the gdb/pwndbg backend; session uses lldb"
                ),
                "type": "GdbOnlyBackendError",
            }
        result = self._execute(translated, timeout_sec=timeout_sec)
        return _shape_mi_result(command, translated, result)

    def drain_async(
        self, timeout_sec: float = 0.0, max_rounds: int = 10
    ) -> Dict[str, Any]:
        responses: List[Dict[str, Any]] = []
        error_found = False
        collected: List[str] = []
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            batch = self._drain_new_lines()
            if batch:
                non_prompt = [line for line in batch if not _PROMPT_PATTERN.match(line)]
                collected.extend(non_prompt)
                self._scan_async("\n".join(non_prompt))
            if time.monotonic() >= deadline or not self._process_running():
                break
            time.sleep(0.005)
        for line in collected:
            if _is_error_line(line):
                error_found = True
        responses = [{"type": "console", "payload": line} for line in collected]
        return {
            "responses": responses,
            "success": not error_found,
            "state": self._state,
        }

    def signal_interrupt(self) -> Dict[str, Any]:
        return self._execute("process interrupt")

    def interrupt(self, timeout_sec: float = 1.0) -> Dict[str, Any]:
        result = self.signal_interrupt()
        drained = self.drain_async(timeout_sec=timeout_sec)
        result["responses"] = result.get("responses", []) + drained.get("responses", [])
        result["success"] = bool(result.get("success") or drained.get("success"))
        result["state"] = self._state
        return result

    # -- lifecycle of the inferior ------------------------------------
    @staticmethod
    def _quote_lldb_path(value: str) -> str:
        """Quote one path for an lldb command line."""
        if any(c in value for c in ("\x00", "\r", "\n")):
            raise ValueError("lldb path contains a forbidden control character")
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    def set_file(self, filepath: str) -> Dict[str, Any]:
        result = self._execute(f"target create {self._quote_lldb_path(filepath)}")
        self._state = "stopped"
        result["state"] = self._state
        return result

    def attach(self, pid: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            pid = validate_attach_pid(pid)
        except ValueError as exc:
            return (
                {
                    "success": False,
                    "command": f"process attach -p {pid}",
                    "state": self._state,
                    "pid": self._inferior_pid,
                    "error": str(exc),
                    "type": "ValueError",
                },
                [],
            )
        result = self._execute(f"process attach -p {pid}")
        if result.get("success"):
            self._inferior_pid = pid
            self._state = "stopped"
            result["state"] = "stopped"
        # get_context("backtrace"/"heap") are pwndbg-only; return no context.
        return result, []

    def run(self, args: str = "", start: bool = False) -> Dict[str, Any]:
        if args:
            self._execute(f"settings set target.run-args {args}")
        command = "process launch --stop-at-entry" if start else "process launch"
        return self._execute(command)

    def continue_execution(self) -> Dict[str, Any]:
        return self._execute("process continue")

    def finish(self) -> Dict[str, Any]:
        return self._execute("thread step-out")

    def next(self) -> Dict[str, Any]:
        return self._execute("thread step-over")

    def step(self) -> Dict[str, Any]:
        return self._execute("thread step-in")

    def nexti(self) -> Dict[str, Any]:
        return self._execute("thread step-instruction-over")

    def stepi(self) -> Dict[str, Any]:
        return self._execute("thread step-instruction")

    def jump(self, locspec: str) -> Dict[str, Any]:
        return self._execute(f"thread jump --addr {locspec}")

    def return_from_function(self) -> Dict[str, Any]:
        return self._execute("thread return")

    def until(self, locspec: Optional[str] = None) -> Dict[str, Any]:
        if locspec:
            return self._execute(f"thread until {locspec}")
        return self._execute("thread step-out")

    def get_quick_context(self) -> Dict[str, Any]:
        parts: Dict[str, Any] = {}
        for ctx_type in ("regs", "stack", "disasm"):
            parts[ctx_type] = self.get_context(ctx_type)
        return {
            "success": all(item.get("success") for item in parts.values()),
            "command": "context all (quick)",
            "state": self.get_state(),
            "contexts": parts,
            "responses": [
                {"type": "console", "payload": f"context {key} rendered"}
                for key in parts
            ],
        }

    def get_context(self, context_type: str) -> Dict[str, Any]:
        mapping = {
            "regs": lambda: self._execute("register read"),
            "stack": lambda: self._execute("thread backtrace"),
            "disasm": lambda: self._execute("disassemble -c 10 -- $pc"),
            "backtrace": lambda: self._execute("thread backtrace"),
        }
        runner = mapping.get(context_type)
        if runner is None:
            return {
                "command": f"context {context_type}",
                "responses": [],
                "success": False,
                "state": self.get_state(),
                "error": f"context {context_type!r} is not available on the lldb backend",
                "type": "GdbOnlyBackendError",
            }
        return runner()

    def read_memory_bytes(self, address: str, num_bytes: int) -> Dict[str, Any]:
        count = max(1, int(num_bytes))
        result = self._execute(f"memory read -f hex -s 1 -c {count} -- {address}")
        hex_bytes: List[str] = []
        for line in self._console_lines(result):
            match = _MEMORY_DUMP_PATTERN.match(line)
            if match:
                hex_bytes.extend(match.group(1).split())
        start = parse_int(address)
        payload: Dict[str, Any] = {}
        if hex_bytes:
            payload["memory"] = [
                {
                    "begin": start if start is not None else address,
                    "end": start + len(hex_bytes) if start is not None else None,
                    "offset": "0x0",
                    "contents": "".join(hex_bytes),
                }
            ]
        result["payload"] = payload
        return result

    def read_memory_mi(
        self,
        address: str,
        word_format: str,
        word_size: int,
        nr_rows: int,
        nr_cols: int,
    ) -> Dict[str, Any]:
        size = max(1, int(nr_rows) * int(nr_cols) * int(word_size))
        return self.read_memory_bytes(address, size)

    def write_memory(self, address: str, data_hex: str) -> Dict[str, Any]:
        return self._execute(f"memory write -f hex -- {address} {data_hex}")

    def write_memory_bytes(self, address: str, data_hex: str) -> Dict[str, Any]:
        return self._execute(f"memory write -f hex -- {address} {data_hex}")

    def get_registers_mi(self) -> Dict[str, Any]:
        return self.execute_mi_command("-data-list-register-names")

    def get_register_values_mi(
        self, register_numbers: Optional[List[int]] = None, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        names = self._register_names_by_number()
        if register_numbers:
            selected = [
                names.get(number) for number in register_numbers if number in names
            ]
            suffix = (
                " -- " + " ".join(str(name) for name in selected if name)
                if selected
                else ""
            )
            command = f"-data-list-register-values x {' '.join(str(n) for n in register_numbers)}"
        else:
            suffix = ""
            command = "-data-list-register-values x"
        result = self._execute("register read" + suffix, timeout_sec=timeout_sec)
        values: List[Dict[str, Any]] = []
        for number, name in names.items():
            for line in self._console_lines(result):
                match = _REGISTER_LINE_PATTERN.match(line.strip())
                if match and match.group(1) == name:
                    values.append({"number": number, "value": match.group(2)})
                    break
        result["payload"] = {"register-values": values}
        return result

    def _register_names_by_number(self) -> Dict[int, str]:
        result = self.execute_mi_command("-data-list-register-names")
        names = result.get("payload", {}).get("register-names", [])
        return {number: str(name) for number, name in enumerate(names)}

    def select_thread(self, thread_id: int) -> Dict[str, Any]:
        return self._execute(f"thread select {thread_id}")

    def list_threads_mi(self, timeout_sec: float = 7.0) -> Dict[str, Any]:
        result = self._execute("thread list", timeout_sec=timeout_sec)
        threads: List[Dict[str, Any]] = []
        for line in self._console_lines(result):
            match = re.search(
                r"thread #(\d+)(?:, name = '([^']*)')?(?:, stop reason = (.*))?$",
                line.strip(),
            )
            if not match:
                continue
            threads.append(
                {
                    "id": match.group(1),
                    "name": match.group(2) or "",
                    "state": "stopped",
                    "reason": (match.group(3) or "").strip(),
                    "target-id": match.group(1),
                }
            )
        result["payload"] = {"threads": threads}
        return result

    def list_thread_groups_mi(
        self, recurse: int = 1, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        result = self._execute("target list", timeout_sec=timeout_sec)
        groups = []
        pid = self._inferior_pid
        if pid is not None:
            groups.append(
                {"id": f"i{pid}", "pid": pid, "type": "process", "exitable": True}
            )
        result["payload"] = {"groups": groups}
        return result

    def list_stack_frames_mi(
        self,
        low: int = 0,
        high: Optional[int] = None,
        timeout_sec: float = 7.0,
    ) -> Dict[str, Any]:
        result = self._execute("thread backtrace", timeout_sec=timeout_sec)
        stack: List[Dict[str, Any]] = []
        for line in self._console_lines(result):
            match = _FRAME_LINE_PATTERN.match(line.strip())
            if not match:
                continue
            stack.append(
                {
                    "level": match.group(1),
                    "addr": f"0x{match.group(2)}",
                    "func": match.group(3) or "",
                    "args": [],
                }
            )
        result["payload"] = {"stack": stack}
        return result

    def select_frame(self, level: int, timeout_sec: float = 7.0) -> Dict[str, Any]:
        return self._execute(f"frame select {level}", timeout_sec=timeout_sec)

    def get_selected_frame_mi(self, timeout_sec: float = 7.0) -> Dict[str, Any]:
        result = self._execute("frame info", timeout_sec=timeout_sec)
        result["payload"] = {"frame": {}}
        return result

    def set_breakpoint(
        self,
        location: str,
        condition: Optional[str] = None,
        temporary: bool = False,
    ) -> Dict[str, Any]:
        loc = location.strip()
        if loc.lower().startswith("*0x") or loc.lower().startswith("0x"):
            parts = ["breakpoint set --address", loc]
        else:
            parts = ["breakpoint set --name", loc]
        if condition:
            parts.extend(["--condition", condition])
        if temporary:
            parts.append("--one-shot true")
        return self._execute(" ".join(parts))

    def list_breakpoints(self) -> Dict[str, Any]:
        return self._execute("breakpoint list")

    def delete_breakpoint(self, number: int) -> Dict[str, Any]:
        return self._execute(f"breakpoint delete {number}")

    def enable_breakpoint(self, number: int) -> Dict[str, Any]:
        return self._execute(f"breakpoint enable {number}")

    def disable_breakpoint(self, number: int) -> Dict[str, Any]:
        return self._execute(f"breakpoint disable {number}")

    def set_watchpoint(self, expression: str, access: str = "write") -> Dict[str, Any]:
        return self._execute(f"watchpoint set expression -- {expression}")

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        result = self._execute(f"expression -- {expression}")
        value: Optional[str] = None
        for line in reversed(self._console_lines(result)):
            match = _EXPRESSION_RESULT_PATTERN.match(line.strip())
            if match:
                value = match.group(2)
                break
        result["payload"] = {"value": value}
        return result

    def disassemble_range(
        self, start: str, byte_count: int, timeout_sec: float = 7.0
    ) -> Dict[str, Any]:
        return self._execute(
            f"disassemble -s {start} -e {start}+{byte_count}",
            timeout_sec=timeout_sec,
        )

    def get_state(self) -> str:
        return self._state

    def get_inferior_pid(self) -> Optional[int]:
        return self._inferior_pid

    def get_inferior_epoch(self) -> int:
        return self._inferior_epoch

    def gdb_process_running(self) -> bool:
        return self._process_running()

    def get_gdb_process_pid(self) -> Optional[int]:
        process = self._process
        return process.pid if process is not None else None

    def is_destroyed(self) -> bool:
        if getattr(self, "_destroyed", False):
            return True
        if not self._process_running():
            self.mark_destroyed("pwndbg-lldb process exited unexpectedly")
            return True
        return False

    def mark_destroyed(self, reason: str) -> None:
        self._destroyed = True
        self._destroyed_reason = str(reason)

    def get_destroyed_reason(self) -> Optional[str]:
        reason = getattr(self, "_destroyed_reason", None)
        return str(reason) if reason is not None else None

    def close(self, timeout_sec: float = DEFAULT_LLDB_CLOSE_TIMEOUT_SEC) -> bool:
        self.mark_destroyed("lldb controller close requested")
        process = self._process
        if process is None:
            self._state = "exited"
            self._initialized = False
            return True
        try:
            if process.stdin is not None:
                try:
                    process.stdin.write(b"quit\n")
                    process.stdin.flush()
                except OSError:
                    pass
        except (ValueError, OSError):
            pass
        if self._process_running():
            try:
                process.wait(timeout=max(0.0, float(timeout_sec)))
            except (subprocess.TimeoutExpired, TimeoutError):
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=max(0.0, float(timeout_sec)))
                except (subprocess.TimeoutExpired, TimeoutError):
                    try:
                        process.kill()
                    except OSError:
                        pass
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    logger.debug(
                        "Failed to close lldb %s pipe", stream_name, exc_info=True
                    )
        self._process = None
        self._state = "exited"
        self._initialized = False
        return True


# ----------------------------------------------------------------------
# MI token translator
# ----------------------------------------------------------------------
def _translate_mi_command(
    command: str, controller: Optional[LldbController] = None
) -> Optional[str]:
    """Map one GDB/MI command token to an lldb command, or ``None``."""
    token, _, rest = command.partition(" ")
    rest = rest.strip()
    if token == "-data-evaluate-expression":
        expr = _unquote(rest)
        return f"expression -- {expr or rest}"
    if token == "-data-read-memory-bytes":
        parts = rest.split()
        if len(parts) >= 2:
            address = _unquote(parts[0])
            count = parts[1]
            return f"memory read -f hex -s 1 -c {count} -- {address}"
        return None
    if token == "-data-write-memory-bytes":
        parts = rest.split()
        if len(parts) >= 2:
            address = _unquote(parts[0])
            data_hex = parts[1]
            return f"memory write -f hex -- {address} {data_hex}"
        return None
    if token == "-data-list-register-names":
        return "register info"
    if token == "-data-list-register-values":
        # Format token then numbers.
        parts = rest.split()
        numbers = [part for part in parts if part.isdigit()]
        if numbers and controller is not None:
            names_by_number = controller._register_names_by_number()
            names = [
                names_by_number.get(int(number), "")
                for number in numbers
                if int(number) in names_by_number
            ]
            if names:
                return "register read -- " + " ".join(names)
        return "register read"
    if token == "-exec-run":
        return "process launch"
    if token == "-exec-continue":
        return "process continue"
    if token == "-exec-next":
        return "thread step-over"
    if token == "-exec-step":
        return "thread step-in"
    if token == "-exec-next-instruction":
        return "thread step-instruction-over"
    if token == "-exec-step-instruction":
        return "thread step-instruction"
    if token == "-exec-finish":
        return "thread step-out"
    if token == "-exec-until":
        return f"thread until {_unquote(rest) or rest}"
    if token == "-exec-return":
        return "thread return"
    if token == "-exec-jump":
        loc = _unquote(rest) or rest
        if loc.startswith("*"):
            loc = loc.lstrip("*")
        return f"thread jump --addr {loc}"
    if token == "-break-insert":
        parts = rest.split()
        one_shot = "-t" in parts
        condition = None
        if "-c" in parts:
            idx = parts.index("-c")
            if idx + 1 < len(parts):
                condition = _unquote(parts[idx + 1])
        # Last token is the location.
        if not parts:
            return None
        location = _unquote(parts[-1])
        if location.startswith("0x") or location.startswith("*0x"):
            cmd = ["breakpoint set --address", location.lstrip("*")]
        else:
            cmd = ["breakpoint set --name", location]
        if condition:
            cmd.extend(["--condition", condition])
        if one_shot:
            cmd.append("--one-shot true")
        return " ".join(cmd)
    if token == "-break-list":
        return "breakpoint list"
    if token == "-break-delete":
        return f"breakpoint delete {_unquote(rest) or rest}"
    if token in ("-break-enable", "-break-disable"):
        action = "enable" if token == "-break-enable" else "disable"
        return f"breakpoint {action} {_unquote(rest) or rest}"
    if token == "-break-watch":
        return f"watchpoint set expression -- {_unquote(rest) or rest}"
    if token == "-target-attach":
        return f"process attach -p {_unquote(rest) or rest}"
    if token == "-target-detach":
        return "process detach"
    if token == "-target-select":
        parts = rest.split()
        if len(parts) >= 2 and parts[0] == "remote":
            return f"gdb-remote {parts[1]}"
        return None
    if token == "-file-exec-and-symbols":
        return f"target create {_unquote(rest) or rest}"
    if token == "-thread-select":
        return f"thread select {_unquote(rest) or rest}"
    if token == "-thread-info":
        return "thread list"
    if token == "-thread-list-ids":
        return "thread list"
    if token == "-stack-select-frame":
        return f"frame select {_unquote(rest) or rest}"
    if token == "-stack-list-frames":
        return "thread backtrace"
    if token == "-stack-info-frame":
        return "frame info"
    if token == "-data-disassemble":
        start = _unquote(rest) if rest else None
        return f"disassemble {start}" if start else "disassemble"
    if token.startswith("-gdb-set"):
        return None  # settings are applied during initialize; ignore stray sets
    if token.startswith("-gdb-show"):
        return "settings show"
    if token.startswith("-inferior") or token.startswith("-list-thread-groups"):
        return None
    if token.startswith("-data-"):
        return None
    if token in ("-environment-cd", "-environment-cwd"):
        return None
    return None


def _unquote(value: str) -> str:
    """Strip one layer of MI quoting from a C-string argument."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def _shape_mi_result(
    mi_command: str, lldb_command: str, result: Dict[str, Any]
) -> Dict[str, Any]:
    """Re-shape an lldb execution result into an MI-like response record."""
    console_lines = [
        str(response.get("payload", ""))
        for response in result.get("responses", [])
        if response.get("type") == "console"
    ]
    token = mi_command.split(" ")[0]
    payload = _payload_for_token(token, console_lines, result.get("payload"))
    responses: List[Dict[str, Any]] = []
    if result.get("success"):
        responses.append({"type": "result", "message": "done", "payload": payload})
    else:
        messages = console_lines or [
            str(response.get("payload", ""))
            for response in result.get("responses", [])
            if response.get("type") == "console"
        ]
        responses.append(
            {
                "type": "result",
                "message": "error",
                "payload": {"msg": messages[-1] if messages else "lldb command failed"},
            }
        )
    return {
        "command": mi_command,
        "responses": responses,
        "success": bool(result.get("success")),
        "state": result.get("state") or "idle",
    }


_NAME_LINE_PATTERN = re.compile(r"^\s*Name:\s*(\S+)")


def _payload_for_token(
    token: str, console_lines: List[str], existing: Optional[Any]
) -> Dict[str, Any]:
    """Build the MI payload shape readers expect for a given MI token."""
    if token == "-data-list-register-names":
        names: List[str] = []
        for line in console_lines:
            match = _NAME_LINE_PATTERN.match(line)
            if match and match.group(1) not in names:
                names.append(match.group(1))
        return {"register-names": names}
    if token == "-data-read-memory-bytes":
        hex_bytes: List[str] = []
        for line in console_lines:
            match = _MEMORY_DUMP_PATTERN.match(line)
            if match:
                hex_bytes.extend(match.group(1).split())
        if not hex_bytes:
            return {}
        return {"memory": [{"contents": "".join(hex_bytes)}]}
    if token == "-data-list-register-values":
        values: List[Dict[str, str]] = []
        for line in console_lines:
            match = _REGISTER_LINE_PATTERN.match(line.strip())
            if match:
                values.append({"name": match.group(1), "value": match.group(2)})
        return {"register-values": values}
    if token == "-data-evaluate-expression":
        value: Optional[str] = None
        for line in reversed(console_lines):
            match = _EXPRESSION_RESULT_PATTERN.match(line.strip())
            if match:
                value = match.group(2)
                # lldb prefixes results with "$N = " -- strip it to the raw value.
                if value.lstrip().startswith("$"):
                    _, _, stripped = value.partition("=")
                    value = stripped.strip()
                break
        return {"value": value}
    if isinstance(existing, Mapping):
        return dict(existing)
    return {}


__all__ = ["LldbController", "BackendUnavailableError"]
