"""
Subprocess execution tools for Pao Tools

Provides tools for running system commands, particularly useful for:
- Compiling binaries with various flags (ASAN, debug symbols, etc.)
- Running helper scripts
- Managing background processes
"""

import logging
import os
import re
import signal as signal_module
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Any, BinaryIO, Callable, Dict, List, Optional

import psutil

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationRegistry, current_operation_id
from paomcp.utils.paths import DEFAULT_WORKSPACE

logger = logging.getLogger(__name__)
KILL_SIGNAL = int(getattr(signal_module, "SIGKILL", signal_module.SIGTERM))
DEFAULT_MAX_MANAGED_PROCESSES = int(os.environ.get("PAO_MAX_MANAGED_PROCESSES", "16"))
DEFAULT_MAX_PROCESS_OUTPUT_BYTES = int(
    os.environ.get("PAO_MAX_PROCESS_OUTPUT_BYTES", str(256 * 1024**2))
)
OUTPUT_READ_CHUNK_BYTES = 64 * 1024
OUTPUT_READER_JOIN_SECONDS = 5.0
# Shared clamps live in timeouts.py; re-exported here so
# existing `from subproc import clamp_timeout` imports keep working.
from paomcp.tools.backends.timeouts import (
    ALLOWED_KILL_SIGNALS,
    DEFAULT_COMMAND_TIMEOUT,
    MAX_COMMAND_TIMEOUT,
    clamp_timeout,
)


class ProcessLimitError(RuntimeError):
    """Raised when no managed background-process slot is available."""


class ProcessOutputLimitError(RuntimeError):
    """Raised when combined process stdout and stderr exceed their byte budget."""


class _BoundedProcessOutput:
    """Drain two process pipes while enforcing one exact combined byte budget."""

    def __init__(
        self,
        stdout_path: str,
        stderr_path: str,
        max_bytes: int,
        terminate_process_tree: Callable[[], None],
    ) -> None:
        self.paths = {"stdout": stdout_path, "stderr": stderr_path}
        self.max_bytes = max_bytes
        self._terminate_process_tree = terminate_process_tree
        self._lock = threading.RLock()
        self._threads: List[threading.Thread] = []
        self._sources: List[BinaryIO] = []
        self._stats: Dict[str, Dict[str, int]] = {
            "stdout": {"observed_bytes": 0, "stored_bytes": 0, "dropped_bytes": 0},
            "stderr": {"observed_bytes": 0, "stored_bytes": 0, "dropped_bytes": 0},
        }
        self._exceeded = False
        self._termination_requested = False
        self._termination_error: Optional[str] = None
        self._reader_errors: List[Dict[str, str]] = []
        self._finished = False
        for path in self.paths.values():
            with open(path, "wb"):
                pass

    def start(self, process: subprocess.Popen[Any]) -> None:
        """Start one daemon drain thread per real subprocess pipe."""
        for stream in ("stdout", "stderr"):
            source = getattr(process, stream, None)
            if source is None or not hasattr(source, "read"):
                continue
            self._sources.append(source)
            thread = threading.Thread(
                target=self._drain,
                args=(stream, source),
                name=f"pao-process-{process.pid}-{stream}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _request_termination(self) -> None:
        with self._lock:
            if self._termination_requested:
                return
            self._termination_requested = True
        try:
            self._terminate_process_tree()
        except (ProcessLookupError, psutil.NoSuchProcess):
            return
        except Exception as exc:
            logger.exception("Failed to terminate output-limited process tree")
            with self._lock:
                self._termination_error = f"{type(exc).__name__}: {exc}"

    def _record_chunk(self, stream: str, output_file: BinaryIO, chunk: bytes) -> None:
        should_terminate = False
        with self._lock:
            stream_stats = self._stats[stream]
            stream_stats["observed_bytes"] += len(chunk)
            total_stored = sum(item["stored_bytes"] for item in self._stats.values())
            accepted_length = min(len(chunk), max(0, self.max_bytes - total_stored))
            written = 0
            if accepted_length:
                accepted = chunk[:accepted_length]
                result = output_file.write(accepted)
                written = accepted_length if result is None else result
                if written != accepted_length:
                    raise OSError(
                        "Could not persist the complete accepted process-output chunk"
                    )
            stream_stats["stored_bytes"] += written
            dropped = len(chunk) - written
            stream_stats["dropped_bytes"] += dropped
            if dropped:
                self._exceeded = True
                should_terminate = not self._termination_requested
        if should_terminate:
            self._request_termination()

    def _drain(self, stream: str, source: BinaryIO) -> None:
        try:
            with open(self.paths[stream], "wb", buffering=0) as output_file:
                while True:
                    read_chunk = getattr(source, "read1", source.read)
                    chunk = read_chunk(OUTPUT_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    self._record_chunk(stream, output_file, chunk)
        except (ValueError, OSError) as exc:
            logger.exception("Failed while draining managed-process %s", stream)
            with self._lock:
                self._reader_errors.append(
                    {"stream": stream, "type": type(exc).__name__, "error": str(exc)}
                )
            self._request_termination()
        finally:
            try:
                source.close()
            except OSError:
                pass

    def finish(self, timeout: float = OUTPUT_READER_JOIN_SECONDS) -> Dict[str, Any]:
        """Boundedly join readers; closing pipes prevents shutdown from hanging."""
        with self._lock:
            if self._finished:
                return self.snapshot()
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        unfinished = [thread for thread in self._threads if thread.is_alive()]
        if unfinished:
            for source in self._sources:
                try:
                    source.close()
                except OSError:
                    pass
            for thread in unfinished:
                thread.join(timeout=0.25)
        with self._lock:
            self._finished = not any(thread.is_alive() for thread in self._threads)
            return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        """Return exact stored/observed/discarded byte accounting."""
        with self._lock:
            streams = {name: dict(stats) for name, stats in self._stats.items()}
            alive = [thread.name for thread in self._threads if thread.is_alive()]
            total_observed = sum(item["observed_bytes"] for item in streams.values())
            total_stored = sum(item["stored_bytes"] for item in streams.values())
            total_dropped = sum(item["dropped_bytes"] for item in streams.values())
            return {
                "limit_bytes": self.max_bytes,
                "exceeded": self._exceeded,
                "total_observed_bytes": total_observed,
                "total_stored_bytes": total_stored,
                "total_dropped_bytes": total_dropped,
                "stdout": streams["stdout"],
                "stderr": streams["stderr"],
                "termination_requested": self._termination_requested,
                "termination_error": self._termination_error,
                "drain_complete": not alive,
                "measurement_complete": not alive and not self._reader_errors,
                "reader_errors": list(self._reader_errors),
                "reader_threads_alive": alive,
            }


class SubprocessTools:
    """Tools for subprocess execution and management"""

    def __init__(
        self,
        process_root: Optional[str] = None,
        workspace_root: str = DEFAULT_WORKSPACE,
        artifact_store: Optional[ArtifactStore] = None,
        operation_registry: Optional[OperationRegistry] = None,
        max_processes: Optional[int] = None,
        max_output_bytes: Optional[int] = None,
    ):
        """Initialize subprocess tools."""
        self.background_processes: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.workspace_root = os.path.normpath(workspace_root)
        self.artifact_store = artifact_store
        self.operation_registry = operation_registry
        self.max_processes = (
            DEFAULT_MAX_MANAGED_PROCESSES if max_processes is None else max_processes
        )
        if self.max_processes <= 0:
            raise ValueError("max_processes must be positive")
        self.max_output_bytes = (
            DEFAULT_MAX_PROCESS_OUTPUT_BYTES
            if max_output_bytes is None
            else max_output_bytes
        )
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be positive")
        self._process_slots = threading.BoundedSemaphore(self.max_processes)
        self._completed_processes: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
        self.process_root = process_root or os.path.join(
            tempfile.gettempdir(), "pao", "processes"
        )
        os.makedirs(self.process_root, exist_ok=True)

    def _release_process_entry(self, entry: Dict[str, Any]) -> None:
        """Release one managed-process capacity token exactly once."""
        if entry.pop("slot_acquired", False):
            self._process_slots.release()

    def _close_process_entry(self, entry: Dict[str, Any]) -> None:
        """Close output handles and release capacity exactly once."""
        output_limiter = entry.get("output_limiter")
        if output_limiter is not None:
            entry["output_limit"] = output_limiter.finish()
        for key in ("stdout_file", "stderr_file"):
            output_file = entry.pop(key, None)
            if output_file is not None:
                try:
                    output_file.close()
                except OSError:
                    pass
        self._release_process_entry(entry)

    @staticmethod
    def _apply_output_evidence(
        result: Dict[str, Any],
        output_limit: Dict[str, Any],
        *,
        require_complete: bool = True,
    ) -> Dict[str, Any]:
        """Attach output accounting and make any loss an explicit tool failure."""
        result["output_limit"] = output_limit
        if output_limit.get("exceeded"):
            result.update(
                {
                    "success": False,
                    "code": "RESOURCE_EXHAUSTED",
                    "type": "ProcessOutputLimitError",
                    "error": (
                        "Combined stdout and stderr exceeded the configured "
                        f"{output_limit['limit_bytes']}-byte process output limit; "
                        "the process tree was terminated"
                    ),
                }
            )
        elif require_complete and not output_limit.get("measurement_complete", True):
            result.update(
                {
                    "success": False,
                    "code": "OUTPUT_CAPTURE_INCOMPLETE",
                    "type": "ProcessOutputCaptureError",
                    "error": "Process output capture did not drain completely",
                }
            )
        return result

    def _remember_completed(self, pid: int, entry: Dict[str, Any]) -> None:
        """Retain a bounded terminal record so polling cannot discard quota evidence."""
        evicted: List[Dict[str, Any]] = []
        with self._lock:
            replaced = self._completed_processes.pop(pid, None)
            if replaced is not None and replaced is not entry:
                evicted.append(replaced)
            self._completed_processes[pid] = entry
            while len(self._completed_processes) > self.max_processes:
                _old_pid, old_entry = self._completed_processes.popitem(last=False)
                evicted.append(old_entry)
        for old_entry in evicted:
            self._finalize_process_entry(old_entry)

    def _remove_process_dir(self, entry: Dict[str, Any]) -> None:
        """Remove terminal-process logs after their output has been delivered."""
        process_dir = entry.get("process_dir")
        if not process_dir:
            return
        resolved_root = os.path.abspath(self.process_root)
        resolved_dir = os.path.abspath(str(process_dir))
        try:
            contained = (
                os.path.commonpath([resolved_root, resolved_dir]) == resolved_root
                and resolved_dir != resolved_root
            )
        except ValueError:
            contained = False
        if not contained:
            logger.error(
                "Refusing to remove process log directory outside process_root: %s",
                process_dir,
            )
            return
        try:
            shutil.rmtree(resolved_dir)
        except FileNotFoundError:
            pass
        except OSError:
            logger.exception("Failed to remove managed process log directory")

    def _finalize_process_entry(self, entry: Dict[str, Any]) -> None:
        """Close one terminal entry, release its slot, and remove temporary logs."""
        self._close_process_entry(entry)
        self._remove_process_dir(entry)

    def _pop_process_entry(
        self, pid: int, expected: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Atomically claim cleanup ownership for one registry entry."""
        with self._lock:
            entry = self.background_processes.get(pid)
            if entry is not None and (expected is None or entry is expected):
                return self.background_processes.pop(pid)
            entry = self._completed_processes.get(pid)
            if entry is None or (expected is not None and entry is not expected):
                return None
            return self._completed_processes.pop(pid)

    def _reap_terminated(self) -> None:
        """Release slots and retain bounded output for processes that exited."""
        with self._lock:
            terminated = [
                (pid, entry)
                for pid, entry in self.background_processes.items()
                if entry["process"].poll() is not None
            ]
            for pid, _entry in terminated:
                self.background_processes.pop(pid, None)
        for pid, entry in terminated:
            self._close_process_entry(entry)
            self._remember_completed(pid, entry)

    @staticmethod
    def _assert_process_owner(entry: Dict[str, Any], session_id: Optional[str]) -> None:
        if entry.get("session_id") != session_id:
            raise PermissionError("The process belongs to a different session")

    def _read_output(
        self,
        path: str,
        stream: str,
        tool: str,
        session_id: Optional[str] = None,
    ) -> Any:
        """Return a small stream inline or persist a large stream incrementally."""
        if (
            self.artifact_store is not None
            and os.path.getsize(path) > self.artifact_store.inline_response_bytes
        ):
            metadata = self.artifact_store.save_file(
                path,
                tool=tool,
                session_id=session_id,
                content_type="text/plain",
                encoding="utf-8",
            )
            return {**metadata.to_dict(), "stream": stream}

        with open(path, "rb") as output_file:
            return output_file.read().decode("utf-8", errors="replace")

    def _is_under_workspace(self, path: str) -> bool:
        workspace_root = self.workspace_root
        candidate = os.path.normpath(path)
        try:
            return os.path.commonpath([workspace_root, candidate]) == workspace_root
        except ValueError:
            return False

    @staticmethod
    def _is_elf_file(path: str) -> bool:
        if not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                return f.read(4) == b"\x7fELF"
        except OSError:
            return False

    @staticmethod
    def _command_segment_heads(cmd_parts: List[str]) -> List[int]:
        """Return the index of every shell segment's executable-position token.

        ``sh -lc`` parses the whole command string, so a first-token-only
        guardrail is trivially bypassed by ``cd x && ./chal``,
        ``env python3 -c ...`` or ``; ./chal``.  This walks the token list
        and collects the head index of each ``&&`` / ``||`` / ``;`` / ``|`` /
        ``&`` segment so every command that will actually run is inspected.
        """
        if not cmd_parts:
            return []
        separators = {"&&", "||", ";", "|", "&"}
        heads = [0]
        for i, token in enumerate(cmd_parts):
            if token in separators and i + 1 < len(cmd_parts):
                nxt = cmd_parts[i + 1]
                if nxt not in separators:
                    heads.append(i + 1)
        return heads

    @staticmethod
    def _contains_dash_c(cmd_parts: List[str], token_index: int) -> bool:
        """Whether ``-c`` follows the python executable at ``token_index``.

        Scans the tokens after the executable: option flags (``-O``, ``-u``)
        are skipped, ``-c`` itself is a hit, and a shell separator or the
        first non-option argument ends the lookahead.  This keeps
        ``python3 -c``, ``python3 -O -c`` and ``env python3 -c`` blocked while
        ``grep -c python3`` (``-c`` precedes python3) and ``python3 -V file``
        are left alone.
        """
        for nxt in cmd_parts[token_index + 1 :]:
            if nxt in {"&&", "||", ";", "|", "&"}:
                break
            if not nxt.startswith("-"):
                break
            if nxt == "-c" or nxt.startswith("-c"):
                return True
        return False

    @staticmethod
    def _resolve_candidate_executable(token: str, cwd: Optional[str]) -> Optional[str]:
        if os.path.isabs(token):
            return os.path.normpath(token)
        if "/" in token:
            base = cwd or os.getcwd()
            return os.path.normpath(os.path.join(base, token))
        return None

    @staticmethod
    def _shell_argv(command: str) -> List[str]:
        """Build an explicit platform shell invocation for documented shell syntax."""
        if os.name == "nt":
            return ["cmd.exe", "/d", "/s", "/c", command]
        return ["/bin/sh", "-lc", command]

    @staticmethod
    def _popen_group_options() -> Dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    @staticmethod
    def _signal_process_tree(process: subprocess.Popen, signal_number: int) -> None:
        """Signal a managed process and all descendants."""
        if process.poll() is not None:
            return
        if os.name != "nt":
            getpgid = getattr(os, "getpgid")
            killpg = getattr(os, "killpg")
            killpg(getpgid(process.pid), signal_number)
            return

        parent = psutil.Process(process.pid)
        descendants = parent.children(recursive=True)
        for child in reversed(descendants):
            try:
                if signal_number == KILL_SIGNAL:
                    child.kill()
                elif signal_number == signal_module.SIGTERM:
                    child.terminate()
                else:
                    child.send_signal(signal_number)
            except psutil.Error:
                pass
        if signal_number == KILL_SIGNAL:
            parent.kill()
        elif signal_number == signal_module.SIGTERM:
            parent.terminate()
        else:
            parent.send_signal(signal_number)

    @staticmethod
    def _is_python_executable(token: str) -> bool:
        name = os.path.basename(token)
        if name in {"python", "python3"}:
            return True
        return bool(re.fullmatch(r"python3\.\d+", name))

    @staticmethod
    def _guardrail_error(
        command: str,
        cwd: Optional[str],
        error: str,
        recommended_tool: str,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "command": command,
            "error": error,
            "type": "ToolUsageError",
            "recommended_tool": recommended_tool,
            "cwd": cwd or os.getcwd(),
        }

    def _check_command_guardrails(
        self,
        command: str,
        cmd_parts: List[str],
        cwd: Optional[str],
        operation: str,
    ) -> Optional[Dict[str, Any]]:
        if not cmd_parts:
            return None

        executable = cmd_parts[0]

        # The command string is executed as ``sh -lc <command>``, so the
        # guardrails must inspect EVERY shell segment's executable position,
        # not just ``cmd_parts[0]`` -- ``cd /workspace && ./chal`` or
        # ``env python3 -c ...`` would otherwise sail past a first-token-only
        # check while the shell happily runs the workspace ELF / python.
        segment_heads = self._command_segment_heads(cmd_parts)

        # Scan every token, not just segment heads: ``env python3 -c ...`` and
        # ``sudo python3 -c ...`` run python from a wrapper argument, and the
        # inline-script branch below already scans the whole sh -c body, so a
        # whole-token scan is the consistent rule.
        for token_index, token in enumerate(cmd_parts):
            if self._is_python_executable(token) and self._contains_dash_c(
                cmd_parts, token_index
            ):
                return self._guardrail_error(
                    command=command,
                    cwd=cwd,
                    error=("Do not use 'python -c' here. Use execute_python_code."),
                    recommended_tool="execute_python_code",
                )

        if executable in {"sh", "bash", "/bin/sh", "/bin/bash"}:
            inline_script = ""
            script_index: Optional[int] = None
            for i, arg in enumerate(cmd_parts[1:], start=1):
                if not arg.startswith("-"):
                    continue
                if "c" in arg:
                    script_index = i + 1
                    break
            if script_index is not None and script_index < len(cmd_parts):
                inline_script = cmd_parts[script_index]
            if re.search(r"\bpython(?:3(?:\.\d+)?)?\b\s+-c\b", inline_script):
                return self._guardrail_error(
                    command=command,
                    cwd=cwd,
                    error=(
                        "Do not wrap 'python -c' in a shell command here. "
                        "Use execute_python_code."
                    ),
                    recommended_tool="execute_python_code",
                )

        for head_index in segment_heads:
            candidate = self._resolve_candidate_executable(cmd_parts[head_index], cwd)
            if (
                candidate
                and self._is_under_workspace(candidate)
                and self._is_elf_file(candidate)
            ):
                action = "run under debugger" if operation == "run_command" else "spawn"
                return self._guardrail_error(
                    command=command,
                    cwd=cwd,
                    error=(
                        f"Do not use {operation} to {action} a target ELF under "
                        f"{self.workspace_root}. "
                        "Use set_file + run (or attach). Use paocli for interactive "
                        "exploit-driver workflows."
                    ),
                    recommended_tool="set_file+run",
                )

        return None

    def run_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 30.0,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a command and wait for it to complete.

        This is primarily intended for compilation commands like:
        - gcc -g -fsanitize=address program.c -o program
        - clang -O0 -g -fno-omit-frame-pointer vuln.c
        - make clean && make

        For Python execution, prefer python_env tools (execute_code / execute_script)
        so callers use the managed shared Python environment.

        Args:
            command: Command to execute (will be shell-parsed)
            cwd: Working directory for the command
            env: Environment variables (merged with current env)
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with execution results
        """
        timeout = clamp_timeout(timeout)
        logger.info(f"Running command: {command}")
        operation_id = current_operation_id()
        if self.operation_registry is not None and operation_id is not None:
            self.operation_registry.update_progress(
                operation_id, current=0, total=2, message="starting command"
            )
        process_dir: Optional[str] = None
        stdout_file = None
        stderr_file = None
        process: Optional[subprocess.Popen[Any]] = None
        output_limiter: Optional[_BoundedProcessOutput] = None

        try:
            # Parse command for safer execution
            cmd_parts = shlex.split(command, posix=os.name != "nt")
            guardrail_result = self._check_command_guardrails(
                command=command,
                cmd_parts=cmd_parts,
                cwd=cwd,
                operation="run_command",
            )
            if guardrail_result:
                return guardrail_result

            # Prepare environment
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)

            process_dir = tempfile.mkdtemp(prefix="command_", dir=self.process_root)
            stdout_path = os.path.join(process_dir, "stdout.log")
            stderr_path = os.path.join(process_dir, "stderr.log")

            process = subprocess.Popen(
                self._shell_argv(command),
                cwd=cwd,
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **self._popen_group_options(),
            )
            output_limiter = _BoundedProcessOutput(
                stdout_path,
                stderr_path,
                self.max_output_bytes,
                lambda: self._signal_process_tree(process, KILL_SIGNAL),
            )
            output_limiter.start(process)
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.update_progress(
                    operation_id, current=1, total=2, message="command running"
                )
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.set_cancel_handler(
                    operation_id,
                    lambda: self._signal_process_tree(process, KILL_SIGNAL),
                )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_process_tree(process, KILL_SIGNAL)
                process.wait(timeout=5)

                timed_out = True
            finally:
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.set_cancel_handler(operation_id, None)

            output_limit = output_limiter.finish()
            stdout = self._read_output(stdout_path, "stdout", "run_command", session_id)
            stderr = self._read_output(stderr_path, "stderr", "run_command", session_id)
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.update_progress(
                    operation_id, current=2, total=2, message="complete"
                )

            result = {
                "success": process.returncode == 0,
                "command": command,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": cwd or os.getcwd(),
                "session_id": session_id,
            }
            if timed_out:
                result["success"] = False
                result["code"] = "TIMEOUT"
                result["type"] = "TimeoutError"
                result["error"] = f"Command timed out after {timeout} seconds"
            cancellation_event = (
                self.operation_registry.cancellation_event(operation_id)
                if self.operation_registry is not None and operation_id is not None
                else None
            )
            if cancellation_event is not None and cancellation_event.is_set():
                result["success"] = False
                result["code"] = "OPERATION_CANCELLED"
                result["type"] = "OperationCancelledError"
                result["error"] = "Command execution was cancelled"
            return self._apply_output_evidence(result, output_limit)

        except Exception as e:
            logger.exception("Failed to run command")
            if process is not None and process.poll() is None:
                try:
                    self._signal_process_tree(process, KILL_SIGNAL)
                    process.wait(timeout=5)
                except Exception:
                    logger.exception("Failed to clean up command after output error")
            final_output_limit = output_limiter.finish() if output_limiter else None
            result = {
                "success": False,
                "command": command,
                "error": str(e),
                "type": type(e).__name__,
                "session_id": session_id,
                "cwd": cwd or os.getcwd(),
            }
            if final_output_limit is not None:
                return self._apply_output_evidence(result, final_output_limit)
            return result
        finally:
            for output_file in (stdout_file, stderr_file):
                if output_file is not None:
                    try:
                        output_file.close()
                    except OSError:
                        pass
            if process_dir is not None:
                self._remove_process_dir({"process_dir": process_dir})
            if process_dir is not None:
                shutil.rmtree(process_dir, ignore_errors=True)

    def spawn_process(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Spawn a background process and return immediately

        Useful for starting long-running processes like:
        - Web servers for exploitation
        - Network listeners
        - Monitoring scripts

        Args:
            command: Command to execute
            cwd: Working directory for the command
            env: Environment variables

        Returns:
            Dictionary with process information including PID
        """
        logger.info(f"Spawning process: {command}")
        stdout_file = None
        stderr_file = None
        slot_acquired = False
        process: Optional[subprocess.Popen[Any]] = None
        stored_entry: Optional[Dict[str, Any]] = None
        output_limiter: Optional[_BoundedProcessOutput] = None

        try:
            # Parse command
            cmd_parts = shlex.split(command, posix=os.name != "nt")
            guardrail_result = self._check_command_guardrails(
                command=command,
                cmd_parts=cmd_parts,
                cwd=cwd,
                operation="spawn_process",
            )
            if guardrail_result:
                return guardrail_result

            self._reap_terminated()
            slot_acquired = self._process_slots.acquire(blocking=False)
            if not slot_acquired:
                raise ProcessLimitError(
                    "Managed process limit reached "
                    f"({self.max_processes}/{self.max_processes})"
                )

            process_dir = tempfile.mkdtemp(prefix="proc_", dir=self.process_root)
            stdout_path = os.path.join(process_dir, "stdout.log")
            stderr_path = os.path.join(process_dir, "stderr.log")

            # Prepare environment
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)

            # Spawn process with stdout and stderr redirected to files
            process = subprocess.Popen(
                self._shell_argv(command),
                cwd=cwd,
                env=cmd_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **self._popen_group_options(),
            )
            output_limiter = _BoundedProcessOutput(
                stdout_path,
                stderr_path,
                self.max_output_bytes,
                lambda: self._signal_process_tree(process, KILL_SIGNAL),
            )
            output_limiter.start(process)

            # Store process reference and file paths
            stored_entry = {
                "process": process,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "output_limiter": output_limiter,
                "process_dir": process_dir,
                "command": command,
                "cwd": cwd or os.getcwd(),
                "session_id": session_id,
                "slot_acquired": True,
            }
            with self._lock:
                stale_entry = self._completed_processes.pop(process.pid, None)
                self.background_processes[process.pid] = stored_entry
            if stale_entry is not None:
                self._finalize_process_entry(stale_entry)

            # Give process a moment to potentially fail
            import time

            time.sleep(0.1)

            if process.poll() is not None:
                # Process already terminated, read outputs
                with self._lock:
                    entry = self.background_processes.pop(process.pid, None)
                if entry is not None:
                    self._close_process_entry(entry)
                    slot_acquired = False
                output_limit = output_limiter.finish()
                stdout = self._read_output(
                    stdout_path, "stdout", "spawn_process", session_id
                )
                stderr = self._read_output(
                    stderr_path, "stderr", "spawn_process", session_id
                )
                if entry is not None:
                    self._remove_process_dir(entry)
                result = {
                    "success": False,
                    "command": command,
                    "error": "Process terminated immediately",
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_path": None,
                    "stderr_path": None,
                    "logs_retained": False,
                    "session_id": session_id,
                }
                return self._apply_output_evidence(result, output_limit)

            result = {
                "success": True,
                "command": command,
                "pid": process.pid,
                "cwd": cwd or os.getcwd(),
                "status": "running",
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "process_dir": process_dir,
                "session_id": session_id,
            }
            return self._apply_output_evidence(
                result, output_limiter.snapshot(), require_complete=False
            )

        except Exception as e:
            logger.error(f"Failed to spawn process: {e}")
            claimed_entry = (
                self._pop_process_entry(process.pid, stored_entry)
                if process is not None and stored_entry is not None
                else None
            )
            if claimed_entry is not None:
                if process is not None and process.poll() is None:
                    try:
                        self._signal_process_tree(process, KILL_SIGNAL)
                        process.wait(timeout=5)
                    except Exception:
                        logger.exception("Failed to clean up partially spawned process")
                self._finalize_process_entry(claimed_entry)
                slot_acquired = False
            elif process is not None and process.poll() is None:
                try:
                    self._signal_process_tree(process, KILL_SIGNAL)
                except Exception:
                    logger.exception("Failed to clean up partially spawned process")
            try:
                if stdout_file:
                    stdout_file.close()
            except Exception:
                pass
            try:
                if stderr_file:
                    stderr_file.close()
            except Exception:
                pass
            if slot_acquired:
                self._process_slots.release()
            result = {
                "success": False,
                "command": command,
                "error": str(e),
                "type": type(e).__name__,
                "code": (
                    "RESOURCE_EXHAUSTED" if isinstance(e, ProcessLimitError) else None
                ),
                "session_id": session_id,
            }
            if output_limiter is not None:
                result = self._apply_output_evidence(
                    result, output_limiter.finish(timeout=1.0)
                )
            return result

    def get_process(self, pid: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of a spawned process

        Args:
            pid: Process ID to check

        Returns:
            Dictionary with process status. When available, includes live
            standard stream contents as strings:
            - ``stdout``: accumulated standard output so far
            - ``stderr``: accumulated standard error so far
            Paths to the backing log files (``stdout_path``, ``stderr_path``)
            are also returned for external tailing.
        """
        try:
            with self._lock:
                entry = self.background_processes.get(pid)
                retained_completed = False
                if entry is None:
                    entry = self._completed_processes.get(pid)
                    retained_completed = entry is not None
                if entry is None:
                    return {
                        "success": False,
                        "pid": pid,
                        "session_id": session_id,
                        "error": "Process is not managed by pao-tools",
                        "type": "UnmanagedProcessError",
                    }
                self._assert_process_owner(entry, session_id)
                if retained_completed:
                    self._completed_processes.pop(pid, None)
                process = entry["process"]
                poll_result = process.poll()
                if poll_result is None:
                    live_stdout = self._read_output(
                        entry["stdout_path"], "stdout", "get_process", session_id
                    )
                    live_stderr = self._read_output(
                        entry["stderr_path"], "stderr", "get_process", session_id
                    )
                    process_info = psutil.Process(pid)
                    result = {
                        "success": True,
                        "pid": pid,
                        "session_id": session_id,
                        "status": "running",
                        "process_dir": entry.get("process_dir"),
                        "stdout_path": entry["stdout_path"],
                        "stderr_path": entry["stderr_path"],
                        "stdout": live_stdout,
                        "stderr": live_stderr,
                        "cpu_percent": process_info.cpu_percent(),
                        "memory_info": process_info.memory_info()._asdict(),
                    }
                    return self._apply_output_evidence(
                        result,
                        entry["output_limiter"].snapshot(),
                        require_complete=False,
                    )
                if not retained_completed:
                    self.background_processes.pop(pid, None)

            self._close_process_entry(entry)
            try:
                stdout = self._read_output(
                    entry["stdout_path"], "stdout", "get_process", session_id
                )
                stderr = self._read_output(
                    entry["stderr_path"], "stderr", "get_process", session_id
                )
            finally:
                self._remove_process_dir(entry)
            result = {
                "success": True,
                "pid": pid,
                "session_id": session_id,
                "status": "terminated",
                "returncode": poll_result,
                "process_dir": entry.get("process_dir"),
                "stdout": stdout,
                "stderr": stderr,
                "stdout_path": None,
                "stderr_path": None,
                "logs_retained": False,
            }
            return self._apply_output_evidence(result, entry["output_limit"])
        except Exception as e:
            logger.exception("Failed to get process status")
            return {
                "success": False,
                "pid": pid,
                "session_id": session_id,
                "error": str(e),
                "type": type(e).__name__,
            }

    def kill_process(
        self, pid: int, signal: int = 15, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Kill a process

        Args:
            pid: Process ID to kill
            signal: Signal to send (default: SIGTERM=15, use 9 for SIGKILL)

        Returns:
            Dictionary with kill result
        """
        if signal not in ALLOWED_KILL_SIGNALS:
            return {
                "success": False,
                "pid": pid,
                "session_id": session_id,
                "error": f"Signal {signal} not allowed; use one of {sorted(ALLOWED_KILL_SIGNALS)}",
                "type": "ValueError",
            }
        try:
            with self._lock:
                entry = self.background_processes.get(pid)
                retained_completed = False
                if entry is None:
                    entry = self._completed_processes.get(pid)
                    retained_completed = entry is not None
                if entry is None:
                    return {
                        "success": False,
                        "pid": pid,
                        "session_id": session_id,
                        "error": "Process is not managed by pao-tools",
                        "type": "UnmanagedProcessError",
                    }
                self._assert_process_owner(entry, session_id)
                if retained_completed:
                    self._completed_processes.pop(pid, None)
                process = entry["process"]
                if process.poll() is None:
                    self._signal_process_tree(process, signal)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._signal_process_tree(process, KILL_SIGNAL)
                        process.wait(timeout=5)
                released_entry = (
                    entry if retained_completed else self.background_processes.pop(pid)
                )
            if released_entry is not None:
                self._close_process_entry(released_entry)
                try:
                    stdout = self._read_output(
                        released_entry["stdout_path"],
                        "stdout",
                        "kill_process",
                        session_id,
                    )
                    stderr = self._read_output(
                        released_entry["stderr_path"],
                        "stderr",
                        "kill_process",
                        session_id,
                    )
                finally:
                    self._remove_process_dir(released_entry)
            else:
                stdout = ""
                stderr = ""

            result = {
                "success": True,
                "pid": pid,
                "signal": signal,
                "session_id": session_id,
                "status": "killed",
                "stdout": stdout,
                "stderr": stderr,
                "stdout_path": None,
                "stderr_path": None,
                "logs_retained": False,
            }
            if released_entry is not None:
                return self._apply_output_evidence(
                    result, released_entry["output_limit"]
                )
            return result

        except ProcessLookupError:
            return {
                "success": False,
                "pid": pid,
                "session_id": session_id,
                "error": "Process not found",
                "type": "ProcessLookupError",
            }
        except Exception as e:
            logger.exception("Failed to kill process")
            return {
                "success": False,
                "pid": pid,
                "session_id": session_id,
                "error": str(e),
                "type": type(e).__name__,
            }

    def list_processes(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        List all tracked background processes

        Returns:
            Dictionary with process list
        """
        processes = []
        self._reap_terminated()

        with self._lock:
            for pid, entry in list(self.background_processes.items()):
                if entry.get("session_id") != session_id:
                    continue
                process = entry["process"]
                try:
                    proc_info = psutil.Process(pid)
                    processes.append(
                        {
                            "pid": pid,
                            "session_id": session_id,
                            "status": "running",
                            "process_dir": entry.get("process_dir"),
                            "name": proc_info.name(),
                            "cmdline": " ".join(proc_info.cmdline()),
                            "cpu_percent": proc_info.cpu_percent(),
                            "memory_mb": proc_info.memory_info().rss / 1024 / 1024,
                        }
                    )
                except psutil.Error:
                    processes.append(
                        {
                            "pid": pid,
                            "session_id": session_id,
                            "status": "running",
                            "error": "Could not get process info",
                        }
                    )
            active_count = len(self.background_processes)
            completed = [
                {
                    "pid": pid,
                    "session_id": session_id,
                    "status": "terminated",
                    "returncode": entry["process"].poll(),
                    "output_limit": entry.get("output_limit")
                    or entry["output_limiter"].snapshot(),
                }
                for pid, entry in self._completed_processes.items()
                if entry.get("session_id") == session_id
            ]

        return {
            "success": True,
            "session_id": session_id,
            "processes": processes,
            "count": len(processes),
            "completed_processes": completed,
            "completed_count": len(completed),
            "capacity": {
                "max_processes": self.max_processes,
                "available": max(0, self.max_processes - active_count),
                "max_output_bytes_per_process": self.max_output_bytes,
            },
        }

    def close_all(self) -> None:
        """Terminate all managed process trees and close their log handles."""
        with self._lock:
            entries = list(self.background_processes.items())
            entries.extend(self._completed_processes.items())
            self.background_processes.clear()
            self._completed_processes.clear()

        for pid, entry in entries:
            process = entry["process"]
            try:
                if process.poll() is None:
                    self._signal_process_tree(process, signal_module.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._signal_process_tree(process, KILL_SIGNAL)
                        process.wait(timeout=3)
            except Exception:
                logger.exception("Failed to stop managed process %s", pid)
            finally:
                self._finalize_process_entry(entry)

    def close_session_processes(self, session_id: str) -> Dict[str, Any]:
        """Stop and remove every managed helper process owned by one session."""
        with self._lock:
            entries = [
                (pid, entry)
                for pid, entry in self.background_processes.items()
                if entry.get("session_id") == session_id
            ]
            entries.extend(
                (pid, entry)
                for pid, entry in self._completed_processes.items()
                if entry.get("session_id") == session_id
            )

        failures: List[Dict[str, Any]] = []
        closed_count = 0
        for pid, entry in entries:
            process = entry["process"]
            try:
                if process.poll() is None:
                    self._signal_process_tree(process, signal_module.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._signal_process_tree(process, KILL_SIGNAL)
                        process.wait(timeout=3)
            except Exception as exc:
                logger.exception(
                    "Failed to stop managed process %s for session %s",
                    pid,
                    session_id,
                )
                failures.append(
                    {"pid": pid, "type": type(exc).__name__, "error": str(exc)}
                )
                if process.poll() is None:
                    continue
            claimed_entry = self._pop_process_entry(pid, entry)
            if claimed_entry is not None:
                self._finalize_process_entry(claimed_entry)
                closed_count += 1

        return {
            "session_id": session_id,
            "attempted_count": len(entries),
            "closed_count": closed_count,
            "failure_count": len(failures),
            "failures": failures,
        }

    def wait_for_pid_marker(
        self, stdout_path: str, timeout: float = 10.0, poll_interval: float = 0.05
    ) -> Dict[str, Any]:
        """
        Wait for a PID marker of the form "<PID>{1234}</PID>" to appear in a stdout log file.

        Args:
            stdout_path: Path to the stdout log file produced by spawn_process
            timeout: Maximum time to wait in seconds
            poll_interval: How often to poll for new output

        Returns:
            Dictionary with keys:
            - success: whether a PID marker was found
            - pid: the captured PID if found
            - elapsed: time spent waiting
            - bytes_scanned: how many bytes of the file were scanned
            - error: error message if any
        """
        start_time = time.time()
        pattern = re.compile(r"<PID>\{(?P<pid>\d+)\}</PID>")
        bytes_scanned = 0

        try:
            if not os.path.exists(stdout_path):
                return {
                    "success": False,
                    "error": "stdout file not found",
                    "elapsed": 0.0,
                    "bytes_scanned": 0,
                }

            with open(stdout_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read()
                    if chunk:
                        bytes_scanned += len(chunk)
                        match = pattern.search(chunk)
                        if match:
                            elapsed = time.time() - start_time
                            return {
                                "success": True,
                                "pid": int(match.group("pid")),
                                "elapsed": elapsed,
                                "bytes_scanned": bytes_scanned,
                            }

                    if time.time() - start_time >= timeout:
                        return {
                            "success": False,
                            "error": "timeout",
                            "elapsed": timeout,
                            "bytes_scanned": bytes_scanned,
                        }
                    time.sleep(poll_interval)
        except Exception as e:
            logger.error(f"Failed while waiting for PID marker: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed": time.time() - start_time,
                "bytes_scanned": bytes_scanned,
            }
