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
from typing import Any, Dict, List, Optional

import psutil

from paomcp.artifacts import ArtifactStore
from paomcp.utils.paths import DEFAULT_WORKSPACE

logger = logging.getLogger(__name__)
KILL_SIGNAL = int(getattr(signal_module, "SIGKILL", signal_module.SIGTERM))


class SubprocessTools:
    """Tools for subprocess execution and management"""

    def __init__(
        self,
        process_root: Optional[str] = None,
        workspace_root: str = DEFAULT_WORKSPACE,
        artifact_store: Optional[ArtifactStore] = None,
    ):
        """Initialize subprocess tools."""
        self.background_processes: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.workspace_root = os.path.normpath(workspace_root)
        self.artifact_store = artifact_store
        self.process_root = process_root or os.path.join(
            tempfile.gettempdir(), "pao", "processes"
        )
        os.makedirs(self.process_root, exist_ok=True)

    def _read_output(self, path: str, stream: str, tool: str) -> Any:
        """Return a small stream inline or persist a large stream incrementally."""
        if (
            self.artifact_store is not None
            and os.path.getsize(path) > self.artifact_store.inline_response_bytes
        ):
            metadata = self.artifact_store.save_file(
                path,
                tool=tool,
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

        if self._is_python_executable(executable) and "-c" in cmd_parts:
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

        candidate = self._resolve_candidate_executable(executable, cwd)
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
        logger.info(f"Running command: {command}")
        process_dir: Optional[str] = None
        stdout_file = None
        stderr_file = None

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
            stdout_file = open(stdout_path, "wb")
            stderr_file = open(stderr_path, "wb")

            process = subprocess.Popen(
                self._shell_argv(command),
                cwd=cwd,
                env=cmd_env,
                stdout=stdout_file,
                stderr=stderr_file,
                **self._popen_group_options(),
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_process_tree(process, KILL_SIGNAL)
                process.wait(timeout=5)

                timed_out = True

            stdout_file.close()
            stdout_file = None
            stderr_file.close()
            stderr_file = None
            stdout = self._read_output(stdout_path, "stdout", "run_command")
            stderr = self._read_output(stderr_path, "stderr", "run_command")

            result = {
                "success": process.returncode == 0,
                "command": command,
                "returncode": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": cwd or os.getcwd(),
            }
            if timed_out:
                result["success"] = False
                result["error"] = f"Command timed out after {timeout} seconds"
            return result

        except Exception as e:
            logger.exception("Failed to run command")
            return {
                "success": False,
                "command": command,
                "error": str(e),
                "cwd": cwd or os.getcwd(),
            }
        finally:
            for output_file in (stdout_file, stderr_file):
                if output_file is not None:
                    try:
                        output_file.close()
                    except OSError:
                        pass
            if process_dir is not None:
                shutil.rmtree(process_dir, ignore_errors=True)

    def spawn_process(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
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

            process_dir = tempfile.mkdtemp(prefix="proc_", dir=self.process_root)
            stdout_path = os.path.join(process_dir, "stdout.log")
            stderr_path = os.path.join(process_dir, "stderr.log")
            stdout_file = open(stdout_path, "wb")
            stderr_file = open(stderr_path, "wb")

            # Prepare environment
            cmd_env = os.environ.copy()
            if env:
                cmd_env.update(env)

            # Spawn process with stdout and stderr redirected to files
            process = subprocess.Popen(
                self._shell_argv(command),
                cwd=cwd,
                env=cmd_env,
                stdout=stdout_file,
                stderr=stderr_file,
                **self._popen_group_options(),
            )

            # Store process reference and file paths
            with self._lock:
                self.background_processes[process.pid] = {
                    "process": process,
                    "stdout_file": stdout_file,
                    "stderr_file": stderr_file,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "process_dir": process_dir,
                    "command": command,
                    "cwd": cwd or os.getcwd(),
                }

            # Give process a moment to potentially fail
            import time

            time.sleep(0.1)

            if process.poll() is not None:
                # Process already terminated, read outputs
                stdout_file.flush()
                stderr_file.flush()
                stdout = self._read_output(stdout_path, "stdout", "spawn_process")
                stderr = self._read_output(stderr_path, "stderr", "spawn_process")
                stdout_file.close()
                stderr_file.close()
                with self._lock:
                    self.background_processes.pop(process.pid, None)
                return {
                    "success": False,
                    "command": command,
                    "error": "Process terminated immediately",
                    "returncode": process.returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }

            return {
                "success": True,
                "command": command,
                "pid": process.pid,
                "cwd": cwd or os.getcwd(),
                "status": "running",
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "process_dir": process_dir,
            }

        except Exception as e:
            logger.error(f"Failed to spawn process: {e}")
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
            return {"success": False, "command": command, "error": str(e)}

    def get_process(self, pid: int) -> Dict[str, Any]:
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
            # Check if we're tracking this process
            if pid in self.background_processes:
                entry = self.background_processes[pid]
                process = entry["process"]
                poll_result = process.poll()

                if poll_result is None:
                    # Still running, return status and output paths
                    # Flush current buffers and read accumulated outputs
                    try:
                        entry["stdout_file"].flush()
                    except Exception:
                        pass
                    try:
                        entry["stderr_file"].flush()
                    except Exception:
                        pass
                    try:
                        live_stdout = self._read_output(
                            entry["stdout_path"], "stdout", "get_process"
                        )
                    except Exception:
                        live_stdout = ""
                    try:
                        live_stderr = self._read_output(
                            entry["stderr_path"], "stderr", "get_process"
                        )
                    except Exception:
                        live_stderr = ""
                    return {
                        "success": True,
                        "pid": pid,
                        "status": "running",
                        "process_dir": entry.get("process_dir"),
                        "stdout_path": entry["stdout_path"],
                        "stderr_path": entry["stderr_path"],
                        "stdout": live_stdout,
                        "stderr": live_stderr,
                        "cpu_percent": psutil.Process(pid).cpu_percent(),
                        "memory_info": psutil.Process(pid).memory_info()._asdict(),
                    }
                else:
                    # Process finished, read outputs
                    entry["stdout_file"].close()
                    entry["stderr_file"].close()
                    stdout = self._read_output(
                        entry["stdout_path"], "stdout", "get_process"
                    )
                    stderr = self._read_output(
                        entry["stderr_path"], "stderr", "get_process"
                    )
                    del self.background_processes[pid]
                    return {
                        "success": True,
                        "pid": pid,
                        "status": "terminated",
                        "returncode": poll_result,
                        "process_dir": entry.get("process_dir"),
                        "stdout": stdout,
                        "stderr": stderr,
                        "stdout_path": entry["stdout_path"],
                        "stderr_path": entry["stderr_path"],
                    }
            else:
                return {
                    "success": False,
                    "pid": pid,
                    "error": "Process is not managed by pao-tools",
                    "type": "UnmanagedProcessError",
                }

        except Exception as e:
            logger.error(f"Failed to get process status: {e}")
            return {"success": False, "pid": pid, "error": str(e)}

    def kill_process(self, pid: int, signal: int = 15) -> Dict[str, Any]:
        """
        Kill a process

        Args:
            pid: Process ID to kill
            signal: Signal to send (default: SIGTERM=15, use 9 for SIGKILL)

        Returns:
            Dictionary with kill result
        """
        try:
            with self._lock:
                entry = self.background_processes.get(pid)
            if not entry:
                return {
                    "success": False,
                    "pid": pid,
                    "error": "Process is not managed by pao-tools",
                    "type": "UnmanagedProcessError",
                }

            process = entry["process"]
            if process.poll() is None:
                self._signal_process_tree(process, signal)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._signal_process_tree(process, KILL_SIGNAL)
                    process.wait(timeout=5)

            return {"success": True, "pid": pid, "signal": signal, "status": "killed"}

        except ProcessLookupError:
            return {"success": False, "pid": pid, "error": "Process not found"}
        except Exception as e:
            logger.error(f"Failed to kill process: {e}")
            return {"success": False, "pid": pid, "error": str(e)}

    def list_processes(self) -> Dict[str, Any]:
        """
        List all tracked background processes

        Returns:
            Dictionary with process list
        """
        processes = []

        for pid, entry in list(self.background_processes.items()):
            process = entry["process"]
            poll_result = process.poll()
            if poll_result is None:
                # Still running
                try:
                    proc_info = psutil.Process(pid)
                    processes.append(
                        {
                            "pid": pid,
                            "status": "running",
                            "process_dir": entry.get("process_dir"),
                            "name": proc_info.name(),
                            "cmdline": " ".join(proc_info.cmdline()),
                            "cpu_percent": proc_info.cpu_percent(),
                            "memory_mb": proc_info.memory_info().rss / 1024 / 1024,
                        }
                    )
                except:
                    processes.append(
                        {
                            "pid": pid,
                            "status": "running",
                            "error": "Could not get process info",
                        }
                    )
            else:
                # Process terminated, clean up
                entry["stdout_file"].close()
                entry["stderr_file"].close()
                del self.background_processes[pid]

        return {"success": True, "processes": processes, "count": len(processes)}

    def close_all(self) -> None:
        """Terminate all managed process trees and close their log handles."""
        with self._lock:
            entries = list(self.background_processes.items())
            self.background_processes.clear()

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
                for key in ("stdout_file", "stderr_file"):
                    try:
                        entry[key].close()
                    except Exception:
                        logger.debug("Failed to close %s for pid %s", key, pid)

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
