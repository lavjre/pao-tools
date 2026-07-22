"""Local Ghidra headless decompilation backend."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil

from paomcp.artifacts import ArtifactStore


class GhidraDecompiler:
    """Run isolated local Ghidra projects and deliver pseudo-C losslessly."""

    def __init__(
        self,
        runtime_dir: str,
        artifact_store: ArtifactStore,
        executable: Optional[str] = None,
        script_dir: Optional[str] = None,
        max_concurrent: Optional[int] = None,
    ) -> None:
        self.runtime_dir = os.path.abspath(runtime_dir)
        self.artifact_store = artifact_store
        self.executable = executable or os.environ.get(
            "PAO_GHIDRA_HEADLESS", "/opt/ghidra/support/analyzeHeadless"
        )
        self.script_dir = script_dir or os.path.join(
            os.path.dirname(__file__), "ghidra_scripts"
        )
        configured_concurrency = max_concurrent or int(
            os.environ.get("PAO_GHIDRA_MAX_CONCURRENT", "1")
        )
        if configured_concurrency <= 0:
            raise ValueError("PAO_GHIDRA_MAX_CONCURRENT must be positive")
        self.max_concurrent = configured_concurrency
        self._slots = threading.BoundedSemaphore(configured_concurrency)
        os.makedirs(self.runtime_dir, exist_ok=True)

    def _resolved_executable(self) -> Optional[str]:
        candidate = os.path.expanduser(self.executable)
        if os.path.isabs(candidate) or os.sep in candidate:
            return os.path.abspath(candidate) if os.path.isfile(candidate) else None
        return shutil.which(candidate)

    def status(self) -> Dict[str, Any]:
        """Report whether the pinned local Ghidra backend is usable."""
        executable = self._resolved_executable()
        script_path = os.path.join(self.script_dir, "PaoDecompile.java")
        reasons: List[str] = []
        if executable is None:
            reasons.append(f"Ghidra headless executable not found: {self.executable}")
        if not os.path.isfile(script_path):
            reasons.append(f"Ghidra decompiler script not found: {script_path}")
        return {
            "backend": "ghidra",
            "ready": not reasons,
            "version": os.environ.get("PAO_GHIDRA_VERSION"),
            "executable": executable or self.executable,
            "script_path": script_path,
            "max_concurrent": self.max_concurrent,
            "max_memory": os.environ.get("PAO_GHIDRA_MAXMEM", "2G"),
            "reasons": reasons,
        }

    @staticmethod
    def _group_options() -> Dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name != "nt":
            try:
                getpgid = getattr(os, "getpgid")
                killpg = getattr(os, "killpg")
                kill_signal = int(getattr(signal, "SIGKILL", signal.SIGTERM))
                killpg(getpgid(process.pid), kill_signal)
            except OSError:
                pass
            return

        try:
            parent = psutil.Process(process.pid)
            for child in reversed(parent.children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            parent.kill()
        except psutil.Error:
            pass

    def _run_process(
        self,
        command: List[str],
        *,
        stdout_path: str,
        stderr_path: str,
        timeout: float,
    ) -> Tuple[int, bool, int]:
        started = time.monotonic()
        environment = os.environ.copy()
        environment["GHIDRA_HEADLESS_MAXMEM"] = os.environ.get(
            "PAO_GHIDRA_MAXMEM", "2G"
        )
        with (
            open(stdout_path, "wb") as stdout_file,
            open(stderr_path, "wb") as stderr_file,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.runtime_dir,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
                env=environment,
                **self._group_options(),
            )
            timed_out = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_process_tree(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        assert process.returncode is not None
        return process.returncode, timed_out, elapsed_ms

    def _deliver_file(self, path: str, *, content_type: str, stream: str) -> Any:
        if not os.path.isfile(path):
            return ""
        if os.path.getsize(path) > self.artifact_store.inline_response_bytes:
            artifact = self.artifact_store.save_file(
                path,
                tool="decompile_binary",
                content_type=content_type,
                encoding="utf-8",
            )
            return {**artifact.to_dict(), "stream": stream}
        with open(path, "rb") as source:
            return source.read().decode("utf-8", errors="replace")

    def _read_manifest(self, path: str) -> Dict[str, Any]:
        program: Optional[Dict[str, Any]] = None
        records: List[Dict[str, Any]] = []
        successful = 0
        failed = 0
        with open(path, "r", encoding="utf-8") as manifest_file:
            for line in manifest_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("type") == "program":
                    program = record
                elif record.get("type") == "function":
                    if record.get("status") == "success":
                        successful += 1
                    else:
                        failed += 1
                elif record.get("type") == "error":
                    failed += 1
                records.append(record)

        if os.path.getsize(path) > self.artifact_store.inline_response_bytes:
            artifact = self.artifact_store.save_file(
                path,
                tool="decompile_binary",
                content_type="application/x-ndjson",
                encoding="utf-8",
            )
            delivered: Any = {**artifact.to_dict(), "stream": "manifest"}
        else:
            delivered = records
        return {
            "program": program,
            "manifest": delivered,
            "function_count": successful,
            "failed_function_count": failed,
        }

    def decompile(
        self,
        binary_path: str,
        *,
        function: Optional[str] = None,
        timeout: float = 300.0,
        function_timeout: int = 30,
    ) -> Dict[str, Any]:
        """Decompile a complete binary or one named/addressed function locally."""
        if timeout <= 0 or timeout > 3600:
            raise ValueError("timeout must be greater than 0 and at most 3600 seconds")
        if function_timeout <= 0 or function_timeout > 600:
            raise ValueError(
                "function_timeout must be greater than 0 and at most 600 seconds"
            )
        if not os.path.isfile(binary_path):
            raise FileNotFoundError(f"binary_path not found: '{binary_path}'")

        backend_status = self.status()
        if not backend_status["ready"]:
            raise RuntimeError("; ".join(backend_status["reasons"]))
        executable = str(backend_status["executable"])

        acquired = self._slots.acquire(timeout=min(timeout, 30.0))
        if not acquired:
            raise TimeoutError("Timed out waiting for a local decompiler slot")
        try:
            with tempfile.TemporaryDirectory(
                prefix="ghidra_", dir=self.runtime_dir
            ) as job_dir:
                project_dir = os.path.join(job_dir, "project")
                os.makedirs(project_dir, exist_ok=True)
                output_path = os.path.join(job_dir, "decompiled.c")
                manifest_path = os.path.join(job_dir, "manifest.jsonl")
                stdout_path = os.path.join(job_dir, "ghidra.stdout.log")
                stderr_path = os.path.join(job_dir, "ghidra.stderr.log")
                project_name = f"PaoDecompile_{os.path.basename(job_dir)}"
                command = [
                    executable,
                    project_dir,
                    project_name,
                    "-import",
                    os.path.abspath(binary_path),
                    "-overwrite",
                    "-analysisTimeoutPerFile",
                    str(max(1, int(timeout))),
                    "-scriptPath",
                    self.script_dir,
                    "-postScript",
                    "PaoDecompile.java",
                    output_path,
                    manifest_path,
                    function or "",
                    str(function_timeout),
                    "-deleteProject",
                ]
                returncode, timed_out, elapsed_ms = self._run_process(
                    command,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout=timeout,
                )
                logs = {
                    "stdout": self._deliver_file(
                        stdout_path, content_type="text/plain", stream="stdout"
                    ),
                    "stderr": self._deliver_file(
                        stderr_path, content_type="text/plain", stream="stderr"
                    ),
                }
                common = {
                    "backend": "ghidra",
                    "backend_version": backend_status.get("version"),
                    "binary_path": os.path.abspath(binary_path),
                    "function": function,
                    "elapsed_ms": elapsed_ms,
                    "returncode": returncode,
                    "logs": logs,
                }
                if timed_out:
                    return {
                        "success": False,
                        "code": "DECOMPILER_TIMEOUT",
                        "type": "TimeoutError",
                        "error": f"Local decompilation exceeded {timeout} seconds",
                        **common,
                    }
                if returncode != 0:
                    return {
                        "success": False,
                        "code": "DECOMPILER_FAILED",
                        "type": "RuntimeError",
                        "error": f"Ghidra exited with code {returncode}",
                        **common,
                    }
                if not os.path.isfile(output_path) or not os.path.isfile(manifest_path):
                    return {
                        "success": False,
                        "code": "DECOMPILER_OUTPUT_MISSING",
                        "type": "RuntimeError",
                        "error": "Ghidra completed without producing decompiler output",
                        **common,
                    }

                manifest = self._read_manifest(manifest_path)
                if manifest["function_count"] == 0:
                    return {
                        "success": False,
                        "code": "FUNCTION_NOT_DECOMPILED",
                        "type": "LookupError",
                        "error": "No function was successfully decompiled",
                        **common,
                        **manifest,
                    }
                code = self._deliver_file(
                    output_path, content_type="text/x-c", stream="decompiled_code"
                )
                warnings = []
                if manifest["failed_function_count"]:
                    warnings.append(
                        f"{manifest['failed_function_count']} function(s) failed to decompile"
                    )
                return {
                    "success": True,
                    **common,
                    **manifest,
                    "decompiled_code": code,
                    "warnings": warnings,
                }
        finally:
            self._slots.release()
