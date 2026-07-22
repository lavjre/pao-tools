"""Local Ghidra headless decompilation backend."""

from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
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
from paomcp.operations import OperationRegistry, current_operation_id


class GhidraDecompiler:
    """Run isolated local Ghidra projects and deliver pseudo-C losslessly."""

    def __init__(
        self,
        runtime_dir: str,
        artifact_store: ArtifactStore,
        executable: Optional[str] = None,
        script_dir: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        operation_registry: Optional[OperationRegistry] = None,
        max_cache_entries: Optional[int] = None,
    ) -> None:
        self.runtime_dir = os.path.abspath(runtime_dir)
        self.artifact_store = artifact_store
        self.operation_registry = operation_registry
        self.model_schema_version = "1"
        self.max_cache_entries = (
            int(os.environ.get("PAO_GHIDRA_CACHE_ENTRIES", "16"))
            if max_cache_entries is None
            else max_cache_entries
        )
        if self.max_cache_entries <= 0:
            raise ValueError("max_cache_entries must be positive")
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_lock = threading.RLock()
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
            "model_schema_version": self.model_schema_version,
            "cache_entries": len(self._cache),
            "max_cache_entries": self.max_cache_entries,
            "reasons": reasons,
        }

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _fingerprint_file(cls, path: str) -> Dict[str, Any]:
        """Hash one stable filesystem object and reject concurrent replacement."""
        before = os.stat(path)
        sha256 = cls._sha256_file(path)
        after = os.stat(path)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise RuntimeError("Binary changed while it was being fingerprinted")
        return {
            "sha256": sha256,
            "size_bytes": str(after.st_size),
            "modified_ns": str(after.st_mtime_ns),
        }

    def _cache_key(
        self,
        binary_sha256: str,
        function: Optional[str],
        function_timeout: int,
        backend_version: Optional[str],
    ) -> str:
        script_path = os.path.join(self.script_dir, "PaoDecompile.java")
        script_sha256 = self._sha256_file(script_path)
        material = json.dumps(
            {
                "binary_sha256": binary_sha256,
                "function": function,
                "function_timeout": function_timeout,
                "backend_version": backend_version,
                "script_sha256": script_sha256,
                "model_schema_version": self.model_schema_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def _artifacts_available(self, value: Any) -> bool:
        if isinstance(value, dict):
            artifact_id = value.get("artifact_id")
            if value.get("delivery") == "artifact" and artifact_id:
                try:
                    self.artifact_store.get(str(artifact_id))
                except (FileNotFoundError, ValueError):
                    return False
            return all(self._artifacts_available(item) for item in value.values())
        if isinstance(value, list):
            return all(self._artifacts_available(item) for item in value)
        return True

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            if not self._artifacts_available(cached):
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            result = copy.deepcopy(cached)
        result["cache"] = {
            "hit": True,
            "key": key,
            "max_entries": self.max_cache_entries,
        }
        return result

    def _cache_put(self, key: str, value: Dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = copy.deepcopy(value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_entries:
                self._cache.popitem(last=False)

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
            operation_id = current_operation_id()
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.set_cancel_handler(
                    operation_id, lambda: self._kill_process_tree(process)
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
            finally:
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.set_cancel_handler(operation_id, None)
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

    def _write_large_model(
        self, manifest_path: str, model_path: str, program: Optional[Dict[str, Any]]
    ) -> None:
        """Convert NDJSON to the public model JSON using bounded working memory."""
        with open(model_path, "w", encoding="utf-8") as model_file:
            model_file.write(
                '{"schema_version":'
                + json.dumps(self.model_schema_version)
                + ',"program":'
                + json.dumps(program, ensure_ascii=True, separators=(",", ":"))
                + ',"functions":['
            )
            first = True
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                for line in manifest_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("type") != "function":
                        continue
                    if not first:
                        model_file.write(",")
                    model_file.write(
                        json.dumps(record, ensure_ascii=True, separators=(",", ":"))
                    )
                    first = False
            model_file.write('],"call_graph":[')
            first = True
            with open(manifest_path, "r", encoding="utf-8") as manifest_file:
                for line in manifest_file:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if (
                        record.get("type") != "function"
                        or record.get("status") != "success"
                    ):
                        continue
                    edge = {
                        "source": record.get("entry"),
                        "source_name": record.get("name"),
                        "targets": record.get("callees", []),
                    }
                    if not first:
                        model_file.write(",")
                    model_file.write(
                        json.dumps(edge, ensure_ascii=True, separators=(",", ":"))
                    )
                    first = False
            model_file.write("]}")

    def _read_manifest(self, path: str) -> Dict[str, Any]:
        program: Optional[Dict[str, Any]] = None
        is_large = os.path.getsize(path) > self.artifact_store.inline_response_bytes
        records: List[Dict[str, Any]] = []
        functions: List[Dict[str, Any]] = []
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
                    if not is_large:
                        functions.append(record)
                    if record.get("status") == "success":
                        successful += 1
                    else:
                        failed += 1
                elif record.get("type") == "error":
                    failed += 1
                if not is_large:
                    records.append(record)

        if is_large:
            artifact = self.artifact_store.save_file(
                path,
                tool="decompile_binary",
                content_type="application/x-ndjson",
                encoding="utf-8",
            )
            delivered: Any = {**artifact.to_dict(), "stream": "manifest"}
            model_path = f"{path}.model.json"
            try:
                self._write_large_model(path, model_path, program)
                model_artifact = self.artifact_store.save_file(
                    model_path,
                    tool="decompile_binary",
                    content_type="application/json",
                    encoding="utf-8",
                )
            finally:
                try:
                    os.unlink(model_path)
                except FileNotFoundError:
                    pass
            model: Any = {
                **model_artifact.to_dict(),
                "stream": "structured_model",
                "schema_version": self.model_schema_version,
            }
        else:
            delivered = records
            model = {
                "schema_version": self.model_schema_version,
                "program": program,
                "functions": functions,
                "call_graph": [
                    {
                        "source": item.get("entry"),
                        "source_name": item.get("name"),
                        "targets": item.get("callees", []),
                    }
                    for item in functions
                    if item.get("status") == "success"
                ],
            }
        return {
            "program": program,
            "model": model,
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
        refresh: bool = False,
    ) -> Dict[str, Any]:
        """Decompile a complete binary or one named/addressed function locally."""
        operation_id = current_operation_id()
        if self.operation_registry is not None and operation_id is not None:
            self.operation_registry.update_progress(
                operation_id, current=0, total=4, message="waiting for Ghidra slot"
            )
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
        binary_fingerprint = self._fingerprint_file(binary_path)
        binary_sha256 = str(binary_fingerprint["sha256"])
        cache_key = self._cache_key(
            binary_sha256,
            function,
            function_timeout,
            backend_status.get("version"),
        )
        if not refresh:
            cached = self._cache_get(cache_key)
            if cached is not None:
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.update_progress(
                        operation_id, current=4, total=4, message="cache hit"
                    )
                return cached

        wait_deadline = time.monotonic() + min(timeout, 30.0)
        acquired = False
        while not acquired:
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.raise_if_cancelled(operation_id)
            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                break
            acquired = self._slots.acquire(timeout=min(0.25, remaining))
        if not acquired:
            raise TimeoutError("Timed out waiting for a local decompiler slot")
        try:
            if self.operation_registry is not None and operation_id is not None:
                self.operation_registry.update_progress(
                    operation_id, current=1, total=4, message="running Ghidra analysis"
                )
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
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.update_progress(
                        operation_id,
                        current=2,
                        total=4,
                        message="collecting Ghidra output",
                    )
                operation_id = current_operation_id()
                cancellation_event = (
                    self.operation_registry.cancellation_event(operation_id)
                    if self.operation_registry is not None and operation_id is not None
                    else None
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
                    "model_schema_version": self.model_schema_version,
                    "binary_path": os.path.abspath(binary_path),
                    "fingerprint": {
                        "sha256": binary_sha256,
                        "size_bytes": binary_fingerprint["size_bytes"],
                        "modified_ns": binary_fingerprint["modified_ns"],
                    },
                    "function": function,
                    "elapsed_ms": elapsed_ms,
                    "returncode": returncode,
                    "logs": logs,
                    "cache": {
                        "hit": False,
                        "refreshed": refresh,
                        "key": cache_key,
                        "max_entries": self.max_cache_entries,
                    },
                }
                if cancellation_event is not None and cancellation_event.is_set():
                    return {
                        "success": False,
                        "code": "OPERATION_CANCELLED",
                        "type": "OperationCancelledError",
                        "error": "Local decompilation was cancelled",
                        **common,
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

                final_fingerprint = self._fingerprint_file(binary_path)
                if final_fingerprint["sha256"] != binary_sha256:
                    return {
                        "success": False,
                        "code": "BINARY_CHANGED_DURING_ANALYSIS",
                        "type": "RuntimeError",
                        "error": "Binary changed while Ghidra analysis was running",
                        **common,
                    }

                manifest = self._read_manifest(manifest_path)
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.update_progress(
                        operation_id,
                        current=3,
                        total=4,
                        message="delivering decompiler artifacts",
                    )
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
                if self.operation_registry is not None and operation_id is not None:
                    self.operation_registry.update_progress(
                        operation_id, current=4, total=4, message="complete"
                    )
                result = {
                    "success": True,
                    **common,
                    **manifest,
                    "decompiled_code": code,
                    "warnings": warnings,
                }
                self._cache_put(cache_key, result)
                return result
        finally:
            self._slots.release()
