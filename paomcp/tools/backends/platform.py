"""Unified platform and runtime capability discovery."""

from __future__ import annotations

import copy
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import psutil

from paomcp import __version__ as MODULE_VERSION
from paomcp.operations import OperationCancelledError

PARSER_VERSION = "1"
PUBLIC_API_VERSION = "1.0"
PAOCLI_VENDORED_VERSION = "1.6-pao"
MAX_PROBE_TIMEOUT_SECONDS = 10.0
MAX_PROBE_TOTAL_TIMEOUT_SECONDS = 60.0
DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS = 15.0
DEFAULT_PROBE_WORKERS = 4
MAX_PROBE_WORKERS = 8
PROBE_OUTPUT_LIMIT_BYTES = 64 * 1024
PROBE_POLL_SECONDS = 0.05
PROBE_CLEANUP_GRACE_SECONDS = 1.0
DEFAULT_PLATFORM_CACHE_ENTRIES = 32


@dataclass(frozen=True)
class ExecutableSpec:
    """Describe one optional executable and its non-interactive version command."""

    candidates: Tuple[str, ...]
    version_args: Tuple[str, ...] = ("--version",)


EXECUTABLE_SPECS: Dict[str, ExecutableSpec] = {
    "gdb": ExecutableSpec(("gdb",)),
    "gdbserver": ExecutableSpec(("gdbserver",)),
    "paocli": ExecutableSpec(("paocli",)),
    "ropper": ExecutableSpec(("ropper",)),
    "ROPgadget": ExecutableSpec(("ROPgadget",)),
    "qemu_user_amd64": ExecutableSpec(("qemu-x86_64",)),
    "qemu_user_i386": ExecutableSpec(("qemu-i386",)),
    "qemu_user_aarch64": ExecutableSpec(("qemu-aarch64",)),
    "qemu_user_arm": ExecutableSpec(("qemu-arm",)),
    "qemu_user_mips": ExecutableSpec(("qemu-mips",)),
    "qemu_user_mipsel": ExecutableSpec(("qemu-mipsel",)),
    "qemu_user_riscv64": ExecutableSpec(("qemu-riscv64",)),
    "qemu_system_amd64": ExecutableSpec(("qemu-system-x86_64",)),
    "qemu_system_aarch64": ExecutableSpec(("qemu-system-aarch64",)),
    "qemu_system_arm": ExecutableSpec(("qemu-system-arm",)),
    "qemu_system_mips": ExecutableSpec(("qemu-system-mips",)),
    "qemu_system_riscv64": ExecutableSpec(("qemu-system-riscv64",)),
}

PACKAGE_NAMES = (
    "pao-tools",
    "fastmcp",
    "pygdbmi",
    "pwntools",
    "psutil",
    "ropper",
    "requests",
)

ARCHITECTURE_ALIASES = {
    "x86_64": "amd64",
    "amd64": "amd64",
    "i386": "i386",
    "i486": "i386",
    "i586": "i386",
    "i686": "i386",
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "armv7l": "arm",
    "armv6l": "arm",
    "mips": "mips",
    "mipsel": "mipsel",
    "riscv64": "riscv64",
}

QEMU_ARCHITECTURES = {
    "qemu_user_amd64": "amd64",
    "qemu_user_i386": "i386",
    "qemu_user_aarch64": "aarch64",
    "qemu_user_arm": "arm",
    "qemu_user_mips": "mips",
    "qemu_user_mipsel": "mipsel",
    "qemu_user_riscv64": "riscv64",
    "qemu_system_amd64": "amd64",
    "qemu_system_aarch64": "aarch64",
    "qemu_system_arm": "arm",
    "qemu_system_mips": "mips",
    "qemu_system_riscv64": "riscv64",
}

WhichFunction = Callable[[str], Optional[str]]
VersionLookup = Callable[[str], Optional[str]]
ModuleLookup = Callable[[str], bool]
EntryPointLookup = Callable[[str], bool]
CancellationCheck = Callable[[], bool]
ProbeRunner = Callable[[Sequence[str], float, CancellationCheck], Dict[str, Any]]
PlatformProvider = Callable[[], Dict[str, Any]]


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _entry_point_available(name: str) -> bool:
    entries = metadata.entry_points()
    if hasattr(entries, "select"):
        selected = entries.select(group="console_scripts", name=name)
        return bool(tuple(selected))
    # Compatibility with older importlib-metadata implementations.
    return any(
        entry.name == name for entry in entries.get("console_scripts", ())  # type: ignore[attr-defined]
    )


def _platform_snapshot() -> Dict[str, Any]:
    machine = platform.machine().lower() or "unknown"
    return {
        "os": {
            "system": platform.system() or "unknown",
            "release": platform.release() or "unknown",
            "version": platform.version() or "unknown",
        },
        "architecture": {
            "machine": machine,
            "normalized": ARCHITECTURE_ALIASES.get(machine, machine),
            "pointer_bits": struct.calcsize("P") * 8,
            "byteorder": sys.byteorder,
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
    }


def _process_group_options() -> Dict[str, Any]:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate a probe and descendants so timeouts cannot leak helpers."""
    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            getpgid = getattr(os, "getpgid")
            killpg = getattr(os, "killpg")
            kill_signal = int(getattr(signal, "SIGKILL", signal.SIGTERM))
            killpg(getpgid(process.pid), kill_signal)
            return
        except OSError:
            pass
    else:
        try:
            parent = psutil.Process(process.pid)
            for child in reversed(parent.children(recursive=True)):
                try:
                    child.kill()
                except psutil.Error:
                    pass
            parent.kill()
            return
        except psutil.Error:
            pass
    try:
        process.kill()
    except OSError:
        pass


def _bounded_text(value: Any, limit: int) -> Tuple[str, int, bool]:
    """Return UTF-8 evidence with explicit byte count and truncation metadata."""
    if isinstance(value, bytes):
        encoded = value
    else:
        encoded = str(value or "").encode("utf-8", errors="replace")
    total = len(encoded)
    clipped = encoded[:limit]
    decoded = clipped.decode("utf-8", errors="replace")
    normalized = decoded.encode("utf-8")
    normalized_truncated = len(normalized) > limit
    if normalized_truncated:
        decoded = normalized[:limit].decode("utf-8", errors="ignore")
    return decoded, total, total > limit or normalized_truncated


def _bound_probe_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Defensively bound evidence from both built-in and injected probe runners."""
    bounded = dict(result)
    for stream in ("stdout", "stderr"):
        text, measured, clipped = _bounded_text(
            bounded.get(stream, ""), PROBE_OUTPUT_LIMIT_BYTES
        )
        total = int(bounded.get(f"{stream}_total_bytes", measured))
        bounded[stream] = text
        bounded[f"{stream}_total_bytes"] = max(total, measured)
        bounded[f"{stream}_truncated"] = bool(
            bounded.get(f"{stream}_truncated", False) or clipped or total > measured
        )
    bounded["output_limit_bytes"] = PROBE_OUTPUT_LIMIT_BYTES
    return bounded


def _run_version_probe(
    command: Sequence[str],
    timeout: float,
    cancellation_requested: CancellationCheck = lambda: False,
) -> Dict[str, Any]:
    """Run one fixed version command with bounded output and tree cleanup."""
    if cancellation_requested():
        return _bound_probe_result(
            {
                "success": False,
                "returncode": None,
                "timed_out": False,
                "cancelled": True,
                "stdout": "",
                "stderr": "",
                "error": "Version probe cancelled before launch",
                "type": "OperationCancelledError",
            }
        )

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["LANG"] = "C"
    try:
        process = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            env=environment,
            **_process_group_options(),
        )
    except OSError as exc:
        return _bound_probe_result(
            {
                "success": False,
                "returncode": None,
                "timed_out": False,
                "cancelled": False,
                "stdout": "",
                "stderr": "",
                "error": str(exc),
                "type": type(exc).__name__,
            }
        )

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    totals = {"stdout": 0, "stderr": 0}

    def drain(stream_name: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(8192)
                if not chunk:
                    break
                totals[stream_name] += len(chunk)
                remaining = PROBE_OUTPUT_LIMIT_BYTES - len(buffers[stream_name])
                if remaining > 0:
                    buffers[stream_name].extend(chunk[:remaining])
        finally:
            stream.close()

    assert process.stdout is not None
    assert process.stderr is not None
    readers = [
        threading.Thread(
            target=drain,
            args=("stdout", process.stdout),
            name="pao-platform-probe-stdout",
            daemon=True,
        ),
        threading.Thread(
            target=drain,
            args=("stderr", process.stderr),
            name="pao-platform-probe-stderr",
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()

    started = time.monotonic()
    timed_out = False
    cancelled = False
    while process.poll() is None:
        if cancellation_requested():
            cancelled = True
            _kill_process_tree(process)
            break
        if time.monotonic() - started >= timeout:
            timed_out = True
            _kill_process_tree(process)
            break
        time.sleep(PROBE_POLL_SECONDS)

    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
    for reader in readers:
        reader.join(timeout=2.0)

    result: Dict[str, Any] = {
        "success": process.returncode == 0 and not timed_out and not cancelled,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "stdout": bytes(buffers["stdout"]),
        "stderr": bytes(buffers["stderr"]),
        "stdout_total_bytes": totals["stdout"],
        "stderr_total_bytes": totals["stderr"],
        "stdout_truncated": totals["stdout"] > PROBE_OUTPUT_LIMIT_BYTES,
        "stderr_truncated": totals["stderr"] > PROBE_OUTPUT_LIMIT_BYTES,
    }
    if timed_out:
        result.update(
            error=f"Version probe exceeded {timeout} seconds",
            type="TimeoutExpired",
        )
    elif cancelled:
        result.update(
            error="Version probe was cancelled",
            type="OperationCancelledError",
        )
    return _bound_probe_result(result)


class PlatformCapabilities:
    """Aggregate runtime capabilities without mutating a debugger session."""

    def __init__(
        self,
        debugger_capabilities: Mapping[str, Any],
        decompiler_status: Mapping[str, Any],
        *,
        which: WhichFunction = shutil.which,
        version_lookup: VersionLookup = _package_version,
        module_lookup: ModuleLookup = _module_available,
        entry_point_lookup: EntryPointLookup = _entry_point_available,
        probe_runner: ProbeRunner = _run_version_probe,
        platform_provider: PlatformProvider = _platform_snapshot,
        environment: Optional[Mapping[str, str]] = None,
        max_probe_workers: Optional[int] = None,
    ) -> None:
        self.debugger_capabilities = copy.deepcopy(dict(debugger_capabilities))
        self.decompiler_status = copy.deepcopy(dict(decompiler_status))
        self.which = which
        self.version_lookup = version_lookup
        self.module_lookup = module_lookup
        self.entry_point_lookup = entry_point_lookup
        self.probe_runner = probe_runner
        self.platform_provider = platform_provider
        self.environment = dict(os.environ if environment is None else environment)
        configured_workers = max_probe_workers
        if configured_workers is None:
            configured_workers = int(
                self.environment.get(
                    "PAO_PLATFORM_PROBE_WORKERS", str(DEFAULT_PROBE_WORKERS)
                )
            )
        if configured_workers <= 0 or configured_workers > MAX_PROBE_WORKERS:
            raise ValueError(
                "Platform probe workers must be greater than zero and at most "
                f"{MAX_PROBE_WORKERS}"
            )
        self.max_probe_workers = configured_workers
        self._cached_report: Optional[Dict[str, Any]] = None
        self._cache_lock = threading.RLock()

    @staticmethod
    def _validate_probe_timeout(timeout: float) -> float:
        value = float(timeout)
        if not math.isfinite(value) or value <= 0 or value > MAX_PROBE_TIMEOUT_SECONDS:
            raise ValueError(
                "probe_timeout must be greater than 0 and at most "
                f"{MAX_PROBE_TIMEOUT_SECONDS:g} seconds"
            )
        return value

    @staticmethod
    def _validate_probe_total_timeout(timeout: float) -> float:
        value = float(timeout)
        if (
            not math.isfinite(value)
            or value <= 0
            or value > MAX_PROBE_TOTAL_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "probe_total_timeout must be greater than 0 and at most "
                f"{MAX_PROBE_TOTAL_TIMEOUT_SECONDS:g} seconds"
            )
        return value

    def _resolve_executables(self) -> Dict[str, Dict[str, Any]]:
        debugger_paths = self.debugger_capabilities.get("executables", {})
        resolved: Dict[str, Dict[str, Any]] = {}
        for name, spec in EXECUTABLE_SPECS.items():
            path: Optional[str] = None
            if isinstance(debugger_paths, dict) and name in debugger_paths:
                candidate = debugger_paths.get(name)
                path = str(candidate) if candidate else None
            if path is None:
                for candidate in spec.candidates:
                    path = self.which(candidate)
                    if path:
                        break
            resolved[name] = {
                "available": path is not None,
                "path": path,
                "candidates": list(spec.candidates),
            }
        return resolved

    def _probe_executables(
        self,
        executables: Dict[str, Dict[str, Any]],
        timeout: float,
        total_timeout: float,
        cancellation_event: Optional[threading.Event],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        available = [
            (name, executable)
            for name, executable in sorted(executables.items())
            if executable.get("path")
        ]
        if not available:
            return {}, {
                "deadline_exceeded": False,
                "cleanup_grace_exceeded": False,
                "cancelled_count": 0,
                "completed_count": 0,
                "worker_count": 0,
                "elapsed_ms": 0,
            }

        deadline_cancel = threading.Event()

        def cancellation_requested() -> bool:
            return deadline_cancel.is_set() or bool(
                cancellation_event is not None and cancellation_event.is_set()
            )

        def run_one(name: str, executable: Mapping[str, Any]) -> Dict[str, Any]:
            path = executable.get("path")
            spec = EXECUTABLE_SPECS[name]
            command = [str(path), *spec.version_args]
            try:
                result = self.probe_runner(command, timeout, cancellation_requested)
            except Exception as exc:
                result = {
                    "success": False,
                    "returncode": None,
                    "timed_out": False,
                    "cancelled": cancellation_requested(),
                    "stdout": "",
                    "stderr": "",
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
            return {"command": command, **_bound_probe_result(result)}

        worker_count = min(self.max_probe_workers, len(available))
        probe_started = time.monotonic()
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="pao-platform-probe",
        )
        futures: Dict[Future[Dict[str, Any]], str] = {
            executor.submit(run_one, name, executable): name
            for name, executable in available
        }
        pending = set(futures)
        while pending:
            if cancellation_event is not None and cancellation_event.is_set():
                break
            remaining = total_timeout - (time.monotonic() - probe_started)
            if remaining <= 0:
                break
            done_now, pending = wait(
                pending,
                timeout=min(PROBE_POLL_SECONDS, remaining),
                return_when=FIRST_COMPLETED,
            )
            if not done_now and remaining <= PROBE_POLL_SECONDS:
                break
        external_cancelled = bool(
            cancellation_event is not None and cancellation_event.is_set()
        )
        deadline_exceeded = bool(pending) and not external_cancelled
        cleanup_grace_exceeded = False
        if pending:
            deadline_cancel.set()
            for future in pending:
                future.cancel()
            _cleaned, pending = wait(
                pending,
                timeout=PROBE_CLEANUP_GRACE_SECONDS,
            )
            cleanup_grace_exceeded = bool(pending)
        executor.shutdown(wait=False, cancel_futures=True)

        evidence: Dict[str, Dict[str, Any]] = {}
        for future, name in futures.items():
            if future.cancelled() or not future.done():
                evidence[name] = _bound_probe_result(
                    {
                        "command": [
                            str(executables[name]["path"]),
                            *EXECUTABLE_SPECS[name].version_args,
                        ],
                        "success": False,
                        "returncode": None,
                        "timed_out": False,
                        "cancelled": True,
                        "stdout": "",
                        "stderr": "",
                        "error": (
                            "Platform probe total deadline exceeded"
                            if deadline_exceeded
                            else "Platform probe was cancelled"
                        ),
                        "type": (
                            "TimeoutError"
                            if deadline_exceeded
                            else "OperationCancelledError"
                        ),
                    }
                )
                continue
            try:
                evidence[name] = future.result()
            except CancelledError:
                continue

        for name, result in evidence.items():
            executable = executables[name]
            raw_text = str(result.get("stdout", "")) or str(result.get("stderr", ""))
            version_line = next(
                (line.strip() for line in raw_text.splitlines() if line.strip()), None
            )
            executable["version"] = version_line
            executable["probe_success"] = bool(result.get("success"))

        if cancellation_event is not None and cancellation_event.is_set():
            raise OperationCancelledError("Platform capability probing was cancelled")

        cancelled_count = sum(
            bool(result.get("cancelled")) for result in evidence.values()
        )
        completed_count = sum(
            not result.get("cancelled") for result in evidence.values()
        )
        return evidence, {
            "deadline_exceeded": deadline_exceeded,
            "cleanup_grace_exceeded": cleanup_grace_exceeded,
            "cancelled_count": cancelled_count,
            "completed_count": completed_count,
            "worker_count": worker_count,
            "elapsed_ms": int((time.monotonic() - probe_started) * 1000),
        }

    def _package_versions(self) -> Dict[str, Dict[str, Any]]:
        packages: Dict[str, Dict[str, Any]] = {}
        for name in PACKAGE_NAMES:
            version = self.version_lookup(name)
            if name == "pao-tools" and version is None:
                version = MODULE_VERSION
            packages[name] = {"installed": version is not None, "version": version}
        return packages

    def _debugger_command_names(self) -> set[str]:
        names: set[str] = set()
        groups = self.debugger_capabilities.get("command_groups", {})
        if isinstance(groups, dict):
            for group in groups.values():
                if not isinstance(group, dict):
                    continue
                available = group.get("available", [])
                if isinstance(available, list):
                    names.update(str(item) for item in available)

        evidence = self.debugger_capabilities.get("evidence", {})
        command_text = (
            evidence.get("commands", "") if isinstance(evidence, dict) else ""
        )
        if isinstance(command_text, str):
            for line in command_text.splitlines():
                candidate = line.strip().split(" ", 1)[0]
                if (
                    candidate
                    and candidate[0].islower()
                    and candidate.replace("-", "").isalnum()
                ):
                    names.add(candidate)
        return names

    @staticmethod
    def _emulated_architectures(
        executables: Mapping[str, Mapping[str, Any]], prefix: str
    ) -> List[str]:
        return sorted(
            {
                QEMU_ARCHITECTURES[name]
                for name, details in executables.items()
                if name.startswith(prefix) and details.get("available")
            }
        )

    def _feature_groups(
        self,
        platform_data: Mapping[str, Any],
        executables: Mapping[str, Mapping[str, Any]],
        paocli: Mapping[str, Any],
    ) -> Dict[str, Any]:
        native = str(platform_data.get("architecture", {}).get("normalized", "unknown"))
        user_arches = self._emulated_architectures(executables, "qemu_user_")
        system_arches = self._emulated_architectures(executables, "qemu_system_")
        commands = self._debugger_command_names()

        allocator_commands = {
            "glibc_ptmalloc": ["arena", "bins", "heap", "malloc-chunk", "tcachebins"],
            "jemalloc": ["jemalloc"],
            "musl_mallocng": ["mallocng"],
        }
        allocators: Dict[str, Any] = {}
        for allocator, expected in allocator_commands.items():
            available = [name for name in expected if name in commands]
            missing = [name for name in expected if name not in commands]
            primitive_status = (
                "ready"
                if available and not missing
                else "partial" if available else "unavailable"
            )
            allocators[allocator] = {
                "runtime_primitives": available,
                "missing_primitives": missing,
                "available": bool(available),
                "typed_api": False,
                "primitive_status": primitive_status,
                "status": "partial" if available else "unavailable",
                "ready": False,
            }

        debugger_ready = bool(self.debugger_capabilities.get("ready"))
        paocli_ready = bool(paocli.get("ready"))
        qemu_user_ready = bool(user_arches)
        qemu_system_ready = bool(system_arches)
        providers = {
            "local_process": {
                "available": True,
                "typed_api": True,
                "status": "ready",
                "ready": True,
            },
            "existing_pid": {
                "available": debugger_ready,
                "typed_api": True,
                "status": "ready" if debugger_ready else "unavailable",
                "ready": debugger_ready,
                "requires": ["debugger"],
            },
            "paocli_driver": {
                "available": paocli_ready,
                "typed_api": True,
                "status": "ready" if paocli_ready else "unavailable",
                "ready": paocli_ready,
                "requires": ["paocli"],
            },
            "remote_tcp": {
                "available": paocli_ready,
                "typed_api": False,
                "status": "partial" if paocli_ready else "unavailable",
                "ready": False,
                "source": "paocli primitive",
            },
            "ssh": {
                "available": False,
                "typed_api": False,
                "status": "unavailable",
                "ready": False,
            },
            "docker": {
                "available": False,
                "typed_api": False,
                "status": "unavailable",
                "ready": False,
            },
            "qemu_user": {
                "available": qemu_user_ready,
                "typed_api": False,
                "status": "partial" if qemu_user_ready else "unavailable",
                "ready": False,
                "architectures": user_arches,
            },
            "qemu_system": {
                "available": qemu_system_ready,
                "typed_api": False,
                "status": "partial" if qemu_system_ready else "unavailable",
                "ready": False,
                "architectures": system_arches,
            },
        }
        return {
            "architectures": {
                "native": native,
                "qemu_user": user_arches,
                "qemu_system": system_arches,
                "schema": "architecture-neutral",
            },
            "providers": providers,
            "allocators": allocators,
            "analysis": {
                "debugger": bool(self.debugger_capabilities.get("ready")),
                "decompiler": bool(self.decompiler_status.get("ready")),
                "rop_helpers": bool(
                    executables.get("ropper", {}).get("available")
                    or executables.get("ROPgadget", {}).get("available")
                ),
            },
        }

    def get_capabilities(
        self,
        *,
        refresh: bool = False,
        probe: bool = False,
        probe_timeout: float = 2.0,
        probe_total_timeout: float = DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Build a deterministic platform manifest, optionally probing tool versions."""
        timeout = self._validate_probe_timeout(probe_timeout)
        total_timeout = self._validate_probe_total_timeout(probe_total_timeout)
        if cancellation_event is not None and cancellation_event.is_set():
            raise OperationCancelledError("Platform capability discovery was cancelled")
        if probe:
            report = self._build_capabilities(
                probe=True,
                probe_timeout=timeout,
                probe_total_timeout=total_timeout,
                cancellation_event=cancellation_event,
            )
            if cancellation_event is not None and cancellation_event.is_set():
                raise OperationCancelledError(
                    "Platform capability discovery was cancelled"
                )
            report["cache"] = {
                "hit": False,
                "stored": False,
                "reason": "active_probe",
            }
            return report

        with self._cache_lock:
            if self._cached_report is not None and not refresh:
                if cancellation_event is not None and cancellation_event.is_set():
                    raise OperationCancelledError(
                        "Platform capability discovery was cancelled"
                    )
                cached = copy.deepcopy(self._cached_report)
                cached["cache"] = {"hit": True, "stored": True, "reason": None}
                return cached
            report = self._build_capabilities(
                probe=False,
                probe_timeout=timeout,
                probe_total_timeout=total_timeout,
                cancellation_event=cancellation_event,
            )
            if cancellation_event is not None and cancellation_event.is_set():
                raise OperationCancelledError(
                    "Platform capability discovery was cancelled"
                )
            report["cache"] = {
                "hit": False,
                "stored": True,
                "reason": "refresh" if refresh else "cache_miss",
            }
            self._cached_report = copy.deepcopy(report)
            return copy.deepcopy(report)

    def _build_capabilities(
        self,
        *,
        probe: bool,
        probe_timeout: float,
        probe_total_timeout: float,
        cancellation_event: Optional[threading.Event],
    ) -> Dict[str, Any]:
        """Build one uncached manifest from this immutable source snapshot."""
        if cancellation_event is not None and cancellation_event.is_set():
            raise OperationCancelledError("Platform capability discovery was cancelled")

        platform_data = self.platform_provider()
        packages = self._package_versions()
        executables = self._resolve_executables()
        probe_evidence: Dict[str, Dict[str, Any]] = {}
        probe_summary = {
            "deadline_exceeded": False,
            "cleanup_grace_exceeded": False,
            "cancelled_count": 0,
            "completed_count": 0,
            "worker_count": 0,
            "elapsed_ms": 0,
        }
        if probe:
            probe_evidence, probe_summary = self._probe_executables(
                executables,
                probe_timeout,
                probe_total_timeout,
                cancellation_event,
            )

        paocli_module = self.module_lookup("paocli")
        paocli_entry = self.entry_point_lookup("paocli")
        paocli_executable = executables["paocli"]
        paocli = {
            "ready": bool(
                paocli_module and (paocli_entry or paocli_executable["available"])
            ),
            "module_available": paocli_module,
            "console_script_registered": paocli_entry,
            "executable": paocli_executable.get("path"),
            "version": self.environment.get(
                "PAO_PAOCLI_VERSION", PAOCLI_VENDORED_VERSION
            ),
        }

        debugger = copy.deepcopy(self.debugger_capabilities)
        debugger_evidence = debugger.pop("evidence", {})
        decompiler = copy.deepcopy(self.decompiler_status)
        readiness = {
            "package": bool(packages["pao-tools"]["installed"]),
            "debugger": bool(debugger.get("ready")),
            "decompiler": bool(decompiler.get("ready")),
            "paocli": bool(paocli["ready"]),
        }

        warnings: List[str] = []
        debugger_warnings = debugger.get("warnings", [])
        if isinstance(debugger_warnings, list):
            warnings.extend(str(item) for item in debugger_warnings)
        if not readiness["debugger"] and not debugger_warnings:
            warnings.append("GDB/pwndbg compatibility is not ready")
        if not readiness["decompiler"]:
            reasons = decompiler.get("reasons", [])
            if isinstance(reasons, list) and reasons:
                warnings.extend(str(item) for item in reasons)
            else:
                warnings.append("Ghidra decompiler is not ready")
        if not readiness["paocli"]:
            warnings.append("paocli module or console entry point is unavailable")
        for name, evidence in probe_evidence.items():
            if not evidence.get("success"):
                warnings.append(f"Version probe failed for {name}")
            truncated_streams = [
                stream
                for stream in ("stdout", "stderr")
                if evidence.get(f"{stream}_truncated")
            ]
            if truncated_streams:
                warnings.append(
                    f"Version probe output was bounded for {name}: "
                    + ", ".join(truncated_streams)
                )
        if probe_summary["deadline_exceeded"]:
            warnings.append(
                "Platform version probes exceeded the total deadline; partial "
                "probe evidence is returned"
            )
        if probe_summary["cleanup_grace_exceeded"]:
            warnings.append(
                "One or more platform probe workers did not stop within the "
                "bounded cleanup grace period"
            )

        report = {
            "success": True,
            "ready": all(readiness.values()),
            "parser_version": PARSER_VERSION,
            "platform": platform_data,
            "packages": packages,
            "runtimes": {
                "debugger": debugger,
                "decompiler": decompiler,
                "paocli": paocli,
            },
            "executables": executables,
            "feature_groups": self._feature_groups(platform_data, executables, paocli),
            "readiness": readiness,
            "versions": {
                "public_api": PUBLIC_API_VERSION,
                "platform_parser": PARSER_VERSION,
                "pwndbg_parser": debugger.get("parser_version"),
                "decompiler_manifest": "1",
            },
            "warnings": list(dict.fromkeys(warnings)),
            "evidence": {
                "debugger": debugger_evidence,
                "executable_probes": probe_evidence,
            },
            "probe": {
                "requested": probe,
                "timeout_seconds": probe_timeout,
                "total_timeout_seconds": probe_total_timeout,
                "probed_count": len(probe_evidence),
                **probe_summary,
                "output_limit_bytes_per_stream": PROBE_OUTPUT_LIMIT_BYTES,
            },
        }
        return report


class PlatformCapabilitiesCache:
    """Bounded per-service cache of immutable platform capability backends."""

    def __init__(self, max_entries: int = DEFAULT_PLATFORM_CACHE_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("Platform capability cache size must be positive")
        self.max_entries = max_entries
        self._entries: "OrderedDict[str, Tuple[str, PlatformCapabilities]]" = (
            OrderedDict()
        )
        self._lock = threading.RLock()

    @staticmethod
    def _fingerprint(
        debugger_capabilities: Mapping[str, Any],
        decompiler_status: Mapping[str, Any],
    ) -> str:
        encoded = json.dumps(
            {
                "debugger": debugger_capabilities,
                "decompiler": decompiler_status,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get(
        self,
        scope: str,
        debugger_capabilities: Mapping[str, Any],
        decompiler_status: Mapping[str, Any],
        *,
        refresh: bool = False,
    ) -> PlatformCapabilities:
        """Reuse one backend while its source manifests remain identical."""
        fingerprint = self._fingerprint(debugger_capabilities, decompiler_status)
        with self._lock:
            cached = self._entries.get(scope)
            if cached is not None and cached[0] == fingerprint and not refresh:
                self._entries.move_to_end(scope)
                return cached[1]
            backend = PlatformCapabilities(
                debugger_capabilities,
                decompiler_status,
            )
            self._entries[scope] = (fingerprint, backend)
            self._entries.move_to_end(scope)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return backend
