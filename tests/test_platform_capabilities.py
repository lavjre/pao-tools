from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import psutil
import pytest

from paomcp.operations import OperationCancelledError
from paomcp.tools.backends.platform import (
    PROBE_OUTPUT_LIMIT_BYTES,
    PlatformCapabilities,
    PlatformCapabilitiesCache,
    _bounded_text,
    _run_version_probe,
)
from paomcp.tools.backends.targets import TargetKind
from paomcp.tools.platform import _service_cache

DEBUGGER_READY = {
    "success": True,
    "ready": True,
    "parser_version": "1",
    "versions": {"gdb": "17.1", "pwndbg": "2026.02.18"},
    "command_groups": {
        "core": {"available": ["context", "vmmap"], "missing": []},
        "heap": {
            "available": ["arena", "bins", "heap", "malloc-chunk", "tcachebins"],
            "missing": [],
        },
    },
    "executables": {"gdb": "/tools/gdb"},
    "warnings": [],
    "evidence": {"version": "Gdb: 17.1\nPwndbg: 2026.02.18\n"},
}

DECOMPILER_READY = {
    "backend": "ghidra",
    "ready": True,
    "version": "11.4.2",
    "executable": "/tools/analyzeHeadless",
    "reasons": [],
}

PLATFORM_DATA = {
    "os": {"system": "Linux", "release": "test", "version": "test"},
    "architecture": {
        "machine": "x86_64",
        "normalized": "amd64",
        "pointer_bits": 64,
        "byteorder": "little",
    },
    "python": {
        "version": "3.12.9",
        "implementation": "CPython",
        "executable": "/usr/bin/python3",
    },
}


def _which(name: str):
    paths = {
        "paocli": "/tools/paocli",
        "ropper": "/tools/ropper",
        "ROPgadget": "/tools/ROPgadget",
        "qemu-aarch64": "/tools/qemu-aarch64",
        "qemu-system-x86_64": "/tools/qemu-system-x86_64",
    }
    return paths.get(name)


def _version(name: str):
    versions = {
        "pao-tools": "0.3.0",
        "fastmcp": "3.1.0",
        "pygdbmi": "0.11.0.0",
        "pwntools": "4.14.1",
        "psutil": "7.0.0",
        "ropper": "1.13.13",
        "requests": "2.32.0",
    }
    return versions.get(name)


def _backend(**overrides):
    arguments = {
        "debugger_capabilities": DEBUGGER_READY,
        "decompiler_status": DECOMPILER_READY,
        "which": _which,
        "version_lookup": _version,
        "module_lookup": lambda name: name == "paocli",
        "entry_point_lookup": lambda name: name == "paocli",
        "platform_provider": lambda: PLATFORM_DATA,
        "environment": {},
    }
    arguments.update(overrides)
    return PlatformCapabilities(**arguments)


def test_platform_manifest_aggregates_runtime_and_feature_groups():
    report = _backend().get_capabilities()

    assert report["success"] is True
    assert report["ready"] is True
    assert report["platform"]["architecture"]["normalized"] == "amd64"
    assert report["packages"]["pao-tools"]["version"] == "0.3.0"
    assert report["runtimes"]["debugger"]["versions"]["pwndbg"] == "2026.02.18"
    assert report["runtimes"]["decompiler"]["version"] == "11.4.2"
    assert report["runtimes"]["paocli"]["version"] == "1.6-pao"
    assert report["feature_groups"]["architectures"]["qemu_user"] == ["aarch64"]
    providers = report["feature_groups"]["providers"]
    canonical_kinds = {kind.value for kind in TargetKind}
    assert canonical_kinds <= set(providers)
    assert report["feature_groups"]["target_provider_kinds"] == sorted(canonical_kinds)
    assert all(providers[kind]["typed_api"] is True for kind in canonical_kinds)
    tcp_remote = providers["tcp_remote"]
    assert tcp_remote["available"] is True
    assert tcp_remote["status"] == "ready"
    assert tcp_remote["ready"] is True
    assert providers["remote_tcp"]["source"] == "paocli primitive"
    assert providers["remote_tcp"]["typed_api"] is False
    assert providers["remote_tcp"]["status"] == "partial"
    assert providers["remote_tcp"]["deprecated"] is True
    assert providers["existing_pid"]["alias_for"] == "attach_pid"
    assert providers["existing_pid"]["deprecated"] is True
    assert providers["paocli_driver"]["category"] == "exploit_driver"
    assert providers["paocli_driver"]["ready"] is True
    assert providers["gdbserver"]["status"] == "unavailable"
    assert providers["ssh"]["status"] == "unavailable"
    assert providers["docker"]["status"] == "unavailable"
    ptmalloc = report["feature_groups"]["allocators"]["glibc_ptmalloc"]
    assert ptmalloc["available"] is True
    assert ptmalloc["typed_api"] is True
    assert ptmalloc["primitive_status"] == "ready"
    assert ptmalloc["status"] == "ready"
    assert ptmalloc["ready"] is True
    assert report["evidence"]["debugger"]["version"].startswith("Gdb:")
    assert report["probe"]["probed_count"] == 0


def test_probe_runs_only_resolved_executables_and_keeps_raw_evidence():
    calls = []

    def probe(command, timeout, cancellation_requested):
        calls.append((command, timeout))
        assert cancellation_requested() is False
        return {
            "success": True,
            "returncode": 0,
            "timed_out": False,
            "stdout": f"{Path(command[0]).name} version 1\n",
            "stderr": "",
        }

    report = _backend(probe_runner=probe).get_capabilities(
        probe=True, probe_timeout=0.5
    )

    available_count = sum(
        1 for details in report["executables"].values() if details["available"]
    )
    assert len(calls) == available_count
    assert all(timeout == 0.5 for _, timeout in calls)
    assert report["probe"]["probed_count"] == available_count
    assert report["probe"]["worker_count"] == min(4, available_count)
    assert report["probe"]["deadline_exceeded"] is False
    assert report["executables"]["paocli"]["version"] == "paocli version 1"
    paocli_evidence = report["evidence"]["executable_probes"]["paocli"]
    assert paocli_evidence["command"] == ["/tools/paocli", "--version"]
    assert paocli_evidence["stdout"] == "paocli version 1\n"


@pytest.mark.parametrize("timeout", [0, -1, 10.01])
def test_probe_timeout_is_bounded(timeout):
    with pytest.raises(ValueError, match="probe_timeout"):
        _backend().get_capabilities(probe_timeout=timeout)


@pytest.mark.parametrize("timeout", [0, -1, 60.01, float("nan")])
def test_probe_total_timeout_is_bounded(timeout):
    with pytest.raises(ValueError, match="probe_total_timeout"):
        _backend().get_capabilities(probe_total_timeout=timeout)


def test_refresh_rebuilds_cached_platform_snapshot():
    snapshots = []

    def platform_provider():
        value = dict(PLATFORM_DATA)
        value["sequence"] = len(snapshots) + 1
        snapshots.append(value)
        return value

    backend = _backend(platform_provider=platform_provider)

    first = backend.get_capabilities()
    cached = backend.get_capabilities()
    refreshed = backend.get_capabilities(refresh=True)

    assert first["platform"]["sequence"] == 1
    assert first["cache"] == {
        "hit": False,
        "stored": True,
        "reason": "cache_miss",
    }
    assert cached["platform"]["sequence"] == 1
    assert cached["cache"] == {"hit": True, "stored": True, "reason": None}
    assert refreshed["platform"]["sequence"] == 2
    assert refreshed["cache"] == {
        "hit": False,
        "stored": True,
        "reason": "refresh",
    }
    assert len(snapshots) == 2


def test_degraded_components_return_actionable_warnings():
    debugger = {
        **DEBUGGER_READY,
        "ready": False,
        "warnings": ["Expected pwndbg 2026.02.18, found unknown"],
    }
    decompiler = {
        **DECOMPILER_READY,
        "ready": False,
        "reasons": ["Ghidra headless executable not found"],
    }
    backend = _backend(
        debugger_capabilities=debugger,
        decompiler_status=decompiler,
        module_lookup=lambda _name: False,
        entry_point_lookup=lambda _name: False,
        which=lambda _name: None,
    )

    report = backend.get_capabilities()

    assert report["success"] is True
    assert report["ready"] is False
    assert report["readiness"] == {
        "package": True,
        "debugger": False,
        "decompiler": False,
        "paocli": False,
    }
    assert "Expected pwndbg 2026.02.18, found unknown" in report["warnings"]
    assert "Ghidra headless executable not found" in report["warnings"]
    assert "paocli module or console entry point is unavailable" in report["warnings"]
    attach_pid = report["feature_groups"]["providers"]["attach_pid"]
    assert attach_pid["typed_api"] is True
    assert attach_pid["available"] is False
    assert attach_pid["status"] == "unavailable"
    assert attach_pid["ready"] is False


def test_allocator_status_distinguishes_partial_primitives_from_typed_readiness():
    debugger = {
        **DEBUGGER_READY,
        "command_groups": {
            "core": {"available": ["context", "vmmap"], "missing": []},
            "heap": {"available": ["arena"], "missing": ["bins"]},
        },
    }

    report = _backend(debugger_capabilities=debugger).get_capabilities()

    ptmalloc = report["feature_groups"]["allocators"]["glibc_ptmalloc"]
    assert ptmalloc["available"] is True
    assert ptmalloc["typed_api"] is True
    assert ptmalloc["primitive_status"] == "partial"
    assert ptmalloc["status"] == "ready"
    assert ptmalloc["ready"] is True


def test_provider_runtime_dependencies_are_distinct_from_typed_registration():
    paths = {
        "docker": "/tools/docker",
        "gdbserver": "/tools/gdbserver",
    }

    def which(name: str):
        return paths.get(name) or _which(name)

    report = _backend(
        which=which,
        module_lookup=lambda name: name in {"paocli", "paramiko"},
    ).get_capabilities()

    providers = report["feature_groups"]["providers"]
    canonical_kinds = {kind.value for kind in TargetKind}
    assert canonical_kinds <= set(providers)
    assert all(providers[kind]["typed_api"] is True for kind in canonical_kinds)
    assert all(providers[kind]["ready"] is True for kind in canonical_kinds)
    assert providers["gdbserver"]["requires"] == ["gdbserver", "debugger"]
    assert providers["docker"]["requires"] == ["docker"]
    assert providers["ssh"]["requires"] == ["paramiko"]


def test_probe_set_runs_in_parallel_with_a_bounded_worker_pool():
    state = {"running": 0, "peak": 0}
    lock = threading.Lock()

    def probe(command, timeout, cancellation_requested):
        with lock:
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        try:
            time.sleep(0.05)
            return {
                "success": not cancellation_requested(),
                "returncode": 0,
                "timed_out": False,
                "cancelled": cancellation_requested(),
                "stdout": f"{Path(command[0]).name} 1\n",
                "stderr": "",
            }
        finally:
            with lock:
                state["running"] -= 1

    report = _backend(probe_runner=probe, max_probe_workers=3).get_capabilities(
        probe=True,
        probe_timeout=1.0,
        probe_total_timeout=2.0,
    )

    assert state["peak"] == 3
    assert report["probe"]["worker_count"] == 3
    assert report["probe"]["deadline_exceeded"] is False


def test_probe_total_deadline_returns_explicit_partial_evidence():
    def probe(command, timeout, cancellation_requested):
        while not cancellation_requested():
            time.sleep(0.005)
        return {
            "success": False,
            "returncode": None,
            "timed_out": False,
            "cancelled": True,
            "stdout": "",
            "stderr": "",
            "error": "cancelled",
        }

    started = time.monotonic()
    report = _backend(probe_runner=probe, max_probe_workers=2).get_capabilities(
        probe=True,
        probe_timeout=1.0,
        probe_total_timeout=0.05,
    )

    assert time.monotonic() - started < 1.0
    assert report["probe"]["deadline_exceeded"] is True
    assert report["probe"]["cancelled_count"] > 0
    assert any("total deadline" in warning for warning in report["warnings"])


def test_default_probe_bounds_output_without_silent_truncation():
    result = _run_version_probe(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('x' * 70000)",
        ],
        2.0,
    )

    assert result["success"] is True
    assert len(result["stdout"].encode("utf-8")) == PROBE_OUTPUT_LIMIT_BYTES
    assert result["stdout_total_bytes"] == 70000
    assert result["stdout_truncated"] is True
    assert result["output_limit_bytes"] == PROBE_OUTPUT_LIMIT_BYTES


def test_probe_text_bound_survives_invalid_utf8_expansion():
    text, total, truncated = _bounded_text(
        b"\xff" * PROBE_OUTPUT_LIMIT_BYTES,
        PROBE_OUTPUT_LIMIT_BYTES,
    )

    assert len(text.encode("utf-8")) <= PROBE_OUTPUT_LIMIT_BYTES
    assert total == PROBE_OUTPUT_LIMIT_BYTES
    assert truncated is True


def test_default_probe_honors_cooperative_cancellation():
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        result = _run_version_probe(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            5.0,
            cancelled.is_set,
        )
    finally:
        timer.cancel()

    assert time.monotonic() - started < 2.0
    assert result["success"] is False
    assert result["cancelled"] is True
    assert result["type"] == "OperationCancelledError"


def test_default_probe_cancellation_cleans_up_descendants():
    cancelled = threading.Event()
    timer = threading.Timer(0.2, cancelled.set)
    timer.start()
    try:
        result = _run_version_probe(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess,sys,time; "
                    "child=subprocess.Popen([sys.executable,'-c',"
                    "'import time; time.sleep(30)']); "
                    "print(child.pid,flush=True); time.sleep(30)"
                ),
            ],
            5.0,
            cancelled.is_set,
        )
    finally:
        timer.cancel()

    child_pid = int(result["stdout"].strip())
    deadline = time.monotonic() + 1.0
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)

    assert result["cancelled"] is True
    assert not psutil.pid_exists(child_pid)


def test_platform_discovery_rejects_an_already_cancelled_operation():
    cancelled = threading.Event()
    cancelled.set()

    with pytest.raises(OperationCancelledError):
        _backend().get_capabilities(cancellation_event=cancelled)


def test_platform_backend_cache_reuses_only_identical_source_snapshots():
    cache = PlatformCapabilitiesCache(max_entries=2)

    first = cache.get("startup", DEBUGGER_READY, DECOMPILER_READY)
    second = cache.get("startup", DEBUGGER_READY, DECOMPILER_READY)
    changed = cache.get(
        "startup",
        DEBUGGER_READY,
        {**DECOMPILER_READY, "version": "12.0"},
    )
    refreshed = cache.get(
        "startup",
        DEBUGGER_READY,
        {**DECOMPILER_READY, "version": "12.0"},
        refresh=True,
    )

    assert second is first
    assert changed is not first
    assert refreshed is not changed


def test_platform_cache_is_attached_to_the_service_lifetime():
    first_services = SimpleNamespace()
    second_services = SimpleNamespace()

    first = _service_cache(first_services)

    assert _service_cache(first_services) is first
    assert _service_cache(second_services) is not first


def test_active_probe_does_not_pollute_cached_non_probe_manifest():
    backend = _backend(
        probe_runner=lambda command, timeout, cancellation_requested: {
            "success": True,
            "returncode": 0,
            "timed_out": False,
            "cancelled": False,
            "stdout": "version 1\n",
            "stderr": "",
        }
    )

    baseline = backend.get_capabilities()
    probed = backend.get_capabilities(probe=True)
    cached = backend.get_capabilities()

    assert baseline["probe"]["requested"] is False
    assert probed["probe"]["requested"] is True
    assert probed["cache"] == {
        "hit": False,
        "stored": False,
        "reason": "active_probe",
    }
    assert cached["probe"]["requested"] is False
    assert cached["cache"]["hit"] is True
    assert cached["evidence"]["executable_probes"] == {}
