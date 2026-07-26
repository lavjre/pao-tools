"""Tests for Stage 8.1 typed target-provider foundation."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from paomcp.tools.backends.targets import (
    DEFAULT_MAX_TARGETS_PER_SESSION,
    TARGET_CREATE_SCHEMA,
    TARGET_LIST_SCHEMA,
    AttachPidProvider,
    LocalProcessProvider,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
)


class FakeGdb:
    """Minimal GdbController double for attach_debugger tests."""

    def __init__(self, *, attach_success: bool = True) -> None:
        self._attach_success = attach_success
        self.attach_calls: List[int] = []

    def attach(self, pid: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        self.attach_calls.append(pid)
        if self._attach_success:
            return (
                {"success": True, "pid": pid, "state": "stopped"},
                [{"kind": "backtrace"}, {"kind": "heap"}],
            )
        return (
            {"success": False, "error": "attach denied", "type": "PermissionError"},
            [],
        )


# ---------------------------------------------------------------------------
# TargetSpec.from_config
# ---------------------------------------------------------------------------
class TestSpecFromConfig:
    def test_local_process_basic(self) -> None:
        spec = TargetSpec.from_config(
            "local_process",
            {"binary_path": "/bin/ls", "argv": ["-la"]},
            session_id="sess",
        )
        assert spec.kind == TargetKind.LOCAL_PROCESS
        assert spec.binary_path == "/bin/ls"
        assert spec.argv == ["-la"]
        assert spec.session_id == "sess"

    def test_local_process_missing_binary_raises(self) -> None:
        with pytest.raises(TargetError, match="binary_path"):
            TargetSpec.from_config("local_process", {})

    def test_attach_pid_requires_positive_pid(self) -> None:
        spec = TargetSpec.from_config("attach_pid", {"pid": 4242})
        assert spec.pid == 4242
        with pytest.raises(TargetError, match="positive 'pid'"):
            TargetSpec.from_config("attach_pid", {"pid": 0})
        with pytest.raises(TargetError, match="positive 'pid'"):
            TargetSpec.from_config("attach_pid", {})

    def test_argv_from_string_is_shlex_split(self) -> None:
        spec = TargetSpec.from_config(
            "local_process",
            {"binary_path": "/bin/echo", "argv": "hello world 'x y'"},
        )
        assert spec.argv == ["hello", "world", "x y"]

    def test_env_must_be_mapping(self) -> None:
        with pytest.raises(TargetError, match="env must be a mapping"):
            TargetSpec.from_config(
                "local_process",
                {"binary_path": "/bin/ls", "env": ["not", "a", "dict"]},
            )

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(TargetError, match="unknown target kind"):
            TargetSpec.from_config("teleporter", {})

    def test_qemu_system_is_explicitly_unsupported_here(self) -> None:
        with pytest.raises(TargetError, match="qemu_system.*not implemented"):
            TargetSpec.from_config("qemu_system", {"binary_path": "/kernel"})


# ---------------------------------------------------------------------------
# TargetRegistry basics
# ---------------------------------------------------------------------------
class TestRegistryLifecycle:
    def test_create_returns_typed_handle(self) -> None:
        registry = TargetRegistry()
        response = registry.create_target_response(
            kind="attach_pid",
            config={"pid": os.getpid()},
            session_id="sess",
        )
        parsed = response["parsed"]
        assert parsed["schema"] == TARGET_CREATE_SCHEMA
        target = parsed["target"]
        assert target["kind"] == "attach_pid"
        assert target["state"] == "created"
        assert target["session_id"] == "sess"
        assert target["target_id"]

    def test_list_filters_by_session(self) -> None:
        registry = TargetRegistry()
        registry.create_target_response(
            kind="attach_pid",
            config={"pid": os.getpid()},
            session_id="alpha",
        )
        registry.create_target_response(
            kind="attach_pid",
            config={"pid": os.getpid()},
            session_id="beta",
        )
        alpha = registry.list_targets_response(session_id="alpha")["parsed"]
        assert alpha["count"] == 1
        assert alpha["schema"] == TARGET_LIST_SCHEMA
        both = registry.list_targets_response(session_id=None)["parsed"]
        assert both["count"] == 2

    def test_per_session_cap_enforced(self) -> None:
        registry = TargetRegistry(max_targets_per_session=2)
        for _ in range(2):
            registry.create_target_response(
                kind="attach_pid",
                config={"pid": os.getpid()},
                session_id="sess",
            )
        with pytest.raises(TargetError, match="cap"):
            registry.create_target_response(
                kind="attach_pid",
                config={"pid": os.getpid()},
                session_id="sess",
            )

    def test_get_status_reports_dead_pid(self) -> None:
        registry = TargetRegistry()
        response = registry.create_target_response(
            kind="attach_pid",
            config={"pid": 1_000_000_007},  # very unlikely to be alive
            session_id="sess",
        )
        target_id = response["parsed"]["target"]["target_id"]
        # Cannot start because the pid is not alive.
        start = registry.start_target_response(target_id=target_id)
        assert start["parsed"]["success"] is False
        assert "not alive" in start["parsed"]["error"]
        # Status reflects the error state.
        status = registry.get_status_response(target_id=target_id)
        assert status["parsed"]["target"]["state"] == "error"

    def test_get_capabilities_shape(self) -> None:
        registry = TargetRegistry()
        response = registry.create_target_response(
            kind="local_process",
            config={"binary_path": "/bin/echo"},
            session_id="sess",
        )
        target_id = response["parsed"]["target"]["target_id"]
        caps = registry.get_capabilities_response(target_id=target_id)["parsed"]
        assert caps["kind"] == "local_process"
        for key in (
            "start",
            "stop",
            "attach_debugger",
            "file_sync",
            "reconnect",
            "port_forward",
        ):
            assert key in caps["capabilities"]

    def test_unknown_target_id_raises(self) -> None:
        registry = TargetRegistry()
        with pytest.raises(TargetError, match="not found"):
            registry.get_status_response(target_id="deadbeef")


# ---------------------------------------------------------------------------
# LocalProcessProvider (uses the current Python interpreter as the fixture)
# ---------------------------------------------------------------------------
class TestLocalProcessProvider:
    def _spec(self, argv: List[str]) -> TargetSpec:
        return TargetSpec.from_config(
            "local_process",
            {"binary_path": sys.executable, "argv": argv},
            session_id="sess",
        )

    def test_start_stops_process(self) -> None:
        registry = TargetRegistry()
        spec = self._spec(["-c", "import time; time.sleep(30)"])
        handle = registry.create(spec)
        registry.start_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.RUNNING
        assert handle.pid is not None
        stop = registry.stop_target_response(
            target_id=handle.target_id, timeout_sec=5.0
        )
        assert stop["parsed"]["success"] is True
        assert handle.state == TargetState.STOPPED

    def test_start_rejects_missing_binary(self) -> None:
        registry = TargetRegistry()
        # by-passing from_config so we can craft an invalid post-hoc spec
        spec = TargetSpec(
            kind=TargetKind.LOCAL_PROCESS,
            binary_path="/tmp/does-not-exist-pao",
            session_id="sess",
        )
        handle = registry.create(spec)
        start = registry.start_target_response(target_id=handle.target_id)
        parsed = start["parsed"]
        assert parsed["success"] is False
        assert "does not exist" in parsed["error"]

    def test_double_start_reports_error(self) -> None:
        registry = TargetRegistry()
        spec = self._spec(["-c", "import time; time.sleep(30)"])
        handle = registry.create(spec)
        registry.start_target_response(target_id=handle.target_id)
        second = registry.start_target_response(target_id=handle.target_id)
        assert second["parsed"]["success"] is False
        assert "already running" in second["parsed"]["error"]
        registry.stop_target_response(target_id=handle.target_id)

    def test_remove_frees_session_cap_slot(self) -> None:
        registry = TargetRegistry(max_targets_per_session=1)
        spec = self._spec(["-c", "print('hi')"])
        handle = registry.create(spec)
        registry.start_target_response(target_id=handle.target_id)
        registry.stop_target_response(target_id=handle.target_id, remove=True)
        # Now the session has zero slots used again.
        second = registry.create(self._spec(["-c", "print('again')"]))
        assert second.target_id != handle.target_id


# ---------------------------------------------------------------------------
# AttachPidProvider
# ---------------------------------------------------------------------------
class TestAttachPidProvider:
    def test_start_verifies_liveness(self) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": os.getpid()}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.RUNNING
        assert handle.pid == os.getpid()

    def test_stop_detaches_never_kills(self) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": os.getpid()}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        registry.stop_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.DETACHED
        # The current interpreter is definitely still alive.
        assert os.getpid() > 0


# ---------------------------------------------------------------------------
# attach_debugger dispatch
# ---------------------------------------------------------------------------
class TestAttachDebugger:
    def test_attach_success_marks_target_attached(self) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": os.getpid()}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        fake_gdb = FakeGdb(attach_success=True)
        response = registry.attach_debugger_response(
            target_id=handle.target_id, gdb=fake_gdb  # type: ignore[arg-type]
        )
        parsed = response["parsed"]
        assert parsed["success"] is True
        assert parsed["attach"]["attach"]["success"] is True
        assert handle.state == TargetState.ATTACHED
        assert fake_gdb.attach_calls == [os.getpid()]

    def test_attach_failure_records_error(self) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": os.getpid()}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        fake_gdb = FakeGdb(attach_success=False)
        response = registry.attach_debugger_response(
            target_id=handle.target_id, gdb=fake_gdb  # type: ignore[arg-type]
        )
        parsed = response["parsed"]
        assert parsed["success"] is False
        assert handle.state == TargetState.ERROR
        assert handle.error


# ---------------------------------------------------------------------------
# close_session cleanup
# ---------------------------------------------------------------------------
class TestCloseSession:
    def test_close_session_stops_and_drops_all(self) -> None:
        registry = TargetRegistry()
        spec = TargetSpec.from_config(
            "local_process",
            {
                "binary_path": sys.executable,
                "argv": ["-c", "import time; time.sleep(30)"],
            },
            session_id="sess",
        )
        for _ in range(2):
            handle = registry.create(spec)
            registry.start_target_response(target_id=handle.target_id)
        assert len(registry.list(session_id="sess")) == 2
        summary = registry.close_session("sess")
        assert len(summary["stopped"]) == 2
        assert summary["errors"] == []
        assert registry.list(session_id="sess") == []


# ---------------------------------------------------------------------------
# max targets default constant
# ---------------------------------------------------------------------------
def test_default_max_targets_per_session_is_documented() -> None:
    assert DEFAULT_MAX_TARGETS_PER_SESSION >= 4
