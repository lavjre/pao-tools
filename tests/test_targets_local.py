"""Tests for typed target-provider foundation."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
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

    def test_qemu_system_requires_kernel_or_disk(self) -> None:
        with pytest.raises(TargetError, match="kernel_path.*disk_path"):
            TargetSpec.from_config("qemu_system", {"qemu_binary": "qemu-system-x86_64"})


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

    def test_get_status_recovers_stale_error_when_target_alive_again(
        self,
    ) -> None:
        # A transient failure pinned the target in ERROR with a stale error
        # string; once the liveness re-poll says the target is back, status
        # must restore the pre-error state and clear the error.
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid",
                {"pid": os.getpid()},
                session_id="sess",
            )
        )
        handle.metadata["pre_error_state"] = TargetState.RUNNING.value
        handle._set_state(TargetState.ERROR)
        handle.error = "transient ssh hiccup"
        handle.error_type = "TargetError"

        status = registry.get_status_response(target_id=handle.target_id)
        payload = status["parsed"]["target"]
        assert payload["state"] == "running"
        assert payload["error"] is None
        assert payload["error_type"] is None

    def test_get_status_keeps_error_when_target_still_down(self) -> None:
        # Liveness re-poll still says dead: the stale error must stay.
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid",
                {"pid": 1_000_000_007},  # very unlikely to be alive
                session_id="sess",
            )
        )
        handle.metadata["pre_error_state"] = TargetState.RUNNING.value
        handle._set_state(TargetState.ERROR)
        handle.error = "connection refused"
        handle.error_type = "TargetError"

        status = registry.get_status_response(target_id=handle.target_id)
        payload = status["parsed"]["target"]
        assert payload["state"] == "error"
        assert payload["error"] == "connection refused"

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


class TestLocalProcessLaunchContainment:
    def test_start_rejects_symlink_swapped_between_create_and_start(
        self, tmp_path
    ) -> None:
        from tests.test_target_containment import (
            make_directory_link,
            retarget_directory_link,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        inside_dir = tmp_path / "inside"
        inside_dir.mkdir()
        shutil.copy(sys.executable, inside_dir / "prog")
        link = workspace / "drop"
        make_directory_link(link, inside_dir)

        registry = TargetRegistry()
        spec = TargetSpec.from_config(
            "local_process",
            {
                "binary_path": str(link / "prog"),
                "argv": ["-c", "pass"],
                "workspace_root": str(workspace),
            },
            session_id="sess",
        )
        handle = registry.create(spec)
        # Swap the contained link for one resolving outside the workspace.
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        shutil.copy(sys.executable, outside_dir / "prog")
        retarget_directory_link(link, outside_dir)

        response = registry.start_target_response(target_id=handle.target_id)

        assert response["parsed"]["success"] is False
        assert "workspace" in response["parsed"]["error"]
        assert handle.state == TargetState.ERROR


# ---------------------------------------------------------------------------
# AttachPidProvider
# ---------------------------------------------------------------------------
@pytest.fixture
def sleeper_pid():
    """Spawn a child sleep process as an attachable non-self PID."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        yield proc.pid
    finally:
        proc.kill()
        proc.wait(timeout=10)


class TestAttachPidProvider:
    def test_start_verifies_liveness(self, sleeper_pid: int) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": sleeper_pid}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.RUNNING
        assert handle.pid == sleeper_pid

    def test_config_rejects_non_positive_pid(self) -> None:
        with pytest.raises(TargetError, match="positive 'pid'"):
            TargetSpec.from_config("attach_pid", {"pid": 0}, session_id="sess")

    def test_start_refuses_init_self_and_parent_pids(self) -> None:
        registry = TargetRegistry()
        for bad_pid in (1, os.getpid(), os.getppid()):
            handle = registry.create(
                TargetSpec.from_config(
                    "attach_pid", {"pid": bad_pid}, session_id="sess"
                )
            )
            response = registry.start_target_response(target_id=handle.target_id)
            assert response["parsed"]["success"] is False
            assert "Refusing to attach" in response["parsed"]["error"]
            assert handle.state == TargetState.ERROR

    def test_stop_detaches_never_kills(self, sleeper_pid: int) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": sleeper_pid}, session_id="sess"
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        registry.stop_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.DETACHED
        # The child is still alive: stop() detaches without killing.
        assert handle.pid == sleeper_pid


# ---------------------------------------------------------------------------
# attach_debugger dispatch
# ---------------------------------------------------------------------------
class TestAttachDebugger:
    def test_attach_success_marks_target_attached(self, sleeper_pid: int) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": sleeper_pid}, session_id="sess"
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
        assert fake_gdb.attach_calls == [sleeper_pid]

    def test_attach_failure_records_error(self, sleeper_pid: int) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": sleeper_pid}, session_id="sess"
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
# close_all shutdown cleanup
# ---------------------------------------------------------------------------
class TestCloseAll:
    def test_close_all_stops_owned_and_detaches_external_targets(self) -> None:
        registry = TargetRegistry()
        owned = registry.create(
            TargetSpec.from_config(
                "local_process",
                {
                    "binary_path": sys.executable,
                    "argv": ["-c", "import time; time.sleep(30)"],
                },
                session_id="owned",
            )
        )
        external = registry.create(
            TargetSpec.from_config(
                "attach_pid",
                {"pid": os.getpid()},
                session_id="external",
            )
        )
        registry.start_target_response(target_id=owned.target_id)
        registry.start_target_response(target_id=external.target_id)

        summary = registry.close_all(stop_timeout_sec=5.0)

        assert summary == {
            "processed": 2,
            "closed": [owned.target_id, external.target_id],
            "errors": [],
        }
        assert owned.state == TargetState.STOPPED
        assert owned.process is not None
        assert owned.process.poll() is not None
        assert external.state == TargetState.DETACHED
        assert os.getpid() > 0
        assert registry.list() == []

    def test_close_all_processes_every_target_after_provider_failure(self) -> None:
        calls: List[str] = []

        class FailingProvider:
            kind = TargetKind.ATTACH_PID

            def start(self, handle: TargetHandle) -> None:
                handle._set_state(TargetState.RUNNING)

            def stop(
                self,
                handle: TargetHandle,
                *,
                timeout_sec: float = 10.0,
            ) -> None:
                calls.append(handle.target_id)
                if handle.spec.name == "fails":
                    raise RuntimeError("injected cleanup failure")
                handle._set_state(TargetState.DETACHED)

            def attach_debugger(self, handle: TargetHandle, gdb: Any) -> Dict[str, Any]:
                raise NotImplementedError

            def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
                return {}

        registry = TargetRegistry()
        registry.register_provider(FailingProvider())
        handles = [
            registry.create(
                TargetSpec(
                    kind=TargetKind.ATTACH_PID,
                    pid=os.getpid(),
                    name=name,
                    session_id="sess",
                )
            )
            for name in ("first", "fails", "last")
        ]

        summary = registry.close_all()

        assert calls == [handle.target_id for handle in handles]
        assert summary["processed"] == 3
        assert summary["closed"] == [handles[0].target_id, handles[2].target_id]
        assert summary["errors"] == [
            {
                "target_id": handles[1].target_id,
                "kind": "attach_pid",
                "error": "injected cleanup failure",
                "type": "RuntimeError",
            }
        ]
        assert registry.list() == []

    def test_close_all_is_idempotent(self) -> None:
        registry = TargetRegistry()
        assert registry.close_all() == {
            "processed": 0,
            "closed": [],
            "errors": [],
        }

    def test_close_all_rejects_targets_created_during_shutdown(self) -> None:
        registry = TargetRegistry()
        registry.close_all()

        with pytest.raises(TargetError, match="registry is closed"):
            registry.create(
                TargetSpec.from_config(
                    "attach_pid",
                    {"pid": os.getpid()},
                    session_id="late",
                )
            )
        assert registry.close_all() == {
            "processed": 0,
            "closed": [],
            "errors": [],
        }

    def test_close_session_and_close_all_do_not_stop_same_target_twice(self) -> None:
        entered_stop = threading.Event()
        release_stop = threading.Event()
        stop_calls: List[str] = []

        class BlockingProvider:
            kind = TargetKind.ATTACH_PID

            def start(self, handle: TargetHandle) -> None:
                handle._set_state(TargetState.RUNNING)

            def stop(
                self,
                handle: TargetHandle,
                *,
                timeout_sec: float = 10.0,
            ) -> None:
                stop_calls.append(handle.target_id)
                entered_stop.set()
                assert release_stop.wait(timeout=5.0)
                handle._set_state(TargetState.DETACHED)

            def attach_debugger(self, handle: TargetHandle, gdb: Any) -> Dict[str, Any]:
                raise NotImplementedError

            def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
                return {}

        registry = TargetRegistry()
        registry.register_provider(BlockingProvider())
        handle = registry.create(
            TargetSpec(
                kind=TargetKind.ATTACH_PID,
                pid=os.getpid(),
                session_id="sess",
            )
        )
        worker = threading.Thread(target=registry.close_session, args=("sess",))
        worker.start()
        assert entered_stop.wait(timeout=5.0)

        shutdown = registry.close_all()
        release_stop.set()
        worker.join(timeout=5.0)

        assert worker.is_alive() is False
        assert shutdown == {"processed": 0, "closed": [], "errors": []}
        assert stop_calls == [handle.target_id]


# ---------------------------------------------------------------------------
# max targets default constant
# ---------------------------------------------------------------------------
def test_default_max_targets_per_session_is_documented() -> None:
    assert DEFAULT_MAX_TARGETS_PER_SESSION >= 4
