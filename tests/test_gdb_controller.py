from datetime import datetime
import signal
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pygdbmi.constants import GdbTimeoutError

from paomcp.tools.backends import gdb as gdb_module
from paomcp.tools.backends.gdb import GdbController


def bare_controller():
    controller = object.__new__(GdbController)
    controller._state = "running"
    controller._inferior_pid = None
    controller._inferior_epoch = 0
    return controller


def test_stopped_notification_allows_null_payload():
    controller = bare_controller()

    controller._handle_notify({"type": "notify", "message": "stopped", "payload": None})

    assert controller.get_state() == "stopped"
    record = controller.get_last_stop_record()
    assert record is not None
    assert record["payload"] == {}
    assert record["is_live"] is True


def test_isolated_controller_init_is_lazy_and_does_not_spawn(monkeypatch):
    # create_services builds the default session just to read an id; it must
    # not Popen a full gdb subprocess at construction.
    spawned: list[bool] = []
    monkeypatch.setattr(gdb_module.shutil, "which", lambda path: f"/fake/{path}")
    monkeypatch.setattr(
        gdb_module._SessionIsolatedController,
        "spawn_new_gdb_subprocess",
        lambda self: spawned.append(True) or 1,
    )

    controller = gdb_module._SessionIsolatedController(
        command=["pwndbg", "--interpreter=mi3", "--quiet"]
    )

    assert controller.gdb_process is None
    assert spawned == []
    controller._ensure_spawned()
    assert spawned == [True]


def test_isolated_controller_init_fails_fast_on_unresolvable_executable(
    monkeypatch,
):
    monkeypatch.setattr(gdb_module.shutil, "which", lambda path: None)
    with pytest.raises(ValueError, match="could not be resolved"):
        gdb_module._SessionIsolatedController(
            command=["pwndbg", "--interpreter=mi3", "--quiet"]
        )


def test_thread_group_started_allows_null_payload():
    controller = bare_controller()

    controller._handle_notify(
        {"type": "notify", "message": "thread-group-started", "payload": None}
    )

    assert controller.get_inferior_pid() is None
    assert controller.get_inferior_epoch() == 1


def test_thread_group_exit_advances_inferior_epoch():
    controller = bare_controller()

    controller._handle_notify(
        {"type": "notify", "message": "thread-group-exited", "payload": {}}
    )

    assert controller.get_state() == "exited"
    assert controller.get_inferior_epoch() == 1


def test_stop_record_preserves_raw_payload_and_returns_independent_copies():
    controller = bare_controller()
    controller._inferior_pid = 31337
    payload = {
        "reason": "signal-received",
        "signal-name": "SIGSEGV",
        "frame": {"addr": "0x41414141", "args": [{"name": "x"}]},
    }

    controller._handle_notify(
        {"type": "notify", "message": "stopped", "payload": payload}
    )
    payload["frame"]["addr"] = "mutated"

    first = controller.get_last_stop_record()
    second = controller.get_last_stop_record()
    assert first is not None
    assert second is not None
    assert first == second
    assert first is not second
    assert first["payload"] is not second["payload"]
    assert first["payload"]["frame"]["addr"] == "0x41414141"
    assert first["payload"]["reason"] == "signal-received"
    assert first["payload"]["signal-name"] == "SIGSEGV"
    assert first["inferior_pid"] == 31337
    assert first["inferior_epoch"] == 0
    assert first["stop_sequence"] == 1
    assert first["identity"] == "0:1"
    datetime.fromisoformat(first["captured_at"])

    first["payload"]["frame"]["addr"] = "also-mutated"
    assert controller.get_last_stop_record()["payload"]["frame"]["addr"] == (
        "0x41414141"
    )


def test_stop_history_is_bounded_and_sequence_remains_monotonic():
    controller = bare_controller()

    for index in range(70):
        controller._handle_notify(
            {
                "type": "notify",
                "message": "stopped",
                "payload": {"reason": "breakpoint-hit", "index": index},
            }
        )

    records = controller.get_stop_records()
    assert len(records) == 64
    assert records[0]["stop_sequence"] == 7
    assert records[-1]["stop_sequence"] == 70
    assert records[-1]["is_live"] is True
    assert all(record["is_live"] is False for record in records[:-1])


def test_stop_record_survives_exit_but_is_not_live():
    controller = bare_controller()
    controller._inferior_pid = 9001
    controller._handle_notify(
        {
            "type": "notify",
            "message": "stopped",
            "payload": {"reason": "exited-normally"},
        }
    )
    identity = controller.get_last_stop_record()["identity"]

    controller._handle_notify(
        {"type": "notify", "message": "thread-group-exited", "payload": {}}
    )

    record = controller.get_last_stop_record()
    assert record is not None
    assert record["identity"] == identity
    assert record["controller_state"] == "exited"
    assert record["is_live"] is False


def test_thread_group_start_clears_stale_records_without_resetting_sequence():
    controller = bare_controller()
    controller._handle_notify(
        {
            "type": "notify",
            "message": "stopped",
            "payload": {"reason": "signal-received"},
        }
    )

    controller._handle_notify(
        {
            "type": "notify",
            "message": "thread-group-started",
            "payload": {"pid": "1234"},
        }
    )

    assert controller.get_last_stop_record() is None
    assert controller.get_inferior_pid() == 1234
    controller._handle_notify(
        {
            "type": "notify",
            "message": "stopped",
            "payload": {"reason": "breakpoint-hit"},
        }
    )
    assert controller.get_last_stop_record()["stop_sequence"] == 2
    assert controller.get_stop_sequence() == 2


def test_run_clears_stale_records_and_keeps_sequence(monkeypatch):
    controller = bare_controller()
    controller._handle_notify(
        {
            "type": "notify",
            "message": "stopped",
            "payload": {"reason": "signal-received"},
        }
    )

    monkeypatch.setattr(
        controller,
        "execute_mi_command",
        lambda command: {
            "command": command,
            "responses": [],
            "success": True,
            "state": "running",
        },
    )

    controller.run()

    assert controller.get_last_stop_record() is None
    controller._handle_notify(
        {
            "type": "notify",
            "message": "stopped",
            "payload": {"reason": "end-stepping-range"},
        }
    )
    assert controller.get_last_stop_record()["stop_sequence"] == 2


def test_stop_record_limit_validation_and_copy_semantics():
    controller = bare_controller()
    for index in range(3):
        controller._handle_notify(
            {
                "type": "notify",
                "message": "stopped",
                "payload": {"index": index},
            }
        )

    assert [item["stop_sequence"] for item in controller.get_stop_records(2)] == [
        2,
        3,
    ]
    assert controller.get_stop_records(0) == []
    with pytest.raises(ValueError, match="non-negative"):
        controller.get_stop_records(-1)


def test_is_destroyed_detects_unexpected_gdb_exit():
    controller = bare_controller()
    controller.controller = SimpleNamespace(gdb_process=SimpleNamespace(poll=lambda: 9))

    assert controller.is_destroyed() is True
    assert controller.get_state() == "exited"
    assert controller.get_destroyed_reason() == "gdb process exited unexpectedly"


def test_mark_destroyed_records_reason_without_process_probe():
    controller = bare_controller()

    controller.mark_destroyed("unsafe MI state")

    assert controller.is_destroyed() is True
    assert controller.get_destroyed_reason() == "unsafe MI state"


def test_apply_fsize_limit_never_raises_existing_soft_or_hard_limit(monkeypatch):
    controller = bare_controller()
    calls = []
    resource = ModuleType("resource")
    resource.RLIMIT_FSIZE = 1
    resource.RLIM_INFINITY = -1

    def prlimit(pid, resource_id, limits=None):
        assert pid == 4242
        assert resource_id == resource.RLIMIT_FSIZE
        if limits is None:
            return (1024, 4096)
        calls.append(limits)
        return (1024, 4096)

    resource.prlimit = prlimit
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(gdb_module, "_POSIX", True)
    monkeypatch.setattr(controller, "get_gdb_process_pid", lambda: 4242)
    monkeypatch.setattr(controller, "gdb_process_running", lambda: True)

    state = controller.apply_fsize_limit(2048)

    assert state["applied"] is True
    assert state["previous_soft"] == 1024
    assert state["previous_hard"] == 4096
    assert state["applied_soft"] == 1024
    assert calls == [(1024, 4096)]


def test_restore_fsize_limit_restores_soft_and_preserves_current_hard(monkeypatch):
    controller = bare_controller()
    calls = []
    resource = ModuleType("resource")
    resource.RLIMIT_FSIZE = 1

    def prlimit(pid, resource_id, limits=None):
        assert pid == 4242
        assert resource_id == resource.RLIMIT_FSIZE
        if limits is None:
            return (512, 8192)
        calls.append(limits)
        return (512, 8192)

    resource.prlimit = prlimit
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(gdb_module, "_POSIX", True)
    monkeypatch.setattr(controller, "get_gdb_process_pid", lambda: 4242)
    monkeypatch.setattr(controller, "gdb_process_running", lambda: True)

    restored = controller.restore_fsize_limit(
        {
            "applied": True,
            "pid": 4242,
            "previous_soft": 2048,
            "previous_hard": 4096,
        }
    )

    assert restored is True
    assert calls == [(2048, 8192)]


def test_restore_fsize_limit_refuses_stale_or_reused_pid(monkeypatch):
    controller = bare_controller()
    resource = ModuleType("resource")
    resource.RLIMIT_FSIZE = 1
    calls = []

    def prlimit(*args):
        calls.append(args)
        return (512, 8192)

    resource.prlimit = prlimit
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(gdb_module, "_POSIX", True)
    monkeypatch.setattr(controller, "get_gdb_process_pid", lambda: 9002)
    monkeypatch.setattr(controller, "gdb_process_running", lambda: True)

    restored = controller.restore_fsize_limit(
        {
            "applied": True,
            "pid": 9001,
            "previous_soft": 2048,
            "previous_hard": 4096,
        }
    )

    assert restored is False
    assert calls == []


class _CloseTestPipe:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _CloseTestProcess:
    def __init__(self, *, stubborn: bool) -> None:
        self.pid = 4242
        self.stdin = _CloseTestPipe()
        self.stdout = _CloseTestPipe()
        self.stderr = _CloseTestPipe()
        self.stubborn = stubborn
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts = []

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.stubborn:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        if not self.stubborn:
            self.returncode = -9

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("gdb", timeout)
        return self.returncode


def _controller_with_close_process(process: _CloseTestProcess) -> GdbController:
    controller = bare_controller()
    controller._initialized = True
    controller._destroyed = False
    controller.controller = SimpleNamespace(gdb_process=process, io_manager=object())
    return controller


def test_close_uses_bounded_wait_and_releases_dead_process_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", False)
    process = _CloseTestProcess(stubborn=False)
    controller = _controller_with_close_process(process)

    closed = controller.close(timeout_sec=0.01)

    assert closed is True
    assert process.terminate_calls == 1
    assert process.kill_calls == 0
    assert process.wait_timeouts == [0.01]
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert process.stderr.closed is True
    assert controller.controller.gdb_process is None
    assert controller.controller.io_manager is None
    assert controller.get_state() == "exited"
    assert controller.is_destroyed() is True


def test_close_retains_live_process_handle_after_bounded_hard_kill_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", False)
    process = _CloseTestProcess(stubborn=True)
    controller = _controller_with_close_process(process)

    closed = controller.close(timeout_sec=0.01)

    assert closed is False
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_timeouts == [0.01, 5]
    assert controller.controller.gdb_process is process
    assert controller.gdb_process_running() is True
    assert controller.is_destroyed() is True


class _RespawnKillableProcess(_CloseTestProcess):
    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9


def _bare_isolated_controller(process):
    controller = object.__new__(gdb_module._SessionIsolatedController)
    controller.command = ["pwndbg", "--interpreter=mi3", "--quiet"]
    controller.time_to_check_for_additional_output_sec = 0.01
    controller.gdb_process = process
    controller.io_manager = object()
    return controller


def test_respawn_uses_bounded_shutdown_instead_of_upstream_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = _RespawnKillableProcess(stubborn=True)
    replacement = _CloseTestProcess(stubborn=False)
    controller = _bare_isolated_controller(previous)
    monkeypatch.setattr(
        controller,
        "exit",
        lambda: pytest.fail("unbounded pygdbmi exit must not be called"),
    )
    monkeypatch.setattr(
        gdb_module.subprocess, "Popen", lambda *args, **kwargs: replacement
    )
    monkeypatch.setattr(gdb_module, "IoManager", lambda *args, **kwargs: object())

    pid = controller.spawn_new_gdb_subprocess()

    assert pid == replacement.pid
    assert previous.terminate_calls == 1
    assert previous.kill_calls == 1
    assert previous.wait_timeouts == [2.0, 2.0]
    assert previous.stdin.closed is True
    assert previous.stdout.closed is True
    assert previous.stderr.closed is True
    assert controller.gdb_process is replacement


def test_spawn_cleans_up_child_when_io_manager_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned = _CloseTestProcess(stubborn=False)
    controller = _bare_isolated_controller(None)
    monkeypatch.setattr(gdb_module.subprocess, "Popen", lambda *args, **kwargs: spawned)

    def fail_io_manager(*args, **kwargs):
        raise RuntimeError("IoManager initialization failed")

    monkeypatch.setattr(gdb_module, "IoManager", fail_io_manager)

    with pytest.raises(RuntimeError, match="IoManager initialization failed"):
        controller.spawn_new_gdb_subprocess()

    assert spawned.terminate_calls == 1
    assert spawned.wait_timeouts == [2.0]
    assert spawned.stdin.closed is True
    assert spawned.stdout.closed is True
    assert spawned.stderr.closed is True
    assert controller.gdb_process is None
    assert controller.io_manager is None


class _SignalTestProcess:
    def __init__(self, *, running: bool = True) -> None:
        self.pid = 4242
        self.returncode = None if running else 9
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, sig) -> None:
        self.signals.append(sig)


def test_signal_interrupt_sends_sigint_without_draining_mi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", True)
    process = _SignalTestProcess()
    controller = bare_controller()
    drains = []
    controller.controller = SimpleNamespace(
        gdb_process=process,
        get_gdb_response=lambda **kwargs: drains.append(kwargs) or [],
    )

    result = controller.signal_interrupt()

    assert result["success"] is True
    assert result["responses"] == []
    assert process.signals == [signal.SIGINT]
    # The documented contract: never consume the MI stream from the signalling
    # thread; the command-owning thread drains it.
    assert drains == []


def test_signal_interrupt_reports_unsupported_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", False)
    process = _SignalTestProcess()
    controller = bare_controller()
    controller.controller = SimpleNamespace(gdb_process=process)

    result = controller.signal_interrupt()

    assert result["success"] is False
    assert result["type"] == "SignalInterruptUnsupported"
    assert process.signals == []


def test_signal_interrupt_without_live_process_is_a_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", True)
    controller = bare_controller()
    controller.controller = SimpleNamespace(gdb_process=None)

    result = controller.signal_interrupt()

    assert result["success"] is False
    assert result["type"] == "SignalInterruptUnavailable"

    exited = _SignalTestProcess(running=False)
    controller.controller = SimpleNamespace(gdb_process=exited)
    result = controller.signal_interrupt()

    assert result["success"] is False
    assert result["type"] == "SignalInterruptUnavailable"
    assert exited.signals == []


def test_signal_interrupt_reports_send_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gdb_module, "_POSIX", True)

    class _FailingSignal(_SignalTestProcess):
        def send_signal(self, sig) -> None:
            raise OSError("no such process")

    controller = bare_controller()
    controller.controller = SimpleNamespace(gdb_process=_FailingSignal())

    result = controller.signal_interrupt()

    assert result["success"] is False
    assert result["type"] == "SignalInterruptError"
    assert "no such process" in result["error"]


class _SharedMiStream:
    """Fake controller whose pending MI record survives a timed-out write."""

    def __init__(self, pending) -> None:
        self.pending = pending
        self.writes = []
        self.reads = 0

    def write(
        self,
        command,
        timeout_sec=7.0,
        raise_error_on_timeout=True,
        read_response=True,
    ):
        self.writes.append(command)
        if command == "-thread-info":
            raise GdbTimeoutError("Did not get response from gdb after 2.5 seconds")
        if self.pending is not None:
            record, self.pending = self.pending, None
            return [record]
        return [
            {
                "type": "result",
                "message": "done",
                "payload": {"command": command},
            }
        ]

    def get_gdb_response(self, timeout_sec=7.0, raise_error_on_timeout=True):
        self.reads += 1
        if self.pending is not None:
            record, self.pending = self.pending, None
            return [record]
        return []


def test_execute_mi_command_consumes_late_result_after_timeout() -> None:
    late = {"type": "result", "message": "done", "payload": {"late": True}}
    stream = _SharedMiStream(late)
    controller = bare_controller()
    controller.controller = stream

    timed_out = controller.execute_mi_command("-thread-info", timeout_sec=2.5)

    # The command really completed (its terminal record was recovered), so
    # success is honest, but the deadline overrun is still reported.
    assert timed_out["success"] is True
    assert timed_out["timed_out"] is True
    assert timed_out["timeout_seconds"] == 2.5
    assert timed_out["responses"] == [late]
    assert stream.reads == 1

    second = controller.execute_mi_command("-data-list-register-names", timeout_sec=2.5)

    assert second["success"] is True
    assert second["responses"] == [
        {
            "type": "result",
            "message": "done",
            "payload": {"command": "-data-list-register-names"},
        }
    ]
    assert "timed_out" not in second


def test_execute_mi_command_flags_desync_when_late_result_never_arrives() -> None:
    stream = _SharedMiStream(None)
    controller = bare_controller()
    controller.controller = stream

    result = controller.execute_mi_command("-thread-info", timeout_sec=2.5)

    assert result["success"] is False
    assert result["timed_out"] is True
    assert result["type"] == "TimeoutError"
    assert "desynchronized after timeout" in result["error"]
    assert result["responses"] == []


class _NormalPathStream:
    def __init__(self, first_responses, pending=None) -> None:
        self.first_responses = list(first_responses)
        self.pending = list(pending or [])
        self.drained = 0

    def write(
        self, command, timeout_sec=7.0, raise_error_on_timeout=True, read_response=True
    ):
        return list(self.first_responses)

    def get_gdb_response(self, timeout_sec=7.0, raise_error_on_timeout=True):
        self.drained += 1
        if self.pending:
            batch, self.pending = self.pending, []
            return batch
        return []


def test_execute_mi_command_normal_path_never_drains() -> None:
    terminal = {"type": "result", "message": "done", "payload": {}}
    stream = _NormalPathStream(
        [
            {"type": "notify", "message": "stopped", "payload": {"reason": "x"}},
            terminal,
        ]
    )
    controller = bare_controller()
    controller.controller = stream

    result = controller.execute_mi_command("-stack-list-frames")

    assert result["success"] is True
    assert result["responses"][-1] == terminal
    assert stream.drained == 0
    assert controller.get_state() == "stopped"


def test_execute_mi_command_recovers_terminal_missing_from_early_return() -> None:
    terminal = {"type": "result", "message": "done", "payload": {}}
    stream = _NormalPathStream(
        [{"type": "notify", "message": "running", "payload": {}}],
        pending=[terminal],
    )
    controller = bare_controller()
    controller.controller = stream

    result = controller.execute_mi_command("-exec-continue")

    assert result["success"] is True
    assert result["responses"][-1] == terminal
    assert stream.drained == 1
    assert controller.get_state() == "running"


def test_execute_mi_command_refuses_success_without_terminal_result() -> None:
    stream = _NormalPathStream([{"type": "console", "payload": "partial"}])
    controller = bare_controller()
    controller.controller = stream

    result = controller.execute_mi_command("-stack-list-frames")

    assert result["success"] is False
    assert result["type"] == "MIStreamError"
    assert "desynchronized" in result["error"]
    assert stream.drained == 1


def test_run_rejects_newline_in_inferior_arguments() -> None:
    controller = bare_controller()
    writes = []
    controller.controller = SimpleNamespace(
        write=lambda command, timeout_sec: writes.append(command) or []
    )

    with pytest.raises(ValueError, match="NUL or newline"):
        controller.run("safe\n-interpreter-exec console quit")

    assert writes == []


def test_run_quotes_inferior_arguments_with_spaces_and_quotes() -> None:
    controller = bare_controller()
    writes = []

    def write(command, timeout_sec=7.0, **kwargs):
        writes.append(command)
        return [{"type": "result", "message": "done", "payload": {}}]

    controller.controller = SimpleNamespace(write=write)

    result = controller.run("first \"two words\" 'three four'")

    assert result["success"] is True
    assert writes == [
        '-exec-arguments "first" "two words" "three four"',
        "-exec-run",
    ]


def test_run_round_trips_embedded_quotes() -> None:
    controller = bare_controller()
    writes = []

    def write(command, timeout_sec=7.0, **kwargs):
        writes.append(command)
        return [{"type": "result", "message": "done", "payload": {}}]

    controller.controller = SimpleNamespace(write=write)

    controller.run('say "hi \\"there\\""')

    assert writes[0] == '-exec-arguments "say" "hi \\"there\\""'


def test_set_breakpoint_rejects_condition_control_characters() -> None:
    controller = bare_controller()

    with pytest.raises(ValueError, match="control character"):
        controller.set_breakpoint("main", condition="x == 1\n-break-delete 1")


def test_memory_mi_string_arguments_are_validated_before_interpolation() -> None:
    controller = bare_controller()

    with pytest.raises(ValueError, match="word_format"):
        controller.read_memory_mi("0x0", "x\n-interpreter-exec console quit", 8, 1, 1)
    with pytest.raises(ValueError, match="hex"):
        controller.write_memory_bytes("0x0", "aa\nbb")


def test_drain_async_surfaces_backend_read_failure() -> None:
    class _RaisingController:
        def get_gdb_response(self, timeout_sec=0.0, raise_error_on_timeout=False):
            raise OSError("pipe is closed")

    controller = bare_controller()
    controller.controller = _RaisingController()

    result = controller.drain_async(timeout_sec=0.0)

    assert result["success"] is False
    assert result["backend_error"] is True
    assert result["type"] == "OSError"
    assert "pipe is closed" in result["error"]
    assert result["responses"] == []


def test_drain_async_marks_destroyed_when_process_is_gone() -> None:
    class _RaisingController:
        def get_gdb_response(self, timeout_sec=0.0, raise_error_on_timeout=False):
            raise OSError("pipe is closed")

    controller = bare_controller()
    controller._destroyed = False
    controller.controller = SimpleNamespace(
        gdb_process=SimpleNamespace(poll=lambda: 9),
        get_gdb_response=_RaisingController().get_gdb_response,
    )

    result = controller.drain_async(timeout_sec=0.0)

    assert result["success"] is False
    assert result["backend_error"] is True
    assert result["state"] == "exited"
    assert controller.is_destroyed() is True
    assert controller.get_destroyed_reason() == "gdb process exited unexpectedly"


def test_drain_async_surfaces_legacy_getter_failure() -> None:
    class _LegacyRaisingController:
        def get_gdb_response(self, timeout_sec=0.0):
            raise RuntimeError("legacy read failure")

    controller = bare_controller()
    controller.controller = _LegacyRaisingController()

    result = controller.drain_async(timeout_sec=0.0)

    assert result["success"] is False
    assert result["backend_error"] is True
    assert result["type"] == "RuntimeError"
    assert "legacy read failure" in result["error"]


def test_execute_command_rejects_embedded_control_characters() -> None:
    writes: list[str] = []

    class _RecordingController:
        def write(self, command, timeout_sec=0.0):
            writes.append(command)
            return []

    controller = bare_controller()
    controller.controller = _RecordingController()

    result = controller.execute_command("x/s 0x1\nshell id")

    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "control character" in result["error"]
    assert writes == []
