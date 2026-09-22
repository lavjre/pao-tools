"""Tests for the minimal pwndbg-lldb controller (Feature B)."""

import subprocess
from typing import Dict, List, Optional

import pytest

import paomcp.tools.backends.lldb as lldb_module
from paomcp.tools.backends.backend import BackendUnavailableError


class _ScriptedStream:
    """A fake stdout pipe yielding preloaded bytes lines to the reader thread."""

    def __init__(self, lines: List[str]) -> None:
        self._lines = [
            line if isinstance(line, bytes) else line.encode() for line in lines
        ]
        self._idx = 0
        self.closed = False

    def __iter__(self) -> "_ScriptedStream":
        return self

    def __next__(self) -> bytes:
        if self._idx >= len(self._lines):
            raise StopIteration
        line = self._lines[self._idx]
        self._idx += 1
        return line

    def close(self) -> None:
        self.closed = True


class _RecordingStdin:
    def __init__(self) -> None:
        self.writes: List[str] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data.decode("utf-8", "replace"))
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _FakePopen:
    def __init__(self, lines: List[str]) -> None:
        self.pid = 4242
        self.stdin = _RecordingStdin()
        self.stdout = _ScriptedStream(lines)
        self.stderr = None
        self._alive = True
        self._returncode: Optional[int] = None
        self.kill_calls = 0
        self.terminate_calls = 0
        self.wait_calls = 0

    def poll(self) -> Optional[int]:
        return None if self._alive else self._returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        self.wait_calls += 1
        self._alive = False
        self._returncode = 0
        return 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._alive = False
        self._returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._alive = False
        self._returncode = -9


def _make_controller(
    monkeypatch: pytest.MonkeyPatch,
    lines: List[str],
    captured: Optional[Dict] = None,
) -> lldb_module.LldbController:
    captured = {} if captured is None else captured

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        popen = _FakePopen(lines)
        captured["popen"] = popen
        return popen

    monkeypatch.setattr(lldb_module.subprocess, "Popen", _fake_popen)
    controller = lldb_module.LldbController(executable="pwndbg-lldb", timeout_sec=2.0)
    controller.initialize()
    return controller


BANNER = ["pwndbg-lldb 0.0.0 (lldb 16.0.0)", "(lldb) "]


class TestLifecycle:
    def test_initialize_spawns_and_reaches_prompt(self, monkeypatch) -> None:
        controller = _make_controller(monkeypatch, list(BANNER))
        assert controller.get_state() == "idle"
        assert not controller.is_destroyed()
        controller.close()
        assert controller.is_destroyed()

    def test_initialize_writes_quit_on_close(self, monkeypatch) -> None:
        captured: Dict = {}
        controller = _make_controller(monkeypatch, list(BANNER), captured)
        controller.close()
        writes = captured["popen"].stdin.writes
        assert any("quit" in write for write in writes)

    def test_vendor_lib_failure_raises_unavailable(self, monkeypatch) -> None:
        captured = {}

        def _fake_popen(args, **kwargs):
            captured["args"] = args
            return _FakePopen(
                [
                    "libm.so.6: cannot open shared object file",
                    "error while loading shared libraries: libm.so.6",
                ]
            )

        monkeypatch.setattr(lldb_module.subprocess, "Popen", _fake_popen)
        controller = lldb_module.LldbController(timeout_sec=1.0)
        with pytest.raises(BackendUnavailableError):
            controller.initialize()

    def test_never_reaching_prompt_raises_unavailable(self, monkeypatch) -> None:
        captured = {}

        def _fake_popen(args, **kwargs):
            captured["args"] = args
            return _FakePopen(["hung with no prompt"])

        monkeypatch.setattr(lldb_module.subprocess, "Popen", _fake_popen)
        controller = lldb_module.LldbController(timeout_sec=1.0)
        with pytest.raises(BackendUnavailableError):
            controller.initialize()


class TestInitializeRecovery:
    def test_failed_initialize_reaps_process_before_retry(self, monkeypatch) -> None:
        processes: List[_FakePopen] = []
        scripts = [["hung with no prompt"], list(BANNER)]

        def _fake_popen(args, **kwargs):
            popen = _FakePopen(scripts[len(processes)])
            popen.pid = 5000 + len(processes)
            processes.append(popen)
            return popen

        monkeypatch.setattr(lldb_module.subprocess, "Popen", _fake_popen)
        controller = lldb_module.LldbController(timeout_sec=0.3)

        with pytest.raises(BackendUnavailableError):
            controller.initialize()

        assert processes[0].kill_calls == 1
        assert processes[0].wait_calls >= 1
        assert processes[0]._alive is False
        assert controller.get_gdb_process_pid() is None

        controller.initialize()

        assert len(processes) == 2
        assert controller.get_gdb_process_pid() == 5001
        assert controller.get_state() == "idle"
        assert controller.is_destroyed() is False


class TestCloseVerification:
    def test_close_confirms_process_exit(self, monkeypatch) -> None:
        controller = _make_controller(monkeypatch, list(BANNER))
        assert controller.close() is True
        assert controller.gdb_process_running() is False

    def test_close_reports_failure_when_process_survives(self, monkeypatch) -> None:
        class _StubbornPopen(_FakePopen):
            def wait(self, timeout: Optional[float] = None) -> int:
                self.wait_calls += 1
                raise subprocess.TimeoutExpired("pwndbg-lldb", timeout or 0.0)

            def terminate(self) -> None:
                self.terminate_calls += 1

            def kill(self) -> None:
                self.kill_calls += 1

        stubborn = _StubbornPopen(list(BANNER))
        controller = lldb_module.LldbController(timeout_sec=0.1)
        controller._process = stubborn

        assert controller.close(timeout_sec=0.05) is False
        assert controller.gdb_process_running() is True
        assert stubborn.terminate_calls == 1
        assert stubborn.kill_calls == 1
        assert controller.is_destroyed() is True


class TestAttachGuardrails:
    def test_attach_refuses_reserved_pids_without_touching_lldb(
        self, monkeypatch
    ) -> None:
        controller = _make_controller(monkeypatch, list(BANNER))
        for bad_pid in (0, 1):
            result, context = controller.attach(bad_pid)
            assert result["success"] is False
            assert result["type"] == "ValueError"
            assert context == []
        import os

        result, _ = controller.attach(os.getpid())
        assert result["success"] is False

    def test_set_file_quotes_paths(self, monkeypatch) -> None:
        captured: Dict = {}
        controller = _make_controller(monkeypatch, list(BANNER), captured)
        controller.set_file("/tmp/my prog/x")
        writes = captured["popen"].stdin.writes
        assert any('target create "/tmp/my prog/x"' in w for w in writes)


class TestCommandExecution:
    def test_execute_command_returns_console_lines(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += ["(lldb) vmmap", "0x0000000000400000 0x0000000000401000", "(lldb) "]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_command("vmmap")
        assert result["success"] is True
        payloads = [r["payload"] for r in result["responses"] if r["type"] == "console"]
        assert payloads == ["0x0000000000400000 0x0000000000401000"]

    def test_echo_line_is_stripped(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += ["(lldb) help", "help output", "(lldb) "]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_command("help")
        payloads = [r["payload"] for r in result["responses"] if r["type"] == "console"]
        assert payloads == ["help output"]

    def test_state_transitions_from_process_markers(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) process launch",
            "Process 4242 launched: 'a.out'",
            "Process 4242 stopped",
            "* thread #1, stop reason = signal SIGSEGV",
            "stop reason = signal SIGSEGV",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_command("process launch")
        assert result["state"] == "stopped"
        assert controller.get_inferior_pid() == 4242
        assert controller.get_state() == "stopped"
        last = controller.get_last_stop_record()
        assert last is not None
        assert last["payload"]["reason"] == "signal SIGSEGV"

    def test_expression_evaluation(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) expression -- (unsigned long)&chunk",
            "(unsigned long) $0 = 140737488346112",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_command("expression -- (unsigned long)&chunk")
        assert result["success"] is True

    def test_exit_marker_clears_pid(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) process launch",
            "Process 4242 launched: 'a.out'",
            "Process 4242 stopped",
            "(lldb) ",
            "(lldb) process continue",
            "Process 4242 exited with status = 0",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        controller.execute_command("process launch")
        assert controller.get_inferior_pid() == 4242
        result = controller.execute_command("process continue")
        assert result["state"] == "exited"
        assert controller.get_inferior_pid() is None


class TestInterrupt:
    def test_interrupt_fails_while_inferior_still_running(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += ["(lldb) process interrupt", "Process 4242 running", "(lldb) "]
        controller = _make_controller(monkeypatch, lines)

        result = controller.interrupt(timeout_sec=0.0)

        assert result["success"] is False
        assert result["state"] == "running"

    def test_interrupt_succeeds_when_inferior_is_stopped(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += ["(lldb) process interrupt", "Process 4242 stopped", "(lldb) "]
        controller = _make_controller(monkeypatch, lines)

        result = controller.interrupt(timeout_sec=0.0)

        assert result["success"] is True
        assert result["state"] == "stopped"


class TestSetFileState:
    def test_set_file_failure_keeps_previous_state(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            '(lldb) target create "/missing"',
            "error: unable to find executable for '/missing'",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)

        result = controller.set_file("/missing")

        assert result["success"] is False
        assert controller.get_state() == "idle"


class TestBreakpointConditionQuoting:
    def test_set_breakpoint_quotes_condition_with_spaces(self, monkeypatch) -> None:
        captured: Dict = {}
        lines = list(BANNER)
        lines += [
            '(lldb) breakpoint set --name main --condition "$rax == 1"',
            "Breakpoint 1: where = a.out`main",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines, captured)

        result = controller.set_breakpoint("main", condition="$rax == 1")

        assert result["success"] is True
        writes = captured["popen"].stdin.writes
        assert 'breakpoint set --name main --condition "$rax == 1"\n' in writes

    def test_break_insert_translates_quoted_condition(self, monkeypatch) -> None:
        captured: Dict = {}
        lines = list(BANNER)
        lines += [
            '(lldb) breakpoint set --name main --condition "$rax == 1"',
            "Breakpoint 1: where = a.out`main",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines, captured)

        result = controller.execute_mi_command('-break-insert -c "$rax == 1" main')

        assert result["success"] is True
        writes = captured["popen"].stdin.writes
        assert 'breakpoint set --name main --condition "$rax == 1"\n' in writes


class TestTypedReads:
    def test_read_registers_surfaces_lldb_register_values(self, monkeypatch) -> None:
        from paomcp.tools.backends.debugger import DebuggerOperations

        lines = list(BANNER)
        lines += [
            "(lldb) register info",
            "       Name: rax",
            "       Name: rbx",
            "(lldb) ",
            "(lldb) register info",
            "       Name: rax",
            "       Name: rbx",
            "(lldb) ",
            "(lldb) register read",
            "rax = 0x0000000000401000",
            "rbx = 0x00007fff00000000",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        adapter = DebuggerOperations(controller, timeout_seconds=3)

        result = adapter.read_registers()

        assert result["success"] is True
        assert [item["name"] for item in result["parsed"]["registers"]] == [
            "rax",
            "rbx",
        ]
        assert result["parsed"]["count"] == 2


class TestMiTranslation:
    def test_register_names_payload(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) register info",
            "       Name: rax",
            "       Name: rbx",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_mi_command("-data-list-register-names")
        assert result["success"] is True
        assert result["responses"][0]["payload"] == {"register-names": ["rax", "rbx"]}

    def test_register_values_payload(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) register info",
            "       Name: rax",
            "       Name: rbx",
            "(lldb) ",
            "(lldb) register read -- rax rbx",
            "rax = 0x0000000000401000",
            "rbx = 0x00007fff00000000",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_mi_command("-data-list-register-values x 0 1")
        assert result["success"] is True
        payload = result["responses"][0]["payload"]
        assert payload["register-values"] == [
            {"name": "rax", "value": "0x0000000000401000"},
            {"name": "rbx", "value": "0x00007fff00000000"},
        ]

    def test_expression_mi_payload(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) expression -- *((unsigned long*)0x404000)",
            "(unsigned long) $0 = 0x41414141",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_mi_command(
            "-data-evaluate-expression *((unsigned long*)0x404000)"
        )
        assert result["success"] is True
        assert result["responses"][0]["payload"] == {"value": "0x41414141"}

    def test_memory_read_payload(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) memory read -f hex -s 1 -c 8 -- 0x404000",
            "0x0000000000404000: 48 89 e5 c3 90 90 90 90",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_mi_command("-data-read-memory-bytes 0x404000 8")
        assert result["success"] is True
        assert result["responses"][0]["payload"] == {
            "memory": [{"contents": "4889e5c390909090"}]
        }

    def test_read_memory_bytes_envelope(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) memory read -f hex -s 1 -c 4 -- 0x404000",
            "0x0000000000404000: 48 89 e5 c3",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.read_memory_bytes("0x404000", 4)
        assert result["success"] is True
        memory = result["payload"]["memory"]
        assert memory[0]["contents"] == "4889e5c3"
        assert memory[0]["begin"] == 0x404000

    def test_unknown_mi_returns_gated_verdict(self, monkeypatch) -> None:
        controller = _make_controller(monkeypatch, list(BANNER))
        result = controller.execute_mi_command("-data-disassemble-mixed foo")
        assert result["success"] is False
        assert result["type"] == "GdbOnlyBackendError"
        assert "requires the gdb/pwndbg backend" in result["error"]

    def test_target_attach_and_set_file(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) target create /tmp/a.out",
            "Target 0: (a.out) loaded.",
            "(lldb) ",
            "(lldb) process attach -p 777",
            "Process 777 stopped",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        result = controller.execute_mi_command("-file-exec-and-symbols /tmp/a.out")
        assert result["success"] is True
        attach, context = controller.attach(777)
        assert attach["success"] is True
        assert controller.get_inferior_pid() == 777
        assert context == []

    def test_stop_record_tracking_and_is_live(self, monkeypatch) -> None:
        lines = list(BANNER)
        lines += [
            "(lldb) process launch",
            "Process 4242 launched: 'a.out'",
            "Process 4242 stopped",
            "* thread #1, stop reason = breakpoint 1",
            "(lldb) ",
        ]
        controller = _make_controller(monkeypatch, lines)
        controller.execute_command("process launch")
        records = controller.get_stop_records()
        assert len(records) == 1
        assert records[0]["is_live"] is True
        assert records[0]["payload"]["reason"] == "breakpoint 1"
