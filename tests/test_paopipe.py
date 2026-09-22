import asyncio
import sys
import json
import threading
import time
from types import SimpleNamespace

import pytest

from paomcp.paopipe import PaoPipe


def test_paopipe_parses_paocli_attach_marker():
    expected = {"successful": True, "attach": {"pid": 31337}}
    payload = json.dumps(expected)
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            f"print('PAOCLI_ATTACH_RESULT:' + {payload!r})",
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        pipe.proc.wait(timeout=2)

        assert ready["ready"] is True
        assert pipe.get_attach_result() == expected
        assert "PAOCLI_ATTACH_RESULT" not in pipe.release()
    finally:
        pipe.kill()


def test_paopipe_releases_prompt_without_newline():
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('prompt: '); sys.stdout.flush(); time.sleep(5)",
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        output = pipe.release()

        assert ready["ready"] is True
        assert "prompt: " in output
    finally:
        pipe.kill()


def test_paopipe_releases_fragmented_prompt_without_newline():
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; sys.stdout.write('P'); sys.stdout.flush(); "
                "time.sleep(.2); sys.stdout.write('rompt: '); sys.stdout.flush(); "
                "time.sleep(5)"
            ),
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        output = pipe.release()

        assert ready["ready"] is True
        assert "Prompt: " in output
    finally:
        pipe.kill()


def test_paopipe_replaces_invalid_utf8():
    pipe = PaoPipe([sys.executable, "-c", "import os; os.write(1, b'bad: \\xff\\n')"])
    try:
        pipe.wait_ready(timeout=2)
        pipe.proc.wait(timeout=2)
        output = pipe.release()

        assert "bad: " in output
        assert "\ufffd" in output
    finally:
        pipe.kill()


def test_paopipe_kill_after_process_exited_does_not_raise():
    pipe = PaoPipe([sys.executable, "-c", "pass"])
    pipe.proc.wait(timeout=5)
    assert pipe.is_alive() is False
    # kill() after exit exercises the ProcessLookupError path (group already
    # gone) and must swallow it rather than aborting shutdown callers.
    pipe.kill()
    assert pipe.get_exit_code() == 0


def test_paopipe_kill_terminates_live_process_tree():
    import subprocess
    import time

    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )
    try:
        pipe.wait_ready(timeout=2)
        assert pipe.is_alive() is True
        pipe.kill()
        assert pipe.is_alive() is False
        assert pipe.proc.poll() is not None
    finally:
        try:
            pipe.kill()
        except Exception:
            pass


def test_paopipe_send_is_bounded_when_driver_stops_reading():
    pipe = PaoPipe([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pipe.wait_ready(timeout=2)
        started = time.monotonic()
        assert pipe.send("A" * (4 << 20), timeout=1.0) is False
        assert time.monotonic() - started < 10
        # Once a write has wedged, later sends must fail fast instead of
        # queueing behind the blocked writer.
        started = time.monotonic()
        assert pipe.send("B", timeout=1.0) is False
        assert time.monotonic() - started < 1
    finally:
        pipe.kill()


def test_paopipe_kill_unblocks_a_wedged_send():
    pipe = PaoPipe([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        pipe.wait_ready(timeout=2)
        assert pipe.send("A" * (4 << 20), timeout=0.5) is False
        started = time.monotonic()
        pipe.kill()
        assert time.monotonic() - started < 15
        assert pipe.is_alive() is False
    finally:
        pipe.kill()


def test_paopipe_exit_event_is_accounted_in_stats():
    pipe = PaoPipe([sys.executable, "-c", "pass"])
    try:
        pipe.proc.wait(timeout=5)
        deadline = time.monotonic() + 2
        while pipe.stats()["queued_events"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.stats()["queued_events"] == 1
        # Let the put land: _put_event increments the counter before the
        # queue insert.
        time.sleep(0.05)
        events = pipe.release_events()
        assert [event["type"] for event in events] == ["exit"]
        assert pipe.stats()["queued_events"] == 0
    finally:
        pipe.kill()


def test_paopipe_event_cap_applies_to_exit_event(monkeypatch):
    import paomcp.paopipe as paopipe_module

    monkeypatch.setattr(paopipe_module, "PIPE_EVENTS_CAP", 0)
    pipe = PaoPipe([sys.executable, "-c", "pass"])
    try:
        pipe.proc.wait(timeout=5)
        deadline = time.monotonic() + 2
        while pipe.stats()["dropped_events"] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pipe.release_events() == []
        stats = pipe.stats()
        assert stats["dropped_events"] == 1
        assert stats["events_capped"] is True
    finally:
        pipe.kill()


class _ToolSession:
    def __init__(self, session_id, runtime_dir):
        self.session_id = session_id
        self.runtime_dir = runtime_dir
        self.driver_pid = None


class _ToolSessionRegistry:
    def __init__(self, sessions):
        self._sessions = {session.session_id: session for session in sessions}

    def get_session(self, session_id):
        return self._sessions.get(session_id)


def _paocli_services(tmp_path, sessions):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "target").write_bytes(b"\x7fELF")
    return SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(workspace)),
        session_registry=_ToolSessionRegistry(sessions),
        paopipe_lock=threading.Lock(),
        paopipe_sessions={},
    )


async def _paocli_tool_fn(name):
    from fastmcp import FastMCP

    from paomcp.tools import paocli as paocli_module

    mcp = FastMCP("paopipe-tests")
    paocli_module.register(mcp)
    tool = await mcp.get_tool(name)
    return tool.fn


@pytest.mark.asyncio
async def test_paocli_replacement_is_atomic_and_records_driver_pid(
    tmp_path, monkeypatch
):
    from paomcp.tools import paocli as paocli_module

    session = _ToolSession("s1", str(tmp_path / "runtime" / "s1"))
    services = _paocli_services(tmp_path, [session])

    class OldPipe:
        def __init__(self):
            self.calls = []
            self.observed = None

        def release(self):
            self.calls.append("release")
            return "evicted output"

        def release_events(self):
            self.calls.append("events")
            return [{"type": "out", "data": "x"}]

        def kill(self):
            self.calls.append("kill")
            self.observed = dict(services.paopipe_sessions)

    old_pipe = OldPipe()
    services.paopipe_sessions["s1"] = old_pipe

    class FakePipe:
        instances = []

        def __init__(self, command, cwd=None, env=None):
            self.command = command
            self.pid = 4242
            FakePipe.instances.append(self)

        def get_pid(self):
            return self.pid

        def wait_ready(self, timeout=3.0):
            return {"ready": True, "reason": "output", "wait_ms": 3}

        def release(self):
            return "fresh output"

        def release_events(self):
            return []

        def get_attach_result(self):
            return None

        def is_alive(self):
            return True

        def kill(self):
            pass

    monkeypatch.setattr(paocli_module, "PaoPipe", FakePipe)
    paocli = await _paocli_tool_fn("paocli")
    response = await paocli(
        file="print('hi')",
        session_id="s1",
        argument="",
        wait_timeout=0.1,
        binary_path="target",
        ctx=SimpleNamespace(lifespan_context={"services": services}),
    )

    new_pipe = FakePipe.instances[0]
    assert response["success"] is True
    assert old_pipe.calls == ["release", "events", "kill"]
    # The evicted driver must be killed before the replacement is published,
    # so a concurrent launch can never observe the new pipe mid-swap.
    assert old_pipe.observed.get("s1") is not new_pipe
    assert services.paopipe_sessions["s1"] is new_pipe
    assert session.driver_pid == 4242
    startup = response["data"]["startup"]
    assert startup["replaced"] is True
    assert startup["replaced_output"] == "evicted output"
    assert startup["replaced_event_count"] == 1
