import threading

import pytest

from paomcp.http.attach import _attach_command_blocked, run_attach_request
from paomcp.http.models import AttachRequest


class FakePwndbgTools:
    def __init__(self, attach_success: bool = True):
        self.attach_success = attach_success
        self.calls = []

    def set_file(self, path: str):
        self.calls.append(("set_file", path))
        return {"success": True, "command": f"set-file {path}"}

    def execute(self, command: str):
        self.calls.append(("execute", command))
        return {"success": True, "command": command}

    def attach(self, pid: int):
        self.calls.append(("attach", pid))
        result = {
            "success": self.attach_success,
            "command": "attach",
            "state": "stopped",
            "pid": pid,
        }
        return result, []


class FakeGdb:
    def get_inferior_pid(self):
        return None


class FakeSession:
    def __init__(self, session_id: str, tools: FakePwndbgTools):
        self.session_id = session_id
        self.tools = tools
        self.lock = threading.RLock()
        self.state = type("State", (), {"pid": None})()
        self.closing = False
        self.closed = False
        self.gdb = FakeGdb()


class FakeRegistry:
    def __init__(self, session=None):
        self.session = session

    def get_session(self, _session_id):
        return self.session

    def create_session(self, _session_id):
        return self.session


class FakeServices:
    def __init__(self, registry):
        self.session_registry = registry
        self.default_session_id = "default"
        self.runtime_paths = type(
            "RuntimePaths", (), {"workspace_root": "/workspace"}
        )()


@pytest.mark.asyncio
async def test_attach_endpoint_runs_pre_and_after():
    tools = FakePwndbgTools(attach_success=True)
    session = FakeSession("chal-a", tools)
    services = FakeServices(FakeRegistry(session=session))

    body = AttachRequest(
        pre=["info registers"],
        pid=1337,
        after=["bt"],
        where="target",
        session_id="chal-a",
    )
    response = await run_attach_request(body, services)

    assert response.successful is True
    assert response.attach is not None
    assert response.attach["pid"] == 1337
    assert response.result["set-file"]["success"] is True
    assert "info registers" in response.result
    assert "bt" in response.result
    assert tools.calls == [
        ("set_file", "/workspace/target"),
        ("execute", "info registers"),
        ("attach", 1337),
        ("execute", "bt"),
    ]


@pytest.mark.asyncio
async def test_attach_endpoint_skips_after_when_attach_fails():
    tools = FakePwndbgTools(attach_success=False)
    session = FakeSession("chal-a", tools)
    services = FakeServices(FakeRegistry(session=session))

    body = AttachRequest(
        pre=["info registers"],
        pid=9001,
        after=["should_not_run"],
        session_id="chal-a",
    )
    response = await run_attach_request(body, services)

    assert response.successful is False
    assert "should_not_run" not in response.result


@pytest.mark.asyncio
async def test_attach_endpoint_rejects_reserved_pid():
    tools = FakePwndbgTools(attach_success=True)
    session = FakeSession("chal-a", tools)
    services = FakeServices(FakeRegistry(session=session))

    # model_construct bypasses the pydantic gt=1 gate so the backend
    # validate_attach_pid guard is what gets exercised here.
    body = AttachRequest.model_construct(pid=1, session_id="chal-a")
    response = await run_attach_request(body, services)
    assert response.successful is False
    assert tools.calls == []


@pytest.mark.asyncio
async def test_attach_endpoint_blocks_host_shell_commands():
    tools = FakePwndbgTools(attach_success=True)
    session = FakeSession("chal-a", tools)
    services = FakeServices(FakeRegistry(session=session))

    body = AttachRequest(
        pre=["shell id", "info registers"],
        pid=1337,
        session_id="chal-a",
    )
    response = await run_attach_request(body, services)
    assert response.successful is True
    assert response.result["shell id"]["success"] is False
    assert "info registers" in response.result
    assert ("execute", "shell id") not in tools.calls


@pytest.mark.asyncio
async def test_attach_endpoint_errors_on_missing_session_mapping():
    services = FakeServices(FakeRegistry(session=None))

    body = AttachRequest(pid=1337, session_id="missing")
    response = await run_attach_request(body, services)

    assert response.successful is False
    assert response.attach is not None
    assert "missing" in response.attach["error"]


class TestAttachCommandFilter:
    @pytest.mark.parametrize(
        "command",
        [
            "shell id",
            "shell",
            "!ls",
            "  !id  ",
            "python",
            "python 1+1",
            "python-interactive",
            "eval 1+1",
            "pipe ls | cat",
            "compile code",
            "interpreter-exec mi 1+1",
            "info registers\nshell id",
            "info registers\rpwd",
            "info\x00registers",
            "bt\npython import os",
        ],
    )
    def test_blocked_commands(self, command):
        assert _attach_command_blocked(command) is True

    @pytest.mark.parametrize(
        "command",
        [
            "break main",
            "continue",
            "info registers",
            "",
            "   ",
            'interpreter-exec console "bt"',
            "interpreter-exec CONSOLE bt",
        ],
    )
    def test_debugger_commands_pass(self, command):
        assert _attach_command_blocked(command) is False

    @pytest.mark.asyncio
    async def test_newline_bypass_is_rejected_without_execution(self):
        tools = FakePwndbgTools(attach_success=True)
        session = FakeSession("chal-a", tools)
        services = FakeServices(FakeRegistry(session=session))

        injected = "info registers\nshell id"
        body = AttachRequest(pre=[injected], pid=1337, session_id="chal-a")
        response = await run_attach_request(body, services)

        assert response.successful is True
        assert response.result[injected]["success"] is False
        assert ("execute", injected) not in tools.calls
        assert ("execute", "info registers") not in tools.calls

    @pytest.mark.asyncio
    async def test_python_and_pipe_bypasses_are_rejected(self):
        tools = FakePwndbgTools(attach_success=True)
        session = FakeSession("chal-a", tools)
        services = FakeServices(FakeRegistry(session=session))

        body = AttachRequest(
            pre=["python import os", "eval 1+1", "pipe cat /etc/passwd | sh"],
            pid=1337,
            session_id="chal-a",
        )
        response = await run_attach_request(body, services)

        assert response.successful is True
        for command in ("python import os", "eval 1+1", "pipe cat /etc/passwd | sh"):
            assert response.result[command]["success"] is False
            assert ("execute", command) not in tools.calls


class TestClosedSessionContract:
    @pytest.mark.asyncio
    async def test_closed_session_returns_attach_response_not_500(self):
        tools = FakePwndbgTools(attach_success=True)
        session = FakeSession("chal-a", tools)
        session.closing = True
        services = FakeServices(FakeRegistry(session=session))

        body = AttachRequest(pid=1337, session_id="chal-a")
        response = await run_attach_request(body, services)

        assert response.successful is False
        assert response.attach is not None
        assert response.attach["success"] is False
        assert "closed" in response.attach["error"]
        assert response.result == {}
        assert tools.calls == []
