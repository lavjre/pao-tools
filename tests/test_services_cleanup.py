"""Regression tests for application service shutdown ordering."""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationRegistry
from paomcp.services import close_services, create_services
from paomcp.tools.backends.targets import (
    TargetHandle,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
)


def test_close_services_continues_after_a_target_provider_cleanup_failure() -> None:
    events: List[str] = []

    class FailingProvider:
        kind = TargetKind.ATTACH_PID

        def start(self, handle: TargetHandle) -> None:
            handle._set_state(TargetState.RUNNING)

        def stop(self, handle: TargetHandle, *, timeout_sec: float = 10.0) -> None:
            events.append(f"target:{handle.spec.name}")
            if handle.spec.name == "fails":
                raise RuntimeError("injected cleanup failure")
            handle._set_state(TargetState.DETACHED)

        def attach_debugger(self, handle: TargetHandle, gdb: Any) -> Dict[str, Any]:
            raise NotImplementedError

        def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
            return {}

    class CloseRecorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def close_all(self) -> None:
            events.append(self.name)

        def close(self) -> None:
            events.append(self.name)

    class Pipe:
        def kill(self) -> None:
            events.append("paopipe")

    registry = TargetRegistry()
    registry.register_provider(FailingProvider())
    for name in ("first", "fails", "last"):
        registry.create(
            TargetSpec(
                kind=TargetKind.ATTACH_PID,
                pid=os.getpid(),
                name=name,
                session_id="sess",
            )
        )

    services = SimpleNamespace(
        paopipe_lock=threading.Lock(),
        paopipe_sessions={"pipe": Pipe()},
        target_registry=registry,
        subprocess_tools=CloseRecorder("subprocess"),
        session_registry=CloseRecorder("debug-sessions"),
        artifact_store=CloseRecorder("artifacts"),
    )

    close_services(services)  # type: ignore[arg-type]

    assert events == [
        "paopipe",
        "target:first",
        "target:fails",
        "target:last",
        "subprocess",
        "debug-sessions",
        "artifacts",
    ]
    assert services.paopipe_sessions == {}
    assert registry.list() == []


def test_close_services_continues_after_a_pipe_kill_failure() -> None:
    """A pipe whose kill() raises must not abort the rest of shutdown."""
    events: List[str] = []

    class CloseRecorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def close_all(self) -> None:
            events.append(self.name)

        def close(self) -> None:
            events.append(self.name)

    class FailingPipe:
        def kill(self) -> None:
            events.append("paopipe")
            raise RuntimeError("injected pipe kill failure")

    registry = TargetRegistry()
    services = SimpleNamespace(
        paopipe_lock=threading.Lock(),
        paopipe_sessions={"stuck": FailingPipe()},
        target_registry=registry,
        subprocess_tools=CloseRecorder("subprocess"),
        session_registry=CloseRecorder("debug-sessions"),
        artifact_store=CloseRecorder("artifacts"),
    )

    close_services(services)  # type: ignore[arg-type]

    assert events == ["paopipe", "subprocess", "debug-sessions", "artifacts"]
    assert services.paopipe_sessions == {}


async def _tool_fn(mcp: Any, name: str):
    tool = await mcp.get_tool(name)
    return tool.fn


def test_close_services_proceeds_while_a_sendinput_write_is_blocked(
    tmp_path,
) -> None:
    """A send stuck on a non-reading driver must not hold the global lock.

    Regression for the pipe swap/shutdown path: ``sendinput`` used to write
    stdin while holding ``paopipe_lock``, so a blocked write wedged every
    paocli tool and ``close_services``. Shutdown must detach the pipes under
    the lock and kill them outside it.
    """
    from fastmcp import FastMCP

    from paomcp.tools import paocli as paocli_module

    class BlockingPipe:
        def __init__(self) -> None:
            self.send_started = threading.Event()
            self.release_send = threading.Event()
            self.killed = False

        def is_alive(self) -> bool:
            return not self.killed

        def send(self, data: str) -> bool:
            self.send_started.set()
            self.release_send.wait(timeout=10)
            return False

        def kill(self) -> None:
            self.killed = True
            self.release_send.set()

    class FastPipe:
        def __init__(self) -> None:
            self.killed = False
            self.sent = []

        def is_alive(self) -> bool:
            return True

        def send(self, data: str) -> bool:
            self.sent.append(data)
            return True

        def kill(self) -> None:
            self.killed = True

    class Registry:
        def __init__(self) -> None:
            self._sessions = {
                "a": SimpleNamespace(session_id="a"),
                "b": SimpleNamespace(session_id="b"),
            }
            self.closed = False

        def get_session(self, session_id: str) -> Any:
            return self._sessions.get(session_id)

        def close_all(self) -> None:
            self.closed = True

    class Closer:
        def __init__(self, name: str) -> None:
            self.name = name

        def close_all(self) -> None:
            pass

        def close(self) -> None:
            pass

    registry = Registry()
    blocking = BlockingPipe()
    fast = FastPipe()
    services = SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(tmp_path / "workspace")),
        session_registry=registry,
        paopipe_lock=threading.Lock(),
        paopipe_sessions={"a": blocking, "b": fast},
        operation_registry=OperationRegistry(),
        artifact_store=ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0),
        target_registry=TargetRegistry(),
        subprocess_tools=Closer("subprocess"),
        ida=Closer("ida"),
    )
    mcp = FastMCP("paopipe-shutdown-tests")
    paocli_module.register(mcp)
    sendinput = asyncio.run(_tool_fn(mcp, "sendinput"))
    ctx = SimpleNamespace(lifespan_context={"services": services})

    blocked_response: Dict[str, Any] = {}

    def call_blocked() -> None:
        blocked_response["response"] = asyncio.run(
            sendinput(data="A", session_id="a", ctx=ctx)
        )

    worker = threading.Thread(target=call_blocked, daemon=True)
    worker.start()
    assert blocking.send_started.wait(timeout=5)

    # A second session must proceed while the first session's write is stuck.
    second: Dict[str, Any] = {}

    def call_second() -> None:
        second["response"] = asyncio.run(sendinput(data="B", session_id="b", ctx=ctx))

    second_thread = threading.Thread(target=call_second, daemon=True)
    second_thread.start()
    second_thread.join(timeout=3)
    assert second_thread.is_alive() is False
    assert second["response"]["success"] is True

    # Shutdown must proceed too: it detaches and kills the blocked pipe
    # instead of queueing behind the global lock.
    closer = threading.Thread(target=lambda: close_services(services), daemon=True)
    closer.start()
    closer.join(timeout=5)
    assert closer.is_alive() is False
    assert blocking.killed is True
    assert fast.killed is True
    assert services.paopipe_sessions == {}

    worker.join(timeout=5)
    assert worker.is_alive() is False
    assert blocked_response["response"]["success"] is False


def test_create_services_degrades_when_default_session_spawn_fails(
    tmp_path, monkeypatch
) -> None:
    """A debugger spawn failure at startup must not kill the whole server."""

    def _fail_create_session(self, *args, **kwargs):
        raise RuntimeError("pwndbg could not be spawned")

    monkeypatch.setattr(
        "paomcp.services.DebugSessionRegistry.create_session",
        _fail_create_session,
    )

    services = create_services(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )

    assert services.default_session_id is None
    assert services.debugger_compatibility["success"] is False
    assert "spawn failed" in services.debugger_compatibility["error"]
    close_services(services)
