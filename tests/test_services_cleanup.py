"""Regression tests for application service shutdown ordering."""

from __future__ import annotations

import os
import threading
from types import SimpleNamespace
from typing import Any, Dict, List

from paomcp.services import close_services
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
