"""Contract tests for the bounded and session-safe S1 foundation."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
import threading
import time
from typing import Any

import pytest

from paomcp import artifacts as artifact_module
from paomcp.artifacts import ArtifactStore
from paomcp.operations import (
    IdempotencyConflictError,
    OperationCancelledError,
    OperationLimitError,
    OperationOwnershipError,
    OperationRegistry,
    bind_operation,
    reset_operation,
)
from paomcp.state import registry as registry_module
from paomcp.tools.common import catch_errors, run_session_locked
from paomcp.tools.operations import _assert_access
from paomcp.tools.backends import subproc as subproc_module
from paomcp.utils.paths import build_runtime_paths


def test_operation_registry_tracks_lifecycle_and_bound_progress() -> None:
    operations = OperationRegistry(max_concurrent=2, history_limit=10, ttl_seconds=0)
    record = operations.begin("analyze_binary", session_id="alpha")

    token = bind_operation(record.operation_id)
    try:
        progress = operations.update_progress(
            current=2,
            total=5,
            message="reading sections",
        )
    finally:
        reset_operation(token)

    assert progress["progress"] == {
        "current": 2,
        "total": 5,
        "message": "reading sections",
    }
    assert operations.status()["running"] == 1

    completed = operations.complete(record.operation_id, success=True)

    assert completed.state == "succeeded"
    public = operations.get(record.operation_id)
    assert public["tool"] == "analyze_binary"
    assert public["session_id"] == "alpha"
    assert public["state"] == "succeeded"
    assert public["finished_at"] is not None
    assert operations.list(session_id="alpha", state="succeeded", limit=10) == [public]
    assert operations.status()["available"] == 2


def test_operation_registry_cancellation_is_cooperative_and_terminal() -> None:
    operations = OperationRegistry(max_concurrent=1, history_limit=10, ttl_seconds=0)
    record = operations.begin("decompile", session_id="alpha")
    callbacks: list[str] = []
    operations.set_cancel_handler(
        record.operation_id,
        lambda: callbacks.append("cancelled"),
    )

    requested = operations.request_cancel(record.operation_id)

    assert requested["cancellation_requested"] is True
    assert callbacks == ["cancelled"]
    cancellation_event = operations.cancellation_event(record.operation_id)
    assert cancellation_event is not None
    assert cancellation_event.is_set()
    with pytest.raises(OperationCancelledError):
        operations.raise_if_cancelled(record.operation_id)

    completed = operations.complete(record.operation_id, success=False)
    assert completed.state == "cancelled"
    assert completed.error_code == "OPERATION_CANCELLED"


def test_operation_registry_releases_concurrency_capacity() -> None:
    operations = OperationRegistry(max_concurrent=1, history_limit=10, ttl_seconds=0)
    first = operations.begin("first")

    with pytest.raises(OperationLimitError):
        operations.begin("second")

    operations.complete(first.operation_id, success=False, error_code="TEST_FAILURE")
    second = operations.begin("second")
    assert operations.get(second.operation_id)["state"] == "running"


def test_operation_registry_idempotency_claim_replay_conflict_and_in_progress() -> None:
    operations = OperationRegistry(max_concurrent=1, history_limit=10, ttl_seconds=0)

    state, response = operations.claim_idempotency(
        "spawn_process", "alpha", "retry-1", "fingerprint-a"
    )
    assert (state, response) == ("claimed", None)
    assert operations.claim_idempotency(
        "spawn_process", "alpha", "retry-1", "fingerprint-a"
    ) == ("in_progress", None)

    expected = {
        "success": True,
        "meta": {"operation_id": "operation-a", "idempotency_replayed": False},
        "data": {"pid": 4242},
    }
    operations.complete_idempotency(
        "spawn_process",
        "alpha",
        "retry-1",
        "fingerprint-a",
        expected,
    )

    replay_state, replay = operations.claim_idempotency(
        "spawn_process", "alpha", "retry-1", "fingerprint-a"
    )
    assert replay_state == "replay"
    assert replay == expected
    assert replay is not expected
    assert replay is not None
    replay["data"]["pid"] = 9999
    assert (
        operations.claim_idempotency(
            "spawn_process", "alpha", "retry-1", "fingerprint-a"
        )[1]
        == expected
    )

    with pytest.raises(IdempotencyConflictError):
        operations.claim_idempotency(
            "spawn_process", "alpha", "retry-1", "fingerprint-b"
        )

    assert operations.claim_idempotency(
        "spawn_process", "beta", "retry-1", "fingerprint-b"
    ) == ("claimed", None)


def _tool_context(tmp_path, *, max_concurrent: int = 2) -> SimpleNamespace:
    services = SimpleNamespace(
        operation_registry=OperationRegistry(
            max_concurrent=max_concurrent,
            history_limit=20,
            ttl_seconds=0,
        ),
        artifact_store=ArtifactStore(
            str(tmp_path / "artifacts"),
            min_free_bytes=0,
        ),
    )
    return SimpleNamespace(lifespan_context={"services": services})


@pytest.mark.asyncio
async def test_catch_errors_replays_and_rejects_conflicting_idempotency_keys(
    tmp_path,
) -> None:
    ctx = _tool_context(tmp_path)
    calls: list[int] = []

    @catch_errors(auto_artifact=False)
    async def mutate(
        value: int,
        session_id: str,
        mode: str = "safe",
        idempotency_key: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        calls.append(value)
        return {
            "success": True,
            "session_id": session_id,
            "value": value,
            "mode": mode,
        }

    first = await mutate(
        7,
        session_id="alpha",
        idempotency_key="retry-1",
        ctx=ctx,
    )
    replay = await mutate(
        7,
        session_id="alpha",
        mode="safe",
        idempotency_key="retry-1",
        ctx=ctx,
    )
    conflict = await mutate(
        8,
        session_id="alpha",
        idempotency_key="retry-1",
        ctx=ctx,
    )

    assert calls == [7]
    assert first["success"] is True
    assert first["meta"]["idempotency_replayed"] is False
    assert replay["success"] is True
    assert replay["meta"]["idempotency_replayed"] is True
    assert replay["meta"]["operation_id"] == first["meta"]["operation_id"]
    assert conflict["success"] is False
    assert conflict["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.asyncio
async def test_catch_errors_reports_idempotency_in_progress(tmp_path) -> None:
    ctx = _tool_context(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    @catch_errors(auto_artifact=False)
    async def mutate(
        value: int,
        session_id: str,
        idempotency_key: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"success": True, "session_id": session_id, "value": value}

    first_task = asyncio.create_task(
        mutate(
            7,
            session_id="alpha",
            idempotency_key="retry-running",
            ctx=ctx,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        duplicate = await mutate(
            7,
            session_id="alpha",
            idempotency_key="retry-running",
            ctx=ctx,
        )
        assert duplicate["success"] is False
        assert duplicate["error"]["code"] == "IDEMPOTENCY_IN_PROGRESS"
        assert calls == 1
    finally:
        release.set()
        await asyncio.wait_for(first_task, timeout=1)


@pytest.mark.asyncio
async def test_cancelled_tool_releases_operation_and_idempotency_slots(
    tmp_path,
) -> None:
    ctx = _tool_context(tmp_path, max_concurrent=1)
    operations = ctx.lifespan_context["services"].operation_registry
    started = asyncio.Event()
    never_finishes = asyncio.Event()

    @catch_errors(auto_artifact=False)
    async def cancellable(
        session_id: str,
        idempotency_key: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        started.set()
        await never_finishes.wait()
        return {"success": True, "session_id": session_id}

    task = asyncio.create_task(
        cancellable(
            session_id="alpha",
            idempotency_key="retry-cancelled",
            ctx=ctx,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    assert operations.status()["running"] == 1

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    status = operations.status()
    assert status["running"] == 0
    assert status["available"] == 1
    cancelled = operations.list(session_id="alpha", state="cancelled", limit=10)
    assert len(cancelled) == 1
    assert cancelled[0]["error_code"] == "OPERATION_CANCELLED"
    assert operations.claim_idempotency(
        "cancellable",
        "alpha",
        "retry-cancelled",
        "new-attempt",
    ) == ("claimed", None)


@pytest.mark.asyncio
async def test_idempotency_replay_reports_expired_artifact_without_rerunning(
    tmp_path,
) -> None:
    ctx = _tool_context(tmp_path)
    store = ctx.lifespan_context["services"].artifact_store
    artifact = store.save_bytes(b"large evidence", tool="mutate")
    calls = 0

    @catch_errors(auto_artifact=False)
    async def mutate(
        idempotency_key: str | None = None,
        ctx: Any = None,
    ) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"success": True, "evidence": artifact.to_dict()}

    first = await mutate(idempotency_key="artifact-retry", ctx=ctx)
    assert first["success"] is True
    assert calls == 1
    store.delete(artifact.artifact_id)

    replay = await mutate(idempotency_key="artifact-retry", ctx=ctx)

    assert replay["success"] is False
    assert replay["error"]["code"] == "IDEMPOTENCY_RESULT_EXPIRED"
    assert replay["error"]["details"]["missing_artifact_ids"] == [artifact.artifact_id]
    assert replay["meta"]["idempotency_replayed"] is True
    assert calls == 1


def test_operation_access_and_listing_enforce_exact_session_ownership() -> None:
    operations = OperationRegistry(max_concurrent=3, history_limit=10, ttl_seconds=0)
    unscoped = operations.begin("global")
    scoped = operations.begin("debug", session_id="alpha")

    _assert_access(operations.get(unscoped.operation_id), None)
    _assert_access(operations.get(scoped.operation_id), "alpha")
    with pytest.raises(OperationOwnershipError):
        _assert_access(operations.get(unscoped.operation_id), "alpha")
    with pytest.raises(OperationOwnershipError):
        _assert_access(operations.get(scoped.operation_id), None)

    assert [
        item["operation_id"] for item in operations.list(session_id=None, limit=10)
    ] == [unscoped.operation_id]
    assert [
        item["operation_id"] for item in operations.list(session_id="alpha", limit=10)
    ] == [scoped.operation_id]


class _FakeGdb:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False
        self.destroyed = False

    def get_state(self) -> str:
        return "stopped"

    def is_destroyed(self) -> bool:
        return self.destroyed

    def initialize(self) -> dict[str, Any]:
        self.initialized = True
        return {"status": "initialized"}

    def gdb_process_running(self) -> bool:
        return not self.closed

    def close(self) -> bool:
        self.closed = True
        return True


class _StubbornDestroyedGdb(_FakeGdb):
    def __init__(self) -> None:
        super().__init__()
        self.alive = True
        self.close_calls = 0

    def gdb_process_running(self) -> bool:
        return self.alive

    def close(self) -> bool:
        self.close_calls += 1
        if self.alive:
            return False
        self.closed = True
        return True


class _FakePwndbgTools:
    def __init__(self, gdb: _FakeGdb, state: Any) -> None:
        self.gdb = gdb
        self.state = state


def test_debug_session_registry_enforces_limit_without_breaking_idempotency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)

    first = registry.create_session("alpha")
    assert getattr(first.gdb, "initialized") is True
    assert registry.create_session("alpha") is first
    with pytest.raises(registry_module.SessionLimitError):
        registry.create_session("beta")

    assert registry.close_session("alpha")["success"] is True
    replacement = registry.create_session("beta")
    assert replacement.session_id == "beta"
    registry.close_all()
    assert getattr(replacement.gdb, "closed") is True


def test_ensure_session_creates_missing_explicit_id_instead_of_using_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=2)
    default = registry.ensure_session()

    explicit = registry.ensure_session("alpha")

    assert explicit.session_id == "alpha"
    assert explicit is not default
    assert registry.default_session_id == "default"
    assert registry.get_session("alpha") is explicit


def test_ensure_session_rebuilds_destroyed_explicit_id_with_live_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=2)
    default = registry.ensure_session()
    destroyed = registry.create_session("alpha")
    destroyed.gdb.destroyed = True

    replacement = registry.ensure_session("alpha")

    assert replacement.session_id == "alpha"
    assert replacement is not destroyed
    assert replacement is not default
    assert destroyed.closed is True
    assert destroyed.gdb.closed is True
    assert registry.default_session_id == "default"
    assert registry.ensure_session() is default


def test_ensure_session_without_id_reuses_live_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=2)
    default = registry.ensure_session()
    registry.create_session("alpha")

    assert default.session_id == "default"
    assert registry.ensure_session() is default


def test_destroyed_live_debugger_tombstone_blocks_capacity_and_id_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _StubbornDestroyedGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    destroyed = registry.create_session("alpha")
    destroyed.gdb.destroyed = True

    assert registry.get_session("alpha") is None
    assert registry.status() == {
        "active": 0,
        "tombstones": 1,
        "occupied": 1,
        "available": 0,
        "max_sessions": 1,
        "default_session_id": None,
    }
    with pytest.raises(registry_module.SessionLimitError, match="pending"):
        registry.create_session("alpha")
    with pytest.raises(registry_module.SessionLimitError, match="limit"):
        registry.create_session("beta")

    destroyed.gdb.alive = False
    assert registry.reap_tombstones() == 1
    assert destroyed.gdb.closed is True
    replacement = registry.create_session("alpha")
    assert replacement is not destroyed
    assert registry.status()["tombstones"] == 0


def test_close_session_surfaces_shutdown_pending_and_keeps_tombstone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _StubbornDestroyedGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")

    result = registry.close_session("alpha")

    assert result["success"] is False
    assert result["code"] == "SHUTDOWN_PENDING"
    assert result["shutdown_pending"] is True
    assert session.closed is True
    assert registry.status()["tombstones"] == 1
    with pytest.raises(registry_module.SessionLimitError):
        registry.create_session("beta")


def test_destroyed_session_detaches_without_closing_under_active_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    action_started = threading.Event()
    release_action = threading.Event()

    def action() -> None:
        action_started.set()
        assert release_action.wait(timeout=2)

    action_thread = threading.Thread(
        target=lambda: run_session_locked(session, action, sync_pid=False)
    )
    action_thread.start()
    assert action_started.wait(timeout=1)
    session.gdb.destroyed = True

    assert registry.get_session("alpha") is None
    assert session.gdb.closed is False
    assert registry.status()["tombstones"] == 1

    release_action.set()
    action_thread.join(timeout=1)
    assert action_thread.is_alive() is False
    assert registry.reap_tombstones() == 1
    assert session.gdb.closed is True


def test_close_session_times_out_without_holding_registry_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    action_started = threading.Event()
    release_action = threading.Event()
    close_results: list[dict[str, Any]] = []
    cleanup_calls: list[str] = []

    def action() -> None:
        action_started.set()
        assert release_action.wait(timeout=2)

    action_thread = threading.Thread(
        target=lambda: run_session_locked(session, action, sync_pid=False)
    )
    close_thread = threading.Thread(
        target=lambda: close_results.append(
            registry.close_session(
                "alpha",
                on_closing=lambda _session: cleanup_calls.append("cleaned"),
                lock_timeout_sec=0.2,
            )
        )
    )
    action_thread.start()
    assert action_started.wait(timeout=1)
    close_thread.start()

    deadline = time.monotonic() + 1
    while not session.closing and time.monotonic() < deadline:
        time.sleep(0.005)
    assert session.closing is True

    status_started = time.monotonic()
    assert registry.status()["tombstones"] == 1
    assert time.monotonic() - status_started < 0.1

    close_thread.join(timeout=1)
    assert close_thread.is_alive() is False
    assert close_results[0]["success"] is False
    assert close_results[0]["code"] == "SHUTDOWN_PENDING"
    assert close_results[0]["action_lock_busy"] is True
    assert session.gdb.closed is False
    assert cleanup_calls == []

    release_action.set()
    action_thread.join(timeout=1)
    assert action_thread.is_alive() is False
    assert registry.reap_tombstones() == 1
    assert session.gdb.closed is True
    assert cleanup_calls == ["cleaned"]


def test_close_session_reports_associated_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    cleanup = {
        "success": False,
        "type": "ManagedProcessCleanupError",
        "error": "one managed process remains alive",
    }

    result = registry.close_session("alpha", on_closing=lambda _session: cleanup)

    assert result["success"] is False
    assert result["code"] == "SESSION_CLEANUP_FAILED"
    assert result["shutdown_pending"] is False
    assert result["cleanup"] == cleanup
    assert session.gdb.closed is True
    assert registry.status()["tombstones"] == 0


@pytest.mark.parametrize(
    "cleanup",
    [
        {
            "session_id": "alpha",
            "attempted_count": 1,
            "closed_count": 0,
            "failure_count": 1,
            "failures": [
                {
                    "pid": 123,
                    "type": "TimeoutExpired",
                    "error": "process remained alive",
                }
            ],
        },
        {
            "session_id": "alpha",
            "attempted_count": 1,
            "closed_count": 0,
            "failure_count": 0,
            "failures": [{"pid": 123}],
        },
    ],
)
def test_close_session_understands_managed_process_cleanup_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    cleanup,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    registry.create_session("alpha")

    result = registry.close_session("alpha", on_closing=lambda _session: cleanup)

    assert result["success"] is False
    assert result["code"] == "SESSION_CLEANUP_FAILED"
    assert result["cleanup"] == cleanup


def test_session_lock_rejects_poisoned_controller_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    session.gdb.destroyed = True
    dispatched = False

    def action() -> None:
        nonlocal dispatched
        dispatched = True

    with pytest.raises(registry_module.SessionClosedError, match="destroyed"):
        run_session_locked(session, action, sync_pid=False)

    assert dispatched is False
    assert session.closed is True


def test_debug_session_close_waits_for_active_action_and_rejects_stale_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    action_started = threading.Event()
    release_action = threading.Event()
    close_started = threading.Event()
    action_errors: list[BaseException] = []
    close_results: list[dict[str, Any]] = []

    def run_active_action() -> None:
        try:

            def action() -> str:
                action_started.set()
                assert release_action.wait(timeout=2)
                return "finished"

            assert run_session_locked(session, action, sync_pid=False) == "finished"
        except BaseException as exc:
            action_errors.append(exc)

    def close_session() -> None:
        close_started.set()
        close_results.append(registry.close_session("alpha"))

    action_thread = threading.Thread(target=run_active_action)
    close_thread = threading.Thread(target=close_session)
    action_thread.start()
    assert action_started.wait(timeout=1)
    close_thread.start()
    assert close_started.wait(timeout=1)
    time.sleep(0.05)

    assert close_thread.is_alive()
    assert getattr(session.gdb, "closed") is False

    release_action.set()
    action_thread.join(timeout=1)
    close_thread.join(timeout=1)
    assert action_thread.is_alive() is False
    assert close_thread.is_alive() is False
    assert action_errors == []
    assert close_results == [{"success": True, "session_id": "alpha"}]
    assert getattr(session.gdb, "closed") is True
    assert session.closed is True

    with pytest.raises(registry_module.SessionClosedError):
        run_session_locked(session, lambda: None, sync_pid=False)


def test_debug_session_close_rejects_action_already_queued_on_stale_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    registry = registry_module.DebugSessionRegistry(paths, max_sessions=1)
    session = registry.create_session("alpha")
    active_started = threading.Event()
    release_active = threading.Event()
    queued_started = threading.Event()
    queued_action_ran = threading.Event()
    queued_errors: list[BaseException] = []
    close_results: list[dict[str, Any]] = []

    def active_worker() -> None:
        def action() -> None:
            active_started.set()
            assert release_active.wait(timeout=2)

        run_session_locked(session, action, sync_pid=False)

    def queued_worker() -> None:
        queued_started.set()
        try:
            run_session_locked(
                session,
                lambda: queued_action_ran.set(),
                sync_pid=False,
            )
        except BaseException as exc:
            queued_errors.append(exc)

    active_thread = threading.Thread(target=active_worker)
    queued_thread = threading.Thread(target=queued_worker)
    close_thread = threading.Thread(
        target=lambda: close_results.append(registry.close_session("alpha"))
    )
    active_thread.start()
    assert active_started.wait(timeout=1)
    queued_thread.start()
    assert queued_started.wait(timeout=1)
    close_thread.start()

    deadline = time.monotonic() + 1
    while not session.closing and time.monotonic() < deadline:
        time.sleep(0.005)
    assert session.closing is True

    release_active.set()
    for thread in (active_thread, queued_thread, close_thread):
        thread.join(timeout=1)
        assert thread.is_alive() is False

    assert queued_action_ran.is_set() is False
    assert len(queued_errors) == 1
    assert isinstance(queued_errors[0], registry_module.SessionClosedError)
    assert close_results == [{"success": True, "session_id": "alpha"}]
    assert session.closed is True


def test_artifact_store_enforces_session_ownership_and_verifies_integrity(
    tmp_path,
) -> None:
    store = ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0)
    artifact = store.save_bytes(
        b"session-owned evidence",
        tool="test",
        session_id="alpha",
    )

    assert store.assert_owner(artifact.artifact_id, "alpha") == artifact
    assert store.get_for_session(artifact.artifact_id, "alpha") == artifact
    read = store.read_for_session(artifact.artifact_id, "alpha")
    assert base64.b64decode(read["content_base64"]) == b"session-owned evidence"
    assert store.verify(artifact.artifact_id, session_id="alpha")["valid"] is True

    for action in (
        lambda: store.assert_owner(artifact.artifact_id, "beta"),
        lambda: store.get_for_session(artifact.artifact_id, "beta"),
        lambda: store.read_for_session(artifact.artifact_id, "beta"),
        lambda: store.verify(artifact.artifact_id, session_id="beta"),
    ):
        with pytest.raises(PermissionError):
            action()

    with open(store._data_path(artifact.artifact_id), "wb") as payload_file:
        payload_file.write(b"tampered")
    assert store.verify(artifact.artifact_id, session_id="alpha")["valid"] is False


def test_artifact_store_treats_unscoped_as_an_exact_owner(tmp_path) -> None:
    store = ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0)
    unscoped = store.save_bytes(b"global", tool="test")
    scoped = store.save_bytes(b"private", tool="test", session_id="alpha")

    assert store.assert_owner(unscoped.artifact_id, None) == unscoped
    assert store.assert_owner(scoped.artifact_id, "alpha") == scoped
    with pytest.raises(PermissionError):
        store.assert_owner(unscoped.artifact_id, "alpha")
    with pytest.raises(PermissionError):
        store.assert_owner(scoped.artifact_id, None)

    assert [item.artifact_id for item in store.list_owned(None)] == [
        unscoped.artifact_id
    ]
    assert [item.artifact_id for item in store.list_owned("alpha")] == [
        scoped.artifact_id
    ]


def test_artifact_reads_are_bounded_and_lossless_by_pagination(tmp_path) -> None:
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        default_chunk_bytes=3,
        max_read_bytes=4,
        min_free_bytes=0,
    )
    artifact = store.save_bytes(b"abcdefghij", tool="test")

    first = store.read(artifact.artifact_id)
    second = store.read(
        artifact.artifact_id,
        offset=first["next_offset"],
        length=4,
    )
    third = store.read(
        artifact.artifact_id,
        offset=second["next_offset"],
        length=4,
    )

    assert [first["length"], second["length"], third["length"]] == [4, 4, 2]
    assert (
        b"".join(
            base64.b64decode(item["content_base64"]) for item in (first, second, third)
        )
        == b"abcdefghij"
    )
    assert third["eof"] is True
    with pytest.raises(ValueError, match="length cannot exceed 4 bytes"):
        store.read(artifact.artifact_id, length=5)


def test_artifact_save_file_copies_only_the_initial_size_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        default_chunk_bytes=3,
        max_read_bytes=4,
        min_free_bytes=0,
    )
    source = tmp_path / "growing.log"
    source.write_bytes(b"abcdef")
    source_path = str(source)
    original_getsize = artifact_module.os.path.getsize

    def snapshot_size(path: str) -> int:
        if artifact_module.os.path.abspath(path) == artifact_module.os.path.abspath(
            source_path
        ):
            return 3
        return original_getsize(path)

    monkeypatch.setattr(artifact_module.os.path, "getsize", snapshot_size)
    artifact = store.save_file(source_path, tool="get_process")

    assert artifact.size_bytes == 3
    result = store.read(artifact.artifact_id, length=3)
    assert base64.b64decode(result["content_base64"]) == b"abc"


def test_artifact_store_reconciles_crash_orphans(tmp_path) -> None:
    store = ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0)
    data_only = "a" * 32
    metadata_only = "b" * 32
    data_path = tmp_path / "artifacts" / f"{data_only}.data"
    metadata_path = tmp_path / "artifacts" / f"{metadata_only}.meta.json"
    temporary_path = tmp_path / "artifacts" / ".artifact_interrupted"
    data_path.write_bytes(b"orphan payload")
    metadata_path.write_text("{}", encoding="utf-8")
    temporary_path.write_bytes(b"partial")

    result = store.reconcile_orphans(grace_seconds=0)

    assert result["removed_count"] == 3
    assert result["removed"] == {
        "temporary": 1,
        "data_without_metadata": 1,
        "metadata_without_data": 1,
    }
    assert result["reclaimed_bytes"] > 0
    assert data_path.exists() is False
    assert metadata_path.exists() is False
    assert temporary_path.exists() is False


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0


def test_subprocess_tools_enforces_managed_process_limit_without_real_processes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_process = _FakeProcess(pid=4242)
    monkeypatch.setattr(
        subproc_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )
    monkeypatch.setattr(subproc_module.time, "sleep", lambda _seconds: None)

    def stop_fake_process(process: _FakeProcess, _signal: int) -> None:
        process.returncode = 0

    monkeypatch.setattr(
        subproc_module.SubprocessTools,
        "_signal_process_tree",
        staticmethod(stop_fake_process),
    )
    tools = subproc_module.SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_processes=1,
    )

    first = tools.spawn_process("echo first", cwd=str(tmp_path))
    limited = tools.spawn_process("echo second", cwd=str(tmp_path))

    assert first["success"] is True
    assert limited["success"] is False
    assert limited["type"] == "ProcessLimitError"
    assert limited["code"] == "RESOURCE_EXHAUSTED"
    assert len(tools.background_processes) == 1

    tools.close_all()
    assert tools.background_processes == {}


def test_subprocess_tools_enforces_exact_process_session_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_processes = [_FakeProcess(pid=4242), _FakeProcess(pid=4343)]
    monkeypatch.setattr(
        subproc_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_processes.pop(0),
    )
    monkeypatch.setattr(subproc_module.time, "sleep", lambda _seconds: None)

    def stop_fake_process(process: _FakeProcess, _signal: int) -> None:
        process.returncode = 0

    monkeypatch.setattr(
        subproc_module.SubprocessTools,
        "_signal_process_tree",
        staticmethod(stop_fake_process),
    )
    tools = subproc_module.SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_processes=2,
    )

    scoped = tools.spawn_process(
        "echo scoped",
        cwd=str(tmp_path),
        session_id="alpha",
    )
    unscoped = tools.spawn_process("echo global", cwd=str(tmp_path))

    assert scoped["success"] is True
    assert unscoped["success"] is True
    assert tools.list_processes("alpha")["count"] == 1
    assert tools.list_processes(None)["count"] == 1
    beta_view = tools.list_processes("beta")
    assert beta_view["count"] == 0
    assert beta_view["capacity"]["available"] == 0

    denied_scoped = tools.get_process(scoped["pid"], None)
    denied_unscoped = tools.kill_process(
        unscoped["pid"],
        session_id="alpha",
    )
    assert denied_scoped["success"] is False
    assert denied_scoped["type"] == "PermissionError"
    assert denied_unscoped["success"] is False
    assert denied_unscoped["type"] == "PermissionError"

    killed_scoped = tools.kill_process(scoped["pid"], session_id="alpha")
    killed_unscoped = tools.kill_process(unscoped["pid"], session_id=None)
    assert killed_scoped["success"] is True
    assert killed_scoped["stdout"] == ""
    assert killed_scoped["stdout_path"] is None
    assert killed_unscoped["success"] is True
    assert subproc_module.os.path.exists(scoped["process_dir"]) is False
    assert subproc_module.os.path.exists(unscoped["process_dir"]) is False
    assert tools.background_processes == {}


def test_subprocess_session_cleanup_stops_processes_removes_logs_and_releases_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_processes = [_FakeProcess(pid=4444), _FakeProcess(pid=4555)]
    monkeypatch.setattr(
        subproc_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_processes.pop(0),
    )
    monkeypatch.setattr(subproc_module.time, "sleep", lambda _seconds: None)

    def stop_fake_process(process: _FakeProcess, _signal: int) -> None:
        process.returncode = 0

    monkeypatch.setattr(
        subproc_module.SubprocessTools,
        "_signal_process_tree",
        staticmethod(stop_fake_process),
    )
    tools = subproc_module.SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_processes=1,
    )
    scoped = tools.spawn_process("echo scoped", cwd=str(tmp_path), session_id="alpha")
    process_dir = scoped["process_dir"]
    assert subproc_module.os.path.isdir(process_dir)

    cleanup = tools.close_session_processes("alpha")

    assert cleanup["closed_count"] == 1
    assert cleanup["failure_count"] == 0
    assert subproc_module.os.path.exists(process_dir) is False
    replacement = tools.spawn_process(
        "echo replacement", cwd=str(tmp_path), session_id="beta"
    )
    assert replacement["success"] is True
    tools.close_all()
    assert subproc_module.os.path.exists(replacement["process_dir"]) is False


def test_run_command_removes_temporary_logs_after_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_process = _FakeProcess(pid=4666)
    monkeypatch.setattr(
        subproc_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake_process,
    )
    tools = subproc_module.SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_processes=1,
    )

    result = tools.run_command("echo complete", cwd=str(tmp_path))

    assert result["success"] is True
    assert list((tmp_path / "processes").iterdir()) == []
