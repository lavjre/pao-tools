from paomcp.state import SessionState
from paomcp.state import registry as registry_module
from paomcp.utils.paths import build_runtime_paths

from tests.test_stage1_foundation import _FakeGdb, _FakePwndbgTools


def test_breakpoint_lifecycle():
    state = SessionState()

    bp = state.add_breakpoint(1, "0xdeadbeef")
    assert state.breakpoints[1] == bp
    assert bp.enabled is True

    assert state.toggle_breakpoint(1) is True
    assert state.breakpoints[1].enabled is False

    assert state.remove_breakpoint(1) is True
    assert 1 not in state.breakpoints
    assert state.remove_breakpoint(1) is False


def test_watch_lifecycle():
    state = SessionState()

    watch = state.add_watch("0x4141", size=32, format="string")
    assert watch in state.watches

    assert state.remove_watch("0x4141") is True
    assert state.watches == []


def test_to_dict_contains_expected_fields():
    state = SessionState()
    state.add_breakpoint(2, "0x1000", condition="eax==0")
    state.add_watch("0x2000")

    payload = state.to_dict()
    assert payload["binary_loaded"] is False
    assert payload["state"] == "idle"
    assert payload["breakpoints"][2]["address"] == "0x1000"
    assert payload["breakpoints"][2]["condition"] == "eax==0"
    assert payload["watches"][0]["address"] == "0x2000"


def _registry(tmp_path, monkeypatch, max_sessions=1):
    monkeypatch.setattr(registry_module, "GdbController", _FakeGdb)
    monkeypatch.setattr(registry_module, "PwndbgTools", _FakePwndbgTools)
    paths = build_runtime_paths(
        workspace_root=str(tmp_path / "workspace"),
        runtime_root=str(tmp_path / "runtime"),
    )
    return registry_module.DebugSessionRegistry(paths, max_sessions=max_sessions)


def test_close_session_keeps_tombstone_when_cleanup_fails_then_reaps(
    tmp_path, monkeypatch
):
    registry = _registry(tmp_path, monkeypatch)
    registry.create_session("alpha")
    attempts = []

    def cleanup(_session):
        attempts.append("cleanup")
        if len(attempts) == 1:
            return {"failure_count": 1, "failures": [{"pid": 4242}]}
        return {"failure_count": 0, "failures": []}

    result = registry.close_session("alpha", on_closing=cleanup)

    assert result["success"] is False
    assert result["code"] == "SESSION_CLEANUP_FAILED"
    # A leaked helper process must keep the slot reserved for a later reap.
    assert registry.status()["tombstones"] == 1
    assert attempts == ["cleanup"]

    assert registry.reap_tombstones() == 1
    assert attempts == ["cleanup", "cleanup"]
    assert registry.status()["tombstones"] == 0
    replacement = registry.create_session("alpha")
    assert replacement.session_id == "alpha"


def test_failed_cleanup_retries_are_bounded_and_release_capacity(tmp_path, monkeypatch):
    registry = _registry(tmp_path, monkeypatch)
    registry.create_session("alpha")
    attempts = []

    def cleanup(_session):
        attempts.append("cleanup")
        return {"success": False, "error": "helper process still alive"}

    result = registry.close_session("alpha", on_closing=cleanup)

    assert result["code"] == "SESSION_CLEANUP_FAILED"
    assert registry.status()["tombstones"] == 1

    # Each reap retries cleanup until the bounded budget is exhausted.
    assert registry.reap_tombstones() == 0
    assert registry.status()["tombstones"] == 1
    assert registry.reap_tombstones() == 1
    assert registry.status()["tombstones"] == 0
    assert len(attempts) == registry_module.MAX_CLEANUP_ATTEMPTS
    assert registry.create_session("alpha").session_id == "alpha"
