import asyncio

import pytest

from paomcp.http.auth import (
    BearerAuthMiddleware,
    allow_remote_enabled,
    check_remote_bind,
    configured_token,
    is_loopback_host,
)


def _clear_env(monkeypatch):
    monkeypatch.delenv("PAO_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("PAO_MCP_TOKEN", raising=False)


def test_loopback_hosts_need_nothing(monkeypatch):
    _clear_env(monkeypatch)
    for host in ("127.0.0.1", "::1", "localhost", "LOCALHOST", " 127.0.0.1 "):
        assert is_loopback_host(host) is True
        assert check_remote_bind(host, what="test") is None
    assert is_loopback_host("0.0.0.0") is False
    assert is_loopback_host("") is False
    assert is_loopback_host(None) is False


def test_remote_bind_refused_by_default(monkeypatch):
    _clear_env(monkeypatch)
    assert allow_remote_enabled() is False
    with pytest.raises(RuntimeError, match="PAO_ALLOW_REMOTE"):
        check_remote_bind("0.0.0.0", what="MCP server")


def test_remote_bind_requires_token(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PAO_ALLOW_REMOTE", "1")
    assert allow_remote_enabled() is True
    assert configured_token() is None
    with pytest.raises(RuntimeError, match="PAO_MCP_TOKEN"):
        check_remote_bind("0.0.0.0", what="attach server")


def test_remote_bind_returns_token_when_allowed(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PAO_ALLOW_REMOTE", "yes")
    monkeypatch.setenv("PAO_MCP_TOKEN", "s3cret")
    assert configured_token() == "s3cret"
    assert check_remote_bind("192.168.1.10", what="MCP server") == "s3cret"


def _run_middleware(mw, path="/mcp", headers=None):
    seen = {}

    async def app(scope, receive, send):
        seen["reached"] = True

    async def receive():
        return {"type": "http.request", "body": b""}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "path": path,
        "headers": headers or [],
    }
    asyncio.run(mw(scope, receive, send))
    return seen, sent


def test_middleware_rejects_missing_and_wrong_tokens():
    mw = BearerAuthMiddleware(None, "s3cret")
    seen, sent = _run_middleware(mw)
    assert seen == {}
    assert sent[0]["status"] == 401
    seen, sent = _run_middleware(mw, headers=[(b"authorization", b"Bearer wrong")])
    assert seen == {}
    assert sent[0]["status"] == 401


def test_middleware_accepts_correct_token_and_opens_healthz():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    mw = BearerAuthMiddleware(app, "s3cret")
    seen, sent = _run_middleware(mw, headers=[(b"authorization", b"Bearer s3cret")])
    assert sent[0]["status"] == 200
    # /healthz stays open for orchestrator probes.
    seen, sent = _run_middleware(mw, path="/healthz")
    assert sent[0]["status"] == 200
