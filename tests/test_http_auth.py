import asyncio
import socket

import pytest

from paomcp.http.auth import (
    BearerAuthMiddleware,
    RemoteBindGateMiddleware,
    allow_remote_enabled,
    check_remote_bind,
    configured_token,
    is_loopback_address,
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


def test_configured_token_is_stripped(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PAO_MCP_TOKEN", "  s3cret\t")
    assert configured_token() == "s3cret"
    monkeypatch.setenv("PAO_ALLOW_REMOTE", "1")
    assert check_remote_bind("10.0.0.5", what="MCP server") == "s3cret"


def _fake_getaddrinfo(addresses):
    def _resolve(host, port, *args, **kwargs):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (addr, 0))
            for addr in addresses
        ]

    return _resolve


def test_loopback_host_resolves_aliases(monkeypatch):
    monkeypatch.setattr(
        "paomcp.http.auth.socket.getaddrinfo", _fake_getaddrinfo(["127.0.0.1", "::1"])
    )
    assert is_loopback_host("pao.internal") is True


def test_non_loopback_alias_is_not_loopback(monkeypatch):
    monkeypatch.setattr(
        "paomcp.http.auth.socket.getaddrinfo",
        _fake_getaddrinfo(["127.0.0.1", "10.0.0.5"]),
    )
    assert is_loopback_host("pao.internal") is False


def test_resolution_failure_fails_closed(monkeypatch):
    def _fail(host, port, *args, **kwargs):
        raise socket.gaierror("nxdomain")

    monkeypatch.setattr("paomcp.http.auth.socket.getaddrinfo", _fail)
    assert is_loopback_host("missing.internal") is False


def test_literal_loopback_hosts_skip_dns(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("literal loopback hosts must not hit DNS")

    monkeypatch.setattr("paomcp.http.auth.socket.getaddrinfo", _fail)
    for host in ("127.0.0.1", "::1", "localhost", "127.0.0.5"):
        assert is_loopback_host(host) is True


def test_is_loopback_address_handles_mapped_ipv4():
    assert is_loopback_address("127.0.0.1") is True
    assert is_loopback_address("::1") is True
    assert is_loopback_address("::ffff:127.0.0.1") is True
    assert is_loopback_address("10.0.0.5") is False
    assert is_loopback_address(None) is False


def _run_middleware(mw, path="/mcp", headers=None, client=("127.0.0.1", 1234)):
    seen = {}
    inner = mw.app

    async def recording_app(scope, receive, send):
        # Wrap the middleware's terminal app so tests can tell whether an
        # authorized request reached it instead of being rejected with 401.
        seen["reached"] = True
        await inner(scope, receive, send)

    mw.app = recording_app

    async def receive():
        return {"type": "http.request", "body": b""}

    sent = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "path": path,
        "headers": headers or [],
        "client": client,
    }
    asyncio.run(mw(scope, receive, send))
    return seen, sent


async def _ok_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_middleware_rejects_missing_and_wrong_tokens():
    mw = BearerAuthMiddleware(_ok_app, "s3cret")
    seen, sent = _run_middleware(mw)
    assert seen == {}
    assert sent[0]["status"] == 401
    seen, sent = _run_middleware(mw, headers=[(b"authorization", b"Bearer wrong")])
    assert seen == {}
    assert sent[0]["status"] == 401


def test_middleware_accepts_correct_token_and_opens_healthz():
    mw = BearerAuthMiddleware(_ok_app, "s3cret")
    seen, sent = _run_middleware(mw, headers=[(b"authorization", b"Bearer s3cret")])
    assert sent[0]["status"] == 200
    # /healthz stays open for orchestrator probes.
    seen, sent = _run_middleware(mw, path="/healthz")
    assert sent[0]["status"] == 200


def test_middleware_non_ascii_credential_is_401_not_500():
    mw = BearerAuthMiddleware(_ok_app, "s3cret")
    headers = [(b"authorization", b"Bearer tok\xe9n")]
    seen, sent = _run_middleware(mw, headers=headers)
    assert seen == {}
    assert sent[0]["status"] == 401


def test_bearer_middleware_accepts_stripped_token(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("PAO_MCP_TOKEN", " s3cret ")
    mw = BearerAuthMiddleware(_ok_app, configured_token())
    seen, sent = _run_middleware(mw, headers=[(b"authorization", b"Bearer s3cret")])
    assert seen == {"reached": True}
    assert sent[0]["status"] == 200


class TestRemoteBindGate:
    def test_loopback_client_needs_no_token(self, monkeypatch):
        _clear_env(monkeypatch)
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(mw, client=("127.0.0.1", 5000))
        assert seen == {"reached": True}
        assert sent[0]["status"] == 200

    def test_remote_client_denied_without_allow_remote(self, monkeypatch):
        _clear_env(monkeypatch)
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(mw, client=("203.0.113.7", 5000))
        assert seen == {}
        assert sent[0]["status"] == 401

    def test_remote_client_denied_without_token(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("PAO_ALLOW_REMOTE", "1")
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(
            mw,
            client=("203.0.113.7", 5000),
            headers=[(b"authorization", b"Bearer guess")],
        )
        assert seen == {}
        assert sent[0]["status"] == 401

    def test_remote_client_allowed_with_valid_token(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("PAO_ALLOW_REMOTE", "1")
        monkeypatch.setenv("PAO_MCP_TOKEN", "s3cret")
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(
            mw,
            client=("203.0.113.7", 5000),
            headers=[(b"authorization", b"Bearer s3cret")],
        )
        assert seen == {"reached": True}
        assert sent[0]["status"] == 200

    def test_remote_client_healthz_stays_open(self, monkeypatch):
        _clear_env(monkeypatch)
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(mw, path="/healthz", client=("203.0.113.7", 5000))
        assert seen == {"reached": True}
        assert sent[0]["status"] == 200

    def test_non_ascii_credential_is_401_not_500(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("PAO_ALLOW_REMOTE", "1")
        monkeypatch.setenv("PAO_MCP_TOKEN", "s3cret")
        mw = RemoteBindGateMiddleware(_ok_app)
        seen, sent = _run_middleware(
            mw,
            client=("203.0.113.7", 5000),
            headers=[(b"authorization", b"Bearer \xfe\xff")],
        )
        assert seen == {}
        assert sent[0]["status"] == 401
