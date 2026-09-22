import socket
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
import uvicorn

from paomcp.asgi import app
from paomcp.http.auth import RemoteBindGateMiddleware
from paomcp.server import create_mcp


class _StubStatus:
    def __init__(self, payload):
        self._payload = payload

    def status(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _StubSessions:
    def status(self):
        return {"active": 0, "occupied": 0, "available": 4, "max_sessions": 4}


class _StubServices:
    def __init__(
        self,
        default_session_id="default",
        decompiler_ready=True,
        ida_ready=True,
        registry=None,
    ):
        self.default_session_id = default_session_id
        self.decompiler = _StubStatus({"ready": decompiler_ready})
        self.ida = _StubStatus({"ready": ida_ready})
        self.session_registry = registry or _StubSessions()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _running_http_app(asgi_app):
    port = _free_port()
    config = uvicorn.Config(
        asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, name="healthz-test-server", daemon=True
    )
    thread.start()
    deadline = time.time() + 15
    while not server.started and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("uvicorn test server failed to start")
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_healthz_route_registered():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/healthz" in paths


def test_asgi_app_installs_request_time_remote_bind_gate():
    middleware_classes = {entry.cls for entry in app.user_middleware}
    assert RemoteBindGateMiddleware in middleware_classes


@pytest.mark.asyncio
async def test_healthz_returns_ok():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_without_services_is_not_ready():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["reasons"]


def test_readyz_with_explicit_services_over_real_uvicorn():
    mcp = create_mcp(services=_StubServices())
    http_app = mcp.http_app(path="/mcp")
    with _running_http_app(http_app) as port:
        response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=10)

    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is True
    assert body["checks"]["debugger_ready"] is True
    assert body["checks"]["sessions"]["max_sessions"] == 4


def test_readyz_resolves_services_from_fastmcp_lifespan(monkeypatch):
    services = _StubServices()
    monkeypatch.setattr("paomcp.lifespan.create_services", lambda: services)
    monkeypatch.setattr("paomcp.lifespan.close_services", lambda _services: None)
    mcp = create_mcp()
    http_app = mcp.http_app(path="/mcp")
    with _running_http_app(http_app) as port:
        response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=10)

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_readyz_degrades_when_default_session_missing():
    mcp = create_mcp(services=_StubServices(default_session_id=None))
    http_app = mcp.http_app(path="/mcp")
    with _running_http_app(http_app) as port:
        response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=10)

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert "no default debug session (debugger unavailable)" in body["reasons"]


def test_readyz_does_not_echo_raw_exception_text():
    services = _StubServices(registry=_StubStatus(RuntimeError("boom-secret-text")))
    mcp = create_mcp(services=services)
    http_app = mcp.http_app(path="/mcp")
    with _running_http_app(http_app) as port:
        response = httpx.get(f"http://127.0.0.1:{port}/readyz", timeout=10)

    assert response.status_code == 503
    body = response.json()
    assert body["ready"] is False
    assert body["reasons"] == ["readiness check failed"]
    assert "boom-secret-text" not in response.text
