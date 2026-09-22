"""Regression tests for runtime wiring: service lifetime, shutdown drain, profiles."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from paomcp.runtime import (
    AttachServerHandle,
    _stop_attach_server,
    run_http,
    run_stdio,
)


class _FakeUvicornServer:
    def __init__(self) -> None:
        self.should_exit = False
        self.force_exit = False


def _fake_mcp_run(monkeypatch, captured, *, interrupt=False):
    def _create_mcp(services=None, profile=None):
        captured["profile"] = profile

        class _MCP:
            def run(self, *args, **kwargs):
                captured["run_kwargs"] = kwargs
                # run_http catches KeyboardInterrupt; run_stdio relies on its
                # finally block, so only use interrupt where it is caught.
                if interrupt:
                    raise KeyboardInterrupt
                return None

        return _MCP()

    monkeypatch.setattr("paomcp.runtime.create_mcp", _create_mcp)


def test_run_http_closes_services_when_attach_server_fails(monkeypatch):
    events = []
    services = SimpleNamespace()

    monkeypatch.setattr("paomcp.runtime.create_services", lambda: services)
    monkeypatch.setattr("paomcp.runtime.create_attach_app", lambda _s: object())
    monkeypatch.setattr(
        "paomcp.runtime.close_services", lambda _s: events.append("closed")
    )
    monkeypatch.setattr(
        "paomcp.runtime._start_attach_server",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("attach boom")),
    )
    _fake_mcp_run(monkeypatch, {})

    with pytest.raises(RuntimeError, match="attach boom"):
        run_http(host="127.0.0.1", attach_host="127.0.0.1")

    assert events == ["closed"]


def test_run_http_defers_close_when_attach_handlers_do_not_drain(monkeypatch):
    events = []
    services = SimpleNamespace()

    monkeypatch.setattr("paomcp.runtime.create_services", lambda: services)
    monkeypatch.setattr("paomcp.runtime.create_attach_app", lambda _s: object())
    monkeypatch.setattr(
        "paomcp.runtime.close_services", lambda _s: events.append("closed")
    )
    monkeypatch.setattr("paomcp.runtime._start_attach_server", lambda *a, **k: None)
    monkeypatch.setattr("paomcp.runtime._stop_attach_server", lambda _h: False)
    _fake_mcp_run(monkeypatch, {}, interrupt=True)

    run_http(host="127.0.0.1", attach_host="127.0.0.1")

    assert events == []


def test_run_http_closes_services_after_clean_drain(monkeypatch):
    events = []
    services = SimpleNamespace()

    monkeypatch.setattr("paomcp.runtime.create_services", lambda: services)
    monkeypatch.setattr("paomcp.runtime.create_attach_app", lambda _s: object())
    monkeypatch.setattr(
        "paomcp.runtime.close_services", lambda _s: events.append("closed")
    )
    monkeypatch.setattr("paomcp.runtime._start_attach_server", lambda *a, **k: None)
    monkeypatch.setattr("paomcp.runtime._stop_attach_server", lambda _h: True)
    _fake_mcp_run(monkeypatch, {}, interrupt=True)

    run_http(host="127.0.0.1", attach_host="127.0.0.1")

    assert events == ["closed"]


def test_stop_attach_server_waits_for_handler_thread_to_finish():
    release = threading.Event()
    started = threading.Event()

    def _handler():
        started.set()
        release.wait(5)

    thread = threading.Thread(target=_handler, daemon=True)
    thread.start()
    assert started.wait(5)

    release.set()
    handle = AttachServerHandle(server=_FakeUvicornServer(), thread=thread)
    assert _stop_attach_server(handle, drain_timeout=2.0) is True
    assert handle.server.should_exit is True
    assert handle.server.force_exit is False


def test_stop_attach_server_reports_wedged_thread_after_bounded_force():
    release = threading.Event()

    def _handler():
        release.wait(10)

    thread = threading.Thread(target=_handler, daemon=True)
    thread.start()

    handle = AttachServerHandle(server=_FakeUvicornServer(), thread=thread)
    try:
        assert (
            _stop_attach_server(handle, drain_timeout=0.05, force_timeout=0.05) is False
        )
        assert handle.server.force_exit is True
    finally:
        release.set()
        thread.join(timeout=5)


def test_stop_attach_server_without_handle_is_clean():
    assert _stop_attach_server(None) is True


@pytest.mark.parametrize(
    ("env_profile", "expected"),
    [("auto", None), ("full", None), ("rev", "rev"), (" PWN ", "pwn")],
)
def test_run_http_passes_validated_profile_to_create_mcp(
    monkeypatch, env_profile, expected
):
    captured = {}
    services = SimpleNamespace()

    monkeypatch.setenv("PAO_PROFILE", env_profile)
    monkeypatch.setattr("paomcp.runtime.create_services", lambda: services)
    monkeypatch.setattr("paomcp.runtime.create_attach_app", lambda _s: object())
    monkeypatch.setattr("paomcp.runtime.close_services", lambda _s: None)
    monkeypatch.setattr("paomcp.runtime._start_attach_server", lambda *a, **k: None)
    monkeypatch.setattr("paomcp.runtime._stop_attach_server", lambda _h: True)
    _fake_mcp_run(monkeypatch, captured, interrupt=True)

    run_http(host="127.0.0.1", attach_host="127.0.0.1")

    assert captured["profile"] == expected


def test_run_stdio_passes_validated_profile_and_closes_services(monkeypatch):
    captured = {}
    events = []
    services = SimpleNamespace()

    monkeypatch.setenv("PAO_PROFILE", "auto")
    monkeypatch.setattr("paomcp.runtime.create_services", lambda: services)
    monkeypatch.setattr(
        "paomcp.runtime.close_services", lambda _s: events.append("closed")
    )
    _fake_mcp_run(monkeypatch, captured)

    run_stdio()

    assert captured["profile"] is None
    assert events == ["closed"]
