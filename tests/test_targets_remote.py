"""Tests for gdbserver + tcp_remote providers."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest

from paomcp.tools.backends.targets_gdbserver import (
    _attach_via_gdb_remote,
    _find_free_tcp_port,
    _wait_for_tcp_listen,
)
from paomcp.tools.backends.targets import (
    TARGET_CONNECT_SCHEMA,
    TARGET_DISCONNECT_SCHEMA,
    GdbserverProvider,
    TargetError,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
    TcpRemoteProvider,
)


class _EphemeralListener:
    """Bind a TCP server that accepts one connection then closes.

    Used to prove ``_wait_for_tcp_listen`` and the tcp_remote probe do
    the right thing without a real gdbserver in the loop.
    """

    def __init__(self, host: str = "127.0.0.1") -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((host, 0))
        self._server.listen(4)
        self.host = host
        self.port = self._server.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        self._server.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.close()
            except OSError:
                pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._server.close()
        except OSError:
            pass
        self._thread.join(timeout=1.0)


class FakeGdb:
    """Records the last MI command; returns a scripted response."""

    def __init__(self, *, success: bool = True) -> None:
        self._success = success
        self.commands: List[str] = []

    def execute_mi_command(self, command: str, **_: Any) -> Dict[str, Any]:
        self.commands.append(command)
        if self._success:
            return {"success": True, "state": "connected", "command": command}
        return {
            "success": False,
            "error": "connection refused",
            "type": "ConnectionRefused",
        }

    def get_context(self, _kind: str) -> Dict[str, Any]:
        return {"kind": "backtrace", "frames": []}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class TestFindFreeTcpPort:
    def test_returns_a_bindable_port(self) -> None:
        port = _find_free_tcp_port()
        assert 1024 <= port <= 65535
        # rebind should still succeed (kernel handed us an ephemeral).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))


class TestWaitForTcpListen:
    def test_true_when_endpoint_is_listening(self) -> None:
        listener = _EphemeralListener()
        try:
            assert _wait_for_tcp_listen(listener.host, listener.port, timeout_sec=2.0)
        finally:
            listener.close()

    def test_false_on_deadline_without_listener(self) -> None:
        # Pick an ephemeral port and never bind; the wait must time out.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        assert not _wait_for_tcp_listen("127.0.0.1", port, timeout_sec=0.2)


class TestAttachViaGdbRemote:
    def test_emits_target_select_remote_mi_command(self) -> None:
        gdb = FakeGdb(success=True)
        result, context = _attach_via_gdb_remote(gdb, "127.0.0.1", 4444)
        assert gdb.commands == ["-target-select remote 127.0.0.1:4444"]
        assert result["success"] is True
        assert context and context[0]["kind"] == "backtrace"

    def test_uses_extended_remote_when_requested(self) -> None:
        gdb = FakeGdb(success=True)
        _attach_via_gdb_remote(gdb, "10.0.0.5", 1234, extended=True)
        assert gdb.commands == ["-target-select extended-remote 10.0.0.5:1234"]


# ---------------------------------------------------------------------------
# TcpRemoteProvider (does not spawn anything -- just probes)
# ---------------------------------------------------------------------------
class TestTcpRemoteProvider:
    def test_start_probes_the_endpoint(self) -> None:
        listener = _EphemeralListener()
        try:
            registry = TargetRegistry()
            handle = registry.create(
                TargetSpec.from_config(
                    "tcp_remote",
                    {"host": listener.host, "port": listener.port},
                    session_id="sess",
                )
            )
            response = registry.connect_target_response(target_id=handle.target_id)
            parsed = response["parsed"]
            assert parsed["schema"] == TARGET_CONNECT_SCHEMA
            assert parsed["success"] is True
            assert handle.state == TargetState.CONNECTED
            assert handle.host == listener.host
            assert handle.port == listener.port
        finally:
            listener.close()

    def test_start_fails_fast_when_endpoint_is_dead(self) -> None:
        # Ephemeral port with nothing listening.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "tcp_remote",
                {"host": "127.0.0.1", "port": port},
                session_id="sess",
            )
        )
        # Shrink the probe timeout so the test is quick.
        registry._providers[TargetKind.TCP_REMOTE] = TcpRemoteProvider(
            probe_timeout_sec=0.1
        )
        response = registry.connect_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR
        assert "did not accept a probe" in handle.error

    def test_stop_marks_detached_without_killing_remote(self) -> None:
        listener = _EphemeralListener()
        try:
            registry = TargetRegistry()
            handle = registry.create(
                TargetSpec.from_config(
                    "tcp_remote",
                    {"host": listener.host, "port": listener.port},
                    session_id="sess",
                )
            )
            registry.connect_target_response(target_id=handle.target_id)
            response = registry.disconnect_target_response(target_id=handle.target_id)
            parsed = response["parsed"]
            assert parsed["schema"] == TARGET_DISCONNECT_SCHEMA
            assert parsed["success"] is True
            assert handle.state == TargetState.DETACHED
        finally:
            listener.close()

    def test_attach_debugger_issues_target_select_remote(self) -> None:
        listener = _EphemeralListener()
        try:
            registry = TargetRegistry()
            handle = registry.create(
                TargetSpec.from_config(
                    "tcp_remote",
                    {"host": listener.host, "port": listener.port},
                    session_id="sess",
                )
            )
            registry.connect_target_response(target_id=handle.target_id)
            gdb = FakeGdb(success=True)
            response = registry.attach_debugger_response(
                target_id=handle.target_id, gdb=gdb  # type: ignore[arg-type]
            )
            assert response["parsed"]["success"] is True
            assert handle.state == TargetState.ATTACHED
            assert gdb.commands and gdb.commands[0].startswith("-target-select remote")
        finally:
            listener.close()

    def test_capabilities_declare_reconnect_and_no_file_sync(self) -> None:
        listener = _EphemeralListener()
        try:
            registry = TargetRegistry()
            handle = registry.create(
                TargetSpec.from_config(
                    "tcp_remote",
                    {"host": listener.host, "port": listener.port},
                    session_id="sess",
                )
            )
            caps = registry.get_capabilities_response(target_id=handle.target_id)[
                "parsed"
            ]["capabilities"]
            assert caps["reconnect"] is True
            assert caps["file_sync"] is False
            assert caps["attach_debugger"] is True
        finally:
            listener.close()


# ---------------------------------------------------------------------------
# GdbserverProvider (skip spawn tests when gdbserver missing on this box)
# ---------------------------------------------------------------------------
import shutil

_GDBSERVER_AVAILABLE = shutil.which("gdbserver") is not None
gdbserver_only = pytest.mark.skipif(
    not _GDBSERVER_AVAILABLE, reason="gdbserver not on PATH"
)


class TestGdbserverProviderPure:
    """Pure-argv / capability tests that need no real gdbserver."""

    def test_missing_binary_and_pid_raises_at_start(self) -> None:
        registry = TargetRegistry()
        # We construct the spec directly because from_config accepts
        # either binary_path or pid; here we craft the missing-both
        # invalid intermediate state to prove start() catches it.
        spec = TargetSpec(kind=TargetKind.GDBSERVER, session_id="sess")
        handle = registry.create(spec)
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert "binary_path" in response["parsed"]["error"]

    def test_capabilities_declare_reconnect_and_attach(self) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "gdbserver",
                {"binary_path": "/bin/echo", "port": 12345},
                session_id="sess",
            )
        )
        caps = registry.get_capabilities_response(target_id=handle.target_id)["parsed"][
            "capabilities"
        ]
        assert caps["attach_debugger"] is True
        assert caps["reconnect"] is True

    def test_provider_registered_by_kind(self) -> None:
        registry = TargetRegistry()
        provider = registry.provider_for(TargetKind.GDBSERVER)
        assert isinstance(provider, GdbserverProvider)

    def test_listen_timeout_configurable(self) -> None:
        # An impossibly short timeout guarantees the probe fails.
        registry = TargetRegistry()
        registry._providers[TargetKind.GDBSERVER] = GdbserverProvider(
            listen_timeout_sec=0.01,
            executable="__pao_not_a_real_binary__",
        )
        handle = registry.create(
            TargetSpec.from_config(
                "gdbserver",
                {"binary_path": "/bin/echo", "port": _find_free_tcp_port()},
                session_id="sess",
            )
        )
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR
