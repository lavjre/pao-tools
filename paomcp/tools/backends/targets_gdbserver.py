"""GDB-server and TCP-remote providers (split from targets.py)."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from paomcp.tools.backends.foundation import build_backend_response, parse_int
from paomcp.tools.backends.gdb import GdbController

from paomcp.tools.backends.targets_base import (
    DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC,
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    DEFAULT_TCP_PROBE_TIMEOUT_SEC,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetState,
)
from paomcp.tools.backends.targets_local import (
    LocalProcessProvider,
)


def _find_free_tcp_port(host: str = "127.0.0.1") -> int:
    """Return an ephemeral TCP port the kernel just handed us.

    Race-prone by nature -- another process could grab the port before
    gdbserver binds it -- so callers that care must pass an explicit
    port instead of relying on this.  Good enough for local test
    fixtures and interactive workflows.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _wait_for_tcp_listen(
    host: str, port: int, *, timeout_sec: float, poll_interval_sec: float = 0.05
) -> bool:
    """Poll until ``host:port`` accepts a connection or ``timeout_sec`` elapses."""
    import socket

    deadline = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=poll_interval_sec):
                return True
        except (OSError, ConnectionRefusedError):
            time.sleep(poll_interval_sec)
    return False


def _attach_via_gdb_remote(
    gdb: GdbController, host: str, port: int, *, extended: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Common ``-target-select remote host:port`` helper.

    Both GdbserverProvider (we own the gdbserver) and TcpRemoteProvider
    (we merely connect to one) end up here, so the MI command shape is
    kept in exactly one place.
    """
    command = "-target-select "
    command += "extended-remote " if extended else "remote "
    command += f"{host}:{port}"
    result = gdb.execute_mi_command(command)
    context: List[Dict[str, Any]] = []
    if result.get("success"):
        try:
            context.append(gdb.get_context("backtrace"))
        except Exception:
            pass
    return result, context


class GdbserverProvider:
    """Spawn a local gdbserver, expose an inferior via gdb remote protocol.

    Two shapes are supported:

    * spawn a fresh binary   -- ``binary_path`` (+ optional ``argv``)
    * attach an existing pid -- ``pid`` (uses ``gdbserver --attach``)

    ``host`` defaults to ``127.0.0.1``; ``port`` is picked from the
    ephemeral range when the spec omits it.  ``start`` blocks until
    the port is accepting connections (bounded by
    :data:`DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC`) so the caller never
    races the child.  ``attach_debugger`` issues
    ``-target-select remote host:port`` via the session's own
    ``GdbController``.
    """

    kind = TargetKind.GDBSERVER
    _executable = "gdbserver"

    def __init__(
        self,
        *,
        listen_timeout_sec: float = DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC,
        executable: Optional[str] = None,
    ) -> None:
        self._listen_timeout_sec = listen_timeout_sec
        if executable:
            self._executable = executable

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            if handle.state in (TargetState.RUNNING, TargetState.CONNECTED):
                raise TargetError(f"target '{handle.target_id}' is already running")
            spec = handle.spec
            host = spec.host or "127.0.0.1"
            port = spec.port or _find_free_tcp_port(host)
            argv: List[str] = [self._executable]
            if spec.pid is not None:
                argv += ["--attach", f"{host}:{port}", str(spec.pid)]
                handle.pid = spec.pid
            else:
                if not spec.binary_path:
                    raise TargetError("gdbserver spec has neither binary_path nor pid")
                binary = os.path.abspath(spec.binary_path)
                if not os.path.isfile(binary):
                    raise TargetError(f"binary '{binary}' does not exist")
                argv += [f"{host}:{port}", binary] + list(spec.argv or [])
            handle._set_state(TargetState.STARTING)
            env_full = dict(os.environ)
            env_full.update(spec.env or {})
            try:
                handle.process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=spec.working_dir or None,
                    env=env_full,
                    start_new_session=True,
                )
                handle.host = host
                handle.port = port
                handle.started_at = datetime.now(timezone.utc).isoformat()
                handle.metadata.setdefault(
                    "start", {"argv": argv, "cwd": spec.working_dir}
                )
                if not _wait_for_tcp_listen(
                    host, port, timeout_sec=self._listen_timeout_sec
                ):
                    # gdbserver did not bind in time; surface an honest
                    # error and mark the target dead so a caller
                    # rebuilds rather than blindly retrying attach.
                    raise TargetError(
                        f"gdbserver did not begin listening on {host}:{port} "
                        f"within {self._listen_timeout_sec:.1f}s"
                    )
                handle._set_state(TargetState.RUNNING)
            except Exception as exc:
                handle._record_error(exc)
                raise

    def stop(
        self,
        handle: TargetHandle,
        *,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> None:
        # Reuse LocalProcessProvider.stop semantics -- same subprocess
        # ownership contract.
        LocalProcessProvider().stop(handle, timeout_sec=timeout_sec)

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        with handle._lock:
            if handle.host is None or handle.port is None:
                raise TargetError(
                    f"target '{handle.target_id}' is not listening; "
                    "call start_target first"
                )
            attach_result, context = _attach_via_gdb_remote(
                gdb, handle.host, handle.port
            )
            if attach_result.get("success"):
                handle._set_state(TargetState.ATTACHED)
                handle.attached_at = datetime.now(timezone.utc).isoformat()
            else:
                handle._set_state(TargetState.ERROR)
                handle.error = attach_result.get("error") or "remote attach failed"
                handle.error_type = attach_result.get("type") or "RemoteAttachError"
            return {
                "attach": attach_result,
                "context": context,
                "state": handle.state.value,
                "host": handle.host,
                "port": handle.port,
            }

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
        return {
            "start": True,
            "stop": True,
            "attach_debugger": True,
            "file_sync": False,
            "reconnect": True,
            "port_forward": False,
            "notes": (
                "spawns a local gdbserver and attaches via "
                "'-target-select remote host:port'"
            ),
        }

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # Same subprocess-ownership contract as LocalProcessProvider.
        return handle.process is not None and handle.process.poll() is None


class TcpRemoteProvider:
    """Attach to an already-running gdbserver-style remote we do not own.

    ``start`` probes the endpoint (bounded by
    :data:`DEFAULT_TCP_PROBE_TIMEOUT_SEC`) so a mis-typed host:port
    fails fast rather than at attach time.  ``stop`` never kills the
    remote process -- we merely detach.
    """

    kind = TargetKind.TCP_REMOTE

    def __init__(
        self, *, probe_timeout_sec: float = DEFAULT_TCP_PROBE_TIMEOUT_SEC
    ) -> None:
        self._probe_timeout_sec = probe_timeout_sec

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            spec = handle.spec
            if not spec.host or spec.port is None:
                raise TargetError("tcp_remote spec requires 'host' and 'port'")
            if not _wait_for_tcp_listen(
                spec.host, spec.port, timeout_sec=self._probe_timeout_sec
            ):
                raise TargetError(
                    f"tcp_remote endpoint {spec.host}:{spec.port} did not "
                    f"accept a probe within {self._probe_timeout_sec:.1f}s"
                )
            handle.host = spec.host
            handle.port = spec.port
            handle.started_at = datetime.now(timezone.utc).isoformat()
            handle._set_state(TargetState.CONNECTED)

    def stop(
        self,
        handle: TargetHandle,
        *,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> None:
        # We never own the remote process; ``stop`` releases the slot.
        with handle._lock:
            handle._set_state(TargetState.DETACHED)
            handle.stopped_at = datetime.now(timezone.utc).isoformat()

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        with handle._lock:
            host = handle.host or handle.spec.host
            port = handle.port or handle.spec.port
            if host is None or port is None:
                raise TargetError(
                    f"tcp_remote target '{handle.target_id}' has no host / port"
                )
            attach_result, context = _attach_via_gdb_remote(gdb, str(host), int(port))
            if attach_result.get("success"):
                handle._set_state(TargetState.ATTACHED)
                handle.attached_at = datetime.now(timezone.utc).isoformat()
            else:
                handle._set_state(TargetState.ERROR)
                handle.error = attach_result.get("error") or "remote attach failed"
                handle.error_type = attach_result.get("type") or "RemoteAttachError"
            return {
                "attach": attach_result,
                "context": context,
                "state": handle.state.value,
                "host": host,
                "port": port,
            }

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
        return {
            "start": True,
            "stop": True,
            "attach_debugger": True,
            "file_sync": False,
            "reconnect": True,
            "port_forward": False,
            "notes": (
                "connects to an already-running gdbserver at host:port; "
                "stop() detaches but never kills the remote"
            ),
        }

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # We do not own a process -- liveness is whether host:port accepts
        # a connection again.
        host = handle.host or handle.spec.host
        port = handle.port or handle.spec.port
        if host is None or port is None:
            return None
        return _wait_for_tcp_listen(host, port, timeout_sec=self._probe_timeout_sec)
