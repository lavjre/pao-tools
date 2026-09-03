"""SSH provider (split from targets.py)."""

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
    DEFAULT_LIBC_FETCH_PATHS,
    DEFAULT_SSH_CONNECT_TIMEOUT_SEC,
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetState,
)


class SshClientFactory(Protocol):
    """Injection point for tests -- returns something paramiko-shaped."""

    def __call__(self) -> Any: ...


def _default_ssh_client_factory() -> Any:
    """Return a real paramiko client that rejects unknown host keys."""
    import paramiko  # type: ignore[import-untyped]

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    return client


class SshProvider:
    """Manage an SSH session over paramiko and expose file-sync ops.

    ``attach_debugger`` is intentionally *not* supported directly on
    this provider -- remote debugging over SSH is properly done by
    booting a gdbserver on the remote box and using
    :class:`TcpRemoteProvider` (optionally through an SSH tunnel) to
    attach.  What ``SshProvider`` uniquely offers is typed file
    transfer (upload / download) and the ``fetch_runtime_libraries``
    convenience that pulls the target's libc / dynamic loader so an
    offline identify_libc has something to score against.

    The paramiko client is injected via ``client_factory`` so tests
    can swap in a fake without needing a real SSH server on the box.
    """

    kind = TargetKind.SSH

    def __init__(
        self,
        *,
        client_factory: SshClientFactory = _default_ssh_client_factory,
        connect_timeout_sec: float = DEFAULT_SSH_CONNECT_TIMEOUT_SEC,
    ) -> None:
        self._client_factory = client_factory
        self._connect_timeout_sec = connect_timeout_sec
        self._clients: Dict[str, Any] = {}
        self._clients_lock = threading.RLock()

    def _client_for(self, handle: TargetHandle) -> Any:
        with self._clients_lock:
            existing = self._clients.get(handle.target_id)
            if existing is not None:
                return existing
            spec = handle.spec
            client = self._client_factory()
            try:
                if spec.ssh_known_hosts_path:
                    known_hosts_path = os.path.abspath(spec.ssh_known_hosts_path)
                    if not os.path.isfile(known_hosts_path):
                        raise TargetError(
                            "ssh_known_hosts_path must name a readable "
                            "known_hosts file"
                        )
                    client.load_host_keys(known_hosts_path)
                kwargs: Dict[str, Any] = {
                    "hostname": spec.host,
                    "port": int(spec.port) if spec.port else 22,
                    "username": spec.ssh_user,
                    "timeout": self._connect_timeout_sec,
                    "allow_agent": False,
                    "look_for_keys": False,
                }
                if spec.ssh_key_path:
                    kwargs["key_filename"] = spec.ssh_key_path
                if spec.ssh_password:
                    kwargs["password"] = spec.ssh_password
                client.connect(**kwargs)
            except Exception:
                try:
                    client.close()
                except Exception:
                    pass
                raise
            self._clients[handle.target_id] = client
            return client

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            spec = handle.spec
            if not spec.host or not spec.ssh_user:
                raise TargetError("ssh spec requires 'host' and 'ssh_user'")
            if not (spec.ssh_key_path or spec.ssh_password):
                raise TargetError("ssh spec requires 'ssh_key_path' or 'ssh_password'")
            handle._set_state(TargetState.STARTING)
            try:
                client = self._client_for(handle)
                # Sanity-check the transport so a broken handshake fails
                # at start(), not at first exec.
                transport = (
                    client.get_transport() if hasattr(client, "get_transport") else None
                )
                if transport is not None and hasattr(transport, "is_active"):
                    if not transport.is_active():
                        raise TargetError("ssh transport is not active after connect")
                handle.host = spec.host
                handle.port = int(spec.port) if spec.port else 22
                handle.started_at = datetime.now(timezone.utc).isoformat()
                handle._set_state(TargetState.CONNECTED)
            except Exception as exc:
                handle._record_error(exc)
                raise

    def stop(
        self,
        handle: TargetHandle,
        *,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> None:
        with handle._lock:
            with self._clients_lock:
                client = self._clients.pop(handle.target_id, None)
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
            handle._set_state(TargetState.DETACHED)
            handle.stopped_at = datetime.now(timezone.utc).isoformat()

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        raise TargetError(
            "ssh provider does not attach a debugger directly; "
            "spawn gdbserver on the remote and use a tcp_remote target "
            "(optionally through an SSH tunnel) to attach"
        )

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
        return {
            "start": True,
            "stop": True,
            "attach_debugger": False,
            "file_sync": True,
            "reconnect": True,
            "port_forward": True,
            "notes": (
                "SSH transport for file sync + remote command exec; "
                "attach_debugger deferred to a tcp_remote target"
            ),
        }

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # _client_for() re-establishes a dropped transport, which is exactly
        # the recovery we want to detect; a failure to (re)connect means the
        # remote is not reachable right now.
        try:
            client = self._client_for(handle)
        except Exception:
            return False
        transport = client.get_transport() if hasattr(client, "get_transport") else None
        if transport is None:
            return False
        if hasattr(transport, "is_active"):
            return bool(transport.is_active())
        return True

    # ------------------------------------------------------------------
    # File sync
    # ------------------------------------------------------------------
    def sync_file(
        self,
        handle: TargetHandle,
        *,
        source: str,
        destination: str,
        direction: str = "download",
    ) -> Dict[str, Any]:
        if direction not in ("download", "upload"):
            raise TargetError(
                f"direction must be 'download' or 'upload', got {direction!r}"
            )
        client = self._client_for(handle)
        sftp = client.open_sftp()
        try:
            if direction == "download":
                local_dir = os.path.dirname(destination)
                if local_dir:
                    os.makedirs(local_dir, exist_ok=True)
                sftp.get(source, destination)
                info = os.stat(destination)
                return {
                    "direction": "download",
                    "remote_path": source,
                    "local_path": os.path.abspath(destination),
                    "bytes": info.st_size,
                }
            # upload
            sftp.put(source, destination)
            return {
                "direction": "upload",
                "local_path": os.path.abspath(source),
                "remote_path": destination,
                "bytes": os.path.getsize(source),
            }
        finally:
            try:
                sftp.close()
            except Exception:
                pass

    def fetch_runtime_libraries(
        self,
        handle: TargetHandle,
        *,
        destination_dir: str,
        candidate_paths: Sequence[str] = DEFAULT_LIBC_FETCH_PATHS,
    ) -> Dict[str, Any]:
        os.makedirs(destination_dir, exist_ok=True)
        client = self._client_for(handle)
        sftp = client.open_sftp()
        fetched: List[Dict[str, Any]] = []
        missing: List[str] = []
        try:
            for remote_path in candidate_paths:
                try:
                    sftp.stat(remote_path)
                except (FileNotFoundError, IOError):
                    missing.append(remote_path)
                    continue
                local_name = os.path.basename(remote_path)
                local_path = os.path.join(destination_dir, local_name)
                sftp.get(remote_path, local_path)
                fetched.append(
                    {
                        "remote_path": remote_path,
                        "local_path": os.path.abspath(local_path),
                        "bytes": os.path.getsize(local_path),
                    }
                )
        finally:
            try:
                sftp.close()
            except Exception:
                pass
        return {
            "destination_dir": os.path.abspath(destination_dir),
            "fetched": fetched,
            "missing": missing,
            "candidate_paths": list(candidate_paths),
        }
