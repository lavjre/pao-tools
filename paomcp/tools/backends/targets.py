"""Target lifecycle facade (providers live in targets_* modules)."""

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
    DEFAULT_DOCKER_TIMEOUT_SEC,
    DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC,
    DEFAULT_LIBC_FETCH_PATHS,
    DEFAULT_MAX_TARGETS_PER_SESSION,
    DEFAULT_QEMU_LISTEN_TIMEOUT_SEC,
    DEFAULT_SSH_CONNECT_TIMEOUT_SEC,
    DEFAULT_TARGET_START_TIMEOUT_SEC,
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    DEFAULT_TCP_PROBE_TIMEOUT_SEC,
    PARSER_VERSION,
    TARGET_ATTACH_SCHEMA,
    TARGET_CAPABILITIES_SCHEMA,
    TARGET_CONNECT_SCHEMA,
    TARGET_CREATE_SCHEMA,
    TARGET_DISCONNECT_SCHEMA,
    TARGET_FETCH_LIBS_SCHEMA,
    TARGET_FILE_SYNC_SCHEMA,
    TARGET_LIST_SCHEMA,
    TARGET_START_SCHEMA,
    TARGET_STATUS_SCHEMA,
    TARGET_STOP_SCHEMA,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetProvider,
    TargetSpec,
    TargetState,
)
from paomcp.tools.backends.targets_docker import (
    DockerProvider,
    DockerRunner,
)
from paomcp.tools.backends.targets_gdbserver import (
    GdbserverProvider,
    TcpRemoteProvider,
)
from paomcp.tools.backends.targets_local import (
    AttachPidProvider,
    LocalProcessProvider,
    _pid_is_alive,
)
from paomcp.tools.backends.targets_qemu import (
    QemuSystemProvider,
    QemuUserProvider,
)
from paomcp.tools.backends.targets_ssh import (
    SshClientFactory,
    SshProvider,
)


class TargetRegistry:
    """Thread-safe, session-scoped, bounded target registry.

    Each target is keyed by its opaque ``target_id`` (a UUID hex slug);
    duplicates by ``name`` are permitted -- a name is a human label,
    not an identity.  A per-session cap prevents a runaway workflow
    from allocating unbounded processes on the box.
    """

    def __init__(
        self,
        *,
        max_targets_per_session: int = DEFAULT_MAX_TARGETS_PER_SESSION,
    ) -> None:
        if max_targets_per_session <= 0:
            raise ValueError("max_targets_per_session must be positive")
        self._max_per_session = max_targets_per_session
        self._targets: Dict[str, TargetHandle] = {}
        self._providers: Dict[TargetKind, TargetProvider] = {
            TargetKind.LOCAL_PROCESS: LocalProcessProvider(),
            TargetKind.ATTACH_PID: AttachPidProvider(),
            TargetKind.GDBSERVER: GdbserverProvider(),
            TargetKind.TCP_REMOTE: TcpRemoteProvider(),
            TargetKind.SSH: SshProvider(),
            TargetKind.DOCKER: DockerProvider(),
            TargetKind.QEMU_USER: QemuUserProvider(),
            TargetKind.QEMU_SYSTEM: QemuSystemProvider(),
        }
        self._lock = threading.RLock()
        self._closed = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_provider(self, provider: TargetProvider) -> None:
        """Attach a new provider implementation (provider extension point)."""
        self._providers[provider.kind] = provider

    def has_provider(self, kind: TargetKind) -> bool:
        return kind in self._providers

    def provider_for(self, kind: TargetKind) -> TargetProvider:
        provider = self._providers.get(kind)
        if provider is None:
            raise TargetError(
                f"no provider registered for kind '{kind.value}'; "
                "this build does not ship it yet"
            )
        return provider

    # ------------------------------------------------------------------
    # Session accounting
    # ------------------------------------------------------------------
    def _count_for_session(self, session_id: Optional[str]) -> int:
        return sum(
            1
            for handle in self._targets.values()
            if handle.spec.session_id == session_id
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def create(self, spec: TargetSpec) -> TargetHandle:
        with self._lock:
            if self._closed:
                raise TargetError("target registry is closed")
            if not self.has_provider(spec.kind):
                raise TargetError(
                    f"provider for kind '{spec.kind.value}' is not registered"
                )
            if self._count_for_session(spec.session_id) >= self._max_per_session:
                raise TargetError(
                    f"session '{spec.session_id}' already owns "
                    f"{self._max_per_session} targets (cap)"
                )
            handle = TargetHandle(
                target_id=uuid.uuid4().hex[:12],
                kind=spec.kind,
                spec=spec,
            )
            self._targets[handle.target_id] = handle
            return handle

    def get(self, target_id: str) -> TargetHandle:
        with self._lock:
            handle = self._targets.get(target_id)
            if handle is None:
                raise TargetError(f"target '{target_id}' not found")
            return handle

    def list(self, *, session_id: Optional[str] = None) -> List[TargetHandle]:
        with self._lock:
            handles = list(self._targets.values())
        if session_id is None:
            return handles
        return [h for h in handles if h.spec.session_id == session_id]

    def remove(self, target_id: str) -> Optional[TargetHandle]:
        with self._lock:
            return self._targets.pop(target_id, None)

    def close_session(
        self,
        session_id: str,
        *,
        stop_timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Stop + drop every target owned by a session.

        Errors are collected (never re-raised) so a caller's own
        cleanup is not aborted by a stubborn subprocess.
        """
        with self._lock:
            handles = [
                handle
                for handle in self._targets.values()
                if handle.spec.session_id == session_id
            ]
            for handle in handles:
                self._targets.pop(handle.target_id, None)

        stopped: List[str] = []
        errors: List[Dict[str, Any]] = []
        for handle in handles:
            try:
                provider = self.provider_for(handle.kind)
                provider.stop(handle, timeout_sec=stop_timeout_sec)
                stopped.append(handle.target_id)
            except Exception as exc:
                errors.append(
                    {
                        "target_id": handle.target_id,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )
        return {
            "session_id": session_id,
            "stopped": stopped,
            "errors": errors,
        }

    def close_all(
        self,
        *,
        stop_timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Release every retained target without aborting on one failure.

        Provider ``stop`` implementations encode the ownership boundary:
        locally spawned processes, containers, and QEMU instances are
        terminated; attach-PID and TCP-remote targets are only detached;
        SSH transports are closed.  The registry is drained atomically
        before invoking providers so every target retained when shutdown
        begins is processed exactly once and no failed provider can leave
        a stale registry entry behind.
        """
        with self._lock:
            self._closed = True
            handles = list(self._targets.values())
            self._targets.clear()

        closed: List[str] = []
        errors: List[Dict[str, Any]] = []
        for handle in handles:
            try:
                provider = self.provider_for(handle.kind)
                provider.stop(handle, timeout_sec=stop_timeout_sec)
                closed.append(handle.target_id)
            except Exception as exc:
                errors.append(
                    {
                        "target_id": handle.target_id,
                        "kind": handle.kind.value,
                        "error": str(exc),
                        "type": type(exc).__name__,
                    }
                )

        return {
            "processed": len(handles),
            "closed": closed,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # High-level typed operations (used by the tool layer)
    # ------------------------------------------------------------------
    def create_target_response(
        self,
        *,
        kind: str,
        config: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        spec = TargetSpec.from_config(kind, config, session_id=session_id)
        handle = self.create(spec)
        return build_backend_response(
            "create_target",
            parsed={
                "schema": TARGET_CREATE_SCHEMA,
                "target": handle.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def list_targets_response(
        self, *, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        handles = self.list(session_id=session_id)
        return build_backend_response(
            "list_targets",
            parsed={
                "schema": TARGET_LIST_SCHEMA,
                "session_id": session_id,
                "targets": [h.to_dict() for h in handles],
                "count": len(handles),
                "max_per_session": self._max_per_session,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    @staticmethod
    def _recover_handle(handle: TargetHandle) -> None:
        """Restore a handle from ERROR to its live state after recovery.

        The state before the failure is captured in ``_record_error``
        (``metadata["pre_error_state"]``); an unrecoverable prior state
        (created/detached/stopped/dead) resolves to ``running`` because the
        remote is demonstrably up again.
        """
        prior = handle.metadata.get("pre_error_state")
        if prior in (
            TargetState.RUNNING.value,
            TargetState.ATTACHED.value,
            TargetState.CONNECTED.value,
        ):
            restored = TargetState(prior)
        else:
            restored = TargetState.RUNNING
        handle._set_state(restored)
        handle._clear_error()

    def get_status_response(self, *, target_id: str) -> Dict[str, Any]:
        handle = self.get(target_id)
        with handle._lock:
            # For process-owned targets, refresh liveness from the OS.
            if handle.process is not None:
                poll = handle.process.poll()
                if poll is not None and handle.state not in (
                    TargetState.STOPPED,
                    TargetState.DEAD,
                    TargetState.ERROR,
                ):
                    handle._set_state(TargetState.STOPPED)
                    handle.stopped_at = datetime.now(timezone.utc).isoformat()
                    handle.metadata["exit_code"] = poll
            elif handle.kind == TargetKind.ATTACH_PID and handle.pid is not None:
                if not _pid_is_alive(handle.pid) and handle.state not in (
                    TargetState.STOPPED,
                    TargetState.DEAD,
                    TargetState.DETACHED,
                ):
                    handle._set_state(TargetState.DEAD)
                    handle.stopped_at = datetime.now(timezone.utc).isoformat()
            # Recovery re-poll for remote targets: a transient start/attach
            # error must not pin an ERROR state (and stale error string)
            # forever once the underlying process / container / ssh transport
            # is reachable again.
            if handle.state == TargetState.ERROR:
                provider = self.provider_for(handle.kind)
                try:
                    alive = provider.check_alive(handle)
                except Exception:
                    alive = None
                if alive is True:
                    self._recover_handle(handle)
            payload = handle.to_dict()
        return build_backend_response(
            "get_target_status",
            parsed={
                "schema": TARGET_STATUS_SCHEMA,
                "target": payload,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def get_capabilities_response(self, *, target_id: str) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        capabilities = provider.get_capabilities(handle)
        return build_backend_response(
            "get_target_capabilities",
            parsed={
                "schema": TARGET_CAPABILITIES_SCHEMA,
                "target_id": target_id,
                "kind": handle.kind.value,
                "capabilities": capabilities,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def start_target_response(self, *, target_id: str) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        try:
            provider.start(handle)
            error: Optional[str] = None
        except Exception as exc:
            handle._record_error(exc)
            error = str(exc)
        return build_backend_response(
            "start_target",
            parsed={
                "schema": TARGET_START_SCHEMA,
                "target": handle.to_dict(),
                "success": error is None,
                "error": error,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def stop_target_response(
        self,
        *,
        target_id: str,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
        remove: bool = False,
    ) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        error: Optional[str] = None
        try:
            provider.stop(handle, timeout_sec=timeout_sec)
        except Exception as exc:
            handle._record_error(exc)
            error = str(exc)
        payload = handle.to_dict()
        if remove and error is None:
            self.remove(target_id)
        return build_backend_response(
            "stop_target",
            parsed={
                "schema": TARGET_STOP_SCHEMA,
                "target": payload,
                "success": error is None,
                "error": error,
                "removed": remove and error is None,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def connect_target_response(self, *, target_id: str) -> Dict[str, Any]:
        """Alias for ``start`` semantically -- a "connect" verb for remote kinds."""
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        error: Optional[str] = None
        try:
            provider.start(handle)
        except Exception as exc:
            handle._record_error(exc)
            error = str(exc)
        return build_backend_response(
            "connect_target",
            parsed={
                "schema": TARGET_CONNECT_SCHEMA,
                "target": handle.to_dict(),
                "success": error is None,
                "error": error,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def disconnect_target_response(
        self,
        *,
        target_id: str,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Alias for ``stop`` for remote kinds; keeps registry slot occupied."""
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        error: Optional[str] = None
        try:
            provider.stop(handle, timeout_sec=timeout_sec)
        except Exception as exc:
            handle._record_error(exc)
            error = str(exc)
        return build_backend_response(
            "disconnect_target",
            parsed={
                "schema": TARGET_DISCONNECT_SCHEMA,
                "target": handle.to_dict(),
                "success": error is None,
                "error": error,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def reconnect_target_response(
        self,
        *,
        target_id: str,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Stop then re-start the same target in place."""
        self.disconnect_target_response(target_id=target_id, timeout_sec=timeout_sec)
        return self.connect_target_response(target_id=target_id)

    def sync_target_files_response(
        self,
        *,
        target_id: str,
        source: str,
        destination: str,
        direction: str = "download",
    ) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        sync_fn = getattr(provider, "sync_file", None)
        if not callable(sync_fn):
            raise TargetError(
                f"provider '{handle.kind.value}' does not support file sync"
            )
        payload = sync_fn(
            handle, source=source, destination=destination, direction=direction
        )
        return build_backend_response(
            "sync_target_files",
            parsed={
                "schema": TARGET_FILE_SYNC_SCHEMA,
                "target_id": target_id,
                "kind": handle.kind.value,
                "result": payload,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def fetch_runtime_libraries_response(
        self,
        *,
        target_id: str,
        destination_dir: str,
        candidate_paths: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        fetch_fn = getattr(provider, "fetch_runtime_libraries", None)
        if not callable(fetch_fn):
            raise TargetError(
                f"provider '{handle.kind.value}' does not support "
                "fetch_runtime_libraries"
            )
        payload = fetch_fn(
            handle,
            destination_dir=destination_dir,
            candidate_paths=(
                list(candidate_paths)
                if candidate_paths is not None
                else DEFAULT_LIBC_FETCH_PATHS
            ),
        )
        return build_backend_response(
            "fetch_runtime_libraries",
            parsed={
                "schema": TARGET_FETCH_LIBS_SCHEMA,
                "target_id": target_id,
                "kind": handle.kind.value,
                "result": payload,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def attach_debugger_response(
        self, *, target_id: str, gdb: GdbController
    ) -> Dict[str, Any]:
        handle = self.get(target_id)
        provider = self.provider_for(handle.kind)
        error: Optional[str] = None
        attach_payload: Optional[Dict[str, Any]] = None
        try:
            attach_payload = provider.attach_debugger(handle, gdb)
        except Exception as exc:
            handle._record_error(exc)
            error = str(exc)
        return build_backend_response(
            "attach_debugger",
            parsed={
                "schema": TARGET_ATTACH_SCHEMA,
                "target": handle.to_dict(),
                "attach": attach_payload,
                "success": error is None
                and (
                    (attach_payload or {}).get("attach", {}).get("success", False)
                    if attach_payload is not None
                    else False
                ),
                "error": error,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = [
    "TargetKind",
    "TargetState",
    "TargetError",
    "TargetSpec",
    "TargetHandle",
    "TargetProvider",
    "TargetRegistry",
    "LocalProcessProvider",
    "AttachPidProvider",
    "GdbserverProvider",
    "TcpRemoteProvider",
    "SshProvider",
    "SshClientFactory",
    "DockerProvider",
    "DockerRunner",
    "QemuUserProvider",
    "QemuSystemProvider",
    "DEFAULT_DOCKER_TIMEOUT_SEC",
    "DEFAULT_QEMU_LISTEN_TIMEOUT_SEC",
    "DEFAULT_SSH_CONNECT_TIMEOUT_SEC",
    "DEFAULT_LIBC_FETCH_PATHS",
    "DEFAULT_MAX_TARGETS_PER_SESSION",
    "DEFAULT_TARGET_START_TIMEOUT_SEC",
    "DEFAULT_TARGET_STOP_TIMEOUT_SEC",
    "DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC",
    "DEFAULT_TCP_PROBE_TIMEOUT_SEC",
    "TARGET_CREATE_SCHEMA",
    "TARGET_LIST_SCHEMA",
    "TARGET_STATUS_SCHEMA",
    "TARGET_CAPABILITIES_SCHEMA",
    "TARGET_START_SCHEMA",
    "TARGET_STOP_SCHEMA",
    "TARGET_ATTACH_SCHEMA",
    "TARGET_CONNECT_SCHEMA",
    "TARGET_DISCONNECT_SCHEMA",
    "TARGET_FILE_SYNC_SCHEMA",
    "TARGET_FETCH_LIBS_SCHEMA",
]
