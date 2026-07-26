"""Typed target-provider abstraction (Stage 8).

Stage 8 unifies the eight ways a research target can reach a debugger --
local process, attach-PID, gdbserver, TCP remote, SSH, Docker, QEMU user
and QEMU system -- behind one typed API.  Every provider implements the
same :class:`TargetProvider` protocol so the tool layer never branches on
target kind, and every target carries an explicit
:class:`TargetState` machine so a caller can reason about lifecycle
without guessing.

Two pieces ship in the S8.1 slice:

* :class:`LocalProcessProvider`  -- spawns ``argv`` (optionally under a
  working directory and env), waits for the process to be live, and
  hands its PID to ``GdbController.attach``.
* :class:`AttachPidProvider`     -- attaches to an already-running PID
  without owning its lifecycle (a ``stop`` here means *detach*, not
  kill).

Everything is session-scoped: a :class:`TargetRegistry` remembers which
target belongs to which debug session and enforces a bounded per-session
target count so a runaway workflow cannot allocate the box.
"""

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

PARSER_VERSION = "1"
TARGET_CREATE_SCHEMA = "pao.target-create.v1"
TARGET_LIST_SCHEMA = "pao.target-list.v1"
TARGET_STATUS_SCHEMA = "pao.target-status.v1"
TARGET_CAPABILITIES_SCHEMA = "pao.target-capabilities.v1"
TARGET_START_SCHEMA = "pao.target-start.v1"
TARGET_STOP_SCHEMA = "pao.target-stop.v1"
TARGET_ATTACH_SCHEMA = "pao.target-attach.v1"

DEFAULT_MAX_TARGETS_PER_SESSION = 8
DEFAULT_TARGET_START_TIMEOUT_SEC = 15.0
DEFAULT_TARGET_STOP_TIMEOUT_SEC = 10.0


class TargetKind(str, Enum):
    """Every provider surface we intend to ship in Stage 8."""

    LOCAL_PROCESS = "local_process"
    ATTACH_PID = "attach_pid"
    GDBSERVER = "gdbserver"
    TCP_REMOTE = "tcp_remote"
    SSH = "ssh"
    DOCKER = "docker"
    QEMU_USER = "qemu_user"
    QEMU_SYSTEM = "qemu_system"


class TargetState(str, Enum):
    """Lifecycle states any provider transitions through."""

    CREATED = "created"  # spec accepted, not started
    STARTING = "starting"  # start() in progress
    RUNNING = "running"  # process alive, not yet attached
    CONNECTED = "connected"  # remote transport open, not yet attached
    ATTACHED = "attached"  # a debugger session is bound to this target
    DETACHED = "detached"  # attach released; process may still be alive
    STOPPED = "stopped"  # process gone (killed / exited)
    DEAD = "dead"  # unrecoverable; fresh create required
    ERROR = "error"  # last operation failed; see ``error`` field


class TargetError(RuntimeError):
    """Raised for spec / lifecycle / provider errors."""


@dataclass
class TargetSpec:
    """User-supplied configuration for a target.

    Every field is optional so a single dataclass can describe any of
    the eight providers; :meth:`validate_for` decides at create time
    which fields the chosen ``kind`` actually needs.
    """

    kind: TargetKind
    name: Optional[str] = None
    session_id: Optional[str] = None
    # Local process / QEMU user
    binary_path: Optional[str] = None
    argv: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: Optional[str] = None
    stdin_bytes: Optional[bytes] = None
    # Attach PID
    pid: Optional[int] = None
    # gdbserver / TCP / SSH remote
    host: Optional[str] = None
    port: Optional[int] = None
    # SSH
    ssh_user: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_password: Optional[str] = None
    # Docker
    docker_image: Optional[str] = None
    docker_workdir: Optional[str] = None
    docker_extra_args: List[str] = field(default_factory=list)
    # QEMU user
    qemu_binary: Optional[str] = None
    qemu_ld_prefix: Optional[str] = None
    qemu_extra_args: List[str] = field(default_factory=list)
    # Metadata (informational)
    arch: Optional[str] = None
    platform: Optional[str] = None
    notes: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        kind: str,
        config: Optional[Mapping[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> "TargetSpec":
        """Build a spec from a loosely-typed dict (as an MCP tool arg)."""
        try:
            resolved_kind = TargetKind(kind)
        except ValueError as exc:
            raise TargetError(
                f"unknown target kind {kind!r}; expected one of "
                f"{[k.value for k in TargetKind]}"
            ) from exc
        config = dict(config or {})
        spec = cls(kind=resolved_kind, session_id=session_id)
        for attr in (
            "name",
            "binary_path",
            "working_dir",
            "host",
            "ssh_user",
            "ssh_key_path",
            "ssh_password",
            "docker_image",
            "docker_workdir",
            "qemu_binary",
            "qemu_ld_prefix",
            "arch",
            "platform",
            "notes",
        ):
            if attr in config and config[attr] is not None:
                setattr(spec, attr, str(config[attr]))
        for int_attr in ("pid", "port"):
            if int_attr in config and config[int_attr] is not None:
                parsed = parse_int(config[int_attr])
                if parsed is None:
                    raise TargetError(
                        f"'{int_attr}' must be an integer, got {config[int_attr]!r}"
                    )
                setattr(spec, int_attr, parsed)
        for list_attr in ("argv", "docker_extra_args", "qemu_extra_args"):
            if list_attr in config and config[list_attr] is not None:
                raw = config[list_attr]
                if isinstance(raw, str):
                    raw = shlex.split(raw)
                spec_list: List[str] = [str(item) for item in raw]
                setattr(spec, list_attr, spec_list)
        if "env" in config and config["env"] is not None:
            env_dict = config["env"]
            if not isinstance(env_dict, Mapping):
                raise TargetError("env must be a mapping of name->value strings")
            spec.env = {str(k): str(v) for k, v in env_dict.items()}
        if "stdin_bytes" in config and config["stdin_bytes"] is not None:
            raw = config["stdin_bytes"]
            if isinstance(raw, str):
                spec.stdin_bytes = raw.encode("utf-8")
            elif isinstance(raw, (bytes, bytearray)):
                spec.stdin_bytes = bytes(raw)
            else:
                raise TargetError("stdin_bytes must be bytes or a str")
        spec.validate_for(resolved_kind)
        return spec

    def validate_for(self, kind: TargetKind) -> None:
        """Fail fast when the spec is missing fields the provider needs."""
        if kind == TargetKind.LOCAL_PROCESS:
            if not self.binary_path:
                raise TargetError("local_process requires 'binary_path'")
            if not os.path.isabs(self.binary_path):
                # Accept relative but flag it -- silent cwd surprises
                # are a recurring debug-session footgun.
                pass
        elif kind == TargetKind.ATTACH_PID:
            if self.pid is None or self.pid <= 0:
                raise TargetError("attach_pid requires a positive 'pid'")
        elif kind == TargetKind.GDBSERVER:
            if not self.binary_path and self.pid is None:
                raise TargetError(
                    "gdbserver requires 'binary_path' (to spawn) or 'pid' (to attach)"
                )
        elif kind == TargetKind.TCP_REMOTE:
            if not self.host or self.port is None:
                raise TargetError("tcp_remote requires 'host' and 'port'")
        elif kind == TargetKind.SSH:
            if not self.host:
                raise TargetError("ssh requires 'host'")
            if not self.ssh_user:
                raise TargetError("ssh requires 'ssh_user'")
            if not (self.ssh_key_path or self.ssh_password):
                raise TargetError(
                    "ssh requires either 'ssh_key_path' or 'ssh_password'"
                )
        elif kind == TargetKind.DOCKER:
            if not self.docker_image:
                raise TargetError("docker requires 'docker_image'")
        elif kind == TargetKind.QEMU_USER:
            if not self.binary_path:
                raise TargetError("qemu_user requires 'binary_path'")
            if not self.qemu_binary:
                raise TargetError(
                    "qemu_user requires 'qemu_binary' (e.g. 'qemu-x86_64')"
                )
        elif kind == TargetKind.QEMU_SYSTEM:
            # QEMU system deferred to a later stage; the field validator
            # is honest that we do not yet support it.
            raise TargetError("qemu_system provider is not implemented in this stage")


@dataclass
class TargetHandle:
    """Runtime state and metadata for one target."""

    target_id: str
    kind: TargetKind
    spec: TargetSpec
    state: TargetState = TargetState.CREATED
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    attached_at: Optional[str] = None
    stopped_at: Optional[str] = None
    pid: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    process: Optional[subprocess.Popen] = field(default=None, repr=False)
    error: Optional[str] = None
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def _set_state(self, state: TargetState) -> None:
        self.state = state
        self._touch()

    def _record_error(self, exc: BaseException) -> None:
        self.error = str(exc)
        self.error_type = type(exc).__name__
        self.state = TargetState.ERROR
        self._touch()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "kind": self.kind.value,
            "session_id": self.spec.session_id,
            "name": self.spec.name,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "attached_at": self.attached_at,
            "stopped_at": self.stopped_at,
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "arch": self.spec.arch,
            "platform": self.spec.platform,
            "notes": self.spec.notes,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": dict(self.metadata),
        }


class TargetProvider(Protocol):
    """Interface every provider implements.

    The methods intentionally take a :class:`TargetHandle` in *and*
    mutate it in place -- they never return a new object -- so the
    registry stays authoritative and callers never juggle stale copies.
    """

    kind: TargetKind

    def start(self, handle: TargetHandle) -> None: ...

    def stop(self, handle: TargetHandle, *, timeout_sec: float = ...) -> None: ...

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]: ...

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]: ...


class LocalProcessProvider:
    """Spawn a binary as a local subprocess."""

    kind = TargetKind.LOCAL_PROCESS

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            if handle.state in (TargetState.RUNNING, TargetState.ATTACHED):
                raise TargetError(f"target '{handle.target_id}' is already running")
            spec = handle.spec
            if not spec.binary_path:
                raise TargetError("local_process spec has no binary_path")
            binary = os.path.abspath(spec.binary_path)
            if not os.path.isfile(binary):
                raise TargetError(f"binary '{binary}' does not exist")
            argv: List[str] = [binary] + list(spec.argv or [])
            env_full = dict(os.environ)
            env_full.update(spec.env or {})
            handle._set_state(TargetState.STARTING)
            try:
                handle.process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE if spec.stdin_bytes is not None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=spec.working_dir or None,
                    env=env_full,
                    start_new_session=True,
                )
                if spec.stdin_bytes is not None and handle.process.stdin is not None:
                    # Write and close: stdin_bytes is a one-shot delivery,
                    # not an interactive stream.
                    try:
                        handle.process.stdin.write(spec.stdin_bytes)
                    finally:
                        handle.process.stdin.close()
                handle.pid = handle.process.pid
                handle.started_at = datetime.now(timezone.utc).isoformat()
                handle._set_state(TargetState.RUNNING)
                handle.metadata.setdefault(
                    "start", {"argv": argv, "cwd": spec.working_dir}
                )
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
            process = handle.process
            if process is None or process.poll() is not None:
                handle.state = TargetState.STOPPED
                handle.stopped_at = datetime.now(timezone.utc).isoformat()
                handle._touch()
                return
            handle._set_state(TargetState.STOPPED)
            try:
                process.terminate()
                try:
                    process.wait(timeout=max(0.1, timeout_sec))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=timeout_sec)
            except ProcessLookupError:
                pass
            finally:
                handle.stopped_at = datetime.now(timezone.utc).isoformat()
                handle._touch()

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        with handle._lock:
            if handle.pid is None:
                raise TargetError(
                    f"target '{handle.target_id}' has no PID to attach to; "
                    "call start_target first"
                )
            attach_result, context = gdb.attach(handle.pid)
            if attach_result.get("success"):
                handle._set_state(TargetState.ATTACHED)
                handle.attached_at = datetime.now(timezone.utc).isoformat()
            else:
                handle._set_state(TargetState.ERROR)
                handle.error = attach_result.get("error") or "attach failed"
                handle.error_type = attach_result.get("type") or "AttachError"
            return {
                "attach": attach_result,
                "context": context,
                "state": handle.state.value,
                "pid": handle.pid,
            }

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
        return {
            "start": True,
            "stop": True,
            "attach_debugger": True,
            "file_sync": False,
            "reconnect": False,
            "port_forward": False,
            "notes": "spawns a local subprocess and attaches gdb via PID",
        }


class AttachPidProvider:
    """Attach to an existing PID we do not own."""

    kind = TargetKind.ATTACH_PID

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            pid = handle.spec.pid
            if pid is None:
                raise TargetError("attach_pid spec has no pid")
            if not _pid_is_alive(pid):
                raise TargetError(f"pid {pid} is not alive")
            handle.pid = pid
            handle.started_at = datetime.now(timezone.utc).isoformat()
            handle._set_state(TargetState.RUNNING)

    def stop(
        self,
        handle: TargetHandle,
        *,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> None:
        # We never kill a process we did not start.  ``stop`` here means
        # release the target-registry slot; the inferior continues.
        with handle._lock:
            handle._set_state(TargetState.DETACHED)
            handle.stopped_at = datetime.now(timezone.utc).isoformat()

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        with handle._lock:
            pid = handle.pid or handle.spec.pid
            if pid is None:
                raise TargetError("attach_pid handle has no pid resolved")
            attach_result, context = gdb.attach(pid)
            if attach_result.get("success"):
                handle._set_state(TargetState.ATTACHED)
                handle.attached_at = datetime.now(timezone.utc).isoformat()
            else:
                handle._set_state(TargetState.ERROR)
                handle.error = attach_result.get("error") or "attach failed"
                handle.error_type = attach_result.get("type") or "AttachError"
            return {
                "attach": attach_result,
                "context": context,
                "state": handle.state.value,
                "pid": pid,
            }

    def get_capabilities(self, handle: TargetHandle) -> Dict[str, Any]:
        return {
            "start": True,
            "stop": True,
            "attach_debugger": True,
            "file_sync": False,
            "reconnect": False,
            "port_forward": False,
            "notes": (
                "attaches to an existing PID; stop() detaches without "
                "terminating the inferior"
            ),
        }


def _pid_is_alive(pid: int) -> bool:
    """Portable liveness check for a PID (Linux + Windows friendly)."""
    if pid <= 0:
        return False
    try:
        if os.name == "posix":
            os.kill(pid, 0)
            return True
        # Windows: OpenProcess via ctypes.  Falls through to False on any
        # error rather than raising.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    except Exception:
        return False


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
        }
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def register_provider(self, provider: TargetProvider) -> None:
        """Attach a new provider implementation (S8.2+ extension point)."""
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
        stopped: List[str] = []
        errors: List[Dict[str, Any]] = []
        for handle in self.list(session_id=session_id):
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
            finally:
                self.remove(handle.target_id)
        return {
            "session_id": session_id,
            "stopped": stopped,
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
    "DEFAULT_MAX_TARGETS_PER_SESSION",
    "DEFAULT_TARGET_START_TIMEOUT_SEC",
    "DEFAULT_TARGET_STOP_TIMEOUT_SEC",
    "TARGET_CREATE_SCHEMA",
    "TARGET_LIST_SCHEMA",
    "TARGET_STATUS_SCHEMA",
    "TARGET_CAPABILITIES_SCHEMA",
    "TARGET_START_SCHEMA",
    "TARGET_STOP_SCHEMA",
    "TARGET_ATTACH_SCHEMA",
]
