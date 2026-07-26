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
TARGET_CONNECT_SCHEMA = "pao.target-connect.v1"
TARGET_DISCONNECT_SCHEMA = "pao.target-disconnect.v1"
TARGET_FILE_SYNC_SCHEMA = "pao.target-file-sync.v1"
TARGET_FETCH_LIBS_SCHEMA = "pao.target-fetch-libs.v1"

DEFAULT_MAX_TARGETS_PER_SESSION = 8
DEFAULT_TARGET_START_TIMEOUT_SEC = 15.0
DEFAULT_TARGET_STOP_TIMEOUT_SEC = 10.0
DEFAULT_GDBSERVER_LISTEN_TIMEOUT_SEC = 10.0
DEFAULT_TCP_PROBE_TIMEOUT_SEC = 3.0
DEFAULT_SSH_CONNECT_TIMEOUT_SEC = 10.0
DEFAULT_DOCKER_TIMEOUT_SEC = 30.0
DEFAULT_QEMU_LISTEN_TIMEOUT_SEC = 10.0
DEFAULT_LIBC_FETCH_PATHS: Tuple[str, ...] = (
    "/lib/x86_64-linux-gnu/libc.so.6",
    "/lib64/libc.so.6",
    "/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
    "/lib64/ld-linux-x86-64.so.2",
    "/lib/ld-musl-x86_64.so.1",
    "/lib/i386-linux-gnu/libc.so.6",
    "/lib/ld-linux.so.2",
)


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


class SshClientFactory(Protocol):
    """Injection point for tests -- returns something paramiko-shaped."""

    def __call__(self) -> Any: ...


def _default_ssh_client_factory() -> Any:
    """Return a real paramiko ``SSHClient`` with a permissive host-key policy.

    The permissive policy is deliberate for research workflows against
    scratch VMs; a production caller should replace this factory with
    a strict-host-key one.
    """
    import paramiko  # type: ignore[import-untyped]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
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
    offline S7 identify_libc has something to score against.

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


DockerRunner = Callable[[List[str], float], Tuple[int, str, str]]


def _default_docker_runner(argv: List[str], timeout_sec: float) -> Tuple[int, str, str]:
    """Run ``docker <argv>`` via subprocess and return (rc, stdout, stderr)."""
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        return 127, "", "docker executable not found on PATH"
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"docker command timed out after {timeout_sec}s: {exc}"
    return completed.returncode, completed.stdout, completed.stderr


class DockerProvider:
    """Spawn a detached container via the ``docker`` CLI.

    ``start`` runs ``docker run -d --rm ... <image>`` and records the
    container id from stdout.  ``stop`` runs ``docker stop`` (falls
    back to ``docker kill`` after ``timeout_sec``).  File sync is via
    ``docker cp``.  ``attach_debugger`` is *not* supported directly --
    the honest CTF pattern is to run a ``gdbserver`` inside the
    container and attach with a :class:`TcpRemoteProvider` after
    forwarding the port.  This provider surfaces
    ``capabilities.file_sync = True`` so ``sync_target_files`` works
    with a docker target the same way it does with an ``ssh`` one.
    """

    kind = TargetKind.DOCKER

    def __init__(
        self,
        *,
        runner: DockerRunner = _default_docker_runner,
        timeout_sec: float = DEFAULT_DOCKER_TIMEOUT_SEC,
    ) -> None:
        self._runner = runner
        self._timeout_sec = timeout_sec

    def _docker(
        self, argv: List[str], *, timeout_sec: Optional[float] = None
    ) -> Tuple[int, str, str]:
        return self._runner(argv, timeout_sec or self._timeout_sec)

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            spec = handle.spec
            if not spec.docker_image:
                raise TargetError("docker spec requires 'docker_image'")
            container_name = spec.name or f"pao-target-{handle.target_id}"
            handle._set_state(TargetState.STARTING)
            argv: List[str] = [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                container_name,
            ]
            if spec.docker_workdir:
                argv += ["-w", spec.docker_workdir]
            for env_name, env_value in spec.env.items():
                argv += ["-e", f"{env_name}={env_value}"]
            argv += list(spec.docker_extra_args or [])
            argv += [spec.docker_image]
            argv += list(spec.argv or [])
            rc, stdout, stderr = self._docker(argv)
            if rc != 0:
                exc = TargetError(
                    f"'docker run' failed (rc={rc}): "
                    f"{(stderr or stdout).strip() or 'no output'}"
                )
                handle._record_error(exc)
                raise exc
            container_id = (
                stdout.strip().splitlines()[-1] if stdout.strip() else container_name
            )
            handle.metadata["container_name"] = container_name
            handle.metadata["container_id"] = container_id
            handle.metadata["run_argv"] = argv
            handle.started_at = datetime.now(timezone.utc).isoformat()
            handle._set_state(TargetState.RUNNING)

    def stop(
        self,
        handle: TargetHandle,
        *,
        timeout_sec: float = DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    ) -> None:
        with handle._lock:
            container_id = handle.metadata.get("container_id") or handle.metadata.get(
                "container_name"
            )
            if container_id:
                rc, _stdout, stderr = self._docker(
                    ["docker", "stop", "-t", str(int(timeout_sec)), str(container_id)],
                    timeout_sec=max(timeout_sec + 5.0, 15.0),
                )
                if rc != 0:
                    # Try a hard kill so a stuck container does not leak
                    # a registry slot forever.
                    self._docker(
                        ["docker", "kill", str(container_id)],
                        timeout_sec=max(timeout_sec, 10.0),
                    )
                    handle.metadata["stop_stderr"] = stderr.strip()
            handle._set_state(TargetState.STOPPED)
            handle.stopped_at = datetime.now(timezone.utc).isoformat()

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        raise TargetError(
            "docker provider does not attach a debugger directly; "
            "run gdbserver inside the container (docker run ... -p PORT) "
            "and use a tcp_remote target to attach"
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
                "spawns a container via 'docker run -d --rm'; "
                "attach_debugger deferred to a tcp_remote target"
            ),
        }

    # ------------------------------------------------------------------
    # File sync via docker cp
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
        container = handle.metadata.get("container_id") or handle.metadata.get(
            "container_name"
        )
        if not container:
            raise TargetError(
                f"docker target '{handle.target_id}' has no running container"
            )
        if direction == "download":
            local_dir = os.path.dirname(destination)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            argv = ["docker", "cp", f"{container}:{source}", destination]
            rc, stdout, stderr = self._docker(argv)
            if rc != 0:
                raise TargetError(
                    f"'docker cp' download failed (rc={rc}): "
                    f"{(stderr or stdout).strip() or 'no output'}"
                )
            info = os.stat(destination)
            return {
                "direction": "download",
                "remote_path": source,
                "local_path": os.path.abspath(destination),
                "bytes": info.st_size,
            }
        argv = ["docker", "cp", source, f"{container}:{destination}"]
        rc, stdout, stderr = self._docker(argv)
        if rc != 0:
            raise TargetError(
                f"'docker cp' upload failed (rc={rc}): "
                f"{(stderr or stdout).strip() or 'no output'}"
            )
        return {
            "direction": "upload",
            "local_path": os.path.abspath(source),
            "remote_path": destination,
            "bytes": os.path.getsize(source),
        }


class QemuUserProvider:
    """Run a cross-arch binary under qemu-user with a gdbserver stub.

    ``start`` spawns ``qemu-<arch> -g <port> <binary> [args]`` (with an
    optional ``qemu_ld_prefix`` for the target's sysroot); QEMU's
    built-in gdbserver stub listens on ``port`` for
    :meth:`attach_debugger` to hit via ``-target-select remote
    localhost:port``.  ``port`` defaults to an ephemeral pick if the
    caller omits it.  Windows hosts have no ``qemu-user`` binary so
    the real path is exercised from WSL / Linux; ``start`` returns
    the typed error surface either way.
    """

    kind = TargetKind.QEMU_USER

    def __init__(
        self,
        *,
        listen_timeout_sec: float = DEFAULT_QEMU_LISTEN_TIMEOUT_SEC,
    ) -> None:
        self._listen_timeout_sec = listen_timeout_sec

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            spec = handle.spec
            if not spec.binary_path:
                raise TargetError("qemu_user spec requires 'binary_path'")
            if not spec.qemu_binary:
                raise TargetError(
                    "qemu_user spec requires 'qemu_binary' (e.g. 'qemu-x86_64')"
                )
            binary = os.path.abspath(spec.binary_path)
            if not os.path.isfile(binary):
                raise TargetError(f"binary '{binary}' does not exist")
            host = spec.host or "127.0.0.1"
            port = spec.port or _find_free_tcp_port(host)
            argv: List[str] = [spec.qemu_binary]
            if spec.qemu_ld_prefix:
                argv += ["-L", spec.qemu_ld_prefix]
            argv += list(spec.qemu_extra_args or [])
            argv += ["-g", str(port), binary] + list(spec.argv or [])
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
                handle.metadata["run_argv"] = argv
                handle.started_at = datetime.now(timezone.utc).isoformat()
                if not _wait_for_tcp_listen(
                    host, port, timeout_sec=self._listen_timeout_sec
                ):
                    raise TargetError(
                        f"qemu-user did not open gdb stub on {host}:{port} "
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
        LocalProcessProvider().stop(handle, timeout_sec=timeout_sec)

    def attach_debugger(
        self, handle: TargetHandle, gdb: GdbController
    ) -> Dict[str, Any]:
        with handle._lock:
            if handle.host is None or handle.port is None:
                raise TargetError(
                    f"qemu_user target '{handle.target_id}' is not listening; "
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
                handle.error = attach_result.get("error") or "qemu attach failed"
                handle.error_type = attach_result.get("type") or "QemuAttachError"
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
                "spawns qemu-<arch> with '-g PORT' and attaches via "
                "'-target-select remote host:port'"
            ),
        }


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
