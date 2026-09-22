"""Shared target descriptor model (providers live in targets_*)."""

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
from paomcp.utils.paths import ensure_realpath_contained, resolve_workspace_path

# System PATH used to resolve provider infrastructure binaries (qemu,
# gdbserver). Deliberately independent of any caller-supplied PATH so a
# spec env cannot redirect which helper executable gets spawned. An
# explicit absolute path in the spec bypasses the lookup (checked below).
SYSTEM_EXECUTABLE_PATH = os.defpath or os.pathsep.join(
    ("/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin", "/sbin", "/bin")
)


def resolve_infra_executable(name: str, *, what: str) -> str:
    """Resolve an infrastructure helper binary to an absolute path.

    Bare names are looked up on the sanitized system PATH only; absolute
    paths must exist and be executable. Anything else raises TargetError
    instead of letting ``Popen`` resolve through a caller-controlled PATH.
    """
    import shutil

    if not isinstance(name, str) or not name.strip():
        raise TargetError(f"{what} must be a non-empty executable name")
    candidate = name.strip()
    if os.path.sep in candidate or (
        os.path.altsep is not None and os.path.altsep in candidate
    ):
        resolved = os.path.abspath(candidate)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        raise TargetError(f"{what} '{candidate}' does not exist or is not executable")
    found = shutil.which(candidate, path=SYSTEM_EXECUTABLE_PATH)
    if found is None:
        raise TargetError(
            f"{what} '{candidate}' not found on the system PATH "
            f"({SYSTEM_EXECUTABLE_PATH}); pass an absolute path to use a "
            "custom build"
        )
    return found


# Spec env vars that hijack the spawned process itself (loader / runtime
# search paths). They apply to our own gdbserver/qemu children as much as
# to the inferior, so they are rejected outright instead of merged: an
# operator needing a shim inside a *local* inferior can stage it via
# run_command, never via the supervised spawn path.
BLOCKED_TARGET_ENV = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "LD_CONFIG",
        "PYTHONPATH",
        "PYTHONHOME",
        "RUBYLIB",
        "PERL5LIB",
        "NODE_PATH",
        # QEMU reads this as an implicit ``-L`` sysroot; allowing it in the
        # spec env would bypass the workspace containment of qemu_ld_prefix.
        "QEMU_LD_PREFIX",
    }
)

# Bytes retained per drained stream; enough for triage, bounded against a
# chattering inferior filling the runtime volume via handle metadata.
TARGET_DRAIN_CAP_BYTES = 1 << 20


def validate_target_env(env: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Reject loader-hijack variables in a target spec env mapping."""
    merged: Dict[str, str] = {}
    for key, value in dict(env or {}).items():
        name = str(key)
        if name.upper() in BLOCKED_TARGET_ENV:
            raise TargetError(
                f"spec env blocks '{name}': loader/runtime search paths "
                "cannot be set on supervised spawns"
            )
        merged[name] = str(value)
    return merged


# ---------------------------------------------------------------------------
# QEMU extra-arg path containment (shared by create-time and launch-time).
# ---------------------------------------------------------------------------
# Flags whose comma-separated operand names a host file path.
_QEMU_COMMA_OPERAND_KEYS = {
    "-drive": "file",
    "-blockdev": "filename",
    "-device": "file",
    "-chardev": "path",
}
# Flags whose operand is a bare host path.
_QEMU_BARE_PATH_FLAGS = frozenset({"-kernel", "-initrd", "-L"})
# ``-serial`` chardev prefixes that name a host file/socket.
_QEMU_SERIAL_FILE_PREFIXES = ("file:", "pipe:", "unix:")
_QEMU_OPERAND_FLAGS = frozenset(_QEMU_COMMA_OPERAND_KEYS) | _QEMU_BARE_PATH_FLAGS


def _canonical_qemu_operand_flag(name: str) -> Optional[str]:
    """Normalize ``--kernel`` / ``-L`` spellings to the flag variant we track."""
    stripped = name.strip().lower().lstrip("-")
    if not stripped:
        return None
    canonical = f"-{stripped}"
    if canonical == "-l":
        # QEMU's sysroot flag is ``-L`` (case-sensitive); ``-l`` is not a flag.
        canonical = "-L"
    if canonical in _QEMU_OPERAND_FLAGS:
        return canonical
    if canonical == "-serial":
        return canonical
    return None


def _contain_qemu_operand(flag: str, value: str, *, workspace_root: str) -> str:
    """Rewrite one QEMU file operand so it stays inside the workspace."""
    if flag in _QEMU_COMMA_OPERAND_KEYS:
        key = _QEMU_COMMA_OPERAND_KEYS[flag]
        kind = f"qemu {flag.lstrip('-')} {key}"
        rebuilt = []
        for part in value.split(","):
            name, sep, val = part.partition("=")
            if sep and name.strip().lower() == key and val.strip():
                resolved = resolve_workspace_path(
                    val.strip(),
                    workspace_root=workspace_root,
                    require_exists=False,
                    kind=kind,
                )
                rebuilt.append(
                    f"{name}="
                    f"{ensure_realpath_contained(resolved, workspace_root, kind=kind)}"
                )
            else:
                rebuilt.append(part)
        return ",".join(rebuilt)
    if flag == "-serial":
        lowered = value.lower()
        for prefix in _QEMU_SERIAL_FILE_PREFIXES:
            if not lowered.startswith(prefix):
                continue
            path, sep, options = value[len(prefix) :].partition(",")
            if not path:
                return value
            resolved = resolve_workspace_path(
                path,
                workspace_root=workspace_root,
                require_exists=False,
                kind="qemu serial path",
            )
            contained = ensure_realpath_contained(
                resolved, workspace_root, kind="qemu serial path"
            )
            return f"{value[:len(prefix)]}{contained}{sep}{options}"
        return value
    resolved = resolve_workspace_path(
        value,
        workspace_root=workspace_root,
        require_exists=False,
        kind=f"qemu {flag} path",
    )
    return ensure_realpath_contained(resolved, workspace_root, kind=f"qemu {flag} path")


def contain_qemu_extra_args(
    extra_args: Sequence[Any], *, workspace_root: str
) -> List[Any]:
    """Contain every file operand in a QEMU extra-args list.

    Handles ``--kernel=/p`` / ``-kernel=/p`` glued values, ``-blockdev
    driver=file,filename=/p``, ``-device loader,file=/p``, ``-serial
    file:/p``, ``-chardev file,path=/p`` and glued ``-L/p``.  A path that
    escapes the workspace raises ValueError (fail closed).
    """
    normalized: List[Any] = []
    pending: Optional[str] = None
    for token in list(extra_args or []):
        if not isinstance(token, str):
            normalized.append(token)
            pending = None
            continue
        if pending is not None:
            normalized.append(
                _contain_qemu_operand(pending, token, workspace_root=workspace_root)
            )
            pending = None
            continue
        if not token.startswith("--") and len(token) > 2 and token[:2] == "-L":
            remainder = token[2:]
            if remainder.startswith("="):
                remainder = remainder[1:]
            normalized.append(
                "-L"
                + _contain_qemu_operand("-L", remainder, workspace_root=workspace_root)
            )
            continue
        name, sep, value = token.partition("=")
        canonical = _canonical_qemu_operand_flag(name)
        if canonical is not None and sep:
            normalized.append(
                f"{name}="
                f"{_contain_qemu_operand(canonical, value, workspace_root=workspace_root)}"
            )
            continue
        if canonical is not None:
            pending = canonical
        normalized.append(token)
    return normalized


_TARGET_PATH_ATTRS = (
    "binary_path",
    "working_dir",
    "kernel_path",
    "initrd_path",
    "disk_path",
    "qemu_ld_prefix",
)


def ensure_target_paths_contained(spec: "TargetSpec") -> None:
    """Re-check every host path in ``spec`` against the workspace (TOCTOU).

    ``_contain_target_config`` validates at create time, but a symlink can
    be swapped afterwards; providers call this right before spawning so a
    swapped link cannot redirect execution outside the workspace.
    """
    workspace_root = getattr(spec, "workspace_root", None)
    if not workspace_root:
        return
    for attr in _TARGET_PATH_ATTRS:
        value = getattr(spec, attr, None)
        if not value:
            continue
        try:
            ensure_realpath_contained(value, workspace_root, kind=attr)
        except ValueError as exc:
            raise TargetError(str(exc)) from exc
    if spec.qemu_extra_args:
        try:
            contain_qemu_extra_args(spec.qemu_extra_args, workspace_root=workspace_root)
        except ValueError as exc:
            raise TargetError(str(exc)) from exc


def ensure_local_path_contained(
    path: str, *, workspace_root: Optional[str], kind: str
) -> str:
    """Realpath-check a provider-local transfer path against the workspace."""
    if not workspace_root:
        return path
    try:
        return ensure_realpath_contained(path, workspace_root, kind=kind)
    except ValueError as exc:
        raise TargetError(str(exc)) from exc


def drain_stream_to_buffer(
    stream: Any, store: List[bytes], cap_bytes: int = TARGET_DRAIN_CAP_BYTES
) -> None:
    """Drain one pipe to a bounded tail buffer (daemon-thread entry)."""
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            store.append(chunk)
            while sum(map(len, store)) > cap_bytes:
                store.pop(0)
    except (ValueError, OSError):
        pass


def start_output_drains(handle: TargetHandle) -> None:
    """Attach bounded drain threads to a freshly spawned handle.process.

    Raw byte tails live on a private ``handle._drain_tails`` attribute (never
    in ``metadata`` so ``to_dict`` stays JSON-safe); :func:`reap_output_drains`
    collapses them to capped text on stop.
    """
    process = handle.process
    if process is None:
        return
    tails: Dict[str, List[bytes]] = {"stdout": [], "stderr": []}
    threads: List[threading.Thread] = []
    for attr, key in (("stdout", "stdout"), ("stderr", "stderr")):
        stream = getattr(process, attr, None)
        if stream is None:
            continue
        thread = threading.Thread(
            target=drain_stream_to_buffer,
            args=(stream, tails[key]),
            name=f"pao-target-drain-{handle.target_id}-{key}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
    handle._drain_tails = tails  # type: ignore[attr-defined]
    handle._drain_threads = threads  # type: ignore[attr-defined]


def reap_output_drains(handle: TargetHandle, *, keep_chars: int = 8192) -> None:
    """Join drain threads and publish a capped text tail into metadata."""
    tails = getattr(handle, "_drain_tails", None)
    threads = getattr(handle, "_drain_threads", None) or []
    for thread in threads:
        thread.join(timeout=2.0)
    if not isinstance(tails, dict):
        return
    text: Dict[str, str] = {}
    for key in ("stdout", "stderr"):
        raw = b"".join(tails.get(key, []))[-TARGET_DRAIN_CAP_BYTES:]
        text[key] = raw.decode("utf-8", errors="replace")[-keep_chars:]
    handle.metadata["output_tail"] = text


_killpg = getattr(os, "killpg", None)
_getpgid = getattr(os, "getpgid", None)
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def kill_process_tree(process: subprocess.Popen) -> None:
    """Terminate a supervised process *and* its children.

    All target spawns use ``start_new_session=True``, so the child PID is
    the process-group leader on POSIX: ``killpg`` reaps forked grandchildren
    that a bare ``terminate()`` would orphan. Windows falls back to
    ``taskkill /T`` for the same tree semantics. (``getattr`` lookups keep
    this importable/type-clean on platforms without ``killpg``.)
    """
    try:
        if os.name == "posix" and _killpg is not None and _getpgid is not None:
            try:
                _killpg(_getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                process.wait(timeout=5.0)
                return
            except subprocess.TimeoutExpired:
                pass
            try:
                _killpg(_getpgid(process.pid), _SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        else:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15.0,
            )
    finally:
        try:
            process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass


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
    """Every provider surface we intend to ship in targets."""

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
    # Workspace the host-side paths below were contained against; providers
    # re-validate against it right before launching (TOCTOU guard).
    workspace_root: Optional[str] = None
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
    ssh_known_hosts_path: Optional[str] = None
    # Docker
    docker_image: Optional[str] = None
    docker_workdir: Optional[str] = None
    docker_extra_args: List[str] = field(default_factory=list)
    # QEMU user
    qemu_binary: Optional[str] = None
    qemu_ld_prefix: Optional[str] = None
    qemu_extra_args: List[str] = field(default_factory=list)
    # QEMU system
    kernel_path: Optional[str] = None
    initrd_path: Optional[str] = None
    disk_path: Optional[str] = None
    kernel_cmdline: Optional[str] = None
    qemu_memory: Optional[str] = None
    qemu_machine: Optional[str] = None
    # Metadata (informational)
    arch: Optional[str] = None
    platform: Optional[str] = None
    notes: Optional[str] = None
    # Debugger backend request ("gdb" | "lldb" | "auto") for the owning session
    backend: Optional[str] = None

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
            "workspace_root",
            "binary_path",
            "working_dir",
            "host",
            "ssh_user",
            "ssh_key_path",
            "ssh_password",
            "ssh_known_hosts_path",
            "docker_image",
            "docker_workdir",
            "qemu_binary",
            "qemu_ld_prefix",
            "kernel_path",
            "initrd_path",
            "disk_path",
            "kernel_cmdline",
            "qemu_memory",
            "qemu_machine",
            "arch",
            "platform",
            "notes",
            "backend",
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
            if not self.qemu_binary:
                raise TargetError(
                    "qemu_system requires 'qemu_binary' " "(e.g. 'qemu-system-x86_64')"
                )
            if not (self.kernel_path or self.disk_path):
                raise TargetError(
                    "qemu_system requires at least one of 'kernel_path' or "
                    "'disk_path'"
                )


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
        # Remember where we were before the failure so get_status_response can
        # restore the live state once a remote target recovers, instead of
        # pinning a stale ERROR (and error string) forever.
        self.metadata["pre_error_state"] = self.state.value
        self.error = str(exc)
        self.error_type = type(exc).__name__
        self.state = TargetState.ERROR
        self._touch()

    def _clear_error(self) -> None:
        self.error = None
        self.error_type = None
        self.metadata.pop("pre_error_state", None)

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

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        """Return a definitive liveness signal, or ``None`` when the provider
        has no cheap probe (callers then leave the handle untouched).

        ``get_status_response`` re-polls this so a transient failure cannot
        pin a target in ERROR forever after the underlying process /
        container / ssh transport has recovered.
        """
        return None
