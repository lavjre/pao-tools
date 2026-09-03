"""Path and runtime directory helpers for Pao Tools."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
import posixpath
import re
import shutil
import stat
import tempfile
from typing import Any, Callable, Iterator, Optional

from paomcp.operations import OperationCancelledError

DEFAULT_WORKSPACE = "/workspace"
DEFAULT_RUNTIME_ROOT = os.environ.get(
    "PAO_RUNTIME_ROOT", os.path.join(tempfile.gettempdir(), "pao")
)


@dataclass
class RuntimePaths:
    """Filesystem layout used by server-managed runtime artifacts."""

    workspace_root: str
    runtime_root: str
    sessions_dir: str
    processes_dir: str
    python_dir: str
    repos_dir: str
    artifacts_dir: str
    decompiler_dir: str


def build_runtime_paths(
    workspace_root: str = DEFAULT_WORKSPACE,
    runtime_root: str = DEFAULT_RUNTIME_ROOT,
) -> RuntimePaths:
    """Build and ensure runtime directories exist."""
    resolved_workspace = os.path.normpath(workspace_root)
    resolved_runtime = os.path.normpath(runtime_root)
    paths = RuntimePaths(
        workspace_root=resolved_workspace,
        runtime_root=resolved_runtime,
        sessions_dir=os.path.join(resolved_runtime, "sessions"),
        processes_dir=os.path.join(resolved_runtime, "processes"),
        python_dir=os.path.join(resolved_runtime, "python"),
        repos_dir=os.path.join(resolved_runtime, "repos"),
        artifacts_dir=os.path.join(resolved_runtime, "artifacts"),
        decompiler_dir=os.path.join(resolved_runtime, "decompiler"),
    )
    for path in [
        paths.workspace_root,
        paths.runtime_root,
        paths.sessions_dir,
        paths.processes_dir,
        paths.python_dir,
        paths.repos_dir,
        paths.artifacts_dir,
        paths.decompiler_dir,
    ]:
        os.makedirs(path, exist_ok=True)
    return paths


def sanitize_session_id(value: Optional[str]) -> str:
    """Return a filesystem-safe session identifier."""
    if value is None:
        return "session"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", value).strip("._")
    return sanitized or "session"


def validate_session_id(value: str) -> str:
    """Validate and return a debugger session identifier.

    Session identifiers are API keys as well as directory names. Rejecting
    unsupported characters avoids two distinct caller-provided identifiers
    being normalized into the same debugger session.
    """
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError(
            "session_id must be 1-64 characters and contain only letters, "
            "numbers, '.', '_' or '-' (starting with a letter or number)"
        )
    return value


def _path_module(root: str):
    """Return path semantics for a server namespace root.

    ``/workspace`` is a container path even when unit tests or clients run on
    Windows, so it must keep POSIX separators on every host.
    """
    return posixpath if root.replace("\\", "/").startswith("/") else os.path


def _within(path: str, root: str) -> bool:
    path_module = _path_module(root)
    try:
        return path_module.commonpath([path, root]) == root
    except ValueError:
        return False


def _map_host_workspace_path(path: str, workspace_root: str) -> Optional[str]:
    path_module = _path_module(workspace_root)
    separator = path_module.sep
    marker = f"{separator}workspace"
    normalized = path.replace("\\", separator).rstrip(separator)
    if normalized.endswith(marker):
        return workspace_root
    marker_with_sep = marker + separator
    idx = normalized.find(marker_with_sep)
    if idx == -1:
        return None
    suffix = normalized[idx + len(marker) :]
    return path_module.normpath(workspace_root + suffix)


def resolve_workspace_path(
    raw_path: str,
    workspace_root: str = DEFAULT_WORKSPACE,
    require_exists: bool = False,
    kind: str = "path",
) -> str:
    """Resolve user-provided paths into the server workspace namespace.

    Accepts:
    - absolute paths already under /workspace
    - relative paths (resolved under /workspace)
    - host absolute paths containing '/workspace/...', remapped to container path

    The server runs inside a container. Tool path arguments should use container
    paths under /workspace (or relative paths that resolve there).
    """
    if not raw_path or not raw_path.strip():
        raise ValueError(f"{kind} cannot be empty")

    path_module = _path_module(workspace_root)
    workspace_root = path_module.normpath(workspace_root)
    candidate = raw_path.strip()

    if path_module.isabs(candidate):
        candidate = path_module.normpath(candidate.replace("\\", path_module.sep))
        if not _within(candidate, workspace_root):
            mapped = _map_host_workspace_path(candidate, workspace_root)
            if mapped and _within(mapped, workspace_root):
                candidate = mapped
            else:
                raise ValueError(
                    f"{kind} must be accessible inside the container under "
                    f"{workspace_root}. Got '{raw_path}'. Host paths are not "
                    f"directly visible; use '{workspace_root}/<name>' or a "
                    "relative path."
                )
    else:
        normalized_rel = candidate.replace("\\", path_module.sep)
        if normalized_rel.startswith(f"workspace{path_module.sep}"):
            normalized_rel = normalized_rel[len("workspace") + 1 :]
        elif normalized_rel == "workspace":
            normalized_rel = ""
        candidate = path_module.normpath(
            path_module.join(workspace_root, normalized_rel)
        )
        if not _within(candidate, workspace_root):
            raise ValueError(
                f"{kind} escapes {workspace_root}: '{raw_path}'. "
                f"Use a safe path under {workspace_root} (or a relative path)."
            )

    if require_exists and not os.path.exists(candidate):
        raise FileNotFoundError(
            f"{kind} not found: '{candidate}'. Ensure it exists in {workspace_root}. "
            f"In Docker, host './workspace/<name>' maps to '{workspace_root}/<name>'."
        )

    return candidate


def resolve_workspace_cwd(
    cwd: Optional[str], workspace_root: str = DEFAULT_WORKSPACE
) -> str:
    """Resolve an optional working directory to a workspace directory."""
    if cwd is None:
        return os.path.normpath(workspace_root)
    resolved = resolve_workspace_path(
        cwd, workspace_root=workspace_root, require_exists=False, kind="cwd"
    )
    os.makedirs(resolved, exist_ok=True)
    # Re-check after makedirs: a symlinked parent (e.g. /workspace/link -> /etc)
    # would otherwise let makedirs create directories outside the workspace.
    workspace_real = os.path.realpath(workspace_root)
    cwd_real = os.path.realpath(resolved)
    try:
        contained = os.path.commonpath([cwd_real, workspace_real]) == workspace_real
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"cwd resolves outside the configured workspace: '{cwd}'")
    return cwd_real


def resolve_workspace_file(
    raw_path: str,
    workspace_root: str = DEFAULT_WORKSPACE,
    *,
    kind: str = "file",
) -> str:
    """Resolve a regular file and reject symlink escapes from the workspace."""
    resolved = resolve_workspace_path(
        raw_path,
        workspace_root=workspace_root,
        require_exists=True,
        kind=kind,
    )
    workspace_real = os.path.realpath(workspace_root)
    file_real = os.path.realpath(resolved)
    try:
        contained = os.path.commonpath([file_real, workspace_real]) == workspace_real
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"{kind} resolves outside the configured workspace")
    if not os.path.isfile(file_real):
        raise ValueError(f"{kind} must identify a regular file")
    return file_real


def _opened_file_path(source: Any, fallback_path: str) -> str:
    """Return the canonical path represented by an already-open file handle."""
    if os.name != "nt":
        return os.path.realpath(fallback_path)

    import ctypes
    from ctypes import wintypes
    import msvcrt

    get_final_path = ctypes.WinDLL(
        "kernel32", use_last_error=True
    ).GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(source.fileno()))
    length = get_final_path(handle, buffer, len(buffer), 0)
    if length == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    if length >= len(buffer):
        raise OSError("Opened file path exceeds the supported Windows path length")
    resolved = buffer.value
    if resolved.startswith("\\\\?\\UNC\\"):
        resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
        resolved = resolved[4:]
    return os.path.realpath(resolved)


@contextmanager
def _open_regular_beneath(
    source_path: str, workspace_root: Optional[str]
) -> Iterator[Any]:
    """Open a regular file without following workspace-relative symlinks."""
    source_absolute = os.path.abspath(source_path)
    root = os.path.realpath(workspace_root or os.path.dirname(source_absolute))
    try:
        contained = os.path.commonpath([source_absolute, root]) == root
    except ValueError:
        contained = False
    if not contained or source_absolute == root:
        raise ValueError("analysis input is outside the configured workspace")

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    supports_openat = os.name == "posix" and nofollow and directory
    if not supports_openat:
        with open(source_absolute, "rb") as source:
            final_path = _opened_file_path(source, source_absolute)
            try:
                still_contained = os.path.commonpath([final_path, root]) == root
            except ValueError:
                still_contained = False
            if not still_contained:
                raise ValueError("analysis input escaped the configured workspace")
            yield source
        return

    relative = os.path.relpath(source_absolute, root)
    components = [item for item in relative.split(os.sep) if item not in {"", "."}]
    if not components or any(item == ".." for item in components):
        raise ValueError("analysis input is outside the configured workspace")

    directory_flags = os.O_RDONLY | directory | nofollow | cloexec
    file_flags = os.O_RDONLY | nofollow | cloexec
    current_fd = os.open(root, directory_flags)
    try:
        for component in components[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        source_fd = os.open(components[-1], file_flags, dir_fd=current_fd)
        try:
            source = os.fdopen(source_fd, "rb")
        except Exception:
            os.close(source_fd)
            raise
        with source:
            yield source
    finally:
        os.close(current_fd)


@contextmanager
def stable_file_snapshot(
    source_path: str,
    runtime_dir: str,
    *,
    workspace_root: Optional[str] = None,
    minimum_free_bytes: int = 0,
    cancellation_requested: Optional[Callable[[], bool]] = None,
) -> Iterator[str]:
    """Copy one stable regular-file view for parsers that reopen pathnames."""
    os.makedirs(runtime_dir, exist_ok=True)
    descriptor, snapshot_path = tempfile.mkstemp(
        prefix="input-snapshot-",
        suffix=os.path.splitext(source_path)[1],
        dir=runtime_dir,
    )
    descriptor_open = True
    try:
        with _open_regular_beneath(source_path, workspace_root) as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("analysis input must be a regular file")
            destination = os.fdopen(descriptor, "wb")
            descriptor_open = False
            written = 0
            with destination:
                while True:
                    if cancellation_requested is not None and cancellation_requested():
                        raise OperationCancelledError(
                            "Analysis input snapshot was cancelled"
                        )
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    destination.write(chunk)
                    written += len(chunk)
                    if shutil.disk_usage(runtime_dir).free < minimum_free_bytes:
                        raise OSError(
                            "Insufficient free disk for stable analysis snapshot"
                        )
                destination.flush()
                os.fsync(destination.fileno())
            after = os.fstat(source.fileno())

        current = os.stat(source_path, follow_symlinks=False)
        if not stat.S_ISREG(current.st_mode):
            raise RuntimeError("analysis input path was replaced while snapshotting")
        expected_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if expected_identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or expected_identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ):
            raise RuntimeError("analysis input changed while it was being snapshotted")
        if written != before.st_size:
            raise RuntimeError("analysis input size changed while being snapshotted")
        os.utime(
            snapshot_path,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        yield snapshot_path
    finally:
        if descriptor_open:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(snapshot_path)
        except FileNotFoundError:
            pass
