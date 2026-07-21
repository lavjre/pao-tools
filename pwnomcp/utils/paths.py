"""Path and runtime directory helpers for Pwno MCP."""

from __future__ import annotations

from dataclasses import dataclass
import os
import posixpath
import re
import tempfile
from typing import Optional

DEFAULT_WORKSPACE = "/workspace"
DEFAULT_RUNTIME_ROOT = os.environ.get(
    "PWNO_RUNTIME_ROOT", os.path.join(tempfile.gettempdir(), "pwno")
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
    )
    for path in [
        paths.workspace_root,
        paths.runtime_root,
        paths.sessions_dir,
        paths.processes_dir,
        paths.python_dir,
        paths.repos_dir,
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
    return resolved
