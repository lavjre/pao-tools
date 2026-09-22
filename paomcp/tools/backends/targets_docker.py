"""Docker provider (split from targets.py)."""

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
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetState,
    ensure_local_path_contained,
    ensure_target_paths_contained,
)

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


# Blocked = host-impact only (privilege, namespaces, mounts, devices,
# runtimes, host files). Constraining flags (--memory, --pids-limit,
# --ulimit, --user, --name, --hostname, --workdir, --entrypoint) stay
# allowed: they narrow what the container can do, and the provider already
# owns --name/-w via its own fields. Blocking limits would push callers
# away from them.
BLOCKED_DOCKER_PREFIXES = (
    "--privileged",
    "--pid",
    "--network",
    "--uts",
    "--ipc",
    "--volume",
    "--volumes-from",
    "--volume-driver",
    "--mount",
    "--device",
    "--device-cgroup-rule",
    "--cap-add",
    "--cap-drop",
    "--security-opt",
    "--cgroupns",
    "--cgroup-parent",
    "--userns",
    "--gpus",
    "--runtime",
    # Host file read/write primitives.
    "--env-file",
    "--label-file",
    "--cidfile",
    # Publishes every exposed port on all interfaces, bypassing the
    # loopback-only publish policy below.
    "--publish-all",
    "-v",
)
# Shorthands that must never appear inside a short-flag cluster. ``P`` is
# ``--publish-all``; ``v`` is ``--volume``.
_BLOCKED_DOCKER_SHORTHANDS = frozenset({"v", "P"})
# Short flags that take a value; the remainder of the cluster (or the next
# argv token) is consumed as that value, so scanning stops there.
_DOCKER_VALUE_SHORTHANDS = frozenset({"a", "c", "e", "h", "l", "m", "p", "u", "v", "w"})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_PUBLISH_HINT = "Use '-p 127.0.0.1:<host_port>:<container_port>'"


def _docker_blocked_error(entry: str) -> TargetError:
    return TargetError(
        f"Blocked unsafe docker_extra_args entry '{entry}': privileged "
        "mode, host mounts, host pid/network, devices, runtimes, GPUs, "
        "host file mapping and capability changes are not allowed"
    )


def _publish_host_part(spec: str) -> Optional[str]:
    """Return the host IP of a publish spec, or None when it is host-less."""
    text = spec.strip()
    if text.startswith("["):
        end = text.find("]")
        if end == -1 or not text[end + 1 :].startswith(":"):
            return None
        if not text[end + 2 :]:
            return None
        return text[1:end]
    parts = text.split(":")
    if len(parts) == 3 and parts[0]:
        return parts[0]
    return None


def _validate_publish_value(value: str, entry: str) -> None:
    """Fail closed unless a ``-p`` mapping names an explicit loopback host."""
    host = _publish_host_part(value)
    if host is None:
        raise TargetError(
            f"Unsafe docker publish spec '{entry}': a host-less port "
            f"mapping binds every interface. {_PUBLISH_HINT}"
        )
    if host not in _LOOPBACK_HOSTS:
        raise TargetError(
            f"Unsafe docker publish spec '{entry}': host '{host}' is not "
            f"loopback. {_PUBLISH_HINT}"
        )


def _validate_docker_token(
    token: str, pending: Optional[str], entry: str
) -> Optional[str]:
    """Validate one normalized token; return a flag awaiting its value."""
    if pending is not None:
        # The pending flag is always --publish/-p in this walk.
        _validate_publish_value(token, entry)
        return None
    if not token.startswith("-") or token == "-":
        # A plain value (env value, port spec, ...): inert without its flag.
        return None
    if token.startswith("--"):
        name, sep, value = token.partition("=")
        name = name.lower()
        if name in BLOCKED_DOCKER_PREFIXES:
            raise _docker_blocked_error(entry)
        if name == "--publish":
            if sep:
                _validate_publish_value(value, entry)
                return None
            return "--publish"
        return None
    body = token[1:]
    index = 0
    while index < len(body):
        char = body[index]
        if char in _BLOCKED_DOCKER_SHORTHANDS:
            raise _docker_blocked_error(entry)
        rest = body[index + 1 :]
        if rest.startswith("="):
            rest = rest[1:]
        if char in _DOCKER_VALUE_SHORTHANDS:
            if char == "p":
                if rest:
                    _validate_publish_value(rest, entry)
                    return None
                return "p"
            return None
        index += 1
    return None


def validate_docker_extra_args(extra_args) -> None:
    """Reject privilege/mount/device/network/host-file escapes in all spellings.

    Long flags are matched after ``--flag=value`` normalization; short
    clusters (``-itv``, ``-itv/:/host``) are scanned character by character
    so a blocked shorthand cannot hide behind benign ones. ``-p`` /
    ``--publish`` is fail-closed: the mapping must name an explicit
    loopback host. A flag that takes its value from the next argv entry
    (``-v /src:/dst``) is caught before the value is even reached.
    """
    pending: Optional[str] = None
    for extra in list(extra_args or []):
        if not isinstance(extra, str):
            pending = None
            continue
        stripped = extra.strip()
        if not stripped:
            continue
        # Accept both list-form tokens and a single string holding
        # "flag value" (the old normalizer split on whitespace).
        tokens = stripped.split() if stripped.startswith("-") else [stripped]
        for token in tokens:
            pending = _validate_docker_token(token, pending, extra)


def _redact_env_assignment(value: str) -> str:
    """Mask the value of ``NAME=VALUE`` while keeping the variable name."""
    body = value[1:] if value.startswith("=") else value
    name, sep, _ = body.partition("=")
    if not sep:
        return "***"
    return f"{name}=***"


def redact_docker_run_argv(argv: Sequence[str]) -> List[str]:
    """Mask ``-e``/``--env`` values in an argv kept for metadata.

    The registry surfaces ``run_argv`` through list/status responses, and
    env values routinely carry secrets; only the names stay visible.
    """
    redacted: List[str] = []
    pending_env = False
    for token in argv:
        if pending_env:
            redacted.append(_redact_env_assignment(token))
            pending_env = False
            continue
        if token in ("-e", "--env"):
            redacted.append(token)
            pending_env = True
            continue
        if token.startswith("--env="):
            redacted.append("--env=" + _redact_env_assignment(token[len("--env=") :]))
            continue
        if token.startswith("-e") and len(token) > 2:
            redacted.append("-e" + _redact_env_assignment(token[2:]))
            continue
        redacted.append(token)
    return redacted


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
            if spec.docker_image.strip().startswith("-"):
                raise TargetError(
                    f"docker_image must be an image name, got flag-like "
                    f"{spec.docker_image!r}"
                )
            # No host-side launch path exists for docker, but the shared
            # start-time re-check keeps every provider on one contract.
            ensure_target_paths_contained(spec)
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
            validate_docker_extra_args(spec.docker_extra_args)
            argv.extend(list(spec.docker_extra_args or []))
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
            # Env values routinely carry secrets and metadata is returned
            # by list_targets/get_target_status; keep names, mask values.
            handle.metadata["run_argv"] = redact_docker_run_argv(argv)
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

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        container_id = handle.metadata.get("container_id") or handle.metadata.get(
            "container_name"
        )
        if not container_id:
            return False
        try:
            rc, stdout, _stderr = self._docker(
                ["docker", "inspect", "-f", "{{.State.Running}}", str(container_id)],
                timeout_sec=10.0,
            )
        except Exception:
            return False
        return rc == 0 and stdout.strip().lower() == "true"

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
            destination = ensure_local_path_contained(
                destination,
                workspace_root=handle.spec.workspace_root,
                kind="destination",
            )
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
        source = ensure_local_path_contained(
            source,
            workspace_root=handle.spec.workspace_root,
            kind="source",
        )
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
