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


BLOCKED_DOCKER_FLAGS = {
    "--privileged",
    "--pid",
    "--network",
    "--uts",
    "--ipc",
    "-v",
    "--volume",
    "--mount",
    "--device",
    "--cap-add",
    "--cap-drop",
    "--security-opt",
}

BLOCKED_DOCKER_TOKENS = ("--privileged", "-v /", "-v/", "--pid=host", "--network=host")


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
            for extra in list(spec.docker_extra_args or []):
                lowered = extra.strip().lower()
                if lowered in BLOCKED_DOCKER_FLAGS or any(
                    tok in lowered for tok in BLOCKED_DOCKER_TOKENS
                ):
                    raise TargetError(
                        f"Blocked unsafe docker_extra_args entry '{extra}': "
                        "privileged mode, host mounts, host pid/network, devices, "
                        "and capability changes are not allowed"
                    )
                argv.append(extra)
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
