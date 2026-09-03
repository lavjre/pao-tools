"""QEMU user/system providers (split from targets.py)."""

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
    DEFAULT_QEMU_LISTEN_TIMEOUT_SEC,
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetSpec,
    TargetState,
    resolve_infra_executable,
)
from paomcp.tools.backends.targets_gdbserver import (
    _attach_via_gdb_remote,
    _find_free_tcp_port,
    _wait_for_tcp_listen,
)
from paomcp.tools.backends.targets_local import (
    LocalProcessProvider,
)


def _qemu_flag_name(entry: str) -> str:
    """Normalize one extra arg to its flag name (``-netdev x`` -> ``-netdev``)."""
    text = entry.strip()
    for separator in ("=",):
        head, sep, _ = text.partition(separator)
        if sep and head.startswith("-"):
            return head.lower()
    return text.split()[0].lower() if text else ""


BLOCKED_QEMU_FLAGS = frozenset(
    {
        "-virtfs",
        "-fsdev",
        "-netdev",
        "-nic",
        "-net",
        "-daemonize",
        "-daemon",
    }
)


def validate_qemu_extra_args(extra_args) -> None:
    """Reject QEMU flags that escape supervision or the declared disk set.

    ``-virtfs``/``-fsdev`` share host trees, ``-netdev``/``-nic``/``-net``
    rewire host networking, and ``-daemonize`` detaches from our process
    supervision. ``-drive``/``-kernel``/``-initrd`` file operands stay under
    workspace containment enforced at target creation.
    """
    for extra in list(extra_args or []):
        if _qemu_flag_name(extra) in BLOCKED_QEMU_FLAGS:
            raise TargetError(
                f"Blocked unsafe qemu_extra_args entry '{extra}': host "
                "filesystem sharing, custom networking, and detaching from "
                "supervision are not allowed"
            )


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
            qemu_executable = resolve_infra_executable(
                spec.qemu_binary, what="qemu_binary"
            )
            host = spec.host or "127.0.0.1"
            port = spec.port or _find_free_tcp_port(host)
            argv: List[str] = [qemu_executable]
            if spec.qemu_ld_prefix:
                argv += ["-L", spec.qemu_ld_prefix]
            validate_qemu_extra_args(spec.qemu_extra_args)
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

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # Same subprocess-ownership contract as LocalProcessProvider.
        return handle.process is not None and handle.process.poll() is None


class QemuSystemProvider:
    """Boot a full-system QEMU VM with a paused gdb stub bound to a TCP port.

    ``start`` spawns ``qemu-system-<arch> -gdb tcp:host:port -S
    [-M machine] [-m memory] [-kernel K] [-initrd I] [-append CMD]
    [-drive file=DISK,format=raw] [-nographic] [extra]`` -- the ``-S``
    flag holds the guest CPU at the reset vector so :meth:`attach_debugger`
    is guaranteed a stopped guest to attach to.  ``attach_debugger``
    uses the shared ``_attach_via_gdb_remote`` helper so the wire
    contract matches every other remote provider.

    A caller is responsible for providing a functional VM
    configuration (kernel / initrd / disk); we do not synthesize one.
    Windows hosts have no qemu-system binaries, so real end-to-end
    exercise happens from WSL / Linux.  The spec validation and error
    paths are covered here regardless of host.
    """

    kind = TargetKind.QEMU_SYSTEM

    def __init__(
        self,
        *,
        listen_timeout_sec: float = DEFAULT_QEMU_LISTEN_TIMEOUT_SEC,
    ) -> None:
        self._listen_timeout_sec = listen_timeout_sec

    @staticmethod
    def _build_argv(spec: TargetSpec, host: str, port: int) -> List[str]:
        if not spec.qemu_binary:
            raise TargetError(
                "qemu_system spec requires 'qemu_binary' " "(e.g. 'qemu-system-x86_64')"
            )
        argv: List[str] = [
            resolve_infra_executable(spec.qemu_binary, what="qemu_binary"),
            "-gdb",
            f"tcp:{host}:{port}",
            "-S",  # freeze the guest at boot so attach never races
            "-nographic",
        ]
        if spec.qemu_machine:
            argv += ["-M", spec.qemu_machine]
        if spec.qemu_memory:
            argv += ["-m", spec.qemu_memory]
        if spec.kernel_path:
            argv += ["-kernel", os.path.abspath(spec.kernel_path)]
        if spec.initrd_path:
            argv += ["-initrd", os.path.abspath(spec.initrd_path)]
        if spec.kernel_cmdline:
            argv += ["-append", spec.kernel_cmdline]
        if spec.disk_path:
            argv += [
                "-drive",
                f"file={os.path.abspath(spec.disk_path)},format=raw",
            ]
        validate_qemu_extra_args(spec.qemu_extra_args)
        argv += list(spec.qemu_extra_args or [])
        return argv

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            spec = handle.spec
            for path_attr in ("kernel_path", "initrd_path", "disk_path"):
                path = getattr(spec, path_attr)
                if path and not os.path.isfile(path):
                    raise TargetError(
                        f"qemu_system {path_attr} '{path}' does not exist"
                    )
            host = spec.host or "127.0.0.1"
            port = spec.port or _find_free_tcp_port(host)
            argv = self._build_argv(spec, host, port)
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
                        f"qemu-system did not open gdb stub on {host}:{port} "
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
                    f"qemu_system target '{handle.target_id}' is not listening; "
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
                handle.error = attach_result.get("error") or "qemu-system attach failed"
                handle.error_type = attach_result.get("type") or "QemuSystemAttachError"
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
                "spawns qemu-system-<arch> with '-gdb tcp:host:port -S' and "
                "attaches via '-target-select remote host:port'"
            ),
        }

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # Same subprocess-ownership contract as LocalProcessProvider.
        return handle.process is not None and handle.process.poll() is None
