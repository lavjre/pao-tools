"""Local process + PID-attach providers (split from targets.py)."""

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

from paomcp.tools.backends.foundation import (
    build_backend_response,
    parse_int,
    validate_attach_pid,
)
from paomcp.tools.backends.gdb import GdbController

from paomcp.tools.backends.targets_base import (
    DEFAULT_TARGET_STOP_TIMEOUT_SEC,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetState,
    kill_process_tree,
    reap_output_drains,
    start_output_drains,
    validate_target_env,
)


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
            env_full.update(validate_target_env(spec.env))
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
                start_output_drains(handle)
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
                    kill_process_tree(process)
                    try:
                        process.wait(timeout=timeout_sec)
                    except subprocess.TimeoutExpired:
                        pass
            except ProcessLookupError:
                pass
            finally:
                reap_output_drains(handle)
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

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        return handle.process is not None and handle.process.poll() is None


class AttachPidProvider:
    """Attach to an existing PID we do not own."""

    kind = TargetKind.ATTACH_PID

    def start(self, handle: TargetHandle) -> None:
        with handle._lock:
            pid = handle.spec.pid
            if pid is None:
                raise TargetError("attach_pid spec has no pid")
            try:
                pid = validate_attach_pid(pid)
            except ValueError as exc:
                raise TargetError(str(exc))
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

    def check_alive(self, handle: TargetHandle) -> Optional[bool]:
        # Before start() populates handle.pid the spec still names the pid;
        # a stale-ERROR handle from a failed start must still be probed.
        pid = handle.pid or handle.spec.pid
        if pid is None:
            return None
        return _pid_is_alive(pid)


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
