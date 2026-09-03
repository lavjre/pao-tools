"""Tests for Stage 8.4 DockerProvider + QemuUserProvider."""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import pytest

from paomcp.tools.backends.targets import (
    DEFAULT_DOCKER_TIMEOUT_SEC,
    DockerProvider,
    QemuUserProvider,
    TargetError,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
)


# ---------------------------------------------------------------------------
# Fake docker runner: records the argv, returns a scripted (rc, stdout, stderr)
# ---------------------------------------------------------------------------
class _FakeDockerRunner:
    def __init__(
        self,
        script: Optional[List[Tuple[int, str, str]]] = None,
        *,
        default: Tuple[int, str, str] = (0, "", ""),
    ) -> None:
        self.calls: List[Tuple[List[str], float]] = []
        self._script = list(script or [])
        self._default = default

    def __call__(self, argv: List[str], timeout_sec: float) -> Tuple[int, str, str]:
        self.calls.append((argv, timeout_sec))
        if self._script:
            return self._script.pop(0)
        return self._default


def _docker_registry(runner: _FakeDockerRunner) -> TargetRegistry:
    registry = TargetRegistry()
    registry._providers[TargetKind.DOCKER] = DockerProvider(runner=runner)
    return registry


def _docker_spec(**overrides: Any) -> TargetSpec:
    config: Dict[str, Any] = {"docker_image": "pao/ctf:latest"}
    config.update(overrides)
    return TargetSpec.from_config("docker", config, session_id="sess")


# ---------------------------------------------------------------------------
# DockerProvider argv construction + lifecycle
# ---------------------------------------------------------------------------
class TestDockerStart:
    def test_argv_includes_run_dash_d_and_image(self) -> None:
        runner = _FakeDockerRunner(script=[(0, "container-abc123\n", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec(name="pao-test"))
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is True
        assert handle.state == TargetState.RUNNING
        argv = runner.calls[0][0]
        assert argv[:4] == ["docker", "run", "-d", "--rm"]
        assert "--name" in argv and "pao-test" in argv
        assert argv[-1] == "pao/ctf:latest"
        assert handle.metadata["container_id"] == "container-abc123"

    def test_workdir_and_env_flags(self) -> None:
        runner = _FakeDockerRunner(script=[(0, "cid\n", "")])
        registry = _docker_registry(runner)
        handle = registry.create(
            _docker_spec(
                docker_workdir="/pwn",
                env={"CTF_FLAG_DIR": "/flags"},
            )
        )
        registry.start_target_response(target_id=handle.target_id)
        argv = runner.calls[0][0]
        assert "-w" in argv and "/pwn" in argv
        assert "-e" in argv
        assert "CTF_FLAG_DIR=/flags" in argv

    def test_docker_run_failure_records_error(self) -> None:
        runner = _FakeDockerRunner(script=[(1, "", "Error: image not found")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR
        assert "image not found" in handle.error

    def test_missing_docker_image_raises(self) -> None:
        with pytest.raises(TargetError, match="'docker_image'"):
            TargetSpec.from_config("docker", {})

    @pytest.mark.parametrize(
        "extra",
        [
            "--privileged",
            "--cap-add=NET_ADMIN",
            "--cap-drop ALL",
            "--device=/dev/kvm",
            "--volume=x:/y",
            "-v",
            "-v/src:/dst",
            "-v /:/host",
            "--mount=type=bind,src=/,dst=/host",
            "--security-opt=seccomp=unconfined",
            "--pid=host",
            "--pid",
            "--network=host",
            "--gpus=all",
            "--runtime=nvidia",
        ],
    )
    def test_privilege_mount_device_flags_rejected(self, extra: str) -> None:
        runner = _FakeDockerRunner()
        provider = DockerProvider(runner=runner)
        handle = _docker_registry(runner).create(
            _docker_spec(docker_extra_args=[extra])
        )
        with pytest.raises(TargetError, match="Blocked unsafe"):
            provider.start(handle)
        assert runner.calls == []

    @pytest.mark.parametrize(
        "extra",
        ["--memory=512m", "--cpus=2", "--read-only", "--shm-size=256m", "--user=1000"],
    )
    def test_benign_resource_flags_pass_through(self, extra: str) -> None:
        runner = _FakeDockerRunner(script=[(0, "cid\n", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec(docker_extra_args=[extra]))
        registry.start_target_response(target_id=handle.target_id)
        assert extra in runner.calls[0][0]


class TestDockerStop:
    def test_stop_runs_docker_stop_then_marks_stopped(self) -> None:
        runner = _FakeDockerRunner(script=[(0, "cid\n", ""), (0, "cid\n", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        registry.start_target_response(target_id=handle.target_id)
        registry.stop_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.STOPPED
        stop_argv = runner.calls[-1][0]
        assert stop_argv[:2] == ["docker", "stop"]

    def test_stop_falls_back_to_docker_kill_on_failure(self) -> None:
        runner = _FakeDockerRunner(
            script=[
                (0, "cid\n", ""),
                (1, "", "stop timed out"),  # stop fails
                (0, "cid\n", ""),  # kill succeeds
            ]
        )
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        registry.start_target_response(target_id=handle.target_id)
        registry.stop_target_response(target_id=handle.target_id)
        argvs = [call[0] for call in runner.calls]
        assert any(a[:2] == ["docker", "kill"] for a in argvs)


class TestDockerAttachDebugger:
    def test_attach_is_unsupported_with_typed_error(self) -> None:
        runner = _FakeDockerRunner(script=[(0, "cid\n", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        registry.start_target_response(target_id=handle.target_id)

        class _Gdb:
            def attach(self, pid: int) -> Any:
                raise AssertionError

            def execute_mi_command(self, *a: Any, **kw: Any) -> Any:
                raise AssertionError

        response = registry.attach_debugger_response(
            target_id=handle.target_id, gdb=_Gdb()  # type: ignore[arg-type]
        )
        assert response["parsed"]["success"] is False
        assert "docker provider" in response["parsed"]["error"]


class TestDockerFileSync:
    def test_download_calls_docker_cp_container_to_local(self, tmp_path) -> None:
        # Second call is docker cp; first was docker run.
        runner = _FakeDockerRunner(script=[(0, "cid\n", ""), (0, "", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        registry.start_target_response(target_id=handle.target_id)
        local = tmp_path / "pulled.bin"
        # docker cp is faked so we must materialize the file ourselves for
        # the stat() call the sync helper performs.
        local.write_bytes(b"OK")
        registry.sync_target_files_response(
            target_id=handle.target_id,
            source="/data/flag.bin",
            destination=str(local),
            direction="download",
        )
        cp_argv = runner.calls[-1][0]
        assert cp_argv[:2] == ["docker", "cp"]
        assert cp_argv[2].endswith(":/data/flag.bin")
        assert cp_argv[3] == str(local)

    def test_upload_calls_docker_cp_local_to_container(self, tmp_path) -> None:
        runner = _FakeDockerRunner(script=[(0, "cid\n", ""), (0, "", "")])
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        registry.start_target_response(target_id=handle.target_id)
        payload = tmp_path / "payload.bin"
        payload.write_bytes(b"PAYLOAD")
        registry.sync_target_files_response(
            target_id=handle.target_id,
            source=str(payload),
            destination="/tmp/pushed.bin",
            direction="upload",
        )
        cp_argv = runner.calls[-1][0]
        assert cp_argv[:2] == ["docker", "cp"]
        assert cp_argv[2] == str(payload)
        assert cp_argv[3].endswith(":/tmp/pushed.bin")


class TestDockerCapabilities:
    def test_declares_file_sync_and_no_attach(self) -> None:
        runner = _FakeDockerRunner()
        registry = _docker_registry(runner)
        handle = registry.create(_docker_spec())
        caps = registry.get_capabilities_response(target_id=handle.target_id)["parsed"][
            "capabilities"
        ]
        assert caps["file_sync"] is True
        assert caps["attach_debugger"] is False


# ---------------------------------------------------------------------------
# QemuUserProvider
# ---------------------------------------------------------------------------
class TestQemuSpec:
    def test_requires_binary_and_qemu_binary(self) -> None:
        with pytest.raises(TargetError, match="binary_path"):
            TargetSpec.from_config("qemu_user", {"qemu_binary": "qemu-x86_64"})
        with pytest.raises(TargetError, match="qemu_binary"):
            TargetSpec.from_config("qemu_user", {"binary_path": "/bin/x"})

    def test_accepts_ld_prefix_and_extra_args(self) -> None:
        spec = TargetSpec.from_config(
            "qemu_user",
            {
                "binary_path": "/bin/x",
                "qemu_binary": "qemu-aarch64",
                "qemu_ld_prefix": "/opt/aarch64-sysroot",
                "qemu_extra_args": "-cpu max -smp 1",
            },
        )
        assert spec.qemu_binary == "qemu-aarch64"
        assert spec.qemu_ld_prefix == "/opt/aarch64-sysroot"
        assert spec.qemu_extra_args == ["-cpu", "max", "-smp", "1"]


class TestQemuStart:
    def test_missing_qemu_binary_reports_error(self, tmp_path) -> None:
        binary = tmp_path / "prog"
        binary.write_bytes(b"\x7fELF" + b"P" * 32)
        # Point at an executable that surely does not exist.
        registry = TargetRegistry()
        # Use an impossibly short timeout so we do not have to wait.
        registry._providers[TargetKind.QEMU_USER] = QemuUserProvider(
            listen_timeout_sec=0.01
        )
        handle = registry.create(
            TargetSpec.from_config(
                "qemu_user",
                {
                    "binary_path": str(binary),
                    "qemu_binary": "__pao_not_a_real_qemu__",
                    "port": 0,
                },
                session_id="sess",
            )
        )
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR

    def test_capabilities_declare_attach_and_no_file_sync(self, tmp_path) -> None:
        binary = tmp_path / "prog"
        binary.write_bytes(b"\x7fELF" + b"Q" * 32)
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "qemu_user",
                {"binary_path": str(binary), "qemu_binary": "qemu-x86_64"},
                session_id="sess",
            )
        )
        caps = registry.get_capabilities_response(target_id=handle.target_id)["parsed"][
            "capabilities"
        ]
        assert caps["attach_debugger"] is True
        assert caps["file_sync"] is False


def test_default_docker_timeout_is_sane() -> None:
    assert DEFAULT_DOCKER_TIMEOUT_SEC >= 5.0
