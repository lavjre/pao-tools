"""Tests for QemuSystemProvider."""

from __future__ import annotations

import os
from typing import Any

import pytest

from paomcp.tools.backends.targets import (
    QemuSystemProvider,
    TargetError,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
)


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------
class TestQemuSystemSpec:
    def test_requires_qemu_binary(self) -> None:
        with pytest.raises(TargetError, match="qemu_binary"):
            TargetSpec.from_config("qemu_system", {"kernel_path": "/boot/vmlinuz"})

    def test_requires_kernel_or_disk(self) -> None:
        with pytest.raises(TargetError, match="kernel_path.*disk_path"):
            TargetSpec.from_config("qemu_system", {"qemu_binary": "qemu-system-x86_64"})

    def test_accepts_kernel_only(self) -> None:
        spec = TargetSpec.from_config(
            "qemu_system",
            {"qemu_binary": "qemu-system-x86_64", "kernel_path": "/boot/vmlinuz"},
        )
        assert spec.kind == TargetKind.QEMU_SYSTEM
        assert spec.qemu_binary == "qemu-system-x86_64"

    def test_accepts_disk_only(self) -> None:
        spec = TargetSpec.from_config(
            "qemu_system",
            {"qemu_binary": "qemu-system-x86_64", "disk_path": "/img/rootfs.img"},
        )
        assert spec.disk_path == "/img/rootfs.img"

    def test_full_config_fields_populated(self) -> None:
        spec = TargetSpec.from_config(
            "qemu_system",
            {
                "qemu_binary": "qemu-system-aarch64",
                "kernel_path": "/boot/vmlinuz",
                "initrd_path": "/boot/initrd.img",
                "disk_path": "/img/rootfs.img",
                "kernel_cmdline": "console=ttyS0",
                "qemu_memory": "256M",
                "qemu_machine": "virt",
                "qemu_extra_args": "-cpu cortex-a53",
            },
        )
        assert spec.qemu_machine == "virt"
        assert spec.qemu_memory == "256M"
        assert spec.qemu_extra_args == ["-cpu", "cortex-a53"]
        assert spec.kernel_cmdline == "console=ttyS0"


# ---------------------------------------------------------------------------
# argv construction (pure)
# ---------------------------------------------------------------------------
class TestQemuSystemArgv:
    @pytest.fixture(autouse=True)
    def _fake_which(self, monkeypatch) -> None:
        import shutil

        monkeypatch.setattr(shutil, "which", lambda name, path=None: f"/usr/bin/{name}")

    def _spec(self, **overrides: Any) -> TargetSpec:
        config = {
            "qemu_binary": "qemu-system-x86_64",
            "kernel_path": "/boot/vmlinuz",
        }
        config.update(overrides)
        return TargetSpec.from_config("qemu_system", config, session_id="sess")

    def test_argv_starts_with_gdb_and_pause_flags(self) -> None:
        argv = QemuSystemProvider._build_argv(self._spec(), "127.0.0.1", 4444)
        assert argv[0] == "/usr/bin/qemu-system-x86_64"
        assert argv[1:5] == ["-gdb", "tcp:127.0.0.1:4444", "-S", "-nographic"]

    def test_machine_and_memory_get_flags(self) -> None:
        argv = QemuSystemProvider._build_argv(
            self._spec(qemu_machine="q35", qemu_memory="512M"),
            "127.0.0.1",
            4444,
        )
        assert "-M" in argv and "q35" in argv
        assert "-m" in argv and "512M" in argv

    def test_kernel_initrd_and_cmdline_all_flow_through(self) -> None:
        argv = QemuSystemProvider._build_argv(
            self._spec(
                kernel_path="/boot/vmlinuz",
                initrd_path="/boot/initrd.img",
                kernel_cmdline="console=ttyS0 quiet",
            ),
            "127.0.0.1",
            4444,
        )
        assert "-kernel" in argv
        assert "-initrd" in argv
        assert "-append" in argv
        # -append value is the raw cmdline; check the next arg after it
        append_index = argv.index("-append")
        assert argv[append_index + 1] == "console=ttyS0 quiet"

    def test_disk_becomes_drive_flag_with_raw_format(self) -> None:
        argv = QemuSystemProvider._build_argv(
            TargetSpec.from_config(
                "qemu_system",
                {
                    "qemu_binary": "qemu-system-x86_64",
                    "disk_path": "/img/rootfs.img",
                },
                session_id="sess",
            ),
            "127.0.0.1",
            4444,
        )
        drive_index = argv.index("-drive")
        drive_value = argv[drive_index + 1]
        assert drive_value.startswith("file=") and drive_value.endswith(",format=raw")

    def test_missing_qemu_binary_at_build_time_raises(self) -> None:
        # Bypass from_config validation by constructing directly.
        spec = TargetSpec(kind=TargetKind.QEMU_SYSTEM, kernel_path="/boot/vmlinuz")
        with pytest.raises(TargetError, match="qemu_binary"):
            QemuSystemProvider._build_argv(spec, "127.0.0.1", 4444)

    @pytest.mark.parametrize(
        "extra",
        [
            "-virtfs",
            "-fsdev",
            "-netdev",
            "-nic",
            "-net",
            "-daemonize",
        ],
    )
    def test_supervision_escaping_flags_rejected(self, extra: str) -> None:
        from paomcp.tools.backends.targets_qemu import validate_qemu_extra_args

        with pytest.raises(TargetError, match="Blocked unsafe"):
            validate_qemu_extra_args([extra, "user,id=n0"])

    def test_benign_qemu_flags_pass_through(self) -> None:
        from paomcp.tools.backends.targets_qemu import validate_qemu_extra_args

        validate_qemu_extra_args(["-cpu", "max", "-smp", "1", "-m", "512M"])

    def test_infra_executable_resolution(self, tmp_path, monkeypatch) -> None:
        import shutil

        from paomcp.tools.backends.targets_base import resolve_infra_executable

        def _which(name, path=None):
            return None if "no_such" in name else "/usr/bin/" + name

        monkeypatch.setattr(shutil, "which", _which)
        assert (
            resolve_infra_executable("qemu-system-x86_64", what="qemu_binary")
            == "/usr/bin/qemu-system-x86_64"
        )
        helper = tmp_path / "gdbserver"
        helper.write_bytes(b"#!/bin/sh\nexit 0\n")
        if os.access(str(helper), os.X_OK):
            assert resolve_infra_executable(str(helper), what="gdbserver") == str(
                helper
            )
        else:
            with pytest.raises(TargetError, match="not executable"):
                resolve_infra_executable(str(helper), what="gdbserver")
        with pytest.raises(TargetError, match="not found"):
            resolve_infra_executable("__pao_no_such_helper__", what="qemu_binary")
        with pytest.raises(TargetError, match="non-empty"):
            resolve_infra_executable("   ", what="qemu_binary")


# ---------------------------------------------------------------------------
# Lifecycle error paths (host has no qemu-system installed, so we
# assert the failure surface, not a real boot)
# ---------------------------------------------------------------------------
class TestQemuSystemLifecycle:
    def test_provider_registered_by_kind(self) -> None:
        registry = TargetRegistry()
        provider = registry.provider_for(TargetKind.QEMU_SYSTEM)
        assert isinstance(provider, QemuSystemProvider)

    def test_missing_kernel_path_reports_at_start(self, tmp_path) -> None:
        registry = TargetRegistry()
        spec = TargetSpec.from_config(
            "qemu_system",
            {
                "qemu_binary": "__pao_not_a_real_qemu_system__",
                "kernel_path": str(tmp_path / "no-such-kernel"),
            },
            session_id="sess",
        )
        handle = registry.create(spec)
        response = registry.start_target_response(target_id=handle.target_id)
        parsed = response["parsed"]
        assert parsed["success"] is False
        assert handle.state == TargetState.ERROR
        assert "does not exist" in parsed["error"]

    def test_missing_qemu_binary_reports_at_start(self, tmp_path) -> None:
        kernel = tmp_path / "vmlinuz"
        kernel.write_bytes(b"NOT A KERNEL")
        registry = TargetRegistry()
        # Impossibly short listen timeout so the test does not hang.
        registry._providers[TargetKind.QEMU_SYSTEM] = QemuSystemProvider(
            listen_timeout_sec=0.01
        )
        spec = TargetSpec.from_config(
            "qemu_system",
            {
                "qemu_binary": "__pao_not_a_real_qemu_system__",
                "kernel_path": str(kernel),
                "port": 0,
            },
            session_id="sess",
        )
        handle = registry.create(spec)
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR

    def test_capabilities_declare_attach_and_no_file_sync(self, tmp_path) -> None:
        kernel = tmp_path / "vmlinuz"
        kernel.write_bytes(b"NOT A KERNEL")
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "qemu_system",
                {"qemu_binary": "qemu-system-x86_64", "kernel_path": str(kernel)},
                session_id="sess",
            )
        )
        caps = registry.get_capabilities_response(target_id=handle.target_id)["parsed"][
            "capabilities"
        ]
        assert caps["attach_debugger"] is True
        assert caps["file_sync"] is False
        assert caps["reconnect"] is True
