import os
import subprocess

import pytest

from paomcp.tools.backends.targets_base import TargetError, validate_target_env
from paomcp.tools.targets import _contain_target_config


def make_directory_link(link, target) -> None:
    """Create a directory link (symlink, or an unprivileged junction).

    Windows symlinks need SeCreateSymbolicLinkPrivilege; junctions do not,
    so containment tests can still exercise a real ``realpath`` escape.
    """
    target.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        if result.returncode == 0:
            return
    pytest.skip("directory links are unavailable")


def retarget_directory_link(link, target) -> None:
    """Re-point an existing directory link (the create/start TOCTOU swap)."""
    try:
        os.rmdir(link)
    except OSError:
        link.unlink()
    make_directory_link(link, target)


def test_workspace_paths_are_rebased(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {
            "binary_path": "chal",
            "working_dir": "work",
            "kernel_path": "bzImage",
            "qemu_extra_args": ["-cpu", "max"],
        },
        workspace_root=workspace,
    )
    assert config["binary_path"].replace("\\", "/").endswith("/workspace/chal")
    assert config["working_dir"].replace("\\", "/").endswith("/workspace/work")
    assert config["kernel_path"].replace("\\", "/").endswith("/workspace/bzImage")
    assert config["qemu_extra_args"] == ["-cpu", "max"]


def test_host_paths_are_rejected(tmp_path):
    workspace = str(tmp_path / "workspace")
    for key in (
        "binary_path",
        "working_dir",
        "kernel_path",
        "initrd_path",
        "disk_path",
        "qemu_ld_prefix",
    ):
        with pytest.raises(ValueError, match="workspace"):
            _contain_target_config({key: "/etc/shadow"}, workspace_root=workspace)


def test_qemu_drive_file_operand_is_contained(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {"qemu_extra_args": ["-drive", "file=rootfs.img,format=raw"]},
        workspace_root=workspace,
    )
    assert config["qemu_extra_args"][1].replace("\\", "/").endswith(
        "/workspace/rootfs.img,format=raw"
    ) or "workspace" in config["qemu_extra_args"][1].replace("\\", "/")
    with pytest.raises(ValueError, match="workspace"):
        _contain_target_config(
            {"qemu_extra_args": ["-drive", "file=/etc/passwd,format=raw"]},
            workspace_root=workspace,
        )
    with pytest.raises(ValueError, match="workspace"):
        _contain_target_config(
            {"qemu_extra_args": ["-kernel", "/boot/vmlinuz"]},
            workspace_root=workspace,
        )


def test_ssh_key_paths_pass_through(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {"ssh_key_path": "/root/.ssh/id_rsa", "docker_workdir": "/pwn"},
        workspace_root=workspace,
    )
    assert config["ssh_key_path"] == "/root/.ssh/id_rsa"
    assert config["docker_workdir"] == "/pwn"


def test_workspace_root_is_stamped_for_launch_recheck(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config({"binary_path": "chal"}, workspace_root=workspace)
    assert config["workspace_root"] == workspace


@pytest.mark.parametrize(
    "extra",
    [
        ["--kernel=/boot/vmlinuz"],
        ["-kernel=/boot/vmlinuz"],
        ["-kernel", "/boot/vmlinuz"],
        ["-initrd=/boot/initrd.img"],
        ["-blockdev", "driver=file,filename=/etc/passwd"],
        ["-device", "loader,file=/etc/passwd"],
        ["-serial", "file:/etc/passwd"],
        ["-serial", "pipe:/etc/passwd"],
        ["-chardev", "file,path=/etc/passwd"],
        ["-chardev", "socket,path=/etc/passwd"],
        ["-L", "/etc"],
        ["-L/etc"],
        ["-L=/etc"],
    ],
)
def test_qemu_file_operand_spellings_rejected(tmp_path, extra):
    workspace = str(tmp_path / "workspace")
    with pytest.raises(ValueError, match="workspace"):
        _contain_target_config({"qemu_extra_args": extra}, workspace_root=workspace)


def test_qemu_glued_kernel_is_rebased(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {"qemu_extra_args": ["--kernel=bzImage", "-initrd=initrd.img"]},
        workspace_root=workspace,
    )
    assert (
        config["qemu_extra_args"][0].replace("\\", "/").endswith("/workspace/bzImage")
    )
    assert (
        config["qemu_extra_args"][1]
        .replace("\\", "/")
        .endswith("/workspace/initrd.img")
    )


def test_qemu_comma_operands_are_rebased(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {
            "qemu_extra_args": [
                "-blockdev",
                "driver=file,filename=rootfs.img",
                "-device",
                "loader,file=loader.img",
                "-serial",
                "file:serial.log",
                "-chardev",
                "file,path=chardev.log",
            ]
        },
        workspace_root=workspace,
    )
    joined = " ".join(config["qemu_extra_args"]).replace("\\", "/")
    for name in ("rootfs.img", "loader.img", "serial.log", "chardev.log"):
        assert f"/workspace/{name}" in joined


def test_qemu_L_prefix_is_rebased(tmp_path):
    workspace = str(tmp_path / "workspace")
    config = _contain_target_config(
        {"qemu_extra_args": ["-L", "sysroot"]}, workspace_root=workspace
    )
    assert (
        config["qemu_extra_args"][1].replace("\\", "/").endswith("/workspace/sysroot")
    )


def test_qemu_ld_prefix_env_is_blocked():
    with pytest.raises(TargetError, match="QEMU_LD_PREFIX"):
        validate_target_env({"QEMU_LD_PREFIX": "/etc"})
