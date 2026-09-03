import pytest

from paomcp.tools.targets import _contain_target_config


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
