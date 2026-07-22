import os

import pytest

from paomcp.utils.paths import (
    build_runtime_paths,
    resolve_workspace_path,
    validate_session_id,
)


def test_resolve_workspace_path_from_relative():
    assert (
        resolve_workspace_path("chall", workspace_root="/workspace")
        == "/workspace/chall"
    )


def test_resolve_workspace_path_from_host_mounted_workspace_path():
    host_style = "/home/user/project/workspace/bin/chall"
    assert (
        resolve_workspace_path(host_style, workspace_root="/workspace")
        == "/workspace/bin/chall"
    )


def test_resolve_workspace_path_rejects_outside_workspace_absolute_path():
    with pytest.raises(ValueError) as exc:
        resolve_workspace_path("/bin/ls", workspace_root="/workspace")

    message = str(exc.value)
    assert "host paths are not directly visible" in message.lower()
    assert "/workspace/<name>" in message


def test_resolve_workspace_path_rejects_escape_from_relative_path():
    with pytest.raises(ValueError) as exc:
        resolve_workspace_path("../../etc/passwd", workspace_root="/workspace")

    message = str(exc.value)
    assert "escapes /workspace" in message
    assert "relative path" in message.lower()


def test_resolve_workspace_path_missing_file_has_mount_hint():
    with pytest.raises(FileNotFoundError) as exc:
        resolve_workspace_path(
            "missing/chal", workspace_root="/workspace", require_exists=True
        )

    message = str(exc.value)
    assert "not found" in message.lower()
    assert "host './workspace/<name>' maps to '/workspace/<name>'" in message


def test_build_runtime_paths_creates_isolated_directories(tmp_path):
    workspace_root = str(tmp_path / "workspace")
    runtime_root = str(tmp_path / "runtime")
    paths = build_runtime_paths(
        workspace_root=workspace_root, runtime_root=runtime_root
    )

    assert paths.workspace_root == workspace_root
    assert paths.runtime_root == runtime_root
    assert os.path.isdir(paths.sessions_dir)
    assert os.path.isdir(paths.processes_dir)
    assert os.path.isdir(paths.python_dir)
    assert os.path.isdir(paths.repos_dir)
    assert os.path.isdir(paths.artifacts_dir)


def test_session_ids_are_validated_instead_of_normalized():
    assert validate_session_id("chal-a.1") == "chal-a.1"

    with pytest.raises(ValueError):
        validate_session_id("chal/a")

    with pytest.raises(ValueError):
        validate_session_id("chal?a")
