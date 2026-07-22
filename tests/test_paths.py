import os

import pytest

from paomcp.utils.paths import (
    build_runtime_paths,
    resolve_workspace_file,
    resolve_workspace_path,
    stable_file_snapshot,
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


def test_workspace_file_rejects_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.elf"
    outside.write_bytes(b"\x7fELF")
    link = workspace / "escape.elf"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="outside"):
        resolve_workspace_file(str(link), str(workspace), kind="binary_path")


def test_stable_file_snapshot_is_independent_from_later_source_changes(tmp_path):
    source = tmp_path / "input.elf"
    runtime = tmp_path / "runtime"
    source.write_bytes(b"original")
    source_mtime_ns = source.stat().st_mtime_ns

    with stable_file_snapshot(str(source), str(runtime)) as snapshot:
        assert os.stat(snapshot).st_mtime_ns == source_mtime_ns
        source.write_bytes(b"replacement")
        with open(snapshot, "rb") as captured:
            assert captured.read() == b"original"

    assert not os.path.exists(snapshot)


def test_stable_snapshot_rejects_file_replaced_by_external_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    source = workspace / "input.elf"
    outside = tmp_path / "outside.elf"
    source.write_bytes(b"inside")
    outside.write_bytes(b"outside")
    resolved = resolve_workspace_file(str(source), str(workspace))
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises((OSError, ValueError, RuntimeError)):
        with stable_file_snapshot(
            resolved,
            str(runtime),
            workspace_root=str(workspace),
        ):
            pass


def test_stable_snapshot_rejects_parent_replaced_by_external_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    source_dir = workspace / "nested"
    outside_dir = tmp_path / "outside"
    source_dir.mkdir(parents=True)
    outside_dir.mkdir()
    source = source_dir / "input.elf"
    source.write_bytes(b"inside")
    (outside_dir / "input.elf").write_bytes(b"outside")
    resolved = resolve_workspace_file(str(source), str(workspace))
    source.unlink()
    source_dir.rmdir()
    try:
        source_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises((OSError, ValueError, RuntimeError)):
        with stable_file_snapshot(
            resolved,
            str(runtime),
            workspace_root=str(workspace),
        ):
            pass
