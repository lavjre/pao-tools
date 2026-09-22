import os

import pytest

from paomcp.tools.backends.git import GitTools, _redact_repo_url


def test_redact_repo_url_strips_embedded_credentials():
    assert (
        _redact_repo_url("https://user:super-secret@github.com/a/b.git")
        == "https://***@github.com/a/b.git"
    )
    assert (
        _redact_repo_url("https://github.com/a/b.git") == "https://github.com/a/b.git"
    )
    assert _redact_repo_url("git@github.com:a/b.git") == "git@github.com:a/b.git"


def test_fetch_repo_rejects_workspace_root_as_target_dir(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "notes.txt"
    evidence.write_text("keep me", encoding="utf-8")
    tools = GitTools(workspace_dir=str(workspace))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git", target_dir=".", overwrite=True
    )

    # The workspace root is not a repository target: overwrite must never
    # rmtree it.
    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
    assert evidence.read_text(encoding="utf-8") == "keep me"
    assert workspace.is_dir()


def test_fetch_repo_rejects_url_that_derives_the_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "notes.txt"
    evidence.write_text("keep me", encoding="utf-8")
    tools = GitTools(workspace_dir=str(workspace))

    result = tools.fetch_repo("https://example.invalid/repo/.", overwrite=True)

    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
    assert evidence.read_text(encoding="utf-8") == "keep me"
    assert workspace.is_dir()


def test_fetch_repo_rejects_symlinked_parent_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    evidence = outside / "victim.txt"
    evidence.write_text("do not delete", encoding="utf-8")
    link = workspace / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not supported on this host")
    tools = GitTools(workspace_dir=str(workspace))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git", target_dir="link", overwrite=True
    )

    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
    assert evidence.read_text(encoding="utf-8") == "do not delete"
    assert outside.is_dir()


def test_fetch_repo_rejects_symlink_that_resolves_to_workspace_root(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    evidence = workspace / "notes.txt"
    evidence.write_text("keep me", encoding="utf-8")
    link = workspace / "self"
    try:
        link.symlink_to(workspace, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not supported on this host")
    tools = GitTools(workspace_dir=str(workspace))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git", target_dir="self", overwrite=True
    )

    # The symlink resolves to the workspace root, which must never be
    # rmtree'd.
    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
    assert evidence.read_text(encoding="utf-8") == "keep me"
    assert workspace.is_dir()


def test_fetch_repo_rechecks_containment_before_rmtree(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "repo"
    target.mkdir()
    evidence = workspace / "notes.txt"
    evidence.write_text("keep me", encoding="utf-8")
    real_realpath = os.path.realpath
    real_target = real_realpath(target)
    real_workspace = real_realpath(workspace)
    calls = {"target": 0}

    def swapped_realpath(path):
        resolved = real_realpath(path)
        if resolved == real_target:
            calls["target"] += 1
            if calls["target"] >= 2:
                # Simulate a symlink swap after the first containment check.
                return real_workspace
        return resolved

    monkeypatch.setattr(os.path, "realpath", swapped_realpath)
    tools = GitTools(workspace_dir=str(workspace))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git", target_dir="repo", overwrite=True
    )

    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "after symlink resolution" in result["error"]
    assert evidence.read_text(encoding="utf-8") == "keep me"
    assert target.is_dir()


def test_fetch_repo_does_not_delete_existing_target(tmp_path):
    target = tmp_path / "analysis"
    target.mkdir()
    evidence = target / "notes.txt"
    evidence.write_text("keep me", encoding="utf-8")
    tools = GitTools(workspace_dir=str(tmp_path))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git", target_dir=str(target)
    )

    assert result["success"] is False
    assert result["type"] == "FileExistsError"
    assert evidence.read_text(encoding="utf-8") == "keep me"


def test_fetch_repo_rejects_target_outside_workspace(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text("do not delete", encoding="utf-8")
    tools = GitTools(workspace_dir=str(tmp_path / "workspace"))

    result = tools.fetch_repo(
        "https://example.invalid/repo.git",
        target_dir=str(outside),
        overwrite=True,
    )

    # Rejected before any clone or rmtree: the outside directory survives.
    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
    assert victim.read_text(encoding="utf-8") == "do not delete"


def test_fetch_repo_rejects_dotdot_escape(tmp_path):
    tools = GitTools(workspace_dir=str(tmp_path / "workspace"))
    result = tools.fetch_repo(
        "https://example.invalid/repo.git",
        target_dir="../escaped",
        overwrite=True,
    )

    assert result["success"] is False
    assert result["type"] == "ValueError"
    assert "escapes the git workspace" in result["error"]
