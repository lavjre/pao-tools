import os

from paomcp.tools.backends.git import GitTools


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
