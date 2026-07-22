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
