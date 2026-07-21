import base64
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from pwnomcp.artifacts import ArtifactPinnedError, ArtifactStore
from pwnomcp.tools.backends.subproc import SubprocessTools
from pwnomcp.tools.common import build_tool_response


def test_large_response_is_preserved_as_artifact(tmp_path):
    store = ArtifactStore(str(tmp_path), inline_response_bytes=32)
    full_output = "A" * 10_000

    response = build_tool_response(
        "run_command",
        {"success": True, "session_id": "chal-a", "stdout": full_output},
        artifact_store=store,
    )

    assert response["success"] is True
    assert response["session_id"] == "chal-a"
    assert response["data"]["delivery"] == "artifact"
    assert response["data"]["tool"] == "run_command"
    assert response["data"]["size_bytes"] > len(full_output)

    artifact_id = response["data"]["artifact_id"]
    stored = store.read(artifact_id)
    assert stored["eof"] is True
    assert json.loads(stored["content"]) == {"stdout": full_output}


def test_small_response_remains_inline(tmp_path):
    store = ArtifactStore(str(tmp_path), inline_response_bytes=1024)

    response = build_tool_response(
        "get_session_info",
        {"success": True, "state": "stopped"},
        artifact_store=store,
    )

    assert response["data"] == {"state": "stopped"}
    assert store.list() == []


def test_artifact_can_be_read_in_chunks(tmp_path):
    store = ArtifactStore(str(tmp_path), inline_response_bytes=1)
    payload = store.serialize_json({"value": "abcdefghij"})
    artifact = store.save_json_bytes(payload, tool="test")

    first = store.read(artifact.artifact_id, offset=0, length=5)
    second = store.read(
        artifact.artifact_id,
        offset=first["next_offset"],
        length=artifact.size_bytes,
    )

    assert first["eof"] is False
    assert second["eof"] is True
    assert first["content"] + second["content"] == payload.decode("utf-8")
    recovered = base64.b64decode(first["content_base64"]) + base64.b64decode(
        second["content_base64"]
    )
    assert recovered == payload


def test_artifact_file_copy_and_byte_ranges_are_lossless(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"), inline_response_bytes=1)
    payload = ("xin chào 🔬" * 1000).encode("utf-8")
    source = tmp_path / "output.bin"
    source.write_bytes(payload)

    artifact = store.save_file(str(source), tool="test")
    chunks = []
    offset = 0
    while offset < artifact.size_bytes:
        chunk = store.read(artifact.artifact_id, offset=offset, length=7)
        chunks.append(base64.b64decode(chunk["content_base64"]))
        offset = chunk["next_offset"]

    assert b"".join(chunks) == payload
    assert artifact.sha256


def test_run_command_spools_large_stdout_to_artifact(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"), inline_response_bytes=128)
    script = tmp_path / "emit_large_output.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(b'A' * 10000)\n", encoding="utf-8"
    )
    if os.name == "nt":
        command = subprocess.list2cmdline([sys.executable, str(script)])
    else:
        command = shlex.join([sys.executable, str(script)])
    tools = SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        artifact_store=store,
    )

    result = tools.run_command(command, cwd=str(tmp_path))

    assert result["success"] is True
    assert result["stdout"]["delivery"] == "artifact"
    artifact_id = result["stdout"]["artifact_id"]
    stored = store.read(artifact_id)
    assert base64.b64decode(stored["content_base64"]) == b"A" * 10000


def test_artifact_id_rejects_path_traversal(tmp_path):
    store = ArtifactStore(str(tmp_path))

    with pytest.raises(ValueError):
        store.read("../../outside")


def test_ttl_cleanup_deletes_expired_but_preserves_pinned(tmp_path):
    store = ArtifactStore(
        str(tmp_path), ttl_seconds=60, quota_bytes=0, min_free_bytes=0
    )
    expired = store.save_bytes(b"old", tool="test")
    pinned = store.save_bytes(b"important", tool="test")
    store.set_pinned(pinned.artifact_id)

    result = store.cleanup(now=datetime.now(timezone.utc) + timedelta(seconds=61))

    assert result["deleted_count"] == 1
    assert result["deletions"][0]["reason"] == "ttl"
    with pytest.raises(FileNotFoundError):
        store.get(expired.artifact_id)
    assert store.get(pinned.artifact_id).pinned is True


def test_cleanup_dry_run_does_not_delete(tmp_path):
    store = ArtifactStore(str(tmp_path), ttl_seconds=1, quota_bytes=0, min_free_bytes=0)
    artifact = store.save_bytes(b"preview", tool="test")

    result = store.cleanup(
        dry_run=True,
        now=datetime.now(timezone.utc) + timedelta(seconds=2),
    )

    assert result["deleted_count"] == 1
    assert store.get(artifact.artifact_id) == artifact


def test_quota_cleanup_evicts_oldest_and_keeps_newest(tmp_path):
    store = ArtifactStore(str(tmp_path), ttl_seconds=0, quota_bytes=0, min_free_bytes=0)
    first = store.save_bytes(b"A" * 40, tool="test")
    second = store.save_bytes(b"B" * 40, tool="test")
    newest = store.save_bytes(b"C" * 40, tool="test")
    store.quota_bytes = 100
    store.cleanup_target_ratio = 0.5

    result = store.cleanup()

    assert result["deleted_count"] == 2
    assert [item["artifact_id"] for item in result["deletions"]] == [
        first.artifact_id,
        second.artifact_id,
    ]
    assert [item.artifact_id for item in store.list()] == [newest.artifact_id]


def test_pinned_artifact_requires_force_to_delete(tmp_path):
    store = ArtifactStore(str(tmp_path), min_free_bytes=0)
    artifact = store.save_bytes(b"important", tool="test")
    store.set_pinned(artifact.artifact_id)

    with pytest.raises(ArtifactPinnedError):
        store.delete(artifact.artifact_id)

    deleted = store.delete(artifact.artifact_id, force=True)
    assert deleted.artifact_id == artifact.artifact_id
