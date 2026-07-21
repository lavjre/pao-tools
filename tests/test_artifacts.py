import base64
import json
import os
import shlex
import subprocess
import sys

import pytest

from pwnomcp.artifacts import ArtifactStore
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
