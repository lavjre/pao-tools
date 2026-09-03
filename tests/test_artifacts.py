import base64
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paomcp.artifacts import (
    BINARY_ARTIFACT_ENCODING,
    ArtifactPinnedError,
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    ArtifactStore,
    WriterNotQuiescentError,
)
from paomcp.tools.backends.subproc import SubprocessTools
from paomcp.tools.common import build_tool_response


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


def test_generated_artifact_persists_provenance_and_loads_legacy_metadata(tmp_path):
    store = ArtifactStore(str(tmp_path), min_free_bytes=0)
    provenance = {
        "schema": "pao.core-provenance.v1",
        "crash_id": "alpha:generation:1:1",
        "session_id": "alpha",
    }

    metadata, _ = store.save_generated_file(
        lambda path: Path(path).write_bytes(b"core"),
        reserve_bytes=4,
        max_size_bytes=4,
        tool="capture_core_dump",
        session_id="alpha",
        content_type="application/x-coredump",
        provenance=provenance,
    )
    provenance["crash_id"] = "mutated-after-save"

    loaded = store.get(metadata.artifact_id)
    assert loaded.provenance == {
        "schema": "pao.core-provenance.v1",
        "crash_id": "alpha:generation:1:1",
        "session_id": "alpha",
    }

    metadata_path = tmp_path / f"{metadata.artifact_id}.meta.json"
    legacy_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy_payload.pop("provenance")
    metadata_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert store.get(metadata.artifact_id).provenance is None


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


def test_generated_file_is_atomically_adopted_with_generator_evidence(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"), quota_bytes=0, min_free_bytes=0)
    payload = b"\x00core\xffpayload"
    generated = {}

    def generator(staging_path):
        assert Path(staging_path).parent == Path(store.root_dir)
        assert Path(staging_path).name.startswith(".artifact_")
        Path(staging_path).write_bytes(payload)
        generated["inode"] = os.stat(staging_path).st_ino
        return {"command": "generate-core-file", "success": True}

    artifact, evidence = store.save_generated_file(
        generator,
        reserve_bytes=len(payload),
        max_size_bytes=1024,
        tool="capture_core_dump",
        session_id="debug-session",
        content_type="application/x-core",
    )

    final_path = Path(store.root_dir) / f"{artifact.artifact_id}.data"
    assert final_path.read_bytes() == payload
    assert os.stat(final_path).st_ino == generated["inode"]
    assert artifact.encoding == BINARY_ARTIFACT_ENCODING
    assert artifact.sha256
    assert evidence == {"command": "generate-core-file", "success": True}
    assert not any(
        path.name.startswith(".artifact_") for path in Path(store.root_dir).iterdir()
    )

    read = store.read(artifact.artifact_id)
    assert "content" not in read
    assert base64.b64decode(read["content_base64"]) == payload


def test_generated_file_failure_always_removes_partial_staging(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)

    def generator(staging_path):
        Path(staging_path).write_bytes(b"partial")
        raise RuntimeError("generator failed")

    with pytest.raises(RuntimeError, match="generator failed"):
        store.save_generated_file(
            generator,
            reserve_bytes=128,
            max_size_bytes=128,
            tool="capture_core_dump",
        )

    assert list(Path(store.root_dir).iterdir()) == []


def test_unquiet_generated_file_is_quarantined_and_accounted(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    payload = b"possibly-still-growing"
    max_size_bytes = len(payload) + 1024

    def generator(staging_path):
        Path(staging_path).write_bytes(payload)
        raise WriterNotQuiescentError("writer state is unknown")

    with pytest.raises(WriterNotQuiescentError, match="unknown"):
        store.save_generated_file(
            generator,
            reserve_bytes=len(payload),
            max_size_bytes=max_size_bytes,
            tool="capture_core_dump",
        )

    entries = list(Path(store.root_dir).iterdir())
    quarantined = [
        entry for entry in entries if entry.name.endswith(".staging.quarantine")
    ]
    markers = [
        entry for entry in entries if entry.name.endswith(".staging.quarantine-marker")
    ]
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == payload
    assert len(markers) == 1
    assert f"reserved_bytes={max_size_bytes}" in markers[0].read_text("utf-8")

    status = store.status()
    assert status["artifact_count"] == 0
    assert status["usage_bytes"] == 0
    assert status["total_usage_bytes"] == max_size_bytes
    assert status["quarantine_count"] == 1
    assert status["quarantine_bytes"] == max_size_bytes
    assert status["quarantine_rename_failure_count"] == 0
    assert status["protected_staging_count"] == 0


def test_failed_quarantine_rename_keeps_durable_protected_staging(
    tmp_path, monkeypatch
):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    payload = b"active-writer"
    max_size_bytes = len(payload) + 64
    real_replace = os.replace

    def fail_quarantine_rename(source, destination):
        if str(destination).endswith(".quarantine"):
            raise PermissionError("open writer prevents rename")
        return real_replace(source, destination)

    monkeypatch.setattr("paomcp.artifacts.os.replace", fail_quarantine_rename)

    def generator(staging_path):
        Path(staging_path).write_bytes(payload)
        raise WriterNotQuiescentError("writer state is unknown")

    with pytest.raises(WriterNotQuiescentError, match="unknown"):
        store.save_generated_file(
            generator,
            reserve_bytes=len(payload),
            max_size_bytes=max_size_bytes,
            tool="capture_core_dump",
        )

    staging_entries = [
        path
        for path in Path(store.root_dir).iterdir()
        if path.name.endswith(".staging")
    ]
    marker_entries = [
        path
        for path in Path(store.root_dir).iterdir()
        if path.name.endswith(".staging.quarantine-marker")
    ]
    assert len(staging_entries) == 1
    assert staging_entries[0].read_bytes() == payload
    assert len(marker_entries) == 1
    assert marker_entries[0].name == (f"{staging_entries[0].name}.quarantine-marker")
    assert f"reserved_bytes={max_size_bytes}" in marker_entries[0].read_text(
        encoding="utf-8"
    )

    # A fresh store discovers the durable marker and cannot mistake the staging
    # payload for an ordinary abandoned temporary file at startup.
    restarted_store = ArtifactStore(
        str(tmp_path),
        quota_bytes=max_size_bytes - 1,
        min_free_bytes=0,
    )
    reconciliation = restarted_store.reconcile_orphans(grace_seconds=0)
    assert reconciliation["removed_count"] == 0
    assert staging_entries[0].is_file()
    assert marker_entries[0].is_file()

    status = restarted_store.status()
    assert status["total_usage_bytes"] == max_size_bytes
    assert status["quarantine_count"] == 1
    assert status["quarantine_bytes"] == max_size_bytes
    assert status["quarantine_rename_failure_count"] == 1
    assert status["protected_staging_count"] == 1
    assert status["protected_staging_bytes"] == len(payload)

    capacity = restarted_store.cleanup()
    assert capacity["quarantine_bytes"] == max_size_bytes
    assert capacity["quota_satisfied"] is False


def test_successful_quarantine_marker_protects_recreated_original_after_restart(
    tmp_path,
):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    payload = b"quarantined"
    recreated_payload = b"recreated"
    max_size_bytes = 64

    def generator(staging_path):
        Path(staging_path).write_bytes(payload)
        raise WriterNotQuiescentError("writer state is unknown")

    with pytest.raises(WriterNotQuiescentError):
        store.save_generated_file(
            generator,
            reserve_bytes=16,
            max_size_bytes=max_size_bytes,
            tool="capture_core_dump",
        )

    marker = next(Path(store.root_dir).glob("*.staging.quarantine-marker"))
    quarantine = next(Path(store.root_dir).glob("*.staging.quarantine"))
    original = Path(store.root_dir) / marker.name.removesuffix(".quarantine-marker")
    original.write_bytes(recreated_payload)

    restarted_store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    result = restarted_store.reconcile_orphans(grace_seconds=0)

    assert result["removed_count"] == 0
    assert result["failed_count"] == 0
    assert marker.is_file()
    assert quarantine.read_bytes() == payload
    assert original.read_bytes() == recreated_payload

    status = restarted_store.status()
    assert status["quarantine_count"] == 1
    assert status["quarantine_bytes"] == max_size_bytes
    assert status["quarantine_rename_failure_count"] == 0
    assert status["protected_staging_count"] == 1
    assert status["protected_staging_bytes"] == len(recreated_payload)


def test_corrupt_or_missing_quarantine_reservation_blocks_quota_after_restart(
    tmp_path,
):
    quota = 128
    store = ArtifactStore(str(tmp_path), quota_bytes=quota, min_free_bytes=0)

    def generator(staging_path):
        Path(staging_path).write_bytes(b"partial")
        raise WriterNotQuiescentError("writer state is unknown")

    with pytest.raises(WriterNotQuiescentError):
        store.save_generated_file(
            generator,
            reserve_bytes=7,
            max_size_bytes=64,
            tool="capture_core_dump",
        )

    marker = next(Path(store.root_dir).glob("*.staging.quarantine-marker"))
    marker.write_text("corrupt=true\n", encoding="utf-8")
    restarted = ArtifactStore(str(tmp_path), quota_bytes=quota, min_free_bytes=0)
    assert restarted.status()["quarantine_bytes"] == quota
    assert restarted.cleanup(incoming_size=1)["quota_satisfied"] is False

    marker.unlink()
    restarted_without_marker = ArtifactStore(
        str(tmp_path), quota_bytes=quota, min_free_bytes=0
    )
    assert restarted_without_marker.status()["quarantine_bytes"] == quota
    assert restarted_without_marker.cleanup(incoming_size=1)["quota_satisfied"] is False


def test_reconciliation_continues_after_corrupt_candidates_and_os_errors(
    tmp_path, monkeypatch
):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    good = tmp_path / ".artifact_good.staging"
    stat_denied = tmp_path / ".artifact_stat_denied.staging"
    unlink_denied = tmp_path / ".artifact_unlink_denied.staging"
    corrupt_directory = tmp_path / ".artifact_directory.staging"
    good.write_bytes(b"good")
    stat_denied.write_bytes(b"stat denied")
    unlink_denied.write_bytes(b"unlink denied")
    corrupt_directory.mkdir()

    real_lstat = os.lstat
    real_unlink = os.unlink

    def selective_lstat(path):
        if Path(path).name == stat_denied.name:
            raise PermissionError("stat denied")
        return real_lstat(path)

    def selective_unlink(path):
        if Path(path).name == unlink_denied.name:
            raise PermissionError("unlink denied")
        return real_unlink(path)

    monkeypatch.setattr("paomcp.artifacts.os.lstat", selective_lstat)
    monkeypatch.setattr("paomcp.artifacts.os.unlink", selective_unlink)

    result = store.reconcile_orphans(grace_seconds=0)

    assert result["removed_count"] == 1
    assert result["removed"]["temporary"] == 1
    assert result["failed_count"] == 3
    assert {item["name"] for item in result["failures"]} == {
        stat_denied.name,
        unlink_denied.name,
        corrupt_directory.name,
    }
    assert not good.exists()
    assert stat_denied.is_file()
    assert unlink_denied.is_file()
    assert corrupt_directory.is_dir()


def test_generated_file_actual_size_is_hard_capped(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)

    with pytest.raises(ArtifactSizeLimitError, match="exceeding"):
        store.save_generated_file(
            lambda staging_path: Path(staging_path).write_bytes(b"too large"),
            reserve_bytes=4,
            max_size_bytes=4,
            tool="capture_core_dump",
        )

    assert list(Path(store.root_dir).iterdir()) == []


def test_generated_file_rejects_inode_swap_between_lstat_and_open(
    tmp_path, monkeypatch
):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    staging = {}
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"replacement")
    real_open = os.open

    def generator(staging_path):
        staging["path"] = staging_path
        Path(staging_path).write_bytes(b"original")

    def swap_before_open(path, flags, *args):
        if path == staging.get("path"):
            os.replace(replacement, path)
        return real_open(path, flags, *args)

    monkeypatch.setattr("paomcp.artifacts.os.open", swap_before_open)

    with pytest.raises(OSError, match="changed before finalization"):
        store.save_generated_file(
            generator,
            reserve_bytes=16,
            max_size_bytes=16,
            tool="capture_core_dump",
        )

    assert store.list() == []
    assert not any(
        path.name.startswith(".artifact_") for path in Path(store.root_dir).iterdir()
    )


def test_generated_file_rejects_inode_swap_during_atomic_adoption(
    tmp_path, monkeypatch
):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    staging = {}
    real_replace = os.replace

    def generator(staging_path):
        staging["path"] = staging_path
        Path(staging_path).write_bytes(b"original")

    def swap_during_adoption(source, destination):
        if source == staging.get("path") and str(destination).endswith(".data"):
            replacement = tmp_path / "adoption-replacement.bin"
            replacement.write_bytes(b"replacement")
            real_replace(replacement, source)
        return real_replace(source, destination)

    monkeypatch.setattr("paomcp.artifacts.os.replace", swap_during_adoption)

    with pytest.raises(OSError, match="identity changed during adoption"):
        store.save_generated_file(
            generator,
            reserve_bytes=16,
            max_size_bytes=16,
            tool="capture_core_dump",
        )

    assert store.list() == []
    assert not list(Path(store.root_dir).glob("*.data"))
    assert not list(Path(store.root_dir).glob("*.meta.json"))


def test_generated_file_cancellation_during_hash_removes_staging(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    checks = 0

    def cancellation_check():
        nonlocal checks
        checks += 1
        if checks == 4:
            raise RuntimeError("cancelled")

    def generator(staging_path):
        Path(staging_path).write_bytes(b"X" * (2 * 1024 * 1024))
        return "generated"

    with pytest.raises(RuntimeError, match="cancelled"):
        store.save_generated_file(
            generator,
            reserve_bytes=2 * 1024 * 1024,
            max_size_bytes=2 * 1024 * 1024,
            tool="capture_core_dump",
            cancellation_check=cancellation_check,
        )

    assert checks == 4
    assert list(Path(store.root_dir).iterdir()) == []


def test_orphan_reconciliation_waits_for_active_generator(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    reconciliation_started = threading.Event()
    reconciliation_finished = threading.Event()
    reconciliation_thread = None

    def generator(staging_path):
        nonlocal reconciliation_thread
        Path(staging_path).write_bytes(b"active staging")

        def reconcile():
            reconciliation_started.set()
            store.reconcile_orphans(grace_seconds=0)
            reconciliation_finished.set()

        reconciliation_thread = threading.Thread(target=reconcile)
        reconciliation_thread.start()
        assert reconciliation_started.wait(timeout=1)
        # Vong 3 contract: reconciliation is no longer blocked by the write
        # lock while a generator runs, but it must skip the protected
        # staging path (grace_seconds=0 would otherwise reap it).
        assert reconciliation_finished.wait(timeout=5) is True
        assert Path(staging_path).is_file()
        return "done"

    artifact, result = store.save_generated_file(
        generator,
        reserve_bytes=64,
        max_size_bytes=64,
        tool="capture_core_dump",
    )

    assert reconciliation_thread is not None
    reconciliation_thread.join(timeout=1)
    assert reconciliation_finished.is_set()
    assert result == "done"
    assert store.get(artifact.artifact_id) == artifact


def test_concurrent_generated_saves_do_not_serialize_on_generator(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    slow_started = threading.Event()
    slow_release = threading.Event()

    def slow_generator(staging_path):
        Path(staging_path).write_bytes(b"slow")
        slow_started.set()
        assert slow_release.wait(timeout=10)
        return "slow"

    results = {}

    def run_slow():
        results["slow"] = store.save_generated_file(
            slow_generator,
            reserve_bytes=16,
            max_size_bytes=64,
            tool="slow",
        )

    slow_thread = threading.Thread(target=run_slow)
    slow_thread.start()
    assert slow_started.wait(timeout=5)

    def fast_generator(staging_path):
        Path(staging_path).write_bytes(b"fast")
        return "fast"

    started = time.monotonic()
    artifact, result = store.save_generated_file(
        fast_generator,
        reserve_bytes=16,
        max_size_bytes=64,
        tool="fast",
    )
    elapsed = time.monotonic() - started
    assert result == "fast"
    # The fast save must not wait for the slow generator: the write lock is
    # only held for reservation/finalization, never during generation.
    assert elapsed < 5
    assert store.get(artifact.artifact_id) == artifact

    slow_release.set()
    slow_thread.join(timeout=10)
    assert results["slow"][1] == "slow"


def test_generated_file_rechecks_quota_for_actual_size(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    pinned = store.save_bytes(b"A" * 8, tool="existing")
    store.set_pinned(pinned.artifact_id)
    store.quota_bytes = 10
    generator_called = False

    def generator(staging_path):
        nonlocal generator_called
        generator_called = True
        Path(staging_path).write_bytes(b"BBB")

    with pytest.raises(ArtifactStorageFullError, match="quota"):
        store.save_generated_file(
            generator,
            reserve_bytes=2,
            max_size_bytes=3,
            tool="capture_core_dump",
        )

    assert generator_called is True
    assert store.list() == [store.get(pinned.artifact_id)]
    assert not any(
        path.name.startswith(".artifact_") for path in Path(store.root_dir).iterdir()
    )


def test_generated_file_reserve_is_preflighted_before_generator(tmp_path):
    store = ArtifactStore(str(tmp_path), quota_bytes=0, min_free_bytes=0)
    pinned = store.save_bytes(b"A" * 8, tool="existing")
    store.set_pinned(pinned.artifact_id)
    store.quota_bytes = 10
    generator_called = False

    def generator(staging_path):
        nonlocal generator_called
        generator_called = True
        Path(staging_path).write_bytes(b"B")

    with pytest.raises(ArtifactStorageFullError, match="quota"):
        store.save_generated_file(
            generator,
            reserve_bytes=3,
            max_size_bytes=3,
            tool="capture_core_dump",
        )

    assert generator_called is False
    assert store.list() == [store.get(pinned.artifact_id)]


def test_save_bytes_refuses_when_pinned_artifacts_block_quota(tmp_path):
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        ttl_seconds=0,
        quota_bytes=10,
        min_free_bytes=0,
    )
    pinned = store.save_bytes(b"A" * 8, tool="existing")
    store.set_pinned(pinned.artifact_id)

    with pytest.raises(ArtifactStorageFullError, match="quota"):
        store.save_bytes(b"B" * 3, tool="new")

    assert store.status()["total_usage_bytes"] == 8
    assert store.list() == [store.get(pinned.artifact_id)]


def test_save_file_refuses_when_pinned_artifacts_block_quota(tmp_path):
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        ttl_seconds=0,
        quota_bytes=10,
        min_free_bytes=0,
    )
    pinned = store.save_bytes(b"A" * 8, tool="existing")
    store.set_pinned(pinned.artifact_id)
    source = tmp_path / "incoming.bin"
    source.write_bytes(b"B" * 3)

    with pytest.raises(ArtifactStorageFullError, match="quota"):
        store.save_file(str(source), tool="new")

    assert store.status()["total_usage_bytes"] == 8
    assert store.list() == [store.get(pinned.artifact_id)]


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
