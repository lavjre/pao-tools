import json
import os
import sys
from pathlib import Path

from paomcp.artifacts import ArtifactStore
from paomcp.tools.backends.decompiler import GhidraDecompiler


def _make_backend(tmp_path: Path, *, inline_bytes: int = 1024 * 1024):
    script_dir = tmp_path / "scripts"
    script_dir.mkdir()
    (script_dir / "PaoDecompile.java").write_text("// test", encoding="utf-8")
    store = ArtifactStore(
        str(tmp_path / "artifacts"),
        inline_response_bytes=inline_bytes,
        quota_bytes=0,
        min_free_bytes=0,
    )
    backend = GhidraDecompiler(
        runtime_dir=str(tmp_path / "runtime"),
        artifact_store=store,
        executable=sys.executable,
        script_dir=str(script_dir),
    )
    return backend, store


def _write_success_outputs(command, stdout_path, stderr_path, code: str) -> None:
    post_script = command.index("-postScript")
    output_path = command[post_script + 2]
    manifest_path = command[post_script + 3]
    Path(output_path).write_bytes(code.encode("utf-8"))
    records = [
        {
            "type": "program",
            "name": "chal",
            "format": "Executable and Linking Format (ELF)",
            "language": "x86:LE:64:default",
            "compiler": "gcc",
        },
        {
            "type": "function",
            "status": "success",
            "name": "main",
            "entry": "00401120",
            "elapsed_ms": 7,
        },
    ]
    Path(manifest_path).write_bytes(
        "".join(json.dumps(record) + "\n" for record in records).encode("utf-8")
    )
    Path(stdout_path).write_bytes(b"analysis complete\n")
    Path(stderr_path).write_bytes(b"")


def test_local_decompiler_returns_inline_code_and_manifest(tmp_path, monkeypatch):
    backend, _ = _make_backend(tmp_path)
    binary = tmp_path / "chal"
    binary.write_bytes(b"\x7fELF")
    captured = {}

    def fake_run(command, *, stdout_path, stderr_path, timeout):
        captured["command"] = command
        _write_success_outputs(command, stdout_path, stderr_path, "int main(void) {}\n")
        return 0, False, 42

    monkeypatch.setattr(backend, "_run_process", fake_run)
    result = backend.decompile(str(binary), function="main")

    assert result["success"] is True
    assert result["backend"] == "ghidra"
    assert result["function_count"] == 1
    assert result["decompiled_code"] == "int main(void) {}\n"
    assert result["program"]["language"] == "x86:LE:64:default"
    post_script = captured["command"].index("-postScript")
    assert captured["command"][post_script + 4] == "main"
    assert "http" not in " ".join(captured["command"])


def test_local_decompiler_spools_large_output_to_artifacts(tmp_path, monkeypatch):
    backend, store = _make_backend(tmp_path, inline_bytes=64)
    binary = tmp_path / "chal"
    binary.write_bytes(b"\x7fELF")

    def fake_run(command, *, stdout_path, stderr_path, timeout):
        _write_success_outputs(command, stdout_path, stderr_path, "A" * 4096)
        return 0, False, 9

    monkeypatch.setattr(backend, "_run_process", fake_run)
    result = backend.decompile(str(binary))

    code = result["decompiled_code"]
    assert code["delivery"] == "artifact"
    assert code["stream"] == "decompiled_code"
    assert store.get(code["artifact_id"]).size_bytes == 4096
    assert result["manifest"]["delivery"] == "artifact"


def test_local_decompiler_timeout_returns_structured_failure(tmp_path, monkeypatch):
    backend, _ = _make_backend(tmp_path)
    binary = tmp_path / "chal"
    binary.write_bytes(b"\x7fELF")

    def fake_run(command, *, stdout_path, stderr_path, timeout):
        Path(stdout_path).write_bytes(b"starting analysis\n")
        Path(stderr_path).write_bytes(b"timeout\n")
        return -9, True, 1000

    monkeypatch.setattr(backend, "_run_process", fake_run)
    result = backend.decompile(str(binary), timeout=1)

    assert result["success"] is False
    assert result["code"] == "DECOMPILER_TIMEOUT"
    assert result["logs"]["stderr"] == "timeout\n"


def test_local_decompiler_status_reports_missing_executable(tmp_path):
    store = ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0)
    backend = GhidraDecompiler(
        runtime_dir=str(tmp_path / "runtime"),
        artifact_store=store,
        executable=os.path.join(str(tmp_path), "missing-analyze-headless"),
        script_dir=str(tmp_path),
    )

    status = backend.status()

    assert status["ready"] is False
    assert any("executable not found" in reason for reason in status["reasons"])
