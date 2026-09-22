import asyncio
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from paomcp.tools.backends.fuzz import FuzzBatch, generate_inputs


def test_generate_inputs_deterministic_and_typed():
    first = generate_inputs(count=9, max_len=64, seed=7)
    second = generate_inputs(count=9, max_len=64, seed=7)
    assert first["parsed"]["inputs"] == second["parsed"]["inputs"]
    strategies = [item["strategy"] for item in first["parsed"]["inputs"]]
    assert strategies == ["zeros", "cyclic", "bitflip"] * 3
    assert all(item["length"] >= 1 for item in first["parsed"]["inputs"])


def test_generate_inputs_rejects_bad_args():
    with pytest.raises(ValueError, match="count"):
        generate_inputs(count=0)
    with pytest.raises(ValueError, match="max_len"):
        generate_inputs(count=1, max_len=0)
    with pytest.raises(ValueError, match="unknown fuzz strategies"):
        generate_inputs(count=1, strategies=["grammar"])


class _FakeCrash:
    def __init__(self, script):
        self._script = list(script)
        self.runs = []

    def run_until_crash(self, *, arguments, mode, timeout_seconds, **kwargs):
        self.runs.append((arguments, mode, timeout_seconds))
        crashed, crash_id, signal = self._script.pop(0)
        return {
            "parsed": {
                "crashed": crashed,
                "crash_id": crash_id,
                "signal": signal,
            }
        }

    def capture_crash_snapshot(self, crash_id, **kwargs):
        return {"parsed": {"signature": {"digest": f"digest:{crash_id}"}}}


class _FakeTools:
    def __init__(self, script):
        self.crash = _FakeCrash(script)
        self.set_file_calls = []

    def set_file(self, binary_path):
        self.set_file_calls.append(binary_path)
        return {"success": True}


def test_batch_runs_inputs_and_dedups_crashes(tmp_path):
    tools = _FakeTools(
        [
            (False, None, None),
            (True, "crash-1", "SIGSEGV"),
            (True, "crash-2", "SIGSEGV"),
        ]
    )
    campaign = FuzzBatch("sess")
    result = campaign.run_batch(
        tools=tools,
        binary_path="/workspace/target",
        corpus_dir=str(tmp_path),
        inputs=[
            {"index": 0, "strategy": "zeros", "hex": "00"},
            {"index": 1, "strategy": "zeros", "hex": "41"},
            {"index": 2, "strategy": "zeros", "hex": "42"},
        ],
        timeout_per_run=5.0,
        max_runs=10,
    )
    parsed = result["parsed"]
    assert result["success"] is True
    assert tools.set_file_calls == ["/workspace/target"]
    assert parsed["runs"] == 3
    assert parsed["crashes"] == 2
    assert parsed["unique_crashes"] == 2
    assert all(r["novel"] for r in parsed["novel"])
    assert len(tools.crash.runs) == 3
    assert all(call[1] == "run" for call in tools.crash.runs)


def test_batch_duplicate_digest_not_novel(tmp_path):
    tools = _FakeTools(
        [
            (True, "same-crash", "SIGSEGV"),
            (True, "same-crash", "SIGSEGV"),
        ]
    )
    campaign = FuzzBatch("sess")
    parsed = campaign.run_batch(
        tools=tools,
        binary_path="/workspace/target",
        corpus_dir=str(tmp_path),
        inputs=[
            {"index": 0, "hex": "00"},
            {"index": 1, "hex": "ff"},
        ],
    )["parsed"]
    assert parsed["unique_crashes"] == 1
    assert [r["novel"] for r in parsed["records"]] == [True, False]


def test_batch_validates_budgets(tmp_path):
    campaign = FuzzBatch("sess")
    with pytest.raises(ValueError, match="max_runs"):
        campaign.run_batch(
            tools=_FakeTools([]),
            binary_path="/x",
            corpus_dir=str(tmp_path),
            inputs=[],
            max_runs=0,
        )
    with pytest.raises(ValueError, match="timeout_per_run"):
        campaign.run_batch(
            tools=_FakeTools([]),
            binary_path="/x",
            corpus_dir=str(tmp_path),
            inputs=[],
            timeout_per_run=9999,
        )


def test_batch_rejects_oversize_corpus_before_writing(tmp_path, monkeypatch):
    import paomcp.tools.backends.fuzz as fuzz_module

    monkeypatch.setattr(fuzz_module, "MAX_TOTAL_CORPUS_BYTES", 1)
    campaign = FuzzBatch("sess")
    with pytest.raises(ValueError, match="batch budget"):
        campaign.run_batch(
            tools=_FakeTools([]),
            binary_path="/x",
            corpus_dir=str(tmp_path),
            inputs=[{"index": 0, "hex": "4141"}],
        )
    assert list(tmp_path.iterdir()) == []


def test_batch_reraises_cancellation(tmp_path):
    from paomcp.operations import OperationCancelledError

    def cancel():
        raise OperationCancelledError("stop")

    campaign = FuzzBatch("sess")
    with pytest.raises(OperationCancelledError):
        campaign.run_batch(
            tools=_FakeTools([(False, None, None)]),
            binary_path="/x",
            corpus_dir=str(tmp_path),
            inputs=[{"index": 0, "hex": "00"}],
            cancellation_check=cancel,
        )


def test_batch_survives_bad_hex_and_run_errors(tmp_path):
    class ExplodingCrash(_FakeCrash):
        def run_until_crash(self, *, arguments, mode, timeout_seconds):
            raise RuntimeError("gdb wedged")

    tools = _FakeTools([])
    tools.crash = ExplodingCrash([])
    campaign = FuzzBatch("sess")
    parsed = campaign.run_batch(
        tools=tools,
        binary_path="/workspace/target",
        corpus_dir=str(tmp_path),
        inputs=[{"index": 0, "hex": "zz"}, {"index": 1, "hex": "00"}],
    )["parsed"]
    assert parsed["runs"] == 0
    assert len(parsed["errors"]) == 2


def _coverage_paths(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    target = workspace / "target"
    target.write_bytes(b"\x7fELF")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    output = tmp_path / "out"
    output.mkdir()
    return target, corpus, output


def _fake_afl_engine(monkeypatch, script, *, fuzz_cov):
    real_popen = subprocess.Popen
    children = []

    def fake_popen(cmd, **kwargs):
        child = real_popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        children.append(child)
        return child

    monkeypatch.setattr(fuzz_cov.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        fuzz_cov.shutil,
        "which",
        lambda name: "/fake/afl-fuzz" if name == "afl-fuzz" else None,
    )
    return children


def test_coverage_campaign_cancellation_kills_child_promptly(tmp_path, monkeypatch):
    import paomcp.tools.backends.fuzz_cov as fuzz_cov
    from paomcp.operations import OperationCancelledError

    target, corpus, output = _coverage_paths(tmp_path)
    children = _fake_afl_engine(
        monkeypatch, "import time; time.sleep(30)", fuzz_cov=fuzz_cov
    )
    cancel_at = time.monotonic() + 0.2

    def check():
        if time.monotonic() >= cancel_at:
            raise OperationCancelledError("stop")

    started = time.monotonic()
    result = fuzz_cov.run_coverage_batch(
        engine="afl",
        target_path=str(target),
        corpus_dir=str(corpus),
        output_dir=str(output),
        time_seconds=60,
        cancellation_check=check,
    )
    elapsed = time.monotonic() - started

    assert result["parsed"]["verdict"] == "cancelled"
    assert elapsed < 10
    assert children and children[0].poll() is not None


def test_coverage_campaign_drains_chatty_engine_without_wedging(tmp_path, monkeypatch):
    import paomcp.tools.backends.fuzz_cov as fuzz_cov

    target, corpus, output = _coverage_paths(tmp_path)
    _fake_afl_engine(
        monkeypatch,
        (
            "import sys;"
            "sys.stdout.write('x' * 200000); sys.stdout.flush();"
            "sys.stderr.write('y' * 200000); sys.stderr.flush()"
        ),
        fuzz_cov=fuzz_cov,
    )

    started = time.monotonic()
    result = fuzz_cov.run_coverage_batch(
        engine="afl",
        target_path=str(target),
        corpus_dir=str(corpus),
        output_dir=str(output),
        time_seconds=30,
    )
    elapsed = time.monotonic() - started

    parsed = result["parsed"]
    assert parsed["verdict"] == "completed"
    assert parsed["exit_code"] == 0
    assert elapsed < 20


def test_coverage_campaign_deadline_is_reported_as_timeout(tmp_path, monkeypatch):
    import paomcp.tools.backends.fuzz_cov as fuzz_cov

    target, corpus, output = _coverage_paths(tmp_path)
    children = _fake_afl_engine(
        monkeypatch, "import time; time.sleep(30)", fuzz_cov=fuzz_cov
    )
    monkeypatch.setattr(fuzz_cov, "DEADLINE_GRACE_SECONDS", 0)

    result = fuzz_cov.run_coverage_batch(
        engine="afl",
        target_path=str(target),
        corpus_dir=str(corpus),
        output_dir=str(output),
        time_seconds=1,
    )

    parsed = result["parsed"]
    assert parsed["verdict"] == "timeout"
    assert parsed["exit_code"] is not None
    assert children and children[0].poll() is not None


@pytest.mark.asyncio
async def test_run_afl_batch_wires_cooperative_cancellation(tmp_path, monkeypatch):
    from fastmcp import FastMCP

    from paomcp.artifacts import ArtifactStore
    from paomcp.operations import (
        OperationCancelledError,
        OperationRegistry,
        current_operation_id,
    )
    from paomcp.tools import fuzz as fuzz_module
    import paomcp.tools.backends.fuzz_cov as fuzz_cov

    started = threading.Event()
    captured: Dict[str, Any] = {}

    def fake_backend(**kwargs):
        check = kwargs.get("cancellation_check")
        captured["has_check"] = check is not None
        if check is None:
            return {"success": True, "parsed": {"verdict": "completed"}}
        captured["operation_id"] = current_operation_id()
        started.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            check()
            time.sleep(0.01)
        raise AssertionError("cancellation was never requested")

    monkeypatch.setattr(fuzz_cov, "run_coverage_batch", fake_backend)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "target").write_bytes(b"\x7fELF")
    (workspace / "corpus").mkdir()
    (workspace / "out").mkdir()
    services = SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(workspace)),
        operation_registry=OperationRegistry(),
        artifact_store=ArtifactStore(str(tmp_path / "artifacts"), min_free_bytes=0),
    )
    mcp = FastMCP("fuzz-cancel-tests")
    fuzz_module.register(mcp)
    tool = await mcp.get_tool("run_afl_batch")
    ctx = SimpleNamespace(lifespan_context={"services": services})

    task = asyncio.create_task(
        tool.fn(
            corpus_dir="corpus",
            output_dir="out",
            engine="afl",
            target_path="target",
            time_seconds=5,
            ctx=ctx,
        )
    )
    assert await asyncio.to_thread(started.wait, 5)
    services.operation_registry.request_cancel(captured["operation_id"])
    response = await asyncio.wait_for(task, timeout=10)

    assert captured["has_check"] is True
    assert response["success"] is False
    assert response["error"]["code"] == "OPERATION_CANCELLED"
