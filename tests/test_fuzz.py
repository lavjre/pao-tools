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
