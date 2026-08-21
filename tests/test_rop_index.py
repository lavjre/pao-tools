"""Tests for the analyzed-gadget cache, mnemonic index, and parallel pass."""

import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pytest

import paomcp.tools.backends.rop as rop_module
from paomcp.tools.backends.rop import GadgetIndex


def _record(address: int, instructions: List[str]) -> Dict[str, Any]:
    return {
        "address": address,
        "bytes": b"\x90",
        "instructions": instructions,
        "arch": "x86_64",
    }


def _fake_loader(
    records: Sequence[Mapping[str, Any]],
) -> Callable[..., Sequence[Mapping[str, Any]]]:
    def _load(binary_path: str, arch: Optional[str], image_base: Optional[int]):
        return list(records)

    return _load


def _tempbin(content: bytes = b"BIN") -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    handle.write(content)
    handle.close()
    return handle.name


class TestAnalyzedCache:
    def test_repeat_search_does_not_reanalyze(self, monkeypatch) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        calls = {"n": 0}
        original = GadgetIndex.analyze_gadget

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(GadgetIndex, "analyze_gadget", counting)
        try:
            idx.find_gadgets(binary_path=binary, arch="x86_64")
            idx.find_gadgets(binary_path=binary, arch="x86_64")
            idx.find_gadget_sequence(
                binary_path=binary, steps=["pop rdi"], arch="x86_64"
            )
            # Three queries, one analysis pass: the first call analyzed the
            # single record, the rest hit the analyzed cache.  (Same arch on
            # every call: the cache key includes arch.)
            assert calls["n"] == 1
        finally:
            os.unlink(binary)

    def test_mtime_bump_reanalyzes(self, monkeypatch) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        calls = {"n": 0}
        original = GadgetIndex.analyze_gadget

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(GadgetIndex, "analyze_gadget", counting)
        try:
            idx.find_gadgets(binary_path=binary, arch="x86_64")
            future = time.time() + 5
            os.utime(binary, (future, future))
            idx.find_gadgets(binary_path=binary, arch="x86_64")
            assert calls["n"] == 2
        finally:
            os.unlink(binary)

    def test_cross_tool_shares_analyzed_cache(self, monkeypatch) -> None:
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        calls = {"n": 0}
        original = GadgetIndex.analyze_gadget

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(GadgetIndex, "analyze_gadget", counting)
        try:
            # inspect_gadget populates the cache; find_gadgets reuses it.
            idx.inspect_gadget(binary_path=binary, address=0x400100, arch="x86_64")
            idx.find_gadgets(binary_path=binary, arch="x86_64")
            assert calls["n"] == 1
        finally:
            os.unlink(binary)


class TestMnemonicIndex:
    def test_leading_mnemonic_for_anchor(self) -> None:
        assert GadgetIndex._leading_mnemonic_for_anchor("^pop rdi ; ret") == "pop"
        assert GadgetIndex._leading_mnemonic_for_anchor("^  ret") == "ret"
        # Not anchored -> no index narrowing.
        assert GadgetIndex._leading_mnemonic_for_anchor("pop rdi ; ret") is None
        # Regex metachar token is not a literal mnemonic.
        assert GadgetIndex._leading_mnemonic_for_anchor("^p.op rdi") is None
        assert GadgetIndex._leading_mnemonic_for_anchor("^") is None

    def test_anchored_step_ignores_mid_text_sighting(self) -> None:
        # The same "pop rdi" text appears mid-gadget, so an unanchored
        # search finds it but an anchored (^) step must not.
        records = [
            _record(0x400100, ["mov rax, 1", "pop rdi", "ret"]),
            _record(0x400200, ["pop rdi", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            anchored = idx.find_gadget_sequence(
                binary_path=binary, steps=["^pop rdi"], arch="x86_64"
            )
            assert anchored["parsed"]["steps"][0]["gadget"]["address"] == 0x400200

            unanchored = idx.find_gadget_sequence(
                binary_path=binary, steps=["pop rdi"], arch="x86_64"
            )
            # Shortest-first ordering picks the 2-instruction gadget.
            assert unanchored["parsed"]["steps"][0]["gadget"]["address"] == 0x400200
        finally:
            os.unlink(binary)

    def test_anchored_step_with_absent_mnemonic_matches_nothing(self) -> None:
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(0x400200, ["ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadget_sequence(
                binary_path=binary, steps=["^syscall"], arch="x86_64"
            )
            assert response["parsed"]["verdict"] == "partial"
            assert response["parsed"]["steps"][0]["matched"] is False
        finally:
            os.unlink(binary)

    def test_index_path_returns_same_gadget_as_full_scan(self) -> None:
        # Same query delivered through the index (anchored) and through the
        # full scan (identical pattern with ^ stripped) must agree whenever
        # the pattern is start-anchored in both spellings.
        records = [
            _record(0x400100, ["pop rdi", "ret"]),
            _record(0x400200, ["pop rsi", "ret"]),
            _record(0x400300, ["pop rdi", "pop rsi", "ret"]),
        ]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            anchored = idx.find_gadget_sequence(
                binary_path=binary, steps=["^pop rdi"], arch="x86_64"
            )
            assert anchored["parsed"]["steps"][0]["gadget"]["address"] == 0x400100
        finally:
            os.unlink(binary)


class TestParallelAnalysis:
    def test_parallel_matches_sequential_output(self, monkeypatch) -> None:
        # Force the parallel threshold down so the pool actually spawns.
        monkeypatch.setattr(rop_module, "PARALLEL_ANALYSIS_THRESHOLD", 1)
        monkeypatch.setattr(rop_module, "_PARALLEL_ANALYSIS_DISABLED", False)
        records = [_record(0x400000 + i * 0x10, ["pop rdi", "ret"]) for i in range(50)]
        records.append(_record(0x404000, ["syscall", "ret"]))

        parallel_idx = GadgetIndex(loader=_fake_loader(records))
        parallel_bin = _tempbin()
        sequential_idx = GadgetIndex(loader=_fake_loader(records))
        sequential_bin = _tempbin()
        try:
            parallel_result = parallel_idx.find_gadgets(
                binary_path=parallel_bin, arch="x86_64", max_instructions=3
            )
            monkeypatch.setattr(rop_module, "_PARALLEL_ANALYSIS_DISABLED", True)
            sequential_result = sequential_idx.find_gadgets(
                binary_path=sequential_bin, arch="x86_64", max_instructions=3
            )
            p_gadgets = parallel_result["parsed"]["gadgets"]
            s_gadgets = sequential_result["parsed"]["gadgets"]
            assert [g["address"] for g in p_gadgets] == [
                g["address"] for g in s_gadgets
            ]
            assert p_gadgets[0]["reads"] == s_gadgets[0]["reads"]
        finally:
            os.unlink(parallel_bin)
            os.unlink(sequential_bin)

    def test_disabled_env_skips_pool(self, monkeypatch) -> None:
        # With PAO_DISABLE_PARALLEL_GADGETS semantics enabled, no pool is
        # ever created; the query still works.
        monkeypatch.setattr(rop_module, "PARALLEL_ANALYSIS_THRESHOLD", 1)
        monkeypatch.setattr(rop_module, "_PARALLEL_ANALYSIS_DISABLED", True)
        records = [_record(0x400100, ["pop rdi", "ret"])]
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, arch="x86_64")
            assert response["parsed"]["matched"] == 1
        finally:
            os.unlink(binary)

    def test_pool_failure_falls_back_to_sequential(self, monkeypatch) -> None:
        # A broken multiprocessing path must degrade to the sequential pass
        # and still return identical, correct results.
        monkeypatch.setattr(rop_module, "PARALLEL_ANALYSIS_THRESHOLD", 1)
        monkeypatch.setattr(rop_module, "_PARALLEL_ANALYSIS_DISABLED", False)
        records = [_record(0x400100, ["pop rdi", "ret"])]

        def broken_pool(*args, **kwargs):
            raise RuntimeError("sandbox forbids multiprocessing")

        monkeypatch.setattr(rop_module.multiprocessing, "get_context", broken_pool)
        idx = GadgetIndex(loader=_fake_loader(records))
        binary = _tempbin()
        try:
            response = idx.find_gadgets(binary_path=binary, arch="x86_64")
            assert response["parsed"]["matched"] == 1
            assert response["parsed"]["gadgets"][0]["address"] == 0x400100
        finally:
            os.unlink(binary)
