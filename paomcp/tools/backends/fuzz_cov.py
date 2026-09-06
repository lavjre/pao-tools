"""Coverage-guided fuzz bridge (P2): AFL++ / libFuzzer orchestration + corpus minimize.

The deterministic ``generate_fuzz_inputs`` / ``run_fuzz_batch`` surface stays
as-is. This module adds the coverage layer on top:

* :func:`minimize_corpus` is pure (sha256 dedup + size cap, deterministic) and
  unit-testable without any fuzzer installed.
* :func:`run_coverage_batch` shells to ``afl-fuzz`` (engine=``afl``) or a
  ``clang -fsanitize=fuzzer`` harness binary (engine=``libfuzzer``) with hard
  caps (time, mem, corpus bytes). When the engine binary is absent it returns
  a typed ``unavailable`` verdict with the exact install hint instead of
  failing startup. Novel crashes are returned as file paths for the caller to
  triage via ``run_until_crash`` and link with ``link_evidence_batch``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.heap_schemas import PARSER_VERSION

logger = logging.getLogger(__name__)

AFL_BATCH_SCHEMA = "pao.fuzz-afl-batch.v1"
CORPUS_MINIMIZE_SCHEMA = "pao.fuzz-corpus-minimize.v1"

MAX_CAMPAIGN_SECONDS = 3600
MAX_CORPUS_FILES = 2048
MAX_TOTAL_CORPUS_BYTES = 256 * 1024 * 1024
MAX_CRASHES_REPORTED = 64


def minimize_corpus(
    inputs: Sequence[Mapping[str, Any]],
    *,
    max_files: int = MAX_CORPUS_FILES,
    max_bytes: int = MAX_TOTAL_CORPUS_BYTES,
) -> Dict[str, Any]:
    """Dedup raw inputs by sha256, keep smallest-first, enforce caps (pure)."""
    if max_files <= 0 or max_files > MAX_CORPUS_FILES:
        raise ValueError(f"max_files must be 1-{MAX_CORPUS_FILES}")
    seen: Dict[str, Mapping[str, Any]] = {}
    for item in inputs:
        blob = item.get("data_hex") or item.get("data") or b""
        if isinstance(blob, str):
            try:
                raw = bytes.fromhex(blob)
            except ValueError:
                raw = blob.encode("utf-8", "ignore")
        elif isinstance(blob, (bytes, bytearray)):
            raw = bytes(blob)
        else:
            raw = str(blob).encode("utf-8", "ignore")
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in seen or len(raw) < _raw_len(seen[digest]):
            seen[digest] = item
    ranked = sorted(seen.values(), key=_raw_len)
    kept: List[Mapping[str, Any]] = []
    total = 0
    for item in ranked:
        size = _raw_len(item)
        if len(kept) >= max_files or total + size > max_bytes:
            break
        kept.append(item)
        total += size
    return {
        "schema": CORPUS_MINIMIZE_SCHEMA,
        "kept": len(kept),
        "dropped_duplicates": len(inputs) - len(seen),
        "dropped_over_budget": len(seen) - len(kept),
        "total_bytes": total,
        "sha256": [hashlib.sha256(_to_raw(i)).hexdigest() for i in kept],
    }


def _raw_len(item: Mapping[str, Any]) -> int:
    return len(_to_raw(item))


def _to_raw(item: Mapping[str, Any]) -> bytes:
    blob = item.get("data_hex") or item.get("data") or b""
    if isinstance(blob, str):
        try:
            return bytes.fromhex(blob)
        except ValueError:
            return blob.encode("utf-8", "ignore")
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    return str(blob).encode("utf-8", "ignore")


def _engine_hint(engine: str) -> str:
    if engine == "afl":
        return "install AFL++ (apt install afl++ / Docker image with afl-fuzz on PATH)"
    return (
        "build the harness with clang -fsanitize=fuzzer,address and pass harness_path"
    )


def run_coverage_batch(
    *,
    engine: str = "afl",
    target_path: Optional[str] = None,
    harness_path: Optional[str] = None,
    corpus_dir: str,
    output_dir: str,
    time_seconds: int = 600,
    mem_mb: int = 2048,
    extra_args: Optional[Sequence[str]] = None,
    runner: Optional[Callable[..., Any]] = None,
    cancellation_check: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run one bounded coverage campaign; honest ``unavailable`` when missing."""
    campaign_id = uuid.uuid4().hex[:12]
    if engine not in ("afl", "libfuzzer"):
        raise ValueError("engine must be 'afl' or 'libfuzzer'")
    if time_seconds <= 0 or time_seconds > MAX_CAMPAIGN_SECONDS:
        raise ValueError(f"time_seconds must be 1-{MAX_CAMPAIGN_SECONDS}")
    binary = target_path if engine == "afl" else harness_path
    if not binary:
        raise ValueError(
            "target_path is required for engine='afl'; harness_path for 'libfuzzer'"
        )
    exe = shutil.which(binary) or (binary if os.path.isfile(binary) else None)
    # For tests a fake runner bypasses the PATH check entirely.
    if runner is None:
        engine_bin = shutil.which("afl-fuzz" if engine == "afl" else harness_path or "")
        if engine == "afl" and engine_bin is None:
            return build_backend_response(
                "run_afl_batch",
                parsed={
                    "schema": AFL_BATCH_SCHEMA,
                    "supported": False,
                    "verdict": "unavailable",
                    "reason": "afl-fuzz not found on PATH",
                    "hint": _engine_hint(engine),
                    "campaign_id": campaign_id,
                },
                raw_responses=[],
                state="idle",
                parser_version=PARSER_VERSION,
            )
        if engine == "libfuzzer" and exe is None:
            return build_backend_response(
                "run_afl_batch",
                parsed={
                    "schema": AFL_BATCH_SCHEMA,
                    "supported": False,
                    "verdict": "unavailable",
                    "reason": f"harness not found: {harness_path}",
                    "hint": _engine_hint(engine),
                    "campaign_id": campaign_id,
                },
                raw_responses=[],
                state="idle",
                parser_version=PARSER_VERSION,
            )
    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    started = time.monotonic()
    if runner is not None:
        result = runner(
            engine=engine,
            binary=binary,
            corpus_dir=corpus_dir,
            output_dir=output_dir,
            time_seconds=time_seconds,
            mem_mb=mem_mb,
            extra_args=list(extra_args or []),
        )
        crashes = list((result or {}).get("crashes") or [])[:MAX_CRASHES_REPORTED]
        return build_backend_response(
            "run_afl_batch",
            parsed={
                "schema": AFL_BATCH_SCHEMA,
                "supported": True,
                "verdict": "completed",
                "campaign_id": campaign_id,
                "engine": engine,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "crash_count": len(crashes),
                "crashes": crashes,
                "next": "triage each crash path with run_until_crash, then link_evidence_batch",
            },
            raw_responses=[],
            state="idle",
            parser_version=PARSER_VERSION,
        )
    if engine == "afl":
        cmd = [
            "afl-fuzz",
            "-i",
            corpus_dir,
            "-o",
            output_dir,
            "-m",
            str(mem_mb),
            "-V",
            str(time_seconds),
            "--",
            binary,
        ]
    else:
        cmd = [
            str(binary),
            "-max_total_time=%d" % int(time_seconds),
            "-rss_limit_mb=%d" % int(mem_mb),
            "-print_final_stats=1",
            corpus_dir,
        ]
        cmd.extend(list(extra_args or []))
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=time_seconds + 60
        )
    except FileNotFoundError:
        return build_backend_response(
            "run_afl_batch",
            parsed={
                "schema": AFL_BATCH_SCHEMA,
                "supported": False,
                "verdict": "unavailable",
                "reason": f"engine binary missing for {engine}",
                "hint": _engine_hint(engine),
                "campaign_id": campaign_id,
            },
            raw_responses=[],
            state="idle",
            parser_version=PARSER_VERSION,
        )
    except subprocess.TimeoutExpired:
        return build_backend_response(
            "run_afl_batch",
            parsed={
                "schema": AFL_BATCH_SCHEMA,
                "supported": True,
                "verdict": "timeout",
                "campaign_id": campaign_id,
                "engine": engine,
            },
            raw_responses=[],
            state="idle",
            parser_version=PARSER_VERSION,
        )
    crash_dir = os.path.join(output_dir, "crashes" if engine == "afl" else "crash-")
    found: List[str] = []
    if os.path.isdir(crash_dir):
        for name in sorted(os.listdir(crash_dir))[:MAX_CRASHES_REPORTED]:
            if name.startswith("README"):
                continue
            found.append(os.path.join(crash_dir, name))
    return build_backend_response(
        "run_afl_batch",
        parsed={
            "schema": AFL_BATCH_SCHEMA,
            "supported": True,
            "verdict": "completed",
            "campaign_id": campaign_id,
            "engine": engine,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": proc.returncode,
            "crash_count": len(found),
            "crashes": found,
            "next": "triage each crash path with run_until_crash, then link_evidence_batch",
        },
        raw_responses=[],
        state="idle",
        parser_version=PARSER_VERSION,
    )


__all__ = [
    "minimize_corpus",
    "run_coverage_batch",
    "AFL_BATCH_SCHEMA",
    "CORPUS_MINIMIZE_SCHEMA",
]
