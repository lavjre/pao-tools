"""Bounded file-input fuzz batches over a live debug session.

This is a *campaign helper*, not a coverage-guided fuzzer: it generates
deterministic inputs (cyclic patterns, byte fills, seeded bitflips),
feeds each one to the target as a file argument under GDB, and funnels
every crash through the existing :class:`CrashTriage` pipeline with
signature-digest dedup. One crashing input costs one snapshot; clean
runs cost one ``run_until_crash`` each. Everything is capped
(``max_runs``, ``timeout_per_run``, ``max_len``) so a batch cannot pin
a worker indefinitely.
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from paomcp.operations import OperationCancelledError
from paomcp.tools.backends.foundation import build_backend_response

logger = logging.getLogger(__name__)

FUZZ_BATCH_SCHEMA = "pao.fuzz-batch.v1"
FUZZ_INPUTS_SCHEMA = "pao.fuzz-inputs.v1"
DEFAULT_MAX_RUNS = 50
MAX_CONFIGURED_RUNS = 500
DEFAULT_TIMEOUT_PER_RUN_SECONDS = 10.0
MAX_TIMEOUT_PER_RUN_SECONDS = 120.0
DEFAULT_MAX_INPUT_LEN = 4096
MAX_CONFIGURED_INPUT_LEN = 1024 * 1024
MAX_TOTAL_CORPUS_BYTES = 256 * 1024 * 1024
INPUT_STRATEGIES = ("zeros", "ones", "cyclic", "bitflip")


def _cyclic_byte(index: int) -> int:
    alphabet = b"abcdefghijklmnopqrstuvwxyz"
    return alphabet[index % len(alphabet)]


def generate_inputs(
    *,
    count: int,
    max_len: int = DEFAULT_MAX_INPUT_LEN,
    strategies: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """Deterministically generate ``count`` file inputs (pure function)."""
    if count <= 0 or count > MAX_CONFIGURED_RUNS:
        raise ValueError(f"count must be between 1 and {MAX_CONFIGURED_RUNS}")
    if max_len <= 0 or max_len > MAX_CONFIGURED_INPUT_LEN:
        raise ValueError(f"max_len must be between 1 and {MAX_CONFIGURED_INPUT_LEN}")
    wanted = list(strategies or ["zeros", "cyclic", "bitflip"])
    unknown = sorted(set(wanted) - set(INPUT_STRATEGIES))
    if unknown:
        raise ValueError(
            f"unknown fuzz strategies {unknown}; expected any of "
            f"{list(INPUT_STRATEGIES)}"
        )
    rng = random.Random(seed)
    seeds: List[bytes] = [b"A" * 64, b"\x00" * 64, b"PAO" * 32]
    inputs: List[Dict[str, Any]] = []
    for index in range(count):
        strategy = wanted[index % len(wanted)]
        length = 1 + rng.randrange(max_len)
        if strategy == "zeros":
            data = b"\x00" * length
        elif strategy == "ones":
            data = b"\xff" * length
        elif strategy == "cyclic":
            data = bytes(_cyclic_byte(i) for i in range(length))
        else:  # bitflip
            base = bytearray(rng.choice(seeds))
            if len(base) < length:
                base.extend(b"\x41" * (length - len(base)))
            else:
                del base[length:]
            flips = max(1, len(base) // 16)
            for _ in range(flips):
                position = rng.randrange(len(base))
                base[position] ^= 1 << rng.randrange(8)
            data = bytes(base)
        inputs.append(
            {
                "index": index,
                "strategy": strategy,
                "length": len(data),
                "hex": data.hex(),
            }
        )
    return build_backend_response(
        "generate_fuzz_inputs",
        parsed={
            "schema": FUZZ_INPUTS_SCHEMA,
            "count": len(inputs),
            "max_len": max_len,
            "strategies": wanted,
            "seed": seed,
            "inputs": inputs,
        },
        raw_responses=[],
        state="idle",
    )


class FuzzBatch:
    """Run one bounded batch of file inputs against a debug session's tools."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.campaign_id = uuid.uuid4().hex[:12]

    def run_batch(
        self,
        *,
        tools: Any,
        binary_path: str,
        corpus_dir: str,
        inputs: Sequence[Mapping[str, Any]],
        timeout_per_run: float = DEFAULT_TIMEOUT_PER_RUN_SECONDS,
        max_runs: int = DEFAULT_MAX_RUNS,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Feed inputs as files to the target; triage every crash.

        ``tools`` is the session's tool bundle (duck-typed: ``set_file`` +
        ``crash.run_until_crash`` / ``crash.capture_crash_snapshot``).
        """
        if max_runs <= 0 or max_runs > MAX_CONFIGURED_RUNS:
            raise ValueError(f"max_runs must be between 1 and {MAX_CONFIGURED_RUNS}")
        try:
            timeout = float(timeout_per_run)
        except (TypeError, ValueError):
            raise ValueError("timeout_per_run must be a number of seconds")
        if timeout <= 0 or timeout > MAX_TIMEOUT_PER_RUN_SECONDS:
            raise ValueError(
                f"timeout_per_run must be between 0 and {MAX_TIMEOUT_PER_RUN_SECONDS}"
            )
        planned = list(inputs)[:max_runs]
        total_bytes = sum(len(str(item.get("hex", ""))) // 2 for item in planned)
        if total_bytes > MAX_TOTAL_CORPUS_BYTES:
            raise ValueError(
                f"planned corpus is ~{total_bytes} bytes, exceeding the "
                f"{MAX_TOTAL_CORPUS_BYTES}-byte batch budget; split the batch"
            )
        os.makedirs(corpus_dir, exist_ok=True)
        set_result = tools.set_file(binary_path)
        if not set_result.get("success"):
            return build_backend_response(
                "run_fuzz_batch",
                parsed=None,
                raw_responses=[],
                state="idle",
                success=False,
                error=f"set_file failed: {set_result.get('error')}",
                error_type=str(set_result.get("type") or "FuzzSetupError"),
            )
        started = time.monotonic()
        seen_digests: Dict[str, Dict[str, Any]] = {}
        run_records: List[Dict[str, Any]] = []
        errors: List[str] = []
        for position, item in enumerate(planned):
            if cancellation_check is not None:
                cancellation_check()
            raw_hex = str(item.get("hex", ""))
            try:
                payload = bytes.fromhex(raw_hex)
            except ValueError:
                errors.append(f"input {position}: invalid hex, skipped")
                continue
            input_path = os.path.join(corpus_dir, f"run_{position:04d}.bin")
            try:
                with open(input_path, "wb") as handle:
                    handle.write(payload)
            except OSError as exc:
                errors.append(f"input {position}: corpus write failed: {exc}")
                continue
            try:
                run_result = tools.crash.run_until_crash(
                    arguments=[input_path],
                    mode="run",
                    timeout_seconds=timeout,
                    cancellation_check=cancellation_check,
                )
            except OperationCancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad run never kills a batch
                logger.exception("Fuzz run %d failed", position)
                errors.append(f"input {position}: run failed: {exc}")
                continue
            parsed = (
                run_result.get("parsed", {}) if isinstance(run_result, dict) else {}
            )
            crashed = bool(parsed.get("crashed", False))
            record: Dict[str, Any] = {
                "index": position,
                "strategy": item.get("strategy"),
                "length": len(payload),
                "input_path": input_path,
                "crashed": crashed,
                "crash_id": parsed.get("crash_id"),
                "signal": parsed.get("signal"),
            }
            if crashed:
                crash_id = parsed.get("crash_id")
                digest = self._crash_digest(
                    tools, crash_id, parsed, cancellation_check=cancellation_check
                )
                record["digest"] = digest
                if digest is not None and digest not in seen_digests:
                    seen_digests[digest] = record
                    record["novel"] = True
                else:
                    record["novel"] = False
            run_records.append(record)
        elapsed = time.monotonic() - started
        return build_backend_response(
            "run_fuzz_batch",
            parsed={
                "schema": FUZZ_BATCH_SCHEMA,
                "campaign_id": self.campaign_id,
                "session_id": self.session_id,
                "binary_path": binary_path,
                "runs": len(run_records),
                "planned": len(planned),
                "crashes": sum(1 for r in run_records if r["crashed"]),
                "unique_crashes": len(seen_digests),
                "novel": list(seen_digests.values()),
                "records": run_records,
                "errors": errors,
                "elapsed_seconds": elapsed,
            },
            raw_responses=[],
            state="idle",
        )

    def _crash_digest(
        self,
        tools: Any,
        crash_id: Optional[str],
        parsed: Mapping[str, Any],
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Optional[str]:
        """Snapshot a crash and return its signature digest for dedup."""
        if not crash_id:
            signal = parsed.get("signal")
            stop = parsed.get("stop_reason") or parsed.get("state")
            if signal is None and stop is None:
                return None
            return f"{signal}:{stop}"
        try:
            snapshot = tools.crash.capture_crash_snapshot(
                crash_id, cancellation_check=cancellation_check
            )
        except OperationCancelledError:
            raise
        except Exception:  # noqa: BLE001 - dedup is best-effort
            logger.exception("Fuzz crash snapshot failed for %s", crash_id)
            return None
        snap_parsed = snapshot.get("parsed", {}) if isinstance(snapshot, dict) else {}
        signature = snap_parsed.get("signature", {}) or {}
        digest = signature.get("digest")
        return str(digest) if digest is not None else None


__all__ = [
    "FuzzBatch",
    "generate_inputs",
    "FUZZ_BATCH_SCHEMA",
    "FUZZ_INPUTS_SCHEMA",
    "INPUT_STRATEGIES",
    "DEFAULT_MAX_RUNS",
    "MAX_CONFIGURED_RUNS",
]
