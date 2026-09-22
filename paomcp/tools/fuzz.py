"""MCP tools for bounded file-input fuzz batches."""

import logging
import os
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.operations import current_operation_id
from paomcp.tools.backends.fuzz import FuzzBatch, generate_inputs
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_binary_path,
    resolve_debug_session,
    run_blocking,
    run_session_action,
)
from paomcp.utils.paths import resolve_workspace_path

logger = logging.getLogger(__name__)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def generate_fuzz_inputs(
        count: int = 20,
        max_len: int = 4096,
        strategies: Optional[List[str]] = None,
        seed: int = 0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Generate deterministic file inputs (zeros, cyclic, bitflips).

        Args:
            count: Inputs to generate (1-500).
            max_len: Max bytes per input.
            strategies: Subset of zeros/ones/cyclic/bitflip, round-robined.
            seed: RNG seed; same seed replays the same corpus.
        """
        _ = ctx
        return generate_inputs(
            count=count, max_len=max_len, strategies=strategies, seed=seed
        )

    @mcp.tool()
    @catch_errors()
    async def run_fuzz_batch(
        session_id: str,
        binary_path: Optional[str] = None,
        inputs: Optional[List[Dict[str, Any]]] = None,
        timeout_per_run: float = 10.0,
        max_runs: int = 50,
        research_session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run file inputs under GDB; triage + dedup every crash.

        Each input is written to ``/workspace/fuzz/<campaign>/`` and fed
        as a file argument. Crashes go through the crash pipeline with
        signature-digest dedup; novel crashes can be linked as crash
        evidence into a research session.

        Args:
            session_id: Debug session (must already exist).
            binary_path: Target binary (defaults to the session binary).
            inputs: Items shaped like ``generate_fuzz_inputs`` entries.
            timeout_per_run: Seconds per run (max 120).
            max_runs: Cap runs executed (max 500).
            research_session_id: Optional research session receiving novel
                crashes as linked evidence.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        resolved_binary = resolve_binary_path(
            binary_path,
            session,
            require_exists=True,
            workspace_root=services.runtime_paths.workspace_root,
        )
        if not inputs:
            raise ValueError("inputs cannot be empty; call generate_fuzz_inputs first")
        campaign = FuzzBatch(session.session_id)
        corpus_dir = resolve_workspace_path(
            os.path.join("fuzz", campaign.campaign_id),
            workspace_root=services.runtime_paths.workspace_root,
            require_exists=False,
            kind="corpus_dir",
        )
        os.makedirs(corpus_dir, exist_ok=True)
        operation_id = current_operation_id()

        def _run() -> Dict[str, Any]:
            return campaign.run_batch(
                tools=session.tools,
                binary_path=resolved_binary,
                corpus_dir=corpus_dir,
                inputs=inputs,
                timeout_per_run=timeout_per_run,
                max_runs=max_runs,
                cancellation_check=(
                    lambda: services.operation_registry.raise_if_cancelled(operation_id)
                ),
            )

        result = await run_session_action(session, _run)
        result["session_id"] = session.session_id
        if research_session_id is not None and result.get("success"):
            items = [
                {
                    "kind": "crash",
                    "ref": str(novel.get("crash_id")),
                    "description": (
                        f"fuzz novel crash signal={novel.get('signal')} "
                        f"input={novel.get('input_path')}"
                    ),
                }
                for novel in (result.get("parsed", {}) or {}).get("novel", []) or []
                if novel.get("crash_id")
            ]
            if not items:
                result["research_linked"] = []
                return result
            try:
                # One batched link: a single timeline event and one persist.
                linked = await run_blocking(
                    lambda: services.research_registry.link_evidence_batch(
                        session_id=research_session_id,
                        items=items,
                    )
                )
                result["research_linked"] = [e.evidence_id for e in linked]
            except Exception:  # noqa: BLE001 - linking never fails a batch
                logger.exception("Fuzz research linking failed")
                result["research_linked"] = []
        return result

    @mcp.tool()
    @catch_errors()
    async def minimize_corpus(
        inputs: List[Dict[str, Any]],
        max_files: int = 2048,
        max_bytes: int = 268435456,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Dedup a fuzz corpus by sha256, smallest-first under caps (pure).

        Args:
            inputs: Items shaped like ``generate_fuzz_inputs`` entries.
            max_files: Cap files kept.
            max_bytes: Cap total bytes kept.
        """
        from paomcp.tools.backends.fuzz_cov import minimize_corpus as _minimize

        _ = ctx
        return await run_blocking(
            lambda: _minimize(inputs, max_files=max_files, max_bytes=max_bytes)
        )

    @mcp.tool()
    @catch_errors()
    async def run_afl_batch(
        corpus_dir: str,
        output_dir: str,
        engine: str = "afl",
        target_path: Optional[str] = None,
        harness_path: Optional[str] = None,
        time_seconds: int = 600,
        mem_mb: int = 2048,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run one bounded AFL++/libFuzzer campaign; honest unavailable when missing.

        Args:
            corpus_dir: Seed corpus dir under /workspace.
            output_dir: Findings dir under /workspace.
            engine: "afl" (afl-fuzz + target_path) or "libfuzzer" (harness_path).
            target_path: Target binary under /workspace (afl).
            harness_path: clang -fsanitize=fuzzer harness (libfuzzer).
            time_seconds: Campaign budget (max 3600).
            mem_mb: Memory limit per child.
        """
        from paomcp.tools.backends.fuzz_cov import run_coverage_batch

        services = get_services(ctx)
        workspace_root = services.runtime_paths.workspace_root
        resolved_corpus = resolve_workspace_path(
            corpus_dir,
            workspace_root=workspace_root,
            require_exists=False,
            kind="corpus_dir",
        )
        resolved_out = resolve_workspace_path(
            output_dir,
            workspace_root=workspace_root,
            require_exists=False,
            kind="corpus_dir",
        )
        resolved_target = None
        if target_path is not None:
            resolved_target = resolve_workspace_path(
                target_path,
                workspace_root=workspace_root,
                require_exists=False,
                kind="binary_path",
            )
        resolved_harness = None
        if harness_path is not None:
            resolved_harness = resolve_workspace_path(
                harness_path,
                workspace_root=workspace_root,
                require_exists=False,
                kind="binary_path",
            )
        os.makedirs(resolved_corpus, exist_ok=True)
        os.makedirs(resolved_out, exist_ok=True)
        operation_id = current_operation_id()

        def _run() -> Dict[str, Any]:
            return run_coverage_batch(
                engine=engine,
                target_path=resolved_target,
                harness_path=resolved_harness,
                corpus_dir=resolved_corpus,
                output_dir=resolved_out,
                time_seconds=time_seconds,
                mem_mb=mem_mb,
                cancellation_check=(
                    lambda: services.operation_registry.raise_if_cancelled(operation_id)
                ),
            )

        return await run_blocking(_run)
