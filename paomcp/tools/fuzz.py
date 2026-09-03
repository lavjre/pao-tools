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
