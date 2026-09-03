"""MCP tools for typed crash triage, snapshots, core files, and reports."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Annotated, Any, Dict, List, Literal, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext
from pydantic import Field

from paomcp.artifacts import ArtifactMetadata
from paomcp.operations import current_operation_id
from paomcp.tools.backends.crash import (
    MAX_CYCLIC_ALPHABET_CHARS,
    MAX_CYCLIC_LENGTH,
    MAX_CYCLIC_SCAN_MATCHES,
    MAX_CYCLIC_STACK_SCAN_BYTES,
    MAX_CYCLIC_VALUE_CHARS,
    MAX_STACK_BYTES,
    CrashTriage,
)
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_blocking,
    run_session_action,
)

StackBytes = Annotated[int, Field(ge=1, le=MAX_STACK_BYTES)]
DisassemblyBytes = Annotated[int, Field(ge=1, le=16 * 1024 * 1024)]
CyclicN = Annotated[int, Field(ge=2, le=8)]
CyclicWidth = Annotated[int, Field(ge=1, le=64)]
CyclicLength = Annotated[int, Field(ge=1, le=MAX_CYCLIC_LENGTH)]
CyclicStackBytes = Annotated[int, Field(ge=0, le=MAX_CYCLIC_STACK_SCAN_BYTES)]
CyclicMatches = Annotated[int, Field(ge=1, le=MAX_CYCLIC_SCAN_MATCHES)]
CyclicValue = Annotated[str, Field(min_length=1, max_length=MAX_CYCLIC_VALUE_CHARS)]
CyclicAlphabet = Annotated[
    str, Field(min_length=2, max_length=MAX_CYCLIC_ALPHABET_CHARS)
]


def _snapshot_report_provenance(
    crash_snapshot: Dict[str, Any], capture_options: Dict[str, Any]
) -> Dict[str, Any]:
    """Describe the evidence actually retained in the reported snapshot."""
    component_status = crash_snapshot.get("component_status", {})
    if not isinstance(component_status, dict):
        component_status = {}
    sources = crash_snapshot.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}
    mappings_requested = bool(capture_options.get("include_mappings"))
    mapping_source = sources.get("mappings") if mappings_requested else None
    if mappings_requested and not mapping_source:
        mapping_source = "unavailable"
    return {
        "capture_options": capture_options,
        "component_status": component_status,
        "sources": sources,
        "mappings": {
            "requested": mappings_requested,
            "available": (
                bool(component_status.get("mappings")) if mappings_requested else None
            ),
            "source": mapping_source,
        },
    }


def _markdown_report(report: Dict[str, Any]) -> bytes:
    snapshot = report.get("crash_snapshot", {})
    analysis = snapshot.get("analysis", {})
    stop = snapshot.get("stop", {})
    signal = stop.get("signal") or {}
    core = report.get("core_artifact")
    classification = report.get("classification") or {}
    primary = (
        classification.get("primary", {}) if isinstance(classification, dict) else {}
    )
    signature = report.get("signature") or {}
    lines = [
        "# Pao crash report",
        "",
        f"- Crash ID: `{report.get('crash_id')}`",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Signal: `{signal.get('name') or 'unknown'}`",
        f"- Category: `{analysis.get('category') or 'unknown'}`",
        f"- Fault address: `{analysis.get('fault_address', {}).get('value')}`",
        f"- Fault access: `{analysis.get('fault_access') or 'unknown'}`",
        f"- Primitive hypothesis: `{primary.get('kind') or 'unknown'}`",
        f"- Classification confidence: `{primary.get('confidence') or 'unknown'}`",
        f"- Crash signature: `{signature.get('digest') or 'unknown'}`",
        f"- Exploitability: `{analysis.get('exploitability') or 'unknown'}`",
        f"- Complete: `{str(bool(report.get('complete'))).lower()}`",
    ]
    if isinstance(core, dict):
        lines.extend(
            [
                f"- Core artifact: `{core.get('artifact_id')}`",
                f"- Core SHA-256: `{core.get('sha256')}`",
                f"- Core bytes: `{core.get('size_bytes')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Integrity manifest",
            "",
            "The complete typed evidence follows as JSON. Core bytes are never "
            "embedded; use the exact session-owned artifact id and verify its hash.",
            "",
            "```json",
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _core_reference_for_crash(
    metadata: ArtifactMetadata,
    *,
    expected_crash_id: str,
    session_id: str,
    verification: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate that a core artifact was captured from this exact crash."""
    if (
        metadata.session_id != session_id
        or metadata.content_type != "application/x-coredump"
    ):
        raise ValueError(
            "core_artifact_id must reference an application/x-coredump artifact "
            "owned by this session"
        )
    provenance = metadata.provenance
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema") != "pao.core-provenance.v1"
        or provenance.get("crash_id") != expected_crash_id
        or provenance.get("session_id") != session_id
    ):
        raise ValueError(
            "core_artifact_id was not captured from expected_crash_id in this "
            "session; capture a new core for this exact crash"
        )
    verified_artifact = verification.get("artifact")
    if (
        verification.get("valid") is not True
        or not isinstance(verified_artifact, dict)
        or verified_artifact.get("artifact_id") != metadata.artifact_id
    ):
        raise ValueError(
            "core_artifact_id failed size/SHA-256 integrity verification; "
            "capture a new core"
        )
    reference = metadata.to_dict()
    reference["integrity_verified"] = True
    return reference


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def get_last_stop(
        session_id: str,
        fatal_signals: Optional[List[str]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return the latest persisted GDB/MI stop, crash id, and classification."""
        session = resolve_debug_session(
            get_services(ctx), session_id=session_id, create_if_missing=False
        )
        result = await run_session_action(
            session, lambda: session.tools.get_last_stop(fatal_signals)
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def run_until_crash(
        session_id: str,
        arguments: Optional[List[str]] = None,
        mode: Literal["auto", "run", "continue", "wait"] = "auto",
        timeout_seconds: Optional[float] = None,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run/continue/wait until a crash, non-crash stop, normal exit, or timeout.

        This tool never auto-resumes through a breakpoint or other non-crash stop.
        Arguments are a typed vector, not a shell command string.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        if operation_id is not None:

            def interrupt_inferior() -> None:
                # Signal-only: the worker thread owns the MI response stream via
                # its drain loop, so the cancel handler must never call the
                # draining ``interrupt``. It just raises SIGINT and lets the
                # worker observe the resulting stop and its cancellation check.
                signal_only = getattr(session.gdb, "signal_interrupt", None)
                if callable(signal_only):
                    signal_only()
                else:
                    # A legacy adapter without signal-only interruption cannot
                    # safely drain MI from this cancellation thread. Destroy the
                    # process boundary and let the worker observe EOF instead.
                    killer = getattr(session.gdb, "kill_process_group", None)
                    if callable(killer):
                        killer()
                    marker = getattr(session.gdb, "mark_destroyed", None)
                    if callable(marker):
                        marker("cancel handler lacked signal-only interruption")

            services.operation_registry.set_cancel_handler(
                operation_id, interrupt_inferior
            )
        try:
            result = await run_session_action(
                session,
                lambda: session.tools.run_until_crash(
                    arguments=arguments,
                    mode=mode,
                    timeout_seconds=timeout_seconds,
                    fatal_signals=fatal_signals,
                    cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                        operation_id
                    ),
                ),
            )
        finally:
            if operation_id is not None:
                services.operation_registry.set_cancel_handler(operation_id, None)
            if session.gdb.is_destroyed():
                evict = getattr(services.session_registry, "evict_session", None)
                if callable(evict):
                    evict(session.session_id)
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def capture_crash_snapshot(
        expected_crash_id: str,
        session_id: str,
        stack_bytes: StackBytes = 256,
        disassembly_bytes: DisassemblyBytes = 64,
        include_mappings: bool = True,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Capture typed MI state for the exact current fatal-signal crash id."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.capture_crash_snapshot(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                fatal_signals=fatal_signals,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def analyze_crash(
        expected_crash_id: str,
        session_id: str,
        stack_bytes: StackBytes = 256,
        disassembly_bytes: DisassemblyBytes = 64,
        include_mappings: bool = True,
        refresh: bool = False,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Analyze one crash conservatively; exploitability remains evidence-driven."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.analyze_crash(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                refresh=refresh,
                fatal_signals=fatal_signals,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def classify_crash(
        expected_crash_id: str,
        session_id: str,
        stack_bytes: StackBytes = 256,
        disassembly_bytes: DisassemblyBytes = 64,
        include_mappings: bool = True,
        refresh: bool = False,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Classify one exact crash conservatively and emit a versioned signature."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.crash.classify_crash(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                refresh=refresh,
                fatal_signals=fatal_signals,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def compare_crashes(
        left_crash_id: str,
        right_crash_id: str,
        session_id: str,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Compare two retained crash descriptors with ASLR-normalized evidence."""
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        result = await run_session_action(
            session,
            lambda: session.tools.crash.compare_crashes(
                left_crash_id,
                right_crash_id,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            ),
        )
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def find_cyclic_offset(
        value: Optional[CyclicValue] = None,
        expected_crash_id: Optional[str] = None,
        session_id: Optional[str] = None,
        n: CyclicN = 4,
        endian: Literal["little", "big"] = "little",
        width: Optional[CyclicWidth] = None,
        max_length: CyclicLength = 1024 * 1024,
        alphabet: CyclicAlphabet = "abcdefghijklmnopqrstuvwxyz",
        scan_registers: bool = True,
        scan_stack: bool = True,
        max_stack_scan_bytes: CyclicStackBytes = 4096,
        max_matches: CyclicMatches = 64,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Find offsets for one value or scan bounded evidence from an exact crash."""
        services = get_services(ctx)
        operation_id = current_operation_id()
        cancellation_check = lambda: services.operation_registry.raise_if_cancelled(
            operation_id
        )
        if expected_crash_id is not None:
            if session_id is None:
                raise ValueError(
                    "session_id is required when expected_crash_id is provided"
                )
            session = resolve_debug_session(
                services, session_id=session_id, create_if_missing=False
            )
            result = await run_session_action(
                session,
                lambda: session.tools.crash.find_cyclic_offsets_for_crash(
                    expected_crash_id,
                    value=value,
                    n=n,
                    endian=endian,
                    width=width,
                    max_length=max_length,
                    alphabet=alphabet,
                    scan_registers=scan_registers,
                    scan_stack=scan_stack,
                    max_stack_scan_bytes=max_stack_scan_bytes,
                    max_matches=max_matches,
                    cancellation_check=cancellation_check,
                ),
            )
            result["session_id"] = session.session_id
            return result
        if value is None:
            raise ValueError("value is required when expected_crash_id is omitted")
        if session_id is not None:
            raise ValueError(
                "session_id is only valid with expected_crash_id in crash-aware mode"
            )
        return await run_blocking(
            lambda: CrashTriage.find_cyclic_offset(
                value,
                n=n,
                endian=endian,
                width=width,
                max_length=max_length,
                alphabet=alphabet,
                cancellation_check=cancellation_check,
            )
        )

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def capture_core_dump(
        expected_crash_id: str,
        session_id: str,
        max_size_bytes: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Capture a session-owned ELF core artifact using GDB CLI fallback.

        GDB/MI has no dedicated core-generation command, so this tool uses the
        built-in raw-filename `generate-core-file` CLI command through the same
        GDB transport. The caller cannot choose an output path.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()
        core_capability = (
            services.debugger_compatibility.get("gdb_features", {})
            .get("generate_core_file", {})
            .get("available")
        )
        if core_capability is False:
            raise RuntimeError(
                "This GDB runtime does not expose the generate-core-file CLI "
                "fallback required by capture_core_dump"
            )

        def capture() -> Dict[str, Any]:
            estimate = session.tools.crash.estimate_core_size()
            configured_cap = session.tools.crash.max_core_dump_bytes
            cap = configured_cap if max_size_bytes is None else int(max_size_bytes)
            if not 1 <= cap <= configured_cap:
                raise ValueError(
                    f"max_size_bytes must be between 1 and {configured_cap}"
                )
            estimated = int(estimate.get("estimated_upper_bound_bytes", cap))
            reserve = min(cap, max(1, estimated))

            def progress(current: int, total: int, message: str) -> None:
                if operation_id is not None:
                    services.operation_registry.update_progress(
                        operation_id,
                        current=min(current, total),
                        total=total,
                        message=message,
                    )

            metadata, generation = services.artifact_store.save_generated_file(
                lambda staging_path: session.tools.generate_core_file(
                    expected_crash_id,
                    staging_path,
                    max_size_bytes=cap,
                    timeout_seconds=timeout_seconds,
                    fatal_signals=fatal_signals,
                    min_free_bytes=services.artifact_store.min_free_bytes,
                    cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                        operation_id
                    ),
                    progress_callback=progress,
                ),
                reserve_bytes=reserve,
                max_size_bytes=cap,
                tool="capture_core_dump",
                session_id=session.session_id,
                content_type="application/x-coredump",
                encoding="binary",
                provenance={
                    "schema": "pao.core-provenance.v1",
                    "crash_id": expected_crash_id,
                    "session_id": session.session_id,
                },
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            )
            return {
                "success": True,
                "command": "capture_core_dump",
                "state": session.gdb.get_state(),
                "parser_version": "1",
                "parsed": {
                    "crash_id": expected_crash_id,
                    "artifact": metadata.to_dict(),
                    "generation": generation.get("parsed", {}),
                    "estimate": estimate,
                    "component_status": {
                        "crash_identity": True,
                        "core_generation": True,
                        "artifact_commit": True,
                    },
                    "complete": True,
                    "source": "gdb-cli",
                },
                "raw_responses": generation.get("raw_responses", []),
            }

        if operation_id is not None:

            def interrupt_core_generation() -> None:
                session.gdb.signal_interrupt()

            services.operation_registry.set_cancel_handler(
                operation_id, interrupt_core_generation
            )
        try:
            result = await run_session_action(session, capture)
        finally:
            if operation_id is not None:
                services.operation_registry.set_cancel_handler(operation_id, None)
            # P2: evict in finally so a destroyed session is always cleaned up,
            # even when run_session_action raises an exception.
            if session.gdb.is_destroyed():
                evict = getattr(services.session_registry, "evict_session", None)
                if callable(evict):
                    evict(session.session_id)
        if session.gdb.is_destroyed():
            result.setdefault("warnings", []).append(
                "The debugger session was destroyed while enforcing the core-size "
                "limit; call create_debug_session with the same session_id before "
                "continuing"
            )
            if isinstance(result.get("parsed"), dict):
                result["parsed"]["session_destroyed"] = True
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def generate_crash_report(
        expected_crash_id: str,
        session_id: str,
        core_artifact_id: Optional[str] = None,
        report_format: Literal["json", "markdown"] = "json",
        refresh_analysis: bool = False,
        stack_bytes: Optional[StackBytes] = None,
        disassembly_bytes: Optional[DisassemblyBytes] = None,
        include_mappings: Optional[bool] = None,
        fatal_signals: Optional[List[str]] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Generate a lossless JSON/Markdown report artifact for one exact crash."""
        if report_format not in {"json", "markdown"}:
            raise ValueError("report_format must be 'json' or 'markdown'")
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        operation_id = current_operation_id()

        def generate() -> Dict[str, Any]:
            analysis_result = session.tools.crash.analyze_crash(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                refresh=refresh_analysis,
                fatal_signals=fatal_signals,
                cancellation_check=lambda: services.operation_registry.raise_if_cancelled(
                    operation_id
                ),
            )
            crash_snapshot = analysis_result.get("parsed", {}).get("crash_snapshot", {})
            classification = analysis_result.get("parsed", {}).get("classification", {})
            signature = analysis_result.get("parsed", {}).get("signature", {})
            capture_options = analysis_result.get("parsed", {}).get(
                "capture_options", {}
            )
            if not isinstance(capture_options, dict):
                capture_options = {}
            snapshot_provenance = _snapshot_report_provenance(
                crash_snapshot if isinstance(crash_snapshot, dict) else {},
                capture_options,
            )
            snapshot_component_status = snapshot_provenance["component_status"]
            snapshot_sources = snapshot_provenance["sources"]
            mappings_provenance = snapshot_provenance["mappings"]
            core_reference = None
            if core_artifact_id is not None:
                core_metadata = services.artifact_store.get_for_session(
                    core_artifact_id, session.session_id
                )
                core_verification = services.artifact_store.verify(
                    core_artifact_id, session.session_id
                )
                core_reference = _core_reference_for_crash(
                    core_metadata,
                    expected_crash_id=expected_crash_id,
                    session_id=session.session_id,
                    verification=core_verification,
                )
            report = {
                "report_schema": "pao.crash-report.v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "crash_id": expected_crash_id,
                "session_id": session.session_id,
                "complete": bool(analysis_result.get("complete")),
                "capture_options": capture_options,
                "classification": classification,
                "signature": signature,
                "component_status": {
                    "analysis": bool(analysis_result.get("success")),
                    "classification": bool(classification),
                    "signature": bool(signature.get("digest")),
                    "mappings": mappings_provenance["available"],
                    "core_requested": core_artifact_id is not None,
                    "core_reference": (
                        core_artifact_id is None or core_reference is not None
                    ),
                },
                "crash_snapshot": crash_snapshot,
                "core_artifact": core_reference,
                "raw_evidence": analysis_result.get("raw_responses", {}),
                "sources": {
                    "analysis": snapshot_sources.get("debugger"),
                    "mappings": mappings_provenance["source"],
                    "core": "gdb-cli" if core_reference is not None else None,
                    "report": "local",
                },
                "snapshot_provenance": snapshot_provenance,
            }
            services.operation_registry.raise_if_cancelled(operation_id)
            if report_format == "json":
                payload = json.dumps(
                    report,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                content_type = "application/json"
            else:
                payload = _markdown_report(report)
                content_type = "text/markdown"
            metadata = services.artifact_store.save_bytes(
                payload,
                tool="generate_crash_report",
                session_id=session.session_id,
                content_type=content_type,
                encoding="utf-8",
                provenance={
                    "schema": "pao.crash-report-provenance.v1",
                    "crash_id": expected_crash_id,
                    "session_id": session.session_id,
                    "format": report_format,
                    "core_artifact_id": core_artifact_id,
                    "signature": signature.get("digest"),
                    "snapshot": snapshot_provenance,
                },
            )
            return {
                "success": bool(analysis_result.get("success")),
                "complete": bool(report["complete"]),
                "command": "generate_crash_report",
                "state": session.gdb.get_state(),
                "parser_version": "1",
                "parsed": {
                    "crash_id": expected_crash_id,
                    "format": report_format,
                    "artifact": metadata.to_dict(),
                    "core_artifact": core_reference,
                    "classification": classification,
                    "signature": signature,
                    "capture_options": capture_options,
                    "component_status": report["component_status"],
                    "complete": report["complete"],
                    "source": "local",
                },
                "raw_responses": analysis_result.get("raw_responses", {}),
                **(
                    {
                        "warnings": analysis_result.get("warnings", []),
                        "error": analysis_result.get(
                            "error", "Crash analysis was not usable"
                        ),
                        "type": analysis_result.get("type", "CrashReportError"),
                    }
                    if not analysis_result.get("success")
                    else (
                        {"warnings": analysis_result.get("warnings", [])}
                        if analysis_result.get("warnings")
                        else {}
                    )
                ),
            }

        result = await run_session_action(session, generate)
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def analyze_kernel_oops(
        log_text: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Triage a Linux kernel oops/panic log (no debugger needed).

        Parses fault kind (NULL deref, KASAN UAF/OOB, GPF, ...), fault
        address, RIP symbol+offset, call trace, CPU/comm, taint, and slab
        cache, with an ASLR-stable fingerprint for dedup. Paste dmesg or
        serial output (max 1MiB).

        Args:
            log_text: Kernel log text containing the oops.
        """
        from paomcp.tools.backends.kernel_oops import analyze_kernel_oops as triage

        _ = ctx
        return await run_blocking(lambda: triage(log_text))


__all__ = ["register"]
