"""MCP tools for typed research workspace."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services
from paomcp.utils.paths import resolve_workspace_path


def _resolve_contained_workspace_path(
    raw_path: str,
    *,
    workspace_root: str,
    require_exists: bool,
    kind: str,
) -> str:
    """Resolve a workspace path and reject symlink escapes after realpath.

    ``resolve_workspace_path`` only rejects lexical escapes (``..``): a
    symlinked directory under the workspace passes that check while the
    realpath lands outside it.  Re-running the containment test on the
    realpath resolves what exists (parents included) before any bundle
    read or write happens.
    """
    resolved = resolve_workspace_path(
        raw_path,
        workspace_root=workspace_root,
        require_exists=require_exists,
        kind=kind,
    )
    workspace_real = os.path.realpath(workspace_root)
    resolved_real = os.path.realpath(resolved)
    try:
        contained = (
            os.path.commonpath([resolved_real, workspace_real]) == workspace_real
        )
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(
            f"{kind} resolves outside the configured workspace "
            f"(symlink escape): '{raw_path}'"
        )
    return resolved


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def create_research_session(
        name: str,
        description: Optional[str] = None,
        owning_debug_session: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Create a typed research workspace for one investigation.

        A research session is the bounded in-memory container for notes,
        hypotheses, evidence links, named snapshots and an append-only
        timeline.  It is independent of any debugger session;
        ``owning_debug_session`` is a soft label so a caller can pair a
        workspace with the debug session that produced its evidence, but
        the two lifecycles do not have to match.  Use
        ``export_research_session`` before shutdown when the workspace must
        survive a server restart.

        Args:
            name: Human-readable label (required).
            description: Optional longer description of the investigation.
            owning_debug_session: Optional debug session id this
                workspace was opened alongside.
            metadata: Optional arbitrary metadata dict (target binary
                path, CTF challenge name, ticket id, etc.).
        """
        services = get_services(ctx)
        return services.research_registry.create_session_response(
            name=name,
            description=description,
            owning_debug_session=owning_debug_session,
            metadata=dict(metadata) if metadata else None,
        )

    @mcp.tool()
    @catch_errors()
    async def get_research_session(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Fetch a research session with notes, hypotheses, evidence and snapshots.

        The response embeds the current workspace payload inline so a caller
        can render it without separate child-list calls.  Timeline access has
        its own filtered ``get_research_timeline`` tool.
        """
        services = get_services(ctx)
        return services.research_registry.get_session_response(session_id=session_id)

    @mcp.tool()
    @catch_errors()
    async def list_research_sessions(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List every research session with summary counts and the cap."""
        services = get_services(ctx)
        return services.research_registry.list_sessions_response()

    @mcp.tool()
    @catch_errors()
    async def add_research_note(
        session_id: str,
        body: str,
        tags: Optional[List[str]] = None,
        hypothesis_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Append a markdown note to a research session.

        Tags are free-form strings a caller can use to slice notes
        later (e.g. ``"heap"``, ``"crash-triage"``, ``"leak"``).  When
        ``hypothesis_id`` is set the note is bound to that hypothesis;
        the id must already exist in the session or the call fails
        with a typed error.
        """
        services = get_services(ctx)
        return services.research_registry.add_note_response(
            session_id=session_id,
            body=body,
            tags=list(tags) if tags else None,
            hypothesis_id=hypothesis_id,
        )

    @mcp.tool()
    @catch_errors()
    async def link_evidence(
        session_id: str,
        kind: str,
        ref: str,
        description: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Attach a typed pointer to something the platform already stored.

        ``kind`` is one of: ``artifact``, ``crash``, ``heap_snapshot``,
        ``tool_result``, ``note``, ``external``.  ``ref`` is the
        opaque id in that store (artifact id, crash id, heap snapshot
        id, or a URL / path for ``external``).  Evidence never
        duplicates the underlying blob -- it just records the
        provenance so a research session stays lean and the eventual
        export bundle knows what to include.

        Args:
            session_id: Research session to attach the evidence to.
            kind: Evidence kind (see list above).
            ref: Opaque reference into the appropriate store.
            description: Optional human-facing note about this link.
            hypothesis_id: Optional hypothesis this evidence supports
                / rejects; must already exist in the session.
            note_id: Optional note this evidence backs; must already
                exist in the session.
            metadata: Optional arbitrary metadata dict.
        """
        services = get_services(ctx)
        return services.research_registry.link_evidence_response(
            session_id=session_id,
            kind=kind,
            ref=ref,
            description=description,
            hypothesis_id=hypothesis_id,
            note_id=note_id,
            metadata=dict(metadata) if metadata else None,
        )

    @mcp.tool()
    @catch_errors()
    async def add_hypothesis(
        session_id: str,
        statement: str,
        state: str = "proposed",
        rationale: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Register a claim under investigation with an explicit state.

        Valid ``state`` values: ``proposed`` (default), ``supported``,
        ``confirmed``, ``rejected``, ``unknown``.  Notes and evidence
        can be attached to a hypothesis afterwards via their
        ``hypothesis_id`` fields.
        """
        services = get_services(ctx)
        return services.research_registry.add_hypothesis_response(
            session_id=session_id,
            statement=statement,
            state=state,
            rationale=rationale,
        )

    @mcp.tool()
    @catch_errors()
    async def update_hypothesis_state(
        session_id: str,
        hypothesis_id: str,
        state: str,
        rationale: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Move a hypothesis to a new state (supported / rejected / ...).

        ``rationale`` is optional but recommended: it captures the
        reasoning behind the state change so the final report
        can distinguish fact / heuristic / hypothesis honestly.
        """
        services = get_services(ctx)
        return services.research_registry.update_hypothesis_state_response(
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            state=state,
            rationale=rationale,
        )

    @mcp.tool()
    @catch_errors()
    async def create_named_snapshot(
        session_id: str,
        name: str,
        slots: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Capture a named checkpoint into a research session.

        Slots are opaque JSON-serializable payloads the caller
        assembles from live-tool outputs.  Recognized slot names:
        ``registers``, ``memory``, ``mappings``, ``heap_snapshot``,
        ``backtrace``, ``notes``, ``metadata``.  Unknown slot names
        are rejected so a typo does not silently vanish into the
        record.

        A typical CTF pattern: capture one snapshot right before the
        vulnerable call and one right after the crash; call
        ``compare_snapshots`` to see exactly which registers /
        mappings / heap chunks moved.

        Args:
            session_id: Research session that owns the snapshot.
            name: Human-readable label (required).
            slots: Optional mapping of slot -> payload.
            description: Optional longer description.
        """
        services = get_services(ctx)
        return services.research_registry.create_snapshot_response(
            session_id=session_id,
            name=name,
            slots=dict(slots) if slots else None,
            description=description,
        )

    @mcp.tool()
    @catch_errors()
    async def list_snapshots(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List every named snapshot in a research session (summaries only)."""
        services = get_services(ctx)
        return services.research_registry.list_snapshots_response(session_id=session_id)

    @mcp.tool()
    @catch_errors()
    async def append_timeline_event(
        session_id: str,
        kind: str,
        summary: str,
        ref: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Append event to research timeline (cap 2000, oldest dropped openly).

        Args:
            session_id: Research session id.
            kind: tool_call/debug_stop/crash/note/snapshot/hypothesis_change/
                reproduction/artifact/other.
            summary: One-line summary.
            ref: Opaque ref (crash/artifact/operation id).
            hypothesis_id: Existing hypothesis id, if relevant.
            note_id: Existing note id, if relevant.
            snapshot_id: Existing snapshot id, if relevant.
            metadata: Extra dict.
        """
        services = get_services(ctx)
        return services.research_registry.append_timeline_event_response(
            session_id=session_id,
            kind=kind,
            summary=summary,
            ref=ref,
            hypothesis_id=hypothesis_id,
            note_id=note_id,
            snapshot_id=snapshot_id,
            metadata=dict(metadata) if metadata else None,
        )

    @mcp.tool()
    @catch_errors()
    async def get_research_timeline(
        session_id: str,
        kinds: Optional[List[str]] = None,
        limit: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return a research session's timeline, optionally filtered.

        Args:
            session_id: Research session whose timeline to fetch.
            kinds: Optional list of event kinds to keep; any other
                kind is filtered out.  An unknown kind here raises a
                typed error rather than silently returning nothing.
            limit: Optional cap on the number of events returned.
                When set, the *most recent* ``limit`` events are
                returned (the interesting tail).
        """
        services = get_services(ctx)
        return services.research_registry.get_timeline_response(
            session_id=session_id,
            kinds=list(kinds) if kinds else None,
            limit=limit,
        )

    @mcp.tool()
    @catch_errors()
    async def compare_snapshots(
        session_id: str,
        before: str,
        after: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Structural diff of two named snapshots in one research session.

        The diff walks the union of slot names and reports one of
        ``absent``, ``unchanged``, ``added``, ``removed`` or
        ``changed`` per slot.  For dict slots (e.g. ``registers``)
        the ``changed`` payload lists added / removed / changed keys.
        For list slots it reports lengths plus symmetric-difference
        membership.  Diffing a snapshot against itself is rejected.
        """
        services = get_services(ctx)
        return services.research_registry.compare_snapshots_response(
            session_id=session_id, before=before, after=after
        )

    @mcp.tool()
    @catch_errors()
    async def export_research_session(
        session_id: str,
        destination: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Serialize a research session (+ referenced artifacts) into a zip.

        The bundle contains ``manifest.json`` (full session payload
        including notes, hypotheses, evidence, snapshots and timeline),
        ``integrity.json`` (SHA-256 per artifact) and one
        content-addressed file per referenced artifact under
        ``artifacts/``.  Crash / heap / tool_result / external / note
        references stay as opaque strings in the manifest -- they
        point at stores the recipient may or may not have; the
        bundle records the reference so a later report can still
        cite them without duplicating their bytes.

        Args:
            session_id: Research session to export.
            destination: Workspace-relative path (under ``/workspace``) for
                the ``.zip`` bundle; relative paths resolve under the
                workspace and host absolute paths are remapped like every
                other file-taking tool.
        """
        services = get_services(ctx)
        destination = _resolve_contained_workspace_path(
            destination,
            workspace_root=services.runtime_paths.workspace_root,
            require_exists=False,
            kind="destination",
        )
        return services.research_bundle.export_response(
            session_id=session_id, destination=destination
        )

    @mcp.tool()
    @catch_errors()
    async def import_research_session(
        source_path: str,
        artifact_destination_dir: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Verify then rehydrate a research bundle into a fresh session.

        The imported session gets a NEW ``session_id`` so an import
        never clobbers an existing session; the origin id is recorded
        in ``metadata.imported_from``.  Every note / hypothesis /
        evidence / snapshot / timeline id is remapped so
        cross-references stay internally consistent.  When
        ``artifact_destination_dir`` is set, referenced artifacts are
        extracted there (named by their SHA-256) so the caller can
        re-ingest them into the local artifact store.

        Args:
            source_path: Workspace-relative path (under ``/workspace``) to
                the bundle ``.zip``.
            artifact_destination_dir: Optional workspace-relative directory
                to extract artifact payloads into.  Omit to skip extraction.
        """
        services = get_services(ctx)
        source_path = _resolve_contained_workspace_path(
            source_path,
            workspace_root=services.runtime_paths.workspace_root,
            require_exists=True,
            kind="source_path",
        )
        if artifact_destination_dir is not None:
            artifact_destination_dir = _resolve_contained_workspace_path(
                artifact_destination_dir,
                workspace_root=services.runtime_paths.workspace_root,
                require_exists=False,
                kind="artifact_destination_dir",
            )
        return services.research_bundle.import_response(
            source_path=source_path,
            artifact_destination_dir=artifact_destination_dir,
        )

    @mcp.tool()
    @catch_errors()
    async def generate_final_report(
        session_id: str,
        formats: Optional[List[str]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Render a research session as a typed final report.

        The report keeps the fact / heuristic / hypothesis
        distinction explicit: crash / heap / tool_result / artifact
        evidence entries and captured snapshots land in **facts**;
        notes tagged ``heuristic`` and evidence whose metadata carries
        ``inference=True`` land in **heuristics**; every
        ResearchHypothesis lands under **hypotheses** with its state
        and (optional) rationale.  Known unknowns explicitly enumerate
        gaps in the record -- proposed hypotheses without rationale,
        state=unknown hypotheses, missing crash evidence, exploit talk
        without a ROP hypothesis -- so a reader never mistakes absence
        for irrelevance.

        Available formats (any subset): ``json`` (machine-readable),
        ``markdown`` (human narrative), ``html`` (self-contained page
        wrapping the markdown in a properly-escaped <pre> block; we
        deliberately do NOT ship a markdown-to-HTML parser).

        Args:
            session_id: Research session to report on.
            formats: Optional list of ``json`` / ``markdown`` / ``html``.
                Defaults to ``["markdown", "json"]``.
        """
        services = get_services(ctx)
        return services.research_report.generate_report_response(
            session_id=session_id,
            formats=list(formats) if formats else None,
        )

    @mcp.tool()
    @catch_errors()
    async def generate_bounty_report(
        session_id: str,
        title: str,
        severity: str,
        affected_target: str,
        remediation: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Render a triager-ready bounty finding from a research session.

        Root cause cites only supported hypotheses; PoC steps come from
        the reproduction guide plus linked crash evidence; gaps stay
        explicit. Severity is critical/high/medium/low/info.

        Args:
            session_id: Research session to report on.
            title: Finding title.
            severity: critical, high, medium, low, or info.
            affected_target: Binary/endpoint/version hit by the finding.
            remediation: Suggested fix; omitted => marked as not proposed.
        """
        services = get_services(ctx)
        return services.research_report.generate_bounty_report_response(
            session_id=session_id,
            title=title,
            severity=severity,
            affected_target=affected_target,
            remediation=remediation,
        )

    @mcp.tool()
    @catch_errors()
    async def verify_research_bundle(
        source_path: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Verify a research bundle without importing it.

        Checks: the zip opens cleanly and carries manifest.json +
        integrity.json; every artifact listed in the integrity file
        exists and its SHA-256 matches; no unexpected top-level files
        are present.  Returns ``verdict = "valid"`` or ``"invalid"``
        with a per-problem list so a caller can decide whether to
        import.

        Args:
            source_path: Workspace-relative path (under ``/workspace``) to
                the bundle ``.zip``.
        """
        services = get_services(ctx)
        source_path = _resolve_contained_workspace_path(
            source_path,
            workspace_root=services.runtime_paths.workspace_root,
            require_exists=True,
            kind="source_path",
        )
        return services.research_bundle.verify_response(source_path=source_path)


__all__ = ["register"]
