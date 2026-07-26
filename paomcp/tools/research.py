"""MCP tools for typed research workspace (Stage 9)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services


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

        A research session is the durable container for notes,
        hypotheses, evidence links and (in later S9 stages) named
        snapshots + a timeline.  It is independent of any debugger
        session; ``owning_debug_session`` is a soft label so a caller
        can pair a workspace with the debug session that produced its
        evidence, but the two lifecycles do not have to match.

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
        """Fetch a research session with every note, hypothesis and evidence.

        The response embeds children inline (no separate list calls
        needed) so a caller can render a full workspace view from one
        response.
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
        ``rejected``, ``unknown``.  Notes and evidence can be attached
        to a hypothesis afterwards via their ``hypothesis_id`` fields.
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
        reasoning behind the state change so the final S9.5 report
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


__all__ = ["register"]
