"""Typed research-report generator (Stage 9.5).

Given a :class:`ResearchSession`, this module renders a final report in
three shapes:

* ``json``      -- the machine-readable structure (fact / heuristic /
                    hypothesis are kept as distinct sections so the
                    reader never confuses them).
* ``markdown``  -- the human-readable narrative -- session metadata,
                    hypotheses (with state + rationale), crashes,
                    heap findings, ROP hypotheses, snapshots,
                    evidence links, timeline highlights, reproduction
                    guide, known unknowns.
* ``html``      -- a self-contained HTML page wrapping the markdown
                    text in a ``<pre>`` block.  We do NOT convert
                    markdown into rendered HTML here: the platform has
                    no markdown library available and shipping half a
                    parser to please a nice-to-have would poison the
                    output.  A caller who wants rendered HTML can pipe
                    the markdown through their own converter.

The generator is a pure function of the session's contents -- no live
tools, no debugger, no artifact reads.  Evidence entries already point
at everything the reader may want to open later; the report cites those
pointers rather than duplicating the underlying blobs.

The Stage-9 fact / heuristic / hypothesis distinction is preserved:

* **facts**       -- direct observations recorded as evidence
                     (kind=crash/artifact/heap_snapshot/tool_result)
                     plus every snapshot slot the caller captured.
* **heuristics**  -- notes tagged with ``heuristic`` or evidence
                     entries whose metadata carries
                     ``inference = True``.
* **hypotheses**  -- explicit :class:`ResearchHypothesis` entries
                     with their state (proposed / supported /
                     rejected / unknown) and rationale.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.research import (
    HYPOTHESIS_STATES,
    ResearchError,
    ResearchRegistry,
    ResearchSession,
)

PARSER_VERSION = "1"
RESEARCH_REPORT_SCHEMA = "pao.research-report.v1"

REPORT_FORMATS: Tuple[str, ...] = ("json", "markdown", "html")


class ReportError(RuntimeError):
    """Raised when a report cannot be generated."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchReport:
    """Generate typed final reports for research sessions.

    Stateless beyond its registry dependency, so a caller can share
    one instance across every session in the process.
    """

    def __init__(self, registry: ResearchRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Pure classifiers (unit-testable without the registry)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_heuristic_note(note: Any) -> bool:
        return any(
            str(tag).strip().lower() == "heuristic"
            for tag in getattr(note, "tags", []) or []
        )

    @staticmethod
    def _is_heuristic_evidence(evidence: Any) -> bool:
        meta = getattr(evidence, "metadata", {}) or {}
        return bool(meta.get("inference")) or bool(meta.get("heuristic"))

    @staticmethod
    def _crash_evidence(session: ResearchSession) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in session.evidence.values() if e.kind == "crash"]

    @staticmethod
    def _heap_evidence(session: ResearchSession) -> List[Dict[str, Any]]:
        return [
            e.to_dict() for e in session.evidence.values() if e.kind == "heap_snapshot"
        ]

    @staticmethod
    def _tool_result_evidence(session: ResearchSession) -> List[Dict[str, Any]]:
        return [
            e.to_dict() for e in session.evidence.values() if e.kind == "tool_result"
        ]

    @staticmethod
    def _artifact_evidence(session: ResearchSession) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in session.evidence.values() if e.kind == "artifact"]

    @staticmethod
    def _rop_hypotheses(session: ResearchSession) -> List[Dict[str, Any]]:
        """Return every hypothesis whose statement / tags mention rop.

        The registry does not model an explicit "rop" hypothesis kind
        -- callers set the shape by wording their hypothesis.  We
        surface any hypothesis mentioning ``rop`` / ``ret2libc`` /
        ``ret2syscall`` / ``srop`` as a ROP hypothesis in the report.
        """
        keywords = ("rop", "ret2libc", "ret2syscall", "srop", "chain")
        picks: List[Dict[str, Any]] = []
        for hyp in session.hypotheses.values():
            statement = (hyp.statement or "").lower()
            if any(kw in statement for kw in keywords):
                picks.append(hyp.to_dict())
        return picks

    @classmethod
    def _classify_by_state(
        cls, session: ResearchSession
    ) -> Dict[str, List[Dict[str, Any]]]:
        by_state: Dict[str, List[Dict[str, Any]]] = {
            state: [] for state in HYPOTHESIS_STATES
        }
        for hyp in session.hypotheses.values():
            by_state.setdefault(hyp.state, []).append(hyp.to_dict())
        return by_state

    @classmethod
    def _known_unknowns(cls, session: ResearchSession) -> List[str]:
        """Enumerate the honest gaps in the record.

        The report explicitly names the shapes of missing information
        so a reader does not mistake absence for irrelevance.  Every
        item here corresponds to a real gap in ``session`` state.
        """
        unknowns: List[str] = []
        for hyp in session.hypotheses.values():
            if hyp.state == "unknown":
                unknowns.append(
                    f"hypothesis '{hyp.statement[:80]}' is state=unknown"
                    + (f" ({hyp.rationale})" if hyp.rationale else "")
                )
            if hyp.state == "proposed" and not hyp.rationale:
                unknowns.append(
                    f"proposed hypothesis '{hyp.statement[:80]}' "
                    "has no rationale recorded"
                )
        # An investigation with no crash evidence is one of the shapes
        # a reader wants to know is a gap, not a lack of interest.
        if not cls._crash_evidence(session):
            unknowns.append("no crash evidence linked to the session")
        if not cls._rop_hypotheses(session) and any(
            "exploit" in (h.statement or "").lower()
            for h in session.hypotheses.values()
        ):
            unknowns.append(
                "exploit-shaped hypothesis present but no ROP hypothesis recorded"
            )
        return unknowns

    # ------------------------------------------------------------------
    # Reproduction guide
    # ------------------------------------------------------------------
    @classmethod
    def _reproduction_guide(cls, session: ResearchSession) -> Dict[str, Any]:
        """Best-effort reproduction guide from what the session actually recorded.

        We do not fabricate a target spec.  ``session.metadata`` and
        the ``metadata`` fields on tool_result / artifact evidence
        entries are the only trusted sources; the guide reports what
        exists and lists what does not so the reader can tell.
        """
        metadata = dict(session.metadata or {})
        target_bits: Dict[str, Any] = {
            "binary": metadata.get("binary"),
            "argv": metadata.get("argv"),
            "env": metadata.get("env"),
            "target_spec": metadata.get("target_spec"),
            "working_dir": metadata.get("working_dir"),
        }
        supplied = {k: v for k, v in target_bits.items() if v is not None}
        missing = sorted(k for k, v in target_bits.items() if v is None)
        # Timeline-derived breadcrumbs
        first_snapshot = next(
            (e for e in session.timeline if e.kind == "snapshot"),
            None,
        )
        first_crash = next((e for e in session.timeline if e.kind == "crash"), None)
        first_repro = next(
            (e for e in session.timeline if e.kind == "reproduction"),
            None,
        )
        return {
            "supplied": supplied,
            "missing": missing,
            "timeline_hints": {
                "first_snapshot_at": (
                    first_snapshot.created_at if first_snapshot else None
                ),
                "first_crash_at": first_crash.created_at if first_crash else None,
                "first_reproduction_at": (
                    first_repro.created_at if first_repro else None
                ),
            },
        }

    # ------------------------------------------------------------------
    # Structured JSON payload (fact / heuristic / hypothesis distinct)
    # ------------------------------------------------------------------
    def _build_json_payload(self, session: ResearchSession) -> Dict[str, Any]:
        crashes = self._crash_evidence(session)
        heap = self._heap_evidence(session)
        tool_results = self._tool_result_evidence(session)
        artifacts = self._artifact_evidence(session)
        rop = self._rop_hypotheses(session)
        by_state = self._classify_by_state(session)
        heuristic_notes = [
            n.to_dict() for n in session.notes.values() if self._is_heuristic_note(n)
        ]
        heuristic_evidence = [
            e.to_dict()
            for e in session.evidence.values()
            if self._is_heuristic_evidence(e)
        ]
        facts = {
            "crashes": crashes,
            "heap_snapshots": heap,
            "tool_results": tool_results,
            "artifacts": artifacts,
            "captured_snapshots": [
                snap.to_dict(include_slots=False) for snap in session.snapshots.values()
            ],
        }
        heuristics = {
            "notes": heuristic_notes,
            "evidence": heuristic_evidence,
        }
        hypotheses = {
            "rop": rop,
            "by_state": by_state,
            "all": [h.to_dict() for h in session.hypotheses.values()],
        }
        timeline = [event.to_dict() for event in session.timeline]
        return {
            "schema": RESEARCH_REPORT_SCHEMA,
            "generated_at": _now_iso(),
            "session": {
                "session_id": session.session_id,
                "name": session.name,
                "description": session.description,
                "owning_debug_session": session.owning_debug_session,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": dict(session.metadata),
            },
            "counts": {
                "notes": len(session.notes),
                "evidence": len(session.evidence),
                "hypotheses": len(session.hypotheses),
                "snapshots": len(session.snapshots),
                "timeline_events": len(session.timeline),
            },
            "facts": facts,
            "heuristics": heuristics,
            "hypotheses": hypotheses,
            "reproduction": self._reproduction_guide(session),
            "timeline": timeline,
            "known_unknowns": self._known_unknowns(session),
        }

    # ------------------------------------------------------------------
    # Markdown renderer (no third-party dependency)
    # ------------------------------------------------------------------
    @staticmethod
    def _md_escape(text: Any) -> str:
        return str(text if text is not None else "").replace("|", "\\|")

    @classmethod
    def _md_section_hypotheses(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Hypotheses", ""]
        by_state = payload["hypotheses"]["by_state"]
        for state in HYPOTHESIS_STATES:
            entries = by_state.get(state, [])
            if not entries:
                continue
            lines.append(f"### state = {state} ({len(entries)})")
            lines.append("")
            for hyp in entries:
                statement = cls._md_escape(hyp.get("statement"))
                lines.append(f"- **{statement}**")
                if hyp.get("rationale"):
                    lines.append(f"  - rationale: {cls._md_escape(hyp['rationale'])}")
                lines.append(
                    f"  - id: `{hyp.get('hypothesis_id')}`, "
                    f"updated {hyp.get('updated_at')}"
                )
            lines.append("")
        if not payload["hypotheses"]["all"]:
            lines.append("_No hypotheses recorded._")
            lines.append("")
        return lines

    @classmethod
    def _md_section_facts(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Facts (direct observations)", ""]
        facts = payload["facts"]
        blocks = (
            ("Crashes", facts["crashes"]),
            ("Heap snapshots", facts["heap_snapshots"]),
            ("Tool results", facts["tool_results"]),
            ("Artifacts", facts["artifacts"]),
        )
        for label, entries in blocks:
            lines.append(f"### {label} ({len(entries)})")
            lines.append("")
            if not entries:
                lines.append("_none linked._")
                lines.append("")
                continue
            for entry in entries:
                lines.append(
                    f"- `{entry.get('ref')}` -- "
                    f"{cls._md_escape(entry.get('description') or entry.get('kind'))}"
                )
            lines.append("")
        snapshots = facts["captured_snapshots"]
        lines.append(f"### Captured snapshots ({len(snapshots)})")
        lines.append("")
        if not snapshots:
            lines.append("_none captured._")
            lines.append("")
        else:
            for snap in snapshots:
                lines.append(
                    f"- **{cls._md_escape(snap.get('name'))}** "
                    f"(`{snap.get('snapshot_id')}`) -- slots: "
                    f"{', '.join(snap.get('slot_names') or []) or '_empty_'}"
                )
            lines.append("")
        return lines

    @classmethod
    def _md_section_heuristics(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Heuristics (inferred, not proven)", ""]
        h = payload["heuristics"]
        if not h["notes"] and not h["evidence"]:
            lines.append("_No heuristics recorded._")
            lines.append("")
            return lines
        if h["notes"]:
            lines.append(f"### Heuristic notes ({len(h['notes'])})")
            lines.append("")
            for note in h["notes"]:
                lines.append(f"- {cls._md_escape(note.get('body'))}")
            lines.append("")
        if h["evidence"]:
            lines.append(f"### Heuristic evidence ({len(h['evidence'])})")
            lines.append("")
            for ev in h["evidence"]:
                lines.append(
                    f"- `{ev.get('ref')}` ({ev.get('kind')}) -- "
                    f"{cls._md_escape(ev.get('description') or '')}"
                )
            lines.append("")
        return lines

    @classmethod
    def _md_section_rop(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## ROP hypotheses", ""]
        rop = payload["hypotheses"]["rop"]
        if not rop:
            lines.append("_No ROP hypotheses recorded._")
            lines.append("")
            return lines
        for hyp in rop:
            lines.append(
                f"- **{cls._md_escape(hyp.get('statement'))}** "
                f"(state = {hyp.get('state')})"
            )
            if hyp.get("rationale"):
                lines.append(f"  - rationale: {cls._md_escape(hyp['rationale'])}")
        lines.append("")
        return lines

    @classmethod
    def _md_section_timeline(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Timeline (highlights)", ""]
        events = payload["timeline"]
        if not events:
            lines.append("_Timeline empty._")
            lines.append("")
            return lines
        # Show up to 20 most recent so a huge session does not blow
        # up the report; the full timeline lives in the JSON payload.
        recent = events[-20:]
        lines.append(f"_showing {len(recent)} of {len(events)} events_")
        lines.append("")
        for event in recent:
            lines.append(
                f"- `{event.get('created_at')}` **{event.get('kind')}**: "
                f"{cls._md_escape(event.get('summary'))}"
            )
        lines.append("")
        return lines

    @classmethod
    def _md_section_reproduction(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Reproduction guide", ""]
        repro = payload["reproduction"]
        supplied = repro["supplied"]
        missing = repro["missing"]
        if supplied:
            lines.append("### Supplied")
            lines.append("")
            for key, value in supplied.items():
                lines.append(f"- **{key}**: `{cls._md_escape(value)}`")
            lines.append("")
        if missing:
            lines.append("### Missing (fill in before rerunning)")
            lines.append("")
            for key in missing:
                lines.append(f"- {key}")
            lines.append("")
        hints = repro["timeline_hints"]
        lines.append("### Timeline hints")
        lines.append("")
        for label, ts in hints.items():
            lines.append(f"- {label}: `{ts or '<none>'}`")
        lines.append("")
        return lines

    @classmethod
    def _md_section_unknowns(cls, payload: Dict[str, Any]) -> List[str]:
        lines: List[str] = ["## Known unknowns", ""]
        unknowns = payload["known_unknowns"]
        if not unknowns:
            lines.append("_No named gaps in the record._")
            lines.append("")
            return lines
        for item in unknowns:
            lines.append(f"- {cls._md_escape(item)}")
        lines.append("")
        return lines

    @classmethod
    def _md_header(cls, payload: Dict[str, Any]) -> List[str]:
        session = payload["session"]
        counts = payload["counts"]
        lines = [
            f"# Research report: {cls._md_escape(session['name'])}",
            "",
            f"- session_id: `{session['session_id']}`",
            f"- generated_at: `{payload['generated_at']}`",
            f"- created_at:  `{session['created_at']}`",
            f"- updated_at:  `{session['updated_at']}`",
        ]
        if session.get("description"):
            lines.append(f"- description: {cls._md_escape(session['description'])}")
        if session.get("owning_debug_session"):
            lines.append(f"- owning_debug_session: `{session['owning_debug_session']}`")
        lines.append("")
        lines.append("**Counts**: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        lines.append("")
        lines.append("---")
        lines.append("")
        return lines

    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        sections = (
            self._md_header(payload)
            + self._md_section_hypotheses(payload)
            + self._md_section_facts(payload)
            + self._md_section_heuristics(payload)
            + self._md_section_rop(payload)
            + self._md_section_reproduction(payload)
            + self._md_section_timeline(payload)
            + self._md_section_unknowns(payload)
        )
        return "\n".join(sections).rstrip() + "\n"

    def _render_html(self, payload: Dict[str, Any], markdown_text: str) -> str:
        # Deliberately conservative: we do NOT ship half a markdown
        # parser.  This wraps the raw markdown in a properly-escaped
        # <pre> block so the output is safe in a browser and callable
        # from any downstream converter.  A caller who wants rendered
        # HTML runs the markdown_text through their own parser.
        session = payload["session"]
        title = html.escape(f"Research report: {session['name']}")
        body = html.escape(markdown_text)
        return (
            "<!DOCTYPE html>\n"
            '<html><head><meta charset="utf-8">'
            f"<title>{title}</title>"
            "<style>"
            "body{font-family:system-ui,sans-serif;max-width:960px;"
            "margin:2rem auto;padding:0 1rem}"
            "pre{white-space:pre-wrap;word-wrap:break-word;"
            "background:#f6f8fa;padding:1rem;border-radius:6px}"
            "</style></head>"
            f"<body><h1>{title}</h1><pre>{body}</pre></body></html>\n"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_report(
        self,
        *,
        session_id: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return the report payload keyed by requested format."""
        requested = list(formats or ["markdown", "json"])
        unknown = sorted(set(requested) - set(REPORT_FORMATS))
        if unknown:
            raise ReportError(
                f"unknown report formats {unknown}; expected any of "
                f"{list(REPORT_FORMATS)}"
            )
        session = self._registry.get_session(session_id)
        with session._lock:
            payload = self._build_json_payload(session)
        rendered: Dict[str, Any] = {"structure": payload}
        markdown_text: Optional[str] = None
        if "markdown" in requested or "html" in requested:
            markdown_text = self._render_markdown(payload)
        if "markdown" in requested:
            rendered["markdown"] = markdown_text
        if "json" in requested:
            rendered["json"] = json.dumps(payload, indent=2, sort_keys=True)
        if "html" in requested:
            assert markdown_text is not None  # set above
            rendered["html"] = self._render_html(payload, markdown_text)
        return {
            "schema": RESEARCH_REPORT_SCHEMA,
            "session_id": session_id,
            "formats": sorted(requested),
            "rendered": rendered,
            "counts": payload["counts"],
        }

    def generate_report_response(
        self,
        *,
        session_id: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        parsed = self.generate_report(session_id=session_id, formats=formats)
        return build_backend_response(
            "generate_final_report",
            parsed=parsed,
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = [
    "ReportError",
    "ResearchReport",
    "RESEARCH_REPORT_SCHEMA",
    "REPORT_FORMATS",
]
