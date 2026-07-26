"""Typed research workspace (Stage 9).

Stage 9 turns a scattered pile of tool results, crash snapshots, heap
diffs and manual notes into a single research session a caller can
serialize, share, and reopen later without losing provenance.

S9.1 ships the foundation: a session dataclass with typed notes, a
hypothesis state machine, and an evidence graph that links notes to
existing artifact / crash / heap / snapshot / tool-result ids.  The
registry is session-scoped and bounded so a runaway workflow cannot
allocate unbounded state, exactly matching the pattern S6.5
(HeapHistory) and S8.1 (TargetRegistry) already use.

Later S9 slices attach snapshots (S9.2), a timeline (S9.3),
export/import (S9.4) and the final report (S9.5) to the same
:class:`ResearchSession` object rather than growing sibling
dataclasses -- so ``export_research_session`` in S9.4 can serialize
everything under one roof.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import build_backend_response

PARSER_VERSION = "1"
RESEARCH_SESSION_SCHEMA = "pao.research-session.v1"
RESEARCH_SESSION_LIST_SCHEMA = "pao.research-session-list.v1"
RESEARCH_NOTE_SCHEMA = "pao.research-note.v1"
RESEARCH_EVIDENCE_SCHEMA = "pao.research-evidence.v1"
RESEARCH_HYPOTHESIS_SCHEMA = "pao.research-hypothesis.v1"

DEFAULT_MAX_RESEARCH_SESSIONS = 32
DEFAULT_MAX_NOTES_PER_SESSION = 500
DEFAULT_MAX_EVIDENCE_PER_SESSION = 1000
DEFAULT_MAX_HYPOTHESES_PER_SESSION = 100

HYPOTHESIS_STATES: Tuple[str, ...] = (
    "proposed",
    "supported",
    "rejected",
    "unknown",
)

EVIDENCE_KINDS: Tuple[str, ...] = (
    "artifact",
    "crash",
    "heap_snapshot",
    "tool_result",
    "note",
    "external",
)


class ResearchError(RuntimeError):
    """Raised for spec / lifecycle / limit errors in the research layer."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResearchNote:
    """One markdown note attached to a research session."""

    note_id: str
    body: str
    tags: List[str] = field(default_factory=list)
    hypothesis_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "body": self.body,
            "tags": list(self.tags),
            "hypothesis_id": self.hypothesis_id,
            "created_at": self.created_at,
        }


@dataclass
class ResearchEvidence:
    """A typed pointer to something already stored in the platform."""

    evidence_id: str
    kind: str  # one of EVIDENCE_KINDS
    ref: str
    description: Optional[str] = None
    hypothesis_id: Optional[str] = None
    note_id: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "ref": self.ref,
            "description": self.description,
            "hypothesis_id": self.hypothesis_id,
            "note_id": self.note_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class ResearchHypothesis:
    """One claim under investigation, with an explicit state."""

    hypothesis_id: str
    statement: str
    state: str = "proposed"
    rationale: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "state": self.state,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ResearchSession:
    """One research workspace: notes + hypotheses + evidence graph."""

    session_id: str
    name: str
    owning_debug_session: Optional[str] = None
    description: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    notes: Dict[str, ResearchNote] = field(default_factory=dict)
    evidence: Dict[str, ResearchEvidence] = field(default_factory=dict)
    hypotheses: Dict[str, ResearchHypothesis] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )

    def _touch(self) -> None:
        self.updated_at = _now_iso()

    def to_dict(self, *, include_children: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": self.session_id,
            "name": self.name,
            "owning_debug_session": self.owning_debug_session,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "note_count": len(self.notes),
            "evidence_count": len(self.evidence),
            "hypothesis_count": len(self.hypotheses),
            "metadata": dict(self.metadata),
        }
        if include_children:
            payload["notes"] = [note.to_dict() for note in self.notes.values()]
            payload["evidence"] = [item.to_dict() for item in self.evidence.values()]
            payload["hypotheses"] = [hyp.to_dict() for hyp in self.hypotheses.values()]
        return payload


class ResearchRegistry:
    """Thread-safe, bounded registry of research sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_RESEARCH_SESSIONS,
        max_notes_per_session: int = DEFAULT_MAX_NOTES_PER_SESSION,
        max_evidence_per_session: int = DEFAULT_MAX_EVIDENCE_PER_SESSION,
        max_hypotheses_per_session: int = DEFAULT_MAX_HYPOTHESES_PER_SESSION,
    ) -> None:
        for name, value in (
            ("max_sessions", max_sessions),
            ("max_notes_per_session", max_notes_per_session),
            ("max_evidence_per_session", max_evidence_per_session),
            ("max_hypotheses_per_session", max_hypotheses_per_session),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._max_sessions = max_sessions
        self._max_notes = max_notes_per_session
        self._max_evidence = max_evidence_per_session
        self._max_hypotheses = max_hypotheses_per_session
        self._sessions: Dict[str, ResearchSession] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------
    def create_session(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        owning_debug_session: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ResearchSession:
        if not isinstance(name, str) or not name.strip():
            raise ResearchError("name must be a non-empty string")
        with self._lock:
            if len(self._sessions) >= self._max_sessions:
                raise ResearchError(
                    f"research session cap reached ({self._max_sessions})"
                )
            session = ResearchSession(
                session_id=uuid.uuid4().hex[:12],
                name=name.strip(),
                description=description,
                owning_debug_session=owning_debug_session,
                metadata=dict(metadata or {}),
            )
            self._sessions[session.session_id] = session
            return session

    def get_session(self, session_id: str) -> ResearchSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ResearchError(f"research session '{session_id}' not found")
            return session

    def list_sessions(self) -> List[ResearchSession]:
        with self._lock:
            return list(self._sessions.values())

    def remove_session(self, session_id: str) -> Optional[ResearchSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------
    def add_note(
        self,
        *,
        session_id: str,
        body: str,
        tags: Optional[Sequence[str]] = None,
        hypothesis_id: Optional[str] = None,
    ) -> ResearchNote:
        if not isinstance(body, str) or not body.strip():
            raise ResearchError("note body must be a non-empty string")
        session = self.get_session(session_id)
        with session._lock:
            if len(session.notes) >= self._max_notes:
                raise ResearchError(
                    f"session '{session_id}' already has "
                    f"{self._max_notes} notes (cap)"
                )
            if hypothesis_id is not None and hypothesis_id not in session.hypotheses:
                raise ResearchError(
                    f"hypothesis '{hypothesis_id}' not found in session"
                )
            note = ResearchNote(
                note_id=uuid.uuid4().hex[:12],
                body=body.strip(),
                tags=[str(t).strip() for t in (tags or []) if str(t).strip()],
                hypothesis_id=hypothesis_id,
            )
            session.notes[note.note_id] = note
            session._touch()
            return note

    # ------------------------------------------------------------------
    # Evidence
    # ------------------------------------------------------------------
    def link_evidence(
        self,
        *,
        session_id: str,
        kind: str,
        ref: str,
        description: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ResearchEvidence:
        if kind not in EVIDENCE_KINDS:
            raise ResearchError(
                f"evidence kind {kind!r} must be one of {list(EVIDENCE_KINDS)}"
            )
        if not isinstance(ref, str) or not ref.strip():
            raise ResearchError("evidence ref must be a non-empty string")
        session = self.get_session(session_id)
        with session._lock:
            if len(session.evidence) >= self._max_evidence:
                raise ResearchError(
                    f"session '{session_id}' already has "
                    f"{self._max_evidence} evidence entries (cap)"
                )
            if hypothesis_id is not None and hypothesis_id not in session.hypotheses:
                raise ResearchError(
                    f"hypothesis '{hypothesis_id}' not found in session"
                )
            if note_id is not None and note_id not in session.notes:
                raise ResearchError(f"note '{note_id}' not found in session")
            evidence = ResearchEvidence(
                evidence_id=uuid.uuid4().hex[:12],
                kind=kind,
                ref=ref.strip(),
                description=description,
                hypothesis_id=hypothesis_id,
                note_id=note_id,
                metadata=dict(metadata or {}),
            )
            session.evidence[evidence.evidence_id] = evidence
            session._touch()
            return evidence

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------
    def add_hypothesis(
        self,
        *,
        session_id: str,
        statement: str,
        state: str = "proposed",
        rationale: Optional[str] = None,
    ) -> ResearchHypothesis:
        if not isinstance(statement, str) or not statement.strip():
            raise ResearchError("hypothesis statement must be a non-empty string")
        if state not in HYPOTHESIS_STATES:
            raise ResearchError(
                f"hypothesis state {state!r} must be one of "
                f"{list(HYPOTHESIS_STATES)}"
            )
        session = self.get_session(session_id)
        with session._lock:
            if len(session.hypotheses) >= self._max_hypotheses:
                raise ResearchError(
                    f"session '{session_id}' already has "
                    f"{self._max_hypotheses} hypotheses (cap)"
                )
            hypothesis = ResearchHypothesis(
                hypothesis_id=uuid.uuid4().hex[:12],
                statement=statement.strip(),
                state=state,
                rationale=rationale,
            )
            session.hypotheses[hypothesis.hypothesis_id] = hypothesis
            session._touch()
            return hypothesis

    def update_hypothesis_state(
        self,
        *,
        session_id: str,
        hypothesis_id: str,
        state: str,
        rationale: Optional[str] = None,
    ) -> ResearchHypothesis:
        if state not in HYPOTHESIS_STATES:
            raise ResearchError(
                f"hypothesis state {state!r} must be one of "
                f"{list(HYPOTHESIS_STATES)}"
            )
        session = self.get_session(session_id)
        with session._lock:
            hypothesis = session.hypotheses.get(hypothesis_id)
            if hypothesis is None:
                raise ResearchError(
                    f"hypothesis '{hypothesis_id}' not found in session"
                )
            hypothesis.state = state
            if rationale is not None:
                hypothesis.rationale = rationale
            hypothesis.updated_at = _now_iso()
            session._touch()
            return hypothesis

    # ------------------------------------------------------------------
    # High-level response wrappers used by the tool layer
    # ------------------------------------------------------------------
    def create_session_response(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        owning_debug_session: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        session = self.create_session(
            name=name,
            description=description,
            owning_debug_session=owning_debug_session,
            metadata=metadata,
        )
        return build_backend_response(
            "create_research_session",
            parsed={
                "schema": RESEARCH_SESSION_SCHEMA,
                "session": session.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def get_session_response(self, *, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        return build_backend_response(
            "get_research_session",
            parsed={
                "schema": RESEARCH_SESSION_SCHEMA,
                "session": session.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def list_sessions_response(self) -> Dict[str, Any]:
        sessions = self.list_sessions()
        return build_backend_response(
            "list_research_sessions",
            parsed={
                "schema": RESEARCH_SESSION_LIST_SCHEMA,
                "sessions": [s.to_dict(include_children=False) for s in sessions],
                "count": len(sessions),
                "max_sessions": self._max_sessions,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def add_note_response(
        self,
        *,
        session_id: str,
        body: str,
        tags: Optional[Sequence[str]] = None,
        hypothesis_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        note = self.add_note(
            session_id=session_id,
            body=body,
            tags=tags,
            hypothesis_id=hypothesis_id,
        )
        return build_backend_response(
            "add_research_note",
            parsed={
                "schema": RESEARCH_NOTE_SCHEMA,
                "session_id": session_id,
                "note": note.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def link_evidence_response(
        self,
        *,
        session_id: str,
        kind: str,
        ref: str,
        description: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence = self.link_evidence(
            session_id=session_id,
            kind=kind,
            ref=ref,
            description=description,
            hypothesis_id=hypothesis_id,
            note_id=note_id,
            metadata=metadata,
        )
        return build_backend_response(
            "link_evidence",
            parsed={
                "schema": RESEARCH_EVIDENCE_SCHEMA,
                "session_id": session_id,
                "evidence": evidence.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def add_hypothesis_response(
        self,
        *,
        session_id: str,
        statement: str,
        state: str = "proposed",
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        hypothesis = self.add_hypothesis(
            session_id=session_id,
            statement=statement,
            state=state,
            rationale=rationale,
        )
        return build_backend_response(
            "add_hypothesis",
            parsed={
                "schema": RESEARCH_HYPOTHESIS_SCHEMA,
                "session_id": session_id,
                "hypothesis": hypothesis.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def update_hypothesis_state_response(
        self,
        *,
        session_id: str,
        hypothesis_id: str,
        state: str,
        rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        hypothesis = self.update_hypothesis_state(
            session_id=session_id,
            hypothesis_id=hypothesis_id,
            state=state,
            rationale=rationale,
        )
        return build_backend_response(
            "update_hypothesis_state",
            parsed={
                "schema": RESEARCH_HYPOTHESIS_SCHEMA,
                "session_id": session_id,
                "hypothesis": hypothesis.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = [
    "ResearchError",
    "ResearchNote",
    "ResearchEvidence",
    "ResearchHypothesis",
    "ResearchSession",
    "ResearchRegistry",
    "HYPOTHESIS_STATES",
    "EVIDENCE_KINDS",
    "DEFAULT_MAX_RESEARCH_SESSIONS",
    "DEFAULT_MAX_NOTES_PER_SESSION",
    "DEFAULT_MAX_EVIDENCE_PER_SESSION",
    "DEFAULT_MAX_HYPOTHESES_PER_SESSION",
    "RESEARCH_SESSION_SCHEMA",
    "RESEARCH_SESSION_LIST_SCHEMA",
    "RESEARCH_NOTE_SCHEMA",
    "RESEARCH_EVIDENCE_SCHEMA",
    "RESEARCH_HYPOTHESIS_SCHEMA",
]
