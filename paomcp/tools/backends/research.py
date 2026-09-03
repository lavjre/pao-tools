"""Typed research workspace.

This workspace turns a scattered pile of tool results, crash snapshots, heap
diffs and manual notes into a single research session a caller can
serialize, share, and reopen later without losing provenance.

The workspace foundation ships: a session dataclass with typed notes, a
hypothesis state machine, and an evidence graph that links notes to
existing artifact / crash / heap / snapshot / tool-result ids.  The
registry is session-scoped and bounded so a runaway workflow cannot
allocate unbounded state, exactly matching the pattern heap snapshots
(HeapHistory) and the target registry (TargetRegistry) already use.

Snapshots, a timeline, bundle export/import and the final report attach to the same
:class:`ResearchSession` object rather than growing sibling
dataclasses -- so ``export_research_session`` can serialize
everything under one roof.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from paomcp.tools.backends.foundation import build_backend_response

if TYPE_CHECKING:  # the store is duck-typed; import only for annotations
    from paomcp.tools.backends.research_store import CaseStore

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"
RESEARCH_SESSION_SCHEMA = "pao.research-session.v1"
RESEARCH_SESSION_LIST_SCHEMA = "pao.research-session-list.v1"
RESEARCH_NOTE_SCHEMA = "pao.research-note.v1"
RESEARCH_EVIDENCE_SCHEMA = "pao.research-evidence.v1"
RESEARCH_HYPOTHESIS_SCHEMA = "pao.research-hypothesis.v1"
RESEARCH_SNAPSHOT_SCHEMA = "pao.research-snapshot.v1"
RESEARCH_SNAPSHOT_LIST_SCHEMA = "pao.research-snapshot-list.v1"
RESEARCH_SNAPSHOT_DIFF_SCHEMA = "pao.research-snapshot-diff.v1"
RESEARCH_TIMELINE_EVENT_SCHEMA = "pao.research-timeline-event.v1"
RESEARCH_TIMELINE_SCHEMA = "pao.research-timeline.v1"

DEFAULT_MAX_RESEARCH_SESSIONS = 32
DEFAULT_MAX_NOTES_PER_SESSION = 500
DEFAULT_MAX_EVIDENCE_PER_SESSION = 1000
DEFAULT_MAX_HYPOTHESES_PER_SESSION = 100
DEFAULT_MAX_SNAPSHOTS_PER_SESSION = 64
DEFAULT_MAX_TIMELINE_EVENTS_PER_SESSION = 2000

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

# Timeline event kinds a caller may append.  A caller-supplied kind
# outside this vocabulary is rejected so timeline stays queryable.
TIMELINE_EVENT_KINDS: Tuple[str, ...] = (
    "tool_call",
    "debug_stop",
    "crash",
    "note",
    "snapshot",
    "hypothesis_change",
    "reproduction",
    "artifact",
    "other",
)


# Slot names a research snapshot may carry.  A caller supplies any
# subset; each slot is an opaque JSON-serializable payload the caller
# controls.  ``compare_snapshots`` walks the union of slot names and
# produces a structural diff for each.
SNAPSHOT_SLOTS: Tuple[str, ...] = (
    "registers",  # {name: hex_or_int_value}
    "memory",  # [{address, bytes_hex, length}]
    "mappings",  # [{start, end, permissions, file}]
    "heap_snapshot",  # {snapshot_id, ...}  (typically a heap snapshot id)
    "backtrace",  # [{level, pc, function, ...}]
    "notes",  # freeform markdown
    "metadata",  # freeform dict for target identity / ASLR base / etc.
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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchNote":
        """Rehydrate a note persisted by :class:`CaseStore`."""
        return cls(
            note_id=str(data["note_id"]),
            body=str(data.get("body", "")),
            tags=[str(t) for t in data.get("tags") or []],
            hypothesis_id=data.get("hypothesis_id"),
            created_at=str(data.get("created_at") or _now_iso()),
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchEvidence":
        """Rehydrate evidence persisted by :class:`CaseStore`."""
        metadata = data.get("metadata")
        return cls(
            evidence_id=str(data["evidence_id"]),
            kind=str(data.get("kind", "")),
            ref=str(data.get("ref", "")),
            description=data.get("description"),
            hypothesis_id=data.get("hypothesis_id"),
            note_id=data.get("note_id"),
            created_at=str(data.get("created_at") or _now_iso()),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


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

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchHypothesis":
        """Rehydrate a hypothesis persisted by :class:`CaseStore`."""
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            statement=str(data.get("statement", "")),
            state=str(data.get("state", "proposed")),
            rationale=data.get("rationale"),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


@dataclass
class TimelineEvent:
    """One point on a research session's append-only timeline."""

    event_id: str
    kind: str  # one of TIMELINE_EVENT_KINDS
    summary: str
    created_at: str = field(default_factory=_now_iso)
    ref: Optional[str] = None
    hypothesis_id: Optional[str] = None
    note_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "summary": self.summary,
            "created_at": self.created_at,
            "ref": self.ref,
            "hypothesis_id": self.hypothesis_id,
            "note_id": self.note_id,
            "snapshot_id": self.snapshot_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimelineEvent":
        """Rehydrate a timeline event persisted by :class:`CaseStore`."""
        metadata = data.get("metadata")
        return cls(
            event_id=str(data["event_id"]),
            kind=str(data.get("kind", "")),
            summary=str(data.get("summary", "")),
            created_at=str(data.get("created_at") or _now_iso()),
            ref=data.get("ref"),
            hypothesis_id=data.get("hypothesis_id"),
            note_id=data.get("note_id"),
            snapshot_id=data.get("snapshot_id"),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )


@dataclass
class ResearchSnapshot:
    """A named checkpoint bundling caller-supplied slots at one instant.

    Slots are intentionally opaque -- a caller composes them from the
    outputs of live tools (registers from ``get_registers``, memory
    from ``read_memory``, mappings from ``memory_maps``, a heap id
    from ``capture_heap_snapshot``) so this backend does not have to
    know how to talk to a debugger.  The value of the snapshot is
    the *name*, the *timestamp*, and the ability to diff two of them
    later.
    """

    snapshot_id: str
    name: str
    slots: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self, *, include_slots: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "snapshot_id": self.snapshot_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "slot_names": sorted(self.slots.keys()),
        }
        if include_slots:
            payload["slots"] = dict(self.slots)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchSnapshot":
        """Rehydrate a snapshot persisted by :class:`CaseStore`."""
        slots = data.get("slots")
        return cls(
            snapshot_id=str(data["snapshot_id"]),
            name=str(data.get("name", "")),
            slots=dict(slots) if isinstance(slots, Mapping) else {},
            description=data.get("description"),
            created_at=str(data.get("created_at") or _now_iso()),
        )


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
    snapshots: Dict[str, ResearchSnapshot] = field(default_factory=dict)
    timeline: List[TimelineEvent] = field(default_factory=list)
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
            "snapshot_count": len(self.snapshots),
            "timeline_length": len(self.timeline),
            "metadata": dict(self.metadata),
        }
        if include_children:
            payload["notes"] = [note.to_dict() for note in self.notes.values()]
            payload["evidence"] = [item.to_dict() for item in self.evidence.values()]
            payload["hypotheses"] = [hyp.to_dict() for hyp in self.hypotheses.values()]
            payload["snapshots"] = [
                snap.to_dict(include_slots=False) for snap in self.snapshots.values()
            ]
        return payload

    @classmethod
    def from_full_dict(
        cls,
        data: Mapping[str, Any],
        *,
        snapshots: Mapping[str, Mapping[str, Any]],
        timeline: List[Mapping[str, Any]],
    ) -> "ResearchSession":
        """Rehydrate a session with children persisted by :class:`CaseStore`."""
        metadata = data.get("metadata")
        session = cls(
            session_id=str(data["session_id"]),
            name=str(data.get("name", "")),
            owning_debug_session=data.get("owning_debug_session"),
            description=data.get("description"),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
        for item in data.get("notes") or []:
            if isinstance(item, Mapping):
                note = ResearchNote.from_dict(item)
                session.notes[note.note_id] = note
        for item in data.get("evidence") or []:
            if isinstance(item, Mapping):
                evidence = ResearchEvidence.from_dict(item)
                session.evidence[evidence.evidence_id] = evidence
        for item in data.get("hypotheses") or []:
            if isinstance(item, Mapping):
                hypothesis = ResearchHypothesis.from_dict(item)
                session.hypotheses[hypothesis.hypothesis_id] = hypothesis
        for snapshot_id, payload in snapshots.items():
            if isinstance(payload, Mapping):
                session.snapshots[snapshot_id] = ResearchSnapshot.from_dict(payload)
        for item in timeline:
            if isinstance(item, Mapping):
                session.timeline.append(TimelineEvent.from_dict(item))
        return session


class ResearchRegistry:
    """Thread-safe, bounded registry of research sessions."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_RESEARCH_SESSIONS,
        max_notes_per_session: int = DEFAULT_MAX_NOTES_PER_SESSION,
        max_evidence_per_session: int = DEFAULT_MAX_EVIDENCE_PER_SESSION,
        max_hypotheses_per_session: int = DEFAULT_MAX_HYPOTHESES_PER_SESSION,
        max_snapshots_per_session: int = DEFAULT_MAX_SNAPSHOTS_PER_SESSION,
        max_timeline_events_per_session: int = DEFAULT_MAX_TIMELINE_EVENTS_PER_SESSION,
        store: Optional["CaseStore"] = None,
    ) -> None:
        for name, value in (
            ("max_sessions", max_sessions),
            ("max_notes_per_session", max_notes_per_session),
            ("max_evidence_per_session", max_evidence_per_session),
            ("max_hypotheses_per_session", max_hypotheses_per_session),
            ("max_snapshots_per_session", max_snapshots_per_session),
            ("max_timeline_events_per_session", max_timeline_events_per_session),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self._max_sessions = max_sessions
        self._max_notes = max_notes_per_session
        self._max_evidence = max_evidence_per_session
        self._max_hypotheses = max_hypotheses_per_session
        self._max_snapshots = max_snapshots_per_session
        self._max_timeline = max_timeline_events_per_session
        self._sessions: Dict[str, ResearchSession] = {}
        self._lock = threading.RLock()
        self._store: Optional["CaseStore"] = store

    def _persist(self, session: ResearchSession) -> None:
        """Write one session to the case store (best-effort, never raises)."""
        store = self._store
        if store is None:
            return
        try:
            store.save_session(session)
        except Exception:  # noqa: BLE001 - durability must not break research
            logger.exception(
                "Case store persist failed for research session '%s'",
                session.session_id,
            )

    def load_existing(self) -> int:
        """Rehydrate persisted sessions into this registry; returns count."""
        store = self._store
        if store is None:
            return 0
        try:
            stored = store.load_all()
        except Exception:  # noqa: BLE001 - degraded startup, not fatal
            logger.exception("Case store load failed; starting with empty registry")
            return 0
        loaded = 0
        with self._lock:
            for data in stored:
                try:
                    session = ResearchSession.from_full_dict(
                        data,
                        snapshots=data.get("snapshots") or {},
                        timeline=data.get("timeline") or [],
                    )
                except Exception:  # noqa: BLE001 - skip corrupt rows
                    logger.exception("Skipping unloadable research session row")
                    continue
                if session.session_id in self._sessions:
                    continue
                if len(self._sessions) >= self._max_sessions:
                    logger.warning(
                        "Research session cap reached during rehydrate; "
                        "skipping '%s'",
                        session.session_id,
                    )
                    break
                self._sessions[session.session_id] = session
                loaded += 1
        return loaded

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
            self._persist(session)
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
            removed = self._sessions.pop(session_id, None)
        if removed is not None and self._store is not None:
            try:
                self._store.delete_session(session_id)
            except Exception:  # noqa: BLE001 - durability must not break research
                logger.exception(
                    "Case store delete failed for research session '%s'", session_id
                )
        return removed

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------
    def _append_event_locked(
        self,
        session: ResearchSession,
        *,
        kind: str,
        summary: str,
        ref: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TimelineEvent:
        """Append one timeline event; caller must already hold ``session._lock``.

        Timeline is bounded: on overflow the oldest event is dropped so
        the append is O(1) amortized and a runaway workflow does not
        allocate unbounded memory.  Oldest-first eviction preserves the
        most recent history, which is what an investigator wants.
        """
        if kind not in TIMELINE_EVENT_KINDS:
            raise ResearchError(
                f"event kind {kind!r} must be one of " f"{list(TIMELINE_EVENT_KINDS)}"
            )
        if not isinstance(summary, str) or not summary.strip():
            raise ResearchError("event summary must be a non-empty string")
        event = TimelineEvent(
            event_id=uuid.uuid4().hex[:12],
            kind=kind,
            summary=summary.strip(),
            ref=ref,
            hypothesis_id=hypothesis_id,
            note_id=note_id,
            snapshot_id=snapshot_id,
            metadata=dict(metadata or {}),
        )
        session.timeline.append(event)
        # Cap-driven eviction of the oldest event.  We drop rather than
        # error because the timeline is auto-populated by note / snapshot
        # / hypothesis mutations, and a hard error there would break the
        # underlying operation.
        if len(session.timeline) > self._max_timeline:
            del session.timeline[0]
            # Note the drop in the newest event's metadata so a caller
            # can tell the timeline was pruned rather than silently
            # complete.
            event.metadata["dropped_oldest_for_cap"] = True
        # Every registry mutation funnels through this append, so one
        # persist hook here covers notes/evidence/hypotheses/snapshots.
        self._persist(session)
        return event

    def append_timeline_event(
        self,
        *,
        session_id: str,
        kind: str,
        summary: str,
        ref: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> TimelineEvent:
        session = self.get_session(session_id)
        with session._lock:
            if hypothesis_id is not None and hypothesis_id not in session.hypotheses:
                raise ResearchError(
                    f"hypothesis '{hypothesis_id}' not found in session"
                )
            if note_id is not None and note_id not in session.notes:
                raise ResearchError(f"note '{note_id}' not found in session")
            if snapshot_id is not None and snapshot_id not in session.snapshots:
                raise ResearchError(f"snapshot '{snapshot_id}' not found in session")
            event = self._append_event_locked(
                session,
                kind=kind,
                summary=summary,
                ref=ref,
                hypothesis_id=hypothesis_id,
                note_id=note_id,
                snapshot_id=snapshot_id,
                metadata=metadata,
            )
            session._touch()
            return event

    def get_timeline(
        self,
        *,
        session_id: str,
        kinds: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[TimelineEvent]:
        session = self.get_session(session_id)
        with session._lock:
            events = list(session.timeline)
        if kinds:
            kinds_set = set(kinds)
            unknown = sorted(kinds_set - set(TIMELINE_EVENT_KINDS))
            if unknown:
                raise ResearchError(
                    f"unknown filter kinds {unknown}; expected any of "
                    f"{list(TIMELINE_EVENT_KINDS)}"
                )
            events = [e for e in events if e.kind in kinds_set]
        if limit is not None:
            if limit <= 0:
                raise ResearchError("limit must be positive when provided")
            events = events[-limit:]
        return events

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
            self._append_event_locked(
                session,
                kind="note",
                summary=note.body[:120],
                note_id=note.note_id,
                hypothesis_id=hypothesis_id,
            )
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
            summary = description or f"linked {kind} '{ref.strip()}'"
            self._append_event_locked(
                session,
                kind=(
                    "crash"
                    if kind == "crash"
                    else "artifact" if kind == "artifact" else "other"
                ),
                summary=summary[:120],
                ref=ref.strip(),
                hypothesis_id=hypothesis_id,
                note_id=note_id,
                metadata={"evidence_id": evidence.evidence_id, "evidence_kind": kind},
            )
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
            self._append_event_locked(
                session,
                kind="hypothesis_change",
                summary=f"proposed: {hypothesis.statement[:100]}",
                hypothesis_id=hypothesis.hypothesis_id,
                metadata={"transition": ["<none>", state]},
            )
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
            previous_state = hypothesis.state
            hypothesis.state = state
            if rationale is not None:
                hypothesis.rationale = rationale
            hypothesis.updated_at = _now_iso()
            self._append_event_locked(
                session,
                kind="hypothesis_change",
                summary=f"{previous_state} -> {state}: {hypothesis.statement[:80]}",
                hypothesis_id=hypothesis.hypothesis_id,
                metadata={"transition": [previous_state, state]},
            )
            session._touch()
            return hypothesis

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def create_snapshot(
        self,
        *,
        session_id: str,
        name: str,
        slots: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
    ) -> ResearchSnapshot:
        if not isinstance(name, str) or not name.strip():
            raise ResearchError("snapshot name must be a non-empty string")
        slots_dict = dict(slots or {})
        unknown = sorted(set(slots_dict.keys()) - set(SNAPSHOT_SLOTS))
        if unknown:
            raise ResearchError(
                f"unknown snapshot slots {unknown}; expected any of "
                f"{list(SNAPSHOT_SLOTS)}"
            )
        session = self.get_session(session_id)
        with session._lock:
            if len(session.snapshots) >= self._max_snapshots:
                raise ResearchError(
                    f"session '{session_id}' already has "
                    f"{self._max_snapshots} snapshots (cap)"
                )
            snapshot = ResearchSnapshot(
                snapshot_id=uuid.uuid4().hex[:12],
                name=name.strip(),
                slots=slots_dict,
                description=description,
            )
            session.snapshots[snapshot.snapshot_id] = snapshot
            self._append_event_locked(
                session,
                kind="snapshot",
                summary=f"captured snapshot '{snapshot.name}'",
                snapshot_id=snapshot.snapshot_id,
                metadata={"slot_names": sorted(slots_dict.keys())},
            )
            session._touch()
            return snapshot

    def list_snapshots(self, *, session_id: str) -> List[ResearchSnapshot]:
        session = self.get_session(session_id)
        with session._lock:
            return list(session.snapshots.values())

    def get_snapshot(self, *, session_id: str, snapshot_id: str) -> ResearchSnapshot:
        session = self.get_session(session_id)
        with session._lock:
            snapshot = session.snapshots.get(snapshot_id)
            if snapshot is None:
                raise ResearchError(f"snapshot '{snapshot_id}' not found in session")
            return snapshot

    @staticmethod
    def _diff_slot(before: Any, after: Any) -> Dict[str, Any]:
        """Structural diff shared across every snapshot slot."""
        if before is None and after is None:
            return {"status": "absent", "before": None, "after": None}
        if before is None:
            return {"status": "added", "before": None, "after": after}
        if after is None:
            return {"status": "removed", "before": before, "after": None}
        if before == after:
            return {"status": "unchanged", "value": before}
        # Both dicts: compute added / removed / changed key sets.
        if isinstance(before, dict) and isinstance(after, dict):
            before_keys = set(before.keys())
            after_keys = set(after.keys())
            # Dict keys may mix types (e.g. int register numbers next to str
            # field names); str-ordering keeps the diff deterministic instead
            # of crashing `sorted` on an int/str comparison.
            key_sort = lambda key: (type(key).__name__, str(key))
            added_keys = sorted(after_keys - before_keys, key=key_sort)
            removed_keys = sorted(before_keys - after_keys, key=key_sort)
            changed_keys = sorted(
                (k for k in before_keys & after_keys if before[k] != after[k]),
                key=key_sort,
            )
            return {
                "status": "changed",
                "kind": "dict",
                "added": {k: after[k] for k in added_keys},
                "removed": {k: before[k] for k in removed_keys},
                "changed": {
                    k: {"before": before[k], "after": after[k]} for k in changed_keys
                },
            }
        # Lists: report simple length + membership deltas.  A fuller
        # per-element diff (LCS / Myers) is intentionally deferred
        # -- callers building on this either know the slot's shape
        # (registers dict) or want the raw values back for their own
        # diff.
        if isinstance(before, list) and isinstance(after, list):
            before_set = {repr(x) for x in before}
            after_set = {repr(x) for x in after}
            return {
                "status": "changed",
                "kind": "list",
                "before_length": len(before),
                "after_length": len(after),
                "only_in_before": sorted(before_set - after_set),
                "only_in_after": sorted(after_set - before_set),
            }
        return {
            "status": "changed",
            "kind": "scalar",
            "before": before,
            "after": after,
        }

    def compare_snapshots(
        self, *, session_id: str, before: str, after: str
    ) -> Dict[str, Any]:
        if before == after:
            raise ResearchError(
                f"cannot compare a snapshot against itself ('{before}')"
            )
        before_snap = self.get_snapshot(session_id=session_id, snapshot_id=before)
        after_snap = self.get_snapshot(session_id=session_id, snapshot_id=after)
        slot_names = sorted(set(before_snap.slots) | set(after_snap.slots))
        diff: Dict[str, Any] = {}
        for slot in slot_names:
            diff[slot] = self._diff_slot(
                before_snap.slots.get(slot), after_snap.slots.get(slot)
            )
        return {
            "before": before_snap.to_dict(include_slots=False),
            "after": after_snap.to_dict(include_slots=False),
            "slot_names": slot_names,
            "diff": diff,
        }

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

    def create_snapshot_response(
        self,
        *,
        session_id: str,
        name: str,
        slots: Optional[Mapping[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.create_snapshot(
            session_id=session_id,
            name=name,
            slots=slots,
            description=description,
        )
        return build_backend_response(
            "create_named_snapshot",
            parsed={
                "schema": RESEARCH_SNAPSHOT_SCHEMA,
                "session_id": session_id,
                "snapshot": snapshot.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def list_snapshots_response(self, *, session_id: str) -> Dict[str, Any]:
        snapshots = self.list_snapshots(session_id=session_id)
        return build_backend_response(
            "list_snapshots",
            parsed={
                "schema": RESEARCH_SNAPSHOT_LIST_SCHEMA,
                "session_id": session_id,
                "snapshots": [s.to_dict(include_slots=False) for s in snapshots],
                "count": len(snapshots),
                "max_snapshots": self._max_snapshots,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def compare_snapshots_response(
        self, *, session_id: str, before: str, after: str
    ) -> Dict[str, Any]:
        diff = self.compare_snapshots(session_id=session_id, before=before, after=after)
        return build_backend_response(
            "compare_snapshots",
            parsed={
                "schema": RESEARCH_SNAPSHOT_DIFF_SCHEMA,
                "session_id": session_id,
                **diff,
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def append_timeline_event_response(
        self,
        *,
        session_id: str,
        kind: str,
        summary: str,
        ref: Optional[str] = None,
        hypothesis_id: Optional[str] = None,
        note_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = self.append_timeline_event(
            session_id=session_id,
            kind=kind,
            summary=summary,
            ref=ref,
            hypothesis_id=hypothesis_id,
            note_id=note_id,
            snapshot_id=snapshot_id,
            metadata=metadata,
        )
        return build_backend_response(
            "append_timeline_event",
            parsed={
                "schema": RESEARCH_TIMELINE_EVENT_SCHEMA,
                "session_id": session_id,
                "event": event.to_dict(),
            },
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def get_timeline_response(
        self,
        *,
        session_id: str,
        kinds: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        events = self.get_timeline(session_id=session_id, kinds=kinds, limit=limit)
        return build_backend_response(
            "get_research_timeline",
            parsed={
                "schema": RESEARCH_TIMELINE_SCHEMA,
                "session_id": session_id,
                "filter_kinds": sorted(kinds) if kinds else None,
                "limit": limit,
                "events": [e.to_dict() for e in events],
                "count": len(events),
                "max_events": self._max_timeline,
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
    "ResearchSnapshot",
    "ResearchSession",
    "ResearchRegistry",
    "TimelineEvent",
    "HYPOTHESIS_STATES",
    "EVIDENCE_KINDS",
    "SNAPSHOT_SLOTS",
    "TIMELINE_EVENT_KINDS",
    "DEFAULT_MAX_RESEARCH_SESSIONS",
    "DEFAULT_MAX_NOTES_PER_SESSION",
    "DEFAULT_MAX_EVIDENCE_PER_SESSION",
    "DEFAULT_MAX_HYPOTHESES_PER_SESSION",
    "DEFAULT_MAX_SNAPSHOTS_PER_SESSION",
    "DEFAULT_MAX_TIMELINE_EVENTS_PER_SESSION",
    "RESEARCH_SESSION_SCHEMA",
    "RESEARCH_SESSION_LIST_SCHEMA",
    "RESEARCH_NOTE_SCHEMA",
    "RESEARCH_EVIDENCE_SCHEMA",
    "RESEARCH_HYPOTHESIS_SCHEMA",
    "RESEARCH_SNAPSHOT_SCHEMA",
    "RESEARCH_SNAPSHOT_LIST_SCHEMA",
    "RESEARCH_SNAPSHOT_DIFF_SCHEMA",
    "RESEARCH_TIMELINE_EVENT_SCHEMA",
    "RESEARCH_TIMELINE_SCHEMA",
]
