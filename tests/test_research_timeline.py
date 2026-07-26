"""Tests for Stage 9.3 research session timeline."""

from __future__ import annotations

import pytest

from paomcp.tools.backends.research import (
    DEFAULT_MAX_TIMELINE_EVENTS_PER_SESSION,
    RESEARCH_TIMELINE_EVENT_SCHEMA,
    RESEARCH_TIMELINE_SCHEMA,
    TIMELINE_EVENT_KINDS,
    ResearchError,
    ResearchRegistry,
)


def _session():
    registry = ResearchRegistry()
    sid = registry.create_session_response(name="s")["parsed"]["session"]["session_id"]
    return registry, sid


class TestAppendTimelineEvent:
    def test_typed_envelope(self) -> None:
        registry, sid = _session()
        response = registry.append_timeline_event_response(
            session_id=sid,
            kind="tool_call",
            summary="ran find_gadgets on /bin/vuln",
            ref="op-42",
            metadata={"gadget_count": 4321},
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_TIMELINE_EVENT_SCHEMA
        event = parsed["event"]
        assert event["kind"] == "tool_call"
        assert event["ref"] == "op-42"
        assert event["metadata"]["gadget_count"] == 4321

    def test_unknown_kind_rejected(self) -> None:
        registry, sid = _session()
        with pytest.raises(ResearchError, match="must be one of"):
            registry.append_timeline_event_response(
                session_id=sid, kind="vibes", summary="x"
            )

    def test_empty_summary_rejected(self) -> None:
        registry, sid = _session()
        with pytest.raises(ResearchError, match="non-empty string"):
            registry.append_timeline_event_response(
                session_id=sid, kind="other", summary="   "
            )

    def test_referring_hypothesis_note_snapshot_must_exist(self) -> None:
        registry, sid = _session()
        with pytest.raises(ResearchError, match="hypothesis .* not found"):
            registry.append_timeline_event_response(
                session_id=sid,
                kind="other",
                summary="x",
                hypothesis_id="deadbeef",
            )
        with pytest.raises(ResearchError, match="note .* not found"):
            registry.append_timeline_event_response(
                session_id=sid,
                kind="other",
                summary="x",
                note_id="deadbeef",
            )
        with pytest.raises(ResearchError, match="snapshot .* not found"):
            registry.append_timeline_event_response(
                session_id=sid,
                kind="other",
                summary="x",
                snapshot_id="deadbeef",
            )

    def test_vocabulary_documented(self) -> None:
        assert "tool_call" in TIMELINE_EVENT_KINDS
        assert "hypothesis_change" in TIMELINE_EVENT_KINDS
        assert "snapshot" in TIMELINE_EVENT_KINDS
        assert "crash" in TIMELINE_EVENT_KINDS


class TestAutomaticEvents:
    def test_note_appends_a_note_event(self) -> None:
        registry, sid = _session()
        registry.add_note_response(session_id=sid, body="crumb 1")
        events = registry.get_timeline(session_id=sid)
        assert len(events) == 1
        assert events[0].kind == "note"
        assert "crumb 1" in events[0].summary

    def test_snapshot_appends_a_snapshot_event(self) -> None:
        registry, sid = _session()
        registry.create_snapshot_response(session_id=sid, name="pre")
        events = registry.get_timeline(session_id=sid)
        assert len(events) == 1
        assert events[0].kind == "snapshot"
        assert events[0].snapshot_id is not None

    def test_hypothesis_add_and_transition_each_append(self) -> None:
        registry, sid = _session()
        hyp = registry.add_hypothesis_response(
            session_id=sid, statement="stack overflow"
        )["parsed"]["hypothesis"]
        registry.update_hypothesis_state_response(
            session_id=sid,
            hypothesis_id=hyp["hypothesis_id"],
            state="supported",
            rationale="reproduced",
        )
        events = registry.get_timeline(session_id=sid)
        assert len(events) == 2
        assert events[0].metadata["transition"] == ["<none>", "proposed"]
        assert events[1].metadata["transition"] == ["proposed", "supported"]

    def test_crash_evidence_appends_crash_event(self) -> None:
        registry, sid = _session()
        registry.link_evidence_response(
            session_id=sid,
            kind="crash",
            ref="crash-abc",
            description="segv at ret",
        )
        events = registry.get_timeline(session_id=sid)
        crash_events = [e for e in events if e.kind == "crash"]
        assert crash_events, "crash evidence must append a crash event"
        assert crash_events[0].ref == "crash-abc"


class TestGetTimeline:
    def _rich_session(self) -> tuple:
        registry, sid = _session()
        registry.add_note_response(session_id=sid, body="note 1")
        registry.add_note_response(session_id=sid, body="note 2")
        registry.create_snapshot_response(session_id=sid, name="pre")
        registry.append_timeline_event_response(
            session_id=sid, kind="tool_call", summary="ran find_gadgets"
        )
        registry.append_timeline_event_response(
            session_id=sid, kind="crash", summary="reproduced"
        )
        return registry, sid

    def test_full_timeline_ordered(self) -> None:
        registry, sid = self._rich_session()
        response = registry.get_timeline_response(session_id=sid)
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_TIMELINE_SCHEMA
        assert parsed["count"] == 5
        kinds = [e["kind"] for e in parsed["events"]]
        assert kinds == ["note", "note", "snapshot", "tool_call", "crash"]

    def test_filter_by_kind(self) -> None:
        registry, sid = self._rich_session()
        parsed = registry.get_timeline_response(
            session_id=sid, kinds=["crash", "tool_call"]
        )["parsed"]
        kinds = [e["kind"] for e in parsed["events"]]
        assert kinds == ["tool_call", "crash"]
        assert parsed["filter_kinds"] == ["crash", "tool_call"]

    def test_unknown_filter_kind_raises(self) -> None:
        registry, sid = self._rich_session()
        with pytest.raises(ResearchError, match="unknown filter kinds"):
            registry.get_timeline_response(session_id=sid, kinds=["vibes"])

    def test_limit_returns_most_recent(self) -> None:
        registry, sid = self._rich_session()
        parsed = registry.get_timeline_response(session_id=sid, limit=2)["parsed"]
        assert parsed["count"] == 2
        kinds = [e["kind"] for e in parsed["events"]]
        assert kinds == ["tool_call", "crash"]

    def test_bad_limit_rejected(self) -> None:
        registry, sid = self._rich_session()
        with pytest.raises(ResearchError, match="limit must be positive"):
            registry.get_timeline_response(session_id=sid, limit=0)


class TestBoundedEviction:
    def test_overflow_drops_oldest_and_marks_new_event(self) -> None:
        registry = ResearchRegistry(max_timeline_events_per_session=3)
        sid = registry.create_session_response(name="s")["parsed"]["session"][
            "session_id"
        ]
        for i in range(4):
            registry.append_timeline_event_response(
                session_id=sid, kind="other", summary=f"crumb {i}"
            )
        events = registry.get_timeline(session_id=sid)
        assert len(events) == 3
        # oldest 'crumb 0' was evicted; newest carries the drop marker
        assert [e.summary for e in events] == [
            "crumb 1",
            "crumb 2",
            "crumb 3",
        ]
        assert events[-1].metadata.get("dropped_oldest_for_cap") is True

    def test_default_cap_documented(self) -> None:
        assert DEFAULT_MAX_TIMELINE_EVENTS_PER_SESSION >= 500


class TestSessionSummaryIncludesTimelineLength:
    def test_summary_reports_length(self) -> None:
        registry, sid = _session()
        registry.append_timeline_event_response(
            session_id=sid, kind="other", summary="x"
        )
        parsed = registry.get_session_response(session_id=sid)["parsed"]["session"]
        assert parsed["timeline_length"] == 1
