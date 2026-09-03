"""Tests for typed research workspace foundation."""

from __future__ import annotations

import pytest

from paomcp.tools.backends.research import (
    DEFAULT_MAX_EVIDENCE_PER_SESSION,
    DEFAULT_MAX_HYPOTHESES_PER_SESSION,
    DEFAULT_MAX_NOTES_PER_SESSION,
    DEFAULT_MAX_RESEARCH_SESSIONS,
    EVIDENCE_KINDS,
    HYPOTHESIS_STATES,
    RESEARCH_EVIDENCE_SCHEMA,
    RESEARCH_HYPOTHESIS_SCHEMA,
    RESEARCH_NOTE_SCHEMA,
    RESEARCH_SESSION_LIST_SCHEMA,
    RESEARCH_SESSION_SCHEMA,
    ResearchError,
    ResearchRegistry,
)


# ---------------------------------------------------------------------------
# session lifecycle
# ---------------------------------------------------------------------------
class TestCreateSession:
    def test_create_returns_typed_envelope(self) -> None:
        registry = ResearchRegistry()
        response = registry.create_session_response(
            name="ret2libc CTF",
            description="pwn/level4",
            owning_debug_session="dbg-abc",
            metadata={"binary": "/pwn/vuln"},
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_SESSION_SCHEMA
        session = parsed["session"]
        assert session["name"] == "ret2libc CTF"
        assert session["owning_debug_session"] == "dbg-abc"
        assert session["metadata"] == {"binary": "/pwn/vuln"}
        assert session["note_count"] == 0

    def test_blank_name_rejected(self) -> None:
        registry = ResearchRegistry()
        with pytest.raises(ResearchError, match="non-empty string"):
            registry.create_session_response(name="   ")

    def test_session_cap_enforced(self) -> None:
        registry = ResearchRegistry(max_sessions=2)
        registry.create_session_response(name="a")
        registry.create_session_response(name="b")
        with pytest.raises(ResearchError, match="cap"):
            registry.create_session_response(name="c")

    def test_list_reports_summaries_and_cap(self) -> None:
        registry = ResearchRegistry(max_sessions=4)
        registry.create_session_response(name="alpha")
        registry.create_session_response(name="beta")
        response = registry.list_sessions_response()
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_SESSION_LIST_SCHEMA
        assert parsed["count"] == 2
        assert parsed["max_sessions"] == 4
        # summaries do NOT embed inline notes/evidence/hypotheses
        assert "notes" not in parsed["sessions"][0]

    def test_get_unknown_session_raises(self) -> None:
        registry = ResearchRegistry()
        with pytest.raises(ResearchError, match="not found"):
            registry.get_session_response(session_id="deadbeef")


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------
class TestNotes:
    def _session(self) -> tuple:
        registry = ResearchRegistry()
        session = registry.create_session_response(name="s")["parsed"]["session"]
        return registry, session["session_id"]

    def test_add_note_returns_typed_note(self) -> None:
        registry, sid = self._session()
        response = registry.add_note_response(
            session_id=sid,
            body="leaked puts address at 0x7ffe1234",
            tags=["leak", "libc"],
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_NOTE_SCHEMA
        note = parsed["note"]
        assert note["body"].startswith("leaked puts")
        assert note["tags"] == ["leak", "libc"]

    def test_blank_note_rejected(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="non-empty string"):
            registry.add_note_response(session_id=sid, body="   ")

    def test_note_cap_enforced(self) -> None:
        registry = ResearchRegistry(max_notes_per_session=2)
        session = registry.create_session_response(name="s")["parsed"]["session"]
        registry.add_note_response(session_id=session["session_id"], body="a")
        registry.add_note_response(session_id=session["session_id"], body="b")
        with pytest.raises(ResearchError, match="cap"):
            registry.add_note_response(session_id=session["session_id"], body="c")

    def test_note_pointing_at_unknown_hypothesis_rejected(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="hypothesis .* not found"):
            registry.add_note_response(session_id=sid, body="x", hypothesis_id="nope")

    def test_updated_at_moves_after_note(self) -> None:
        registry, sid = self._session()
        before = registry.get_session(sid).updated_at
        registry.add_note_response(session_id=sid, body="crumb")
        after = registry.get_session(sid).updated_at
        assert after >= before


# ---------------------------------------------------------------------------
# hypotheses
# ---------------------------------------------------------------------------
class TestHypotheses:
    def _session(self) -> tuple:
        registry = ResearchRegistry()
        session = registry.create_session_response(name="s")["parsed"]["session"]
        return registry, session["session_id"]

    def test_add_hypothesis_returns_typed_envelope(self) -> None:
        registry, sid = self._session()
        response = registry.add_hypothesis_response(
            session_id=sid,
            statement="bug is a 1-byte off-by-one on the stack",
            state="proposed",
            rationale="crash overwrites RBP low byte only",
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_HYPOTHESIS_SCHEMA
        assert parsed["hypothesis"]["state"] == "proposed"
        assert parsed["hypothesis"]["hypothesis_id"]

    def test_state_must_be_from_vocabulary(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="must be one of"):
            registry.add_hypothesis_response(
                session_id=sid, statement="x", state="probably"
            )

    def test_state_transition(self) -> None:
        registry, sid = self._session()
        hyp = registry.add_hypothesis_response(session_id=sid, statement="claim")[
            "parsed"
        ]["hypothesis"]
        registry.update_hypothesis_state_response(
            session_id=sid,
            hypothesis_id=hyp["hypothesis_id"],
            state="rejected",
            rationale="reproduced without the bug",
        )
        latest = registry.get_session(sid).hypotheses[hyp["hypothesis_id"]]
        assert latest.state == "rejected"
        assert latest.rationale == "reproduced without the bug"

    def test_state_vocabulary_documented(self) -> None:
        assert set(HYPOTHESIS_STATES) == {
            "proposed",
            "supported",
            "rejected",
            "unknown",
        }

    def test_hypothesis_cap_enforced(self) -> None:
        registry = ResearchRegistry(max_hypotheses_per_session=1)
        session = registry.create_session_response(name="s")["parsed"]["session"]
        registry.add_hypothesis_response(
            session_id=session["session_id"], statement="a"
        )
        with pytest.raises(ResearchError, match="cap"):
            registry.add_hypothesis_response(
                session_id=session["session_id"], statement="b"
            )


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------
class TestEvidence:
    def _session(self) -> tuple:
        registry = ResearchRegistry()
        session = registry.create_session_response(name="s")["parsed"]["session"]
        return registry, session["session_id"]

    def test_link_evidence_returns_typed_envelope(self) -> None:
        registry, sid = self._session()
        response = registry.link_evidence_response(
            session_id=sid,
            kind="crash",
            ref="crash-abc123",
            description="segv at ret with cyclic pattern 'daaa'",
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_EVIDENCE_SCHEMA
        assert parsed["evidence"]["kind"] == "crash"
        assert parsed["evidence"]["ref"] == "crash-abc123"

    def test_evidence_kind_must_be_from_vocabulary(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="must be one of"):
            registry.link_evidence_response(session_id=sid, kind="spooky", ref="x")

    def test_evidence_bound_to_hypothesis(self) -> None:
        registry, sid = self._session()
        hyp = registry.add_hypothesis_response(
            session_id=sid, statement="stack overflow"
        )["parsed"]["hypothesis"]
        response = registry.link_evidence_response(
            session_id=sid,
            kind="crash",
            ref="crash-1",
            hypothesis_id=hyp["hypothesis_id"],
        )
        assert response["parsed"]["evidence"]["hypothesis_id"] == hyp["hypothesis_id"]

    def test_evidence_referring_to_missing_note_rejected(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="note .* not found"):
            registry.link_evidence_response(
                session_id=sid, kind="crash", ref="x", note_id="nope"
            )

    def test_evidence_kinds_documented(self) -> None:
        expected = {
            "artifact",
            "crash",
            "heap_snapshot",
            "tool_result",
            "note",
            "external",
        }
        assert set(EVIDENCE_KINDS) == expected

    def test_evidence_cap_enforced(self) -> None:
        registry = ResearchRegistry(max_evidence_per_session=1)
        session = registry.create_session_response(name="s")["parsed"]["session"]
        registry.link_evidence_response(
            session_id=session["session_id"], kind="crash", ref="a"
        )
        with pytest.raises(ResearchError, match="cap"):
            registry.link_evidence_response(
                session_id=session["session_id"], kind="crash", ref="b"
            )


class TestGetSessionEmbedsChildren:
    def test_get_returns_notes_hypotheses_evidence_inline(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="s")["parsed"]["session"][
            "session_id"
        ]
        hyp = registry.add_hypothesis_response(session_id=sid, statement="x")["parsed"][
            "hypothesis"
        ]
        registry.add_note_response(
            session_id=sid,
            body="note-one",
            hypothesis_id=hyp["hypothesis_id"],
        )
        registry.link_evidence_response(
            session_id=sid,
            kind="crash",
            ref="crash-1",
            hypothesis_id=hyp["hypothesis_id"],
        )
        parsed = registry.get_session_response(session_id=sid)["parsed"]["session"]
        assert parsed["note_count"] == 1
        assert parsed["evidence_count"] == 1
        assert parsed["hypothesis_count"] == 1
        assert parsed["notes"][0]["body"] == "note-one"
        assert parsed["hypotheses"][0]["state"] == "proposed"


class TestDefaults:
    def test_caps_are_sane(self) -> None:
        assert DEFAULT_MAX_RESEARCH_SESSIONS >= 8
        assert DEFAULT_MAX_NOTES_PER_SESSION >= 100
        assert DEFAULT_MAX_EVIDENCE_PER_SESSION >= 100
        assert DEFAULT_MAX_HYPOTHESES_PER_SESSION >= 10


class TestLinkEvidenceBatch:
    def _session(self, registry=None):
        registry = registry or ResearchRegistry()
        session = registry.create_session(name="batch")
        return registry, session.session_id

    def test_batch_links_with_one_timeline_event(self) -> None:
        registry, sid = self._session()
        linked = registry.link_evidence_batch(
            session_id=sid,
            items=[
                {"kind": "crash", "ref": "crash-1"},
                {"kind": "artifact", "ref": "art-1", "description": "core"},
            ],
        )
        assert len(linked) == 2
        session = registry.get_session(sid)
        assert len(session.evidence) == 2
        # Exactly one timeline event for the whole batch.
        assert len(session.timeline) == 1
        assert "linked 2 evidence" in session.timeline[0].summary

    def test_batch_validates_before_recording(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="kind"):
            registry.link_evidence_batch(
                session_id=sid,
                items=[
                    {"kind": "crash", "ref": "ok-1"},
                    {"kind": "bogus", "ref": "bad"},
                ],
            )
        assert len(registry.get_session(sid).evidence) == 0

    def test_batch_respects_caps(self) -> None:
        registry, sid = self._session()
        with pytest.raises(ResearchError, match="max_items"):
            registry.link_evidence_batch(
                session_id=sid,
                items=[{"kind": "other", "ref": f"r{i}"} for i in range(501)],
                max_items=500,
            )
        with pytest.raises(ResearchError, match="items cannot be empty"):
            registry.link_evidence_batch(session_id=sid, items=[])
