import os
import time

from paomcp.tools.backends.research import ResearchRegistry
from paomcp.tools.backends.research_store import CaseStore


def _populated_registry(store=None):
    registry = ResearchRegistry(store=store)
    session = registry.create_session(name="case-a", description="d")
    registry.add_note(session_id=session.session_id, body="note one", tags=["t"])
    hyp = registry.add_hypothesis(
        session_id=session.session_id, statement="heap overflow in parse()"
    )
    registry.update_hypothesis_state(
        session_id=session.session_id,
        hypothesis_id=hyp.hypothesis_id,
        state="supported",
    )
    registry.link_evidence(
        session_id=session.session_id,
        kind="external",
        ref="https://example.test/advisory",
    )
    registry.create_snapshot(
        session_id=session.session_id, name="s1", slots={"notes": "n"}
    )
    registry.append_timeline_event(
        session_id=session.session_id, kind="other", summary="manual mark"
    )
    return registry, session.session_id


def test_roundtrip_preserves_session_and_children(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    _, session_id = _populated_registry(store=store)
    assert session_id in store.session_ids()

    fresh = ResearchRegistry(store=store)
    assert fresh.load_existing() == 1
    session = fresh.get_session(session_id)
    assert session.name == "case-a"
    assert len(session.notes) == 1
    assert len(session.hypotheses) == 1
    assert list(session.hypotheses.values())[0].state == "supported"
    assert len(session.evidence) == 1
    by_name = {s.name: s for s in session.snapshots.values()}
    assert by_name["s1"].slots == {"notes": "n"}
    assert len(session.timeline) >= 6

    # Mutations on the rehydrated registry persist again.
    fresh.add_note(session_id=session_id, body="note two")
    again = ResearchRegistry(store=store)
    assert again.load_existing() == 1
    assert len(again.get_session(session_id).notes) == 2
    store.close()


def test_remove_session_deletes_rows(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    registry, session_id = _populated_registry(store=store)
    registry.remove_session(session_id)
    assert store.session_ids() == []
    store.close()


def test_registry_without_store_works_memory_only():
    registry, session_id = _populated_registry(store=None)
    assert registry.get_session(session_id).name == "case-a"
    assert registry.load_existing() == 0


def test_store_failure_never_breaks_research(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    store.close()
    # Closed handle: every persist fails inside, operations must survive.
    registry = ResearchRegistry(store=store)
    session = registry.create_session(name="resilient")
    registry.add_note(session_id=session.session_id, body="kept")
    assert len(registry.get_session(session.session_id).notes) == 1
    assert os.path.exists(str(tmp_path / "cases.db"))


def _store_sessions(store, count):
    registry = ResearchRegistry(store=store)
    session_ids = []
    for index in range(count):
        session = registry.create_session(name=f"case-{index}")
        registry.add_note(session_id=session.session_id, body=f"note-{index}")
        session_ids.append(session.session_id)
        # Distinct updated_at values make the newest-first order decidable.
        time.sleep(0.003)
    return session_ids


def test_rehydrate_at_cap_keeps_newest_session(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    session_ids = _store_sessions(store, 3)

    fresh = ResearchRegistry(store=store, max_sessions=1)
    assert fresh.load_existing() == 1
    assert [session.session_id for session in fresh.list_sessions()] == [
        session_ids[-1]
    ]
    store.close()


def test_rehydrate_at_cap_keeps_newest_in_deterministic_order(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    session_ids = _store_sessions(store, 5)

    fresh = ResearchRegistry(store=store, max_sessions=3)
    assert fresh.load_existing() == 3
    assert [session.session_id for session in fresh.list_sessions()] == session_ids[
        -1:-4:-1
    ]
    assert [row["session_id"] for row in store.load_all()] == session_ids[::-1]
    store.close()


def test_persisted_updated_at_reflects_last_mutation(tmp_path):
    store = CaseStore(str(tmp_path / "cases.db"))
    registry = ResearchRegistry(store=store)
    session = registry.create_session(name="touch-order")
    session_id = session.session_id

    def assert_persisted_touch_matches() -> None:
        row = next(
            item for item in store.load_all() if item["session_id"] == session_id
        )
        assert row["updated_at"] == registry.get_session(session_id).updated_at

    time.sleep(0.003)
    registry.add_note(session_id=session_id, body="note")
    assert_persisted_touch_matches()

    time.sleep(0.003)
    hypothesis = registry.add_hypothesis(session_id=session_id, statement="claim")
    assert_persisted_touch_matches()

    time.sleep(0.003)
    registry.update_hypothesis_state(
        session_id=session_id,
        hypothesis_id=hypothesis.hypothesis_id,
        state="supported",
    )
    assert_persisted_touch_matches()

    time.sleep(0.003)
    registry.link_evidence(session_id=session_id, kind="crash", ref="crash-1")
    assert_persisted_touch_matches()

    time.sleep(0.003)
    registry.link_evidence_batch(
        session_id=session_id,
        items=[{"kind": "external", "ref": "https://example.test/advisory"}],
    )
    assert_persisted_touch_matches()

    time.sleep(0.003)
    registry.create_snapshot(session_id=session_id, name="snap", slots={"notes": "n"})
    assert_persisted_touch_matches()

    time.sleep(0.003)
    registry.append_timeline_event(session_id=session_id, kind="other", summary="mark")
    assert_persisted_touch_matches()
    store.close()
