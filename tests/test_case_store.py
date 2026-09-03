import os

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
