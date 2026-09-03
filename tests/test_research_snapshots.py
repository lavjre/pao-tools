"""Tests for named research snapshots and typed diff."""

from __future__ import annotations

import pytest

from paomcp.tools.backends.research import (
    DEFAULT_MAX_SNAPSHOTS_PER_SESSION,
    RESEARCH_SNAPSHOT_DIFF_SCHEMA,
    RESEARCH_SNAPSHOT_LIST_SCHEMA,
    RESEARCH_SNAPSHOT_SCHEMA,
    SNAPSHOT_SLOTS,
    ResearchError,
    ResearchRegistry,
)


def _session():
    registry = ResearchRegistry()
    sid = registry.create_session_response(name="s")["parsed"]["session"]["session_id"]
    return registry, sid


class TestSnapshotCreate:
    def test_create_returns_typed_envelope(self) -> None:
        registry, sid = _session()
        response = registry.create_snapshot_response(
            session_id=sid,
            name="pre-crash",
            slots={
                "registers": {"rip": 0x400100, "rsp": 0x7FFF1000},
                "notes": "right before ret",
            },
            description="captured at 0x400100",
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_SNAPSHOT_SCHEMA
        snap = parsed["snapshot"]
        assert snap["name"] == "pre-crash"
        assert snap["description"] == "captured at 0x400100"
        assert set(snap["slot_names"]) == {"registers", "notes"}
        assert snap["slots"]["registers"]["rip"] == 0x400100

    def test_blank_name_rejected(self) -> None:
        registry, sid = _session()
        with pytest.raises(ResearchError, match="non-empty string"):
            registry.create_snapshot_response(session_id=sid, name="  ")

    def test_unknown_slot_names_rejected(self) -> None:
        registry, sid = _session()
        with pytest.raises(ResearchError, match="unknown snapshot slots"):
            registry.create_snapshot_response(
                session_id=sid, name="x", slots={"registers": {}, "vibes": 42}
            )

    def test_snapshot_cap_enforced(self) -> None:
        registry = ResearchRegistry(max_snapshots_per_session=2)
        sid = registry.create_session_response(name="s")["parsed"]["session"][
            "session_id"
        ]
        registry.create_snapshot_response(session_id=sid, name="a")
        registry.create_snapshot_response(session_id=sid, name="b")
        with pytest.raises(ResearchError, match="cap"):
            registry.create_snapshot_response(session_id=sid, name="c")

    def test_slot_vocabulary_documented(self) -> None:
        expected = {
            "registers",
            "memory",
            "mappings",
            "heap_snapshot",
            "backtrace",
            "notes",
            "metadata",
        }
        assert set(SNAPSHOT_SLOTS) == expected


class TestSnapshotList:
    def test_list_returns_summaries_not_slots(self) -> None:
        registry, sid = _session()
        registry.create_snapshot_response(
            session_id=sid,
            name="pre",
            slots={"registers": {"rip": 1}},
        )
        response = registry.list_snapshots_response(session_id=sid)
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_SNAPSHOT_LIST_SCHEMA
        assert parsed["count"] == 1
        # summaries carry ``slot_names`` but not the payload itself
        assert parsed["snapshots"][0]["slot_names"] == ["registers"]
        assert "slots" not in parsed["snapshots"][0]

    def test_max_snapshots_reported(self) -> None:
        registry, sid = _session()
        parsed = registry.list_snapshots_response(session_id=sid)["parsed"]
        assert parsed["max_snapshots"] == DEFAULT_MAX_SNAPSHOTS_PER_SESSION


class TestDiffLogic:
    def test_dict_slot_reports_added_removed_changed(self) -> None:
        diff = ResearchRegistry._diff_slot(
            {"rip": 0x100, "rsp": 0x200, "gone": 42},
            {"rip": 0x101, "rsp": 0x200, "new": 99},
        )
        assert diff["status"] == "changed"
        assert diff["kind"] == "dict"
        assert diff["added"] == {"new": 99}
        assert diff["removed"] == {"gone": 42}
        assert diff["changed"] == {"rip": {"before": 0x100, "after": 0x101}}

    def test_list_slot_reports_length_and_membership_delta(self) -> None:
        diff = ResearchRegistry._diff_slot(
            [{"addr": 1}, {"addr": 2}],
            [{"addr": 2}, {"addr": 3}, {"addr": 4}],
        )
        assert diff["status"] == "changed"
        assert diff["kind"] == "list"
        assert diff["before_length"] == 2

    def test_dict_slot_with_mixed_key_types_does_not_crash(self) -> None:
        # Register-number keys are ints while named fields are str; sorting
        # the diff sets must not compare int to str.
        diff = ResearchRegistry._diff_slot(
            {0: "rax", 1: "rbx", "symbol": "main"},
            {0: "rax", 2: "rcx", "symbol": "main", "module": "libc"},
        )
        assert diff["status"] == "changed"
        assert diff["kind"] == "dict"
        assert diff["added"] == {2: "rcx", "module": "libc"}
        assert diff["removed"] == {1: "rbx"}
        assert diff["changed"] == {}

    def test_unchanged_slot_reports_unchanged(self) -> None:
        diff = ResearchRegistry._diff_slot({"rip": 0x100}, {"rip": 0x100})
        assert diff["status"] == "unchanged"
        assert diff["value"] == {"rip": 0x100}

    def test_added_and_removed_when_slot_only_on_one_side(self) -> None:
        assert ResearchRegistry._diff_slot(None, {"rip": 1})["status"] == "added"
        assert ResearchRegistry._diff_slot({"rip": 1}, None)["status"] == "removed"

    def test_scalar_change_returns_before_after(self) -> None:
        diff = ResearchRegistry._diff_slot("pre", "post")
        assert diff["status"] == "changed"
        assert diff["kind"] == "scalar"
        assert diff["before"] == "pre"
        assert diff["after"] == "post"


class TestCompareSnapshots:
    def test_typed_envelope_walks_union_of_slots(self) -> None:
        registry, sid = _session()
        a = registry.create_snapshot_response(
            session_id=sid,
            name="pre",
            slots={
                "registers": {"rip": 0x400100},
                "mappings": [{"start": 0x400000, "end": 0x401000}],
            },
        )["parsed"]["snapshot"]
        b = registry.create_snapshot_response(
            session_id=sid,
            name="post",
            slots={
                "registers": {"rip": 0x400200},
                "backtrace": [{"level": 0, "pc": 0x400200}],
            },
        )["parsed"]["snapshot"]
        response = registry.compare_snapshots_response(
            session_id=sid,
            before=a["snapshot_id"],
            after=b["snapshot_id"],
        )
        parsed = response["parsed"]
        assert parsed["schema"] == RESEARCH_SNAPSHOT_DIFF_SCHEMA
        assert set(parsed["slot_names"]) == {
            "registers",
            "mappings",
            "backtrace",
        }
        assert parsed["diff"]["registers"]["status"] == "changed"
        assert parsed["diff"]["mappings"]["status"] == "removed"
        assert parsed["diff"]["backtrace"]["status"] == "added"

    def test_self_compare_rejected(self) -> None:
        registry, sid = _session()
        snap = registry.create_snapshot_response(session_id=sid, name="only")["parsed"][
            "snapshot"
        ]
        with pytest.raises(ResearchError, match="itself"):
            registry.compare_snapshots_response(
                session_id=sid,
                before=snap["snapshot_id"],
                after=snap["snapshot_id"],
            )

    def test_missing_snapshot_raises(self) -> None:
        registry, sid = _session()
        snap = registry.create_snapshot_response(session_id=sid, name="only")["parsed"][
            "snapshot"
        ]
        with pytest.raises(ResearchError, match="not found"):
            registry.compare_snapshots_response(
                session_id=sid,
                before=snap["snapshot_id"],
                after="deadbeef00ff",
            )


class TestSessionEmbedsSnapshotSummaries:
    def test_get_session_includes_snapshot_summaries(self) -> None:
        registry, sid = _session()
        registry.create_snapshot_response(
            session_id=sid, name="pre", slots={"registers": {"rip": 1}}
        )
        parsed = registry.get_session_response(session_id=sid)["parsed"]["session"]
        assert parsed["snapshot_count"] == 1
        # inline summary carries slot_names but no full payload
        assert parsed["snapshots"][0]["slot_names"] == ["registers"]
        assert "slots" not in parsed["snapshots"][0]
