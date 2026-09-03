"""Tests for typed heap snapshot history and diff."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends import heap as heap_module
from paomcp.tools.backends.heap import (
    DEFAULT_MAX_HEAP_SNAPSHOTS,
    DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
    HEAP_SNAPSHOT_DIFF_SCHEMA,
    HEAP_SNAPSHOT_LINK_SCHEMA,
    HEAP_SNAPSHOT_LIST_SCHEMA,
    HEAP_SNAPSHOT_SCHEMA,
    HeapHistory,
    HeapSnapshotError,
)


def _pin_classification(monkeypatch, classification: Dict[str, Any]) -> None:
    """Force ``AllocatorDiscovery.classify`` to return ``classification``.

    :func:`HeapHistory._prepare_context` calls the classify static method
    directly (not through the injected discovery stub) so the stubs alone
    cannot decide the reported family; a monkeypatch is the cleanest way
    to steer that decision in a unit test.
    """
    monkeypatch.setattr(
        heap_module.AllocatorDiscovery,
        "classify",
        staticmethod(lambda mappings, libcinfo: dict(classification)),
    )


# ---------------------------------------------------------------------------
# Minimal fakes.  The pure helpers do not need any fakes; the storage /
# capture paths only need a stub GDB (state, pid, epoch, last-stop record)
# and canned sub-component responses.
# ---------------------------------------------------------------------------
class StubGdb:
    def __init__(self, *, live: bool = True, pid: int = 4242, epoch: int = 3) -> None:
        self._live = live
        self._pid = pid
        self._epoch = epoch

    def get_state(self) -> str:
        return "STOPPED"

    def get_inferior_pid(self) -> Optional[int]:
        return self._pid if self._live else None

    def get_inferior_epoch(self) -> int:
        return self._epoch

    def get_last_stop_record(self) -> Optional[Dict[str, Any]]:
        if not self._live:
            return None
        return {
            "is_live": True,
            "reason": "signal-received",
            "signal-name": "SIGSTOP",
            "inferior_pid": self._pid,
            "inferior_epoch": self._epoch,
        }


class StubDiscovery:
    """A collect_evidence / target_profile stub for a chosen classification."""

    def __init__(self, classification: Dict[str, Any], profile: Dict[str, Any]) -> None:
        self._classification = classification
        self._profile = profile

    def collect_evidence(self, _cancellation_check):
        # (mappings, libcinfo_parsed, component_status)
        return ([], {"resolved": True}, {"mappings": "ok", "libcinfo": "ok"})

    def target_profile(self, _mappings):
        return dict(self._profile)


class StubArenas:
    def __init__(
        self,
        list_response: Optional[Dict[str, Any]] = None,
        inspect_response_by_addr: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> None:
        self.list_response = list_response or _canned_backend_response(
            "list_arenas", {"arenas": []}
        )
        self.inspect_response_by_addr = inspect_response_by_addr or {}

    def list_arenas(self, *, cancellation_check=None):
        return self.list_response

    def inspect_arena(self, *, arena_address=None, cancellation_check=None):
        addr = int(arena_address) if arena_address is not None else None
        return self.inspect_response_by_addr.get(
            addr,
            _canned_backend_response(
                "inspect_arena",
                {
                    "arena_address": addr,
                    "is_main_arena": addr == 0x7FFF_F7E0_0000,
                    "malloc_state": {},
                    "verdict": "parsed",
                },
            ),
        )


class StubBins:
    def __init__(
        self,
        fastbins_response: Optional[Dict[str, Any]] = None,
        bins_response: Optional[Dict[str, Any]] = None,
        tcache_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fastbins_response = fastbins_response or _canned_backend_response(
            "inspect_fastbins",
            {
                "arena_address": 0x7FFF_F7E0_0000,
                "is_main_arena": True,
                "fastbins": [],
                "verdict": "parsed",
            },
        )
        self.bins_response = bins_response or _canned_backend_response(
            "inspect_bins",
            {
                "arena_address": 0x7FFF_F7E0_0000,
                "is_main_arena": True,
                "bins": [],
                "verdict": "parsed",
            },
        )
        self.tcache_response = tcache_response or _canned_backend_response(
            "inspect_tcache",
            {"bins": [], "verdict": "parsed"},
        )

    def inspect_fastbins(
        self, *, arena_address=None, max_chunks_per_bin=None, cancellation_check=None
    ):
        return self.fastbins_response

    def inspect_bins(
        self, *, arena_address=None, max_chunks_per_bin=None, cancellation_check=None
    ):
        return self.bins_response

    def inspect_tcache(self, *, max_chunks_per_bin=None, cancellation_check=None):
        return self.tcache_response


def _canned_backend_response(command: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        "command": command,
        "state": "STOPPED",
        "parser_version": "1",
        "parsed": parsed,
        "raw_responses": [],
    }


def _glibc_classification(version: str = "2.35") -> Dict[str, Any]:
    return {
        "family": "glibc",
        "version": version,
        "confidence": "confirmed",
        "safe_linking": "enabled",
    }


def _musl_classification() -> Dict[str, Any]:
    return {
        "family": "musl",
        "version": "1.2.4",
        "confidence": "likely",
        "safe_linking": "not_applicable",
    }


def _profile_64() -> Dict[str, Any]:
    return {
        "pointer_bits": 64,
        "byte_order": "little",
        "architecture": {"name": "x86_64", "source": "checksec"},
    }


def _build_history(
    *,
    classification: Optional[Dict[str, Any]] = None,
    live: bool = True,
    arenas_response: Optional[Dict[str, Any]] = None,
    inspect_arena_by_addr: Optional[Dict[int, Dict[str, Any]]] = None,
    fastbins_response: Optional[Dict[str, Any]] = None,
    bins_response: Optional[Dict[str, Any]] = None,
    tcache_response: Optional[Dict[str, Any]] = None,
    max_snapshots: int = DEFAULT_MAX_HEAP_SNAPSHOTS,
    max_snapshot_bytes: int = DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
) -> HeapHistory:
    return HeapHistory(
        StubGdb(live=live),
        None,  # inspection not used directly
        None,  # debugger not used directly
        "sess",
        discovery=StubDiscovery(
            classification or _glibc_classification(),
            _profile_64(),
        ),
        arenas=StubArenas(arenas_response, inspect_arena_by_addr),
        bins=StubBins(fastbins_response, bins_response, tcache_response),
        max_snapshots=max_snapshots,
        max_snapshot_bytes=max_snapshot_bytes,
    )


# ---------------------------------------------------------------------------
# Pure-logic tests (no fakes required beyond dicts)
# ---------------------------------------------------------------------------
class TestDiffLedgers:
    def test_reports_new_removed_and_moved(self) -> None:
        before = [
            {
                "chunk_address": 0x1000,
                "size": 32,
                "location": {"kind": "fastbin", "index": 0},
            },
            {
                "chunk_address": 0x2000,
                "size": 64,
                "location": {"kind": "tcache", "index": 2},
            },
            {
                "chunk_address": 0x3000,
                "size": 96,
                "location": {"kind": "unsorted", "index": 1},
            },
        ]
        after = [
            {
                "chunk_address": 0x2000,
                "size": 64,
                "location": {"kind": "tcache", "index": 2},
            },
            {
                "chunk_address": 0x3000,
                "size": 96,
                "location": {"kind": "small", "index": 5},
            },
            {
                "chunk_address": 0x4000,
                "size": 128,
                "location": {"kind": "fastbin", "index": 1},
            },
        ]
        diff = HeapHistory.diff_ledgers(before, after)
        assert [c["chunk_address"] for c in diff["new_chunks"]] == [0x4000]
        assert [c["chunk_address"] for c in diff["removed_chunks"]] == [0x1000]
        assert [c["chunk_address"] for c in diff["moved_chunks"]] == [0x3000]
        moved = diff["moved_chunks"][0]
        assert moved["locations_before"] == [{"kind": "unsorted", "index": 1}]
        assert moved["locations_after"] == [{"kind": "small", "index": 5}]
        assert diff["summary"] == {
            "new_count": 1,
            "removed_count": 1,
            "moved_count": 1,
            "before_chunk_count": 3,
            "after_chunk_count": 3,
        }

    def test_empty_inputs_yield_empty_diff(self) -> None:
        diff = HeapHistory.diff_ledgers([], [])
        assert diff["new_chunks"] == []
        assert diff["removed_chunks"] == []
        assert diff["moved_chunks"] == []
        assert diff["summary"]["before_chunk_count"] == 0
        assert diff["summary"]["after_chunk_count"] == 0

    def test_same_location_and_size_is_not_moved(self) -> None:
        before = [
            {
                "chunk_address": 0x9000,
                "size": 48,
                "location": {"kind": "tcache", "index": 3},
            },
        ]
        after = [
            {
                "chunk_address": 0x9000,
                "size": 48,
                "location": {"kind": "tcache", "index": 3},
            },
        ]
        diff = HeapHistory.diff_ledgers(before, after)
        assert diff["moved_chunks"] == []
        assert diff["summary"]["moved_count"] == 0

    def test_duplicate_addresses_preserved_across_locations(self) -> None:
        before = [
            {
                "chunk_address": 0xA000,
                "size": 24,
                "location": {"kind": "fastbin", "index": 0},
            },
            {
                "chunk_address": 0xA000,
                "size": 24,
                "location": {"kind": "tcache", "index": 0},
            },
        ]
        after: List[Dict[str, Any]] = []
        diff = HeapHistory.diff_ledgers(before, after)
        assert diff["removed_chunks"][0]["locations_before"] == [
            {"kind": "fastbin", "index": 0},
            {"kind": "tcache", "index": 0},
        ]
        assert diff["summary"]["before_chunk_count"] == 2

    def test_entries_missing_chunk_address_are_ignored(self) -> None:
        before = [{"chunk_address": None, "size": 8, "location": {}}]
        after = [{"chunk_address": 0x11, "size": 16, "location": {}}]
        diff = HeapHistory.diff_ledgers(before, after)
        assert len(diff["new_chunks"]) == 1
        assert diff["summary"]["before_chunk_count"] == 0


class TestDiffArenaSummaries:
    def test_reports_top_delta_and_presence(self) -> None:
        before = [
            {
                "arena_address": 0x7FFF_F7E0_0000,
                "top": 0x5555_5555_9000,
                "last_remainder": 0,
            },
        ]
        after = [
            {
                "arena_address": 0x7FFF_F7E0_0000,
                "top": 0x5555_5555_9800,
                "last_remainder": 0x5555_5555_9ABC,
            },
            {
                "arena_address": 0x7FFF_F7E1_0000,
                "top": 0x7FFF_F0F0_0000,
                "last_remainder": 0,
            },
        ]
        deltas = HeapHistory.diff_arena_summaries(before, after)
        main = next(d for d in deltas if d["arena_address"] == 0x7FFF_F7E0_0000)
        extra = next(d for d in deltas if d["arena_address"] == 0x7FFF_F7E1_0000)
        assert main["top_delta"] == 0x800
        assert main["last_remainder_after"] == 0x5555_5555_9ABC
        assert main["present_before"] and main["present_after"]
        assert extra["present_before"] is False and extra["present_after"] is True
        assert extra["top_delta"] is None  # no before top

    def test_missing_arena_address_entries_are_skipped(self) -> None:
        deltas = HeapHistory.diff_arena_summaries(
            [{"arena_address": None, "top": 0}],
            [{"arena_address": None, "top": 0}],
        )
        assert deltas == []


class TestLedgerFromArenaBins:
    def test_walks_parsed_fastbins_and_bins(self) -> None:
        fastbins = {
            "verdict": "parsed",
            "fastbins": [
                {
                    "index": 0,
                    "chain": [
                        {"chunk_address": 0x1000, "size": 32},
                        {"chunk_address": 0x1020, "size": 32},
                    ],
                }
            ],
        }
        bins = {
            "verdict": "parsed",
            "bins": [
                {
                    "index": 1,
                    "category": "unsorted",
                    "chain": [{"chunk_address": 0x2000, "size": 96}],
                },
                {
                    "index": 60,
                    "category": "small",
                    "chain": [{"chunk_address": 0x3000, "size": 128}],
                },
            ],
        }
        ledger = HeapHistory._chunk_ledger_from_arena_bins(
            arena_address=0x7FFF_F7E0_0000,
            fastbins_parsed=fastbins,
            bins_parsed=bins,
        )
        assert [e["chunk_address"] for e in ledger] == [0x1000, 0x1020, 0x2000, 0x3000]
        assert ledger[0]["arena_address"] == 0x7FFF_F7E0_0000
        assert ledger[0]["location"] == {"kind": "fastbin", "index": 0}
        assert ledger[2]["location"] == {"kind": "unsorted", "index": 1}
        assert ledger[3]["location"] == {"kind": "small", "index": 60}

    def test_unparsed_bins_produce_empty_ledger(self) -> None:
        assert (
            HeapHistory._chunk_ledger_from_arena_bins(
                arena_address=0x1000,
                fastbins_parsed={"verdict": "unresolvable"},
                bins_parsed={"verdict": "unreadable"},
            )
            == []
        )


class TestLedgerFromTcache:
    def test_walks_parsed_tcache_bins(self) -> None:
        tcache = {
            "verdict": "parsed",
            "bins": [
                {"index": 0, "chain": [{"chunk_address": 0x5000, "size": 24}]},
                {"index": 3, "chain": [{"chunk_address": 0x6000, "size": 48}]},
            ],
        }
        ledger = HeapHistory._chunk_ledger_from_tcache(tcache)
        assert [e["chunk_address"] for e in ledger] == [0x5000, 0x6000]
        assert all(e["arena_address"] is None for e in ledger)
        assert ledger[0]["location"] == {"kind": "tcache", "index": 0}

    def test_uninitialized_tcache_yields_empty_ledger(self) -> None:
        assert (
            HeapHistory._chunk_ledger_from_tcache(
                {"verdict": "uninitialized", "bins": []}
            )
            == []
        )


# ---------------------------------------------------------------------------
# Storage / lookup tests (inject payloads via _remember directly)
# ---------------------------------------------------------------------------
def _fake_payload(
    snapshot_id: str, *, supported: bool = True, chunks: int = 0
) -> Dict[str, Any]:
    return {
        "schema": HEAP_SNAPSHOT_SCHEMA,
        "snapshot_id": snapshot_id,
        "name": None,
        "captured_at": "2026-01-01T00:00:00+00:00",
        "session_id": "sess",
        "inferior": {"pid": 1, "epoch": 1},
        "allocator": {"family": "glibc" if supported else "musl", "version": "2.35"},
        "target_profile": _profile_64(),
        "supported": supported,
        "verdict": "captured" if supported else "unsupported",
        "arenas": [],
        "chunk_ledger": [],
        "tcache_summary": None,
        "component_status": {},
        "summary": {"chunk_count": chunks, "arena_count": 0},
    }


class TestBoundedStorage:
    def test_count_eviction_removes_oldest(self) -> None:
        history = _build_history(max_snapshots=3)
        first = history._remember(name="a", payload=_fake_payload("id_a"))
        history._remember(name="b", payload=_fake_payload("id_b"))
        history._remember(name="c", payload=_fake_payload("id_c"))
        history._remember(name="d", payload=_fake_payload("id_d"))  # evicts "a"
        listing = history.list_heap_snapshots()["parsed"]
        ids = [entry["snapshot_id"] for entry in listing["snapshots"]]
        assert first.snapshot_id not in ids
        assert ids == ["id_b", "id_c", "id_d"]
        assert listing["budget"]["count"] == 3
        assert listing["budget"]["max_count"] == 3

    def test_byte_budget_evicts_before_insert(self) -> None:
        history = _build_history(max_snapshot_bytes=2048)
        big_payload = _fake_payload("id_big", chunks=0)
        big_payload["fill"] = "x" * 900
        history._remember(name=None, payload=big_payload)
        big_payload2 = _fake_payload("id_big2", chunks=0)
        big_payload2["fill"] = "y" * 900
        history._remember(name=None, payload=big_payload2)
        big_payload3 = _fake_payload("id_big3", chunks=0)
        big_payload3["fill"] = "z" * 900
        history._remember(name=None, payload=big_payload3)
        listing = history.list_heap_snapshots()["parsed"]
        ids = [entry["snapshot_id"] for entry in listing["snapshots"]]
        assert "id_big" not in ids  # oldest evicted under byte budget
        assert listing["budget"]["bytes"] <= 2048

    def test_repeat_name_evicts_previous(self) -> None:
        history = _build_history()
        history._remember(name="ckpt", payload=_fake_payload("first_id"))
        history._remember(name="ckpt", payload=_fake_payload("second_id"))
        listing = history.list_heap_snapshots()["parsed"]
        ids = [entry["snapshot_id"] for entry in listing["snapshots"]]
        assert ids == ["second_id"]

    def test_oversized_snapshot_is_rejected(self) -> None:
        history = _build_history(max_snapshot_bytes=64)
        payload = _fake_payload("too_big")
        payload["fill"] = "z" * 1024
        with pytest.raises(HeapSnapshotError, match="bounded history byte budget"):
            history._remember(name=None, payload=payload)


class TestResolve:
    def test_resolve_by_id_and_name(self) -> None:
        history = _build_history()
        history._remember(name="ckpt", payload=_fake_payload("abc123"))
        by_id = history.get_heap_snapshot(identifier="abc123")["parsed"]
        by_name = history.get_heap_snapshot(identifier="ckpt")["parsed"]
        assert by_id["snapshot_id"] == "abc123"
        assert by_name["snapshot_id"] == "abc123"

    def test_get_nonexistent_raises(self) -> None:
        history = _build_history()
        with pytest.raises(HeapSnapshotError, match="not found"):
            history.get_heap_snapshot(identifier="never")

    def test_empty_identifier_raises(self) -> None:
        history = _build_history()
        with pytest.raises(HeapSnapshotError, match="non-empty string"):
            history.get_heap_snapshot(identifier="")


class TestLinkSnapshotToCrash:
    def test_link_deduplicates(self) -> None:
        history = _build_history()
        history._remember(name="ckpt", payload=_fake_payload("snap1"))
        history.link_snapshot_to_crash(identifier="ckpt", crash_id="crash-1")
        second = history.link_snapshot_to_crash(identifier="snap1", crash_id="crash-1")
        assert second["parsed"]["linked_crash_ids"] == ["crash-1"]
        history.link_snapshot_to_crash(identifier="snap1", crash_id="crash-2")
        listing = history.list_heap_snapshots()["parsed"]
        assert listing["snapshots"][0]["linked_crash_ids"] == ["crash-1", "crash-2"]

    def test_empty_crash_id_raises(self) -> None:
        history = _build_history()
        history._remember(name=None, payload=_fake_payload("snap-x"))
        with pytest.raises(HeapSnapshotError, match="non-empty string"):
            history.link_snapshot_to_crash(identifier="snap-x", crash_id="")


class TestDiffThroughStore:
    def test_diff_against_self_is_rejected(self) -> None:
        history = _build_history()
        history._remember(name="only", payload=_fake_payload("s1"))
        with pytest.raises(
            HeapSnapshotError, match="cannot diff a snapshot against itself"
        ):
            history.diff_heap_snapshots(before="only", after="s1")

    def test_diff_unsupported_snapshot_is_unsupported(self) -> None:
        history = _build_history()
        history._remember(
            name="glibc-ok", payload=_fake_payload("s_glibc", supported=True)
        )
        history._remember(name="musl", payload=_fake_payload("s_musl", supported=False))
        result = history.diff_heap_snapshots(before="glibc-ok", after="musl")
        parsed = result["parsed"]
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert parsed["schema"] == HEAP_SNAPSHOT_DIFF_SCHEMA

    def test_diff_two_glibc_snapshots_returns_verdict_diffed(self) -> None:
        history = _build_history()
        before_payload = _fake_payload("s_before")
        before_payload["chunk_ledger"] = [
            {
                "chunk_address": 0x1000,
                "size": 32,
                "location": {"kind": "fastbin", "index": 0},
            },
        ]
        before_payload["arenas"] = [
            {"arena_address": 0x7FFF_F7E0_0000, "top": 0x5000, "last_remainder": 0}
        ]
        after_payload = _fake_payload("s_after")
        after_payload["chunk_ledger"] = [
            {
                "chunk_address": 0x2000,
                "size": 64,
                "location": {"kind": "tcache", "index": 2},
            },
        ]
        after_payload["arenas"] = [
            {"arena_address": 0x7FFF_F7E0_0000, "top": 0x5100, "last_remainder": 0}
        ]
        history._remember(name="a", payload=before_payload)
        history._remember(name="b", payload=after_payload)
        parsed = history.diff_heap_snapshots(before="a", after="b")["parsed"]
        assert parsed["verdict"] == "diffed"
        assert parsed["chunk_diff"]["summary"]["new_count"] == 1
        assert parsed["chunk_diff"]["summary"]["removed_count"] == 1
        assert parsed["arena_deltas"][0]["top_delta"] == 0x100


# ---------------------------------------------------------------------------
# Integration tests (composed stubs simulate a full capture path)
# ---------------------------------------------------------------------------
class TestCaptureNonGlibc:
    def test_musl_snapshot_is_stored_unsupported(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _musl_classification())
        history = _build_history(classification=_musl_classification())
        response = history.capture_heap_snapshot(name="pre-fuzz")
        parsed = response["parsed"]
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert parsed["allocator"]["family"] == "musl"
        assert parsed["chunk_ledger"] == []
        assert parsed["arenas"] == []
        # snapshot is retained under the given name
        listing = history.list_heap_snapshots()["parsed"]
        assert listing["snapshots"][0]["name"] == "pre-fuzz"
        assert listing["snapshots"][0]["allocator_family"] == "musl"


class TestCaptureGlibc:
    def test_glibc_snapshot_flattens_bins_into_ledger(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _glibc_classification())
        arena_addr = 0x7FFF_F7E0_0000
        arenas_response = _canned_backend_response(
            "list_arenas",
            {"arenas": [{"address": arena_addr, "is_main": True, "index": 0}]},
        )
        inspect_arena_by_addr = {
            arena_addr: _canned_backend_response(
                "inspect_arena",
                {
                    "arena_address": arena_addr,
                    "is_main": True,
                    "fields": {
                        "top": 0x5555_5559_0000,
                        "last_remainder": 0,
                        "system_mem": 0x21000,
                        "next": arena_addr,
                    },
                    "verdict": "parsed",
                },
            )
        }
        fastbins_response = _canned_backend_response(
            "inspect_fastbins",
            {
                "arena_address": arena_addr,
                "is_main_arena": True,
                "verdict": "parsed",
                "fastbins": [
                    {"index": 0, "chain": [{"chunk_address": 0x1000, "size": 32}]}
                ],
            },
        )
        bins_response = _canned_backend_response(
            "inspect_bins",
            {
                "arena_address": arena_addr,
                "is_main_arena": True,
                "verdict": "parsed",
                "bins": [
                    {
                        "index": 1,
                        "category": "unsorted",
                        "chain": [{"chunk_address": 0x2000, "size": 96}],
                    }
                ],
            },
        )
        tcache_response = _canned_backend_response(
            "inspect_tcache",
            {
                "verdict": "parsed",
                "bins": [
                    {"index": 0, "chain": [{"chunk_address": 0x3000, "size": 24}]}
                ],
            },
        )
        history = _build_history(
            arenas_response=arenas_response,
            inspect_arena_by_addr=inspect_arena_by_addr,
            fastbins_response=fastbins_response,
            bins_response=bins_response,
            tcache_response=tcache_response,
        )
        response = history.capture_heap_snapshot(name="baseline")
        parsed = response["parsed"]
        assert parsed["supported"] is True
        assert parsed["verdict"] == "captured"
        assert parsed["allocator"]["family"] == "glibc"
        assert parsed["summary"]["arena_count"] == 1
        assert parsed["summary"]["chunk_count"] == 3
        ledger_addrs = [entry["chunk_address"] for entry in parsed["chunk_ledger"]]
        assert ledger_addrs == [0x1000, 0x2000, 0x3000]
        assert parsed["arenas"][0]["top"] == 0x5555_5559_0000
        assert parsed["arenas"][0]["last_remainder"] == 0
        assert parsed["arenas"][0]["system_mem"] == 0x21000
        assert parsed["arenas"][0]["next"] == arena_addr
        assert parsed["arenas"][0]["is_main_arena"] is True
        assert parsed["arenas"][0]["arena_address"] == arena_addr
        assert parsed["tcache_summary"]["verdict"] == "parsed"
        assert parsed["inferior"] == {"pid": 4242, "epoch": 3}

    def test_capture_requires_live_inferior(self) -> None:
        history = _build_history(live=False)
        with pytest.raises(LiveInferiorError):
            history.capture_heap_snapshot()

    def test_capture_rejects_empty_name(self) -> None:
        history = _build_history()
        with pytest.raises(ValueError, match="empty or whitespace"):
            history.capture_heap_snapshot(name="   ")

    def test_capture_rejects_non_positive_arena_cap(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _glibc_classification())
        history = _build_history()
        with pytest.raises(ValueError, match="max_arenas must be between"):
            history.capture_heap_snapshot(max_arenas=0)
        with pytest.raises(ValueError, match="max_arenas must be between"):
            history.capture_heap_snapshot(max_arenas=10**9)
        with pytest.raises(ValueError, match="max_chunks_per_bin must be between"):
            history.capture_heap_snapshot(max_chunks_per_bin=10**9)
        with pytest.raises(ValueError, match="max_ledger_entries must be between"):
            history.capture_heap_snapshot(max_ledger_entries=0)
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            history.capture_heap_snapshot(timeout_seconds=0)

    def test_capture_truncates_arena_walk_explicitly(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _glibc_classification())
        addrs = [0x7FFF_F7E0_0000, 0x7FFF_F7E1_0000, 0x7FFF_F7E2_0000]
        arenas_response = _canned_backend_response(
            "list_arenas",
            {
                "arenas": [
                    {"address": addr, "is_main": i == 0, "index": i}
                    for i, addr in enumerate(addrs)
                ]
            },
        )
        history = _build_history(arenas_response=arenas_response)
        response = history.capture_heap_snapshot(max_arenas=2)
        parsed = response["parsed"]
        assert parsed["verdict"] == "captured"
        assert parsed["summary"]["arena_count"] == 2
        assert parsed["summary"]["arenas_truncated"] is True
        assert [a["arena_address"] for a in parsed["arenas"]] == addrs[:2]
        assert any("max_arenas" in w for w in response.get("warnings") or [])

    def test_capture_truncates_ledger_explicitly(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _glibc_classification())
        arena_addr = 0x7FFF_F7E0_0000
        arenas_response = _canned_backend_response(
            "list_arenas",
            {"arenas": [{"address": arena_addr, "is_main": True, "index": 0}]},
        )
        fastbins_response = _canned_backend_response(
            "inspect_fastbins",
            {
                "verdict": "parsed",
                "fastbins": [
                    {
                        "index": 0,
                        "chain": [
                            {"chunk_address": 0x1000, "size": 32},
                            {"chunk_address": 0x2000, "size": 32},
                            {"chunk_address": 0x3000, "size": 32},
                        ],
                    }
                ],
            },
        )
        history = _build_history(
            arenas_response=arenas_response, fastbins_response=fastbins_response
        )
        response = history.capture_heap_snapshot(max_ledger_entries=1)
        parsed = response["parsed"]
        assert parsed["summary"]["ledger_truncated"] is True
        assert parsed["summary"]["chunk_count"] == 1
        assert any("ledger" in w for w in response.get("warnings") or [])

    def test_capture_deadline_expires(self, monkeypatch) -> None:
        _pin_classification(monkeypatch, _glibc_classification())
        history = _build_history()
        ticks = iter([1000.0, 1000.0 + 400.0])
        monkeypatch.setattr(
            "paomcp.tools.backends.heap_snapshots.time.monotonic",
            lambda: next(ticks, 1000.0 + 400.0),
        )
        with pytest.raises(TimeoutError, match="exceeded its .* budget"):
            history.capture_heap_snapshot(timeout_seconds=300.0)

    def test_capture_rejects_non_positive_chunk_cap(self) -> None:
        history = _build_history()
        with pytest.raises(ValueError, match="must be between"):
            history.capture_heap_snapshot(max_chunks_per_bin=0)


class TestListSchema:
    def test_list_response_carries_schema_and_budget(self) -> None:
        history = _build_history()
        parsed = history.list_heap_snapshots()["parsed"]
        assert parsed["schema"] == HEAP_SNAPSHOT_LIST_SCHEMA
        assert parsed["budget"]["max_count"] == DEFAULT_MAX_HEAP_SNAPSHOTS
        assert parsed["budget"]["max_bytes"] == DEFAULT_MAX_HEAP_SNAPSHOT_BYTES


class TestLinkSchema:
    def test_link_response_carries_schema(self) -> None:
        history = _build_history()
        history._remember(name="snap", payload=_fake_payload("snap-id"))
        parsed = history.link_snapshot_to_crash(identifier="snap", crash_id="crash-9")[
            "parsed"
        ]
        assert parsed["schema"] == HEAP_SNAPSHOT_LINK_SCHEMA
        assert parsed["snapshot_id"] == "snap-id"
        assert parsed["linked_crash_ids"] == ["crash-9"]
