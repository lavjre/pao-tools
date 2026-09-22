"""Tests for typed glibc heap corruption classifier."""

from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.foundation import LiveInferiorError
from paomcp.tools.backends.heap import (
    FREE_VALIDATION_SCHEMA,
    HEAP_CORRUPTION_SCHEMA,
    NBINS,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    GlibcHeapCorruption,
    HeapTools,
)

from tests.test_glibc_arenas import (
    ARENA2_ADDR,
    GLIBC_MAPS,
    MAIN_ARENA_ADDR,
    MUSL_MAPS,
    ArenaFakeDebugger,
    ArenaFakeGdb,
    ArenaFakeInspection,
    make_arena_bytes,
)
from tests.test_glibc_regular_bins import (
    HEAP_BASE,
    _bin_head_address,
    _extend_maps_with_heap,
    _libc_version,
    _place_free_chunk,
    _place_self_referential_bins,
)
from tests.test_glibc_tcache import (
    TCACHE_ADDR,
    TcacheAwareFakeDebugger,
    make_tcache_bytes,
    place_tcache_chain,
)


# ---------------------------------------------------------------------------
# Fixture: build a fully-wired GlibcHeapCorruption over the arena fakes.
# ---------------------------------------------------------------------------
def build_corruption(
    *,
    live: bool = True,
    mappings: Optional[List[Dict[str, Any]]] = None,
    libc: Optional[Dict[str, Any]] = None,
    debugger: Optional[ArenaFakeDebugger] = None,
):
    gdb = ArenaFakeGdb(live=live)
    resolved_libc = libc if libc is not None else _libc_version("2.35")
    resolved_maps = mappings if mappings is not None else _extend_maps_with_heap()
    inspection = ArenaFakeInspection(
        gdb,
        mappings=resolved_maps,
        libc=None if libc is None and mappings is not None else resolved_libc,
    )
    dbg = debugger if debugger is not None else ArenaFakeDebugger()
    discovery = AllocatorDiscovery(
        gdb, inspection, dbg, "sess", binary_path_provider=lambda: "/bin/chal"
    )
    arenas = GlibcArenas(gdb, inspection, dbg, "sess", discovery=discovery)
    bins_ = GlibcBins(gdb, inspection, dbg, "sess", discovery=discovery, arenas=arenas)
    glibc = GlibcHeap(gdb, inspection, dbg, "sess", discovery=discovery)
    corruption = GlibcHeapCorruption(
        gdb,
        inspection,
        dbg,
        "sess",
        discovery=discovery,
        arenas=arenas,
        bins=bins_,
        glibc=glibc,
    )
    return corruption, gdb


def _place_chunk_header(
    gdb: ArenaFakeGdb,
    chunk_addr: int,
    size_field: int,
    *,
    prev_size: int = 0,
    trailing_bytes: int = 16,
    byteorder: str = "little",
) -> None:
    """Place a minimal readable chunk header + trailing zeros on 64-bit.

    ``trailing_bytes`` reserves space beyond the 16-byte header so a fastbin
    walker's fd read at ``chunk + 16`` succeeds and terminates on the zero fd.
    """
    total = 16 + trailing_bytes
    buf = bytearray(total)
    buf[0:8] = int(prev_size).to_bytes(8, byteorder)
    buf[8:16] = int(size_field).to_bytes(8, byteorder)
    gdb.place(chunk_addr, bytes(buf))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestAggregateVerdict:
    def test_empty_findings_is_valid(self):
        assert GlibcHeapCorruption._aggregate_verdict([]) == "valid"

    def test_any_corrupted_severity_yields_corrupted(self):
        findings = [{"severity": "unreadable"}, {"severity": "corrupted"}]
        assert GlibcHeapCorruption._aggregate_verdict(findings) == "corrupted"

    def test_only_unreadable_yields_suspicious(self):
        findings = [{"severity": "unreadable"}, {"severity": "unreadable"}]
        assert GlibcHeapCorruption._aggregate_verdict(findings) == "suspicious"


class TestFindingShape:
    def test_finding_with_evidence(self):
        f = GlibcHeapCorruption._finding(
            category="cycle",
            severity="corrupted",
            location={"bin_type": "fastbin", "bin_index": 3},
            reason="loops",
            evidence={"chain_length": 5},
        )
        assert f["category"] == "cycle"
        assert f["severity"] == "corrupted"
        assert f["location"]["bin_index"] == 3
        assert f["evidence"]["chain_length"] == 5

    def test_finding_without_evidence_omits_key(self):
        f = GlibcHeapCorruption._finding(
            category="alignment",
            severity="corrupted",
            location={"chunk_address": 0x101},
            reason="misaligned",
        )
        assert "evidence" not in f


# ---------------------------------------------------------------------------
# classify_heap_corruption -- guards and coarse verdicts
# ---------------------------------------------------------------------------
class TestClassifyGuards:
    def test_requires_live_inferior(self):
        corruption, _ = build_corruption(live=False)
        with pytest.raises(LiveInferiorError):
            corruption.classify_heap_corruption()

    def test_musl_target_returns_unsupported(self):
        corruption, _ = build_corruption(mappings=MUSL_MAPS, libc=None)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        assert parsed["schema"] == HEAP_CORRUPTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert parsed["findings"] == []
        assert parsed["chunks_inspected"] == 0

    def test_missing_bins_offset_yields_unresolvable(self):
        dbg = ArenaFakeDebugger(unresolved_fields=["bins"])
        corruption, _ = build_corruption(debugger=dbg)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        assert parsed["verdict"] == "unresolvable"
        assert parsed["findings"] == []

    def test_invalid_arena_address_raises(self):
        corruption, _ = build_corruption()
        with pytest.raises(ValueError):
            corruption.classify_heap_corruption(arena_address="not-a-number")


class TestClassifyValidCase:
    def test_all_bins_empty_yields_valid(self):
        corruption, gdb = build_corruption()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        assert parsed["schema"] == HEAP_CORRUPTION_SCHEMA
        assert parsed["supported"] is True
        assert parsed["verdict"] == "valid"
        assert parsed["findings"] == []
        assert parsed["chunks_inspected"] == 0
        assert parsed["bins_scanned"] == {"fastbins": 0, "tcache": 0, "regular": 0}
        assert set(parsed["checks_run"]) == set(GlibcHeapCorruption._CHECKS_RUN)
        # Explicit arena_address bookkeeping is preserved.
        assert parsed["is_main_arena"] is True
        assert parsed["arena_address"] == MAIN_ARENA_ADDR

    def test_single_valid_chunk_in_unsorted_bin_no_findings(self):
        corruption, gdb = build_corruption()
        bin_index = 1  # unsorted
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        chunk_addr = HEAP_BASE + 0x400  # aligned
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(chunk_addr)
                bins_array.append(chunk_addr)
            else:
                oh = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(oh)
                bins_array.append(oh)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        # fd=head, bk=head -> length-1 chain, doubly-linked back to head.
        _place_free_chunk(gdb, chunk_addr, size_field=0xA1, fd=head, bk=head)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        assert parsed["verdict"] == "valid"
        assert parsed["findings"] == []
        assert parsed["chunks_inspected"] == 1
        assert parsed["bins_scanned"]["regular"] == 1


# ---------------------------------------------------------------------------
# classify_heap_corruption -- specific corruption categories
# ---------------------------------------------------------------------------
class TestClassifyCycles:
    def test_regular_bin_cycle_flagged(self):
        corruption, gdb = build_corruption()
        bin_index = 3
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        a = HEAP_BASE + 0x400
        b = HEAP_BASE + 0x500
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(a)
                bins_array.append(a)
            else:
                oh = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(oh)
                bins_array.append(oh)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        _place_free_chunk(gdb, a, size_field=0x31, fd=b, bk=head)
        _place_free_chunk(gdb, b, size_field=0x31, fd=a, bk=head)  # cycles
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        assert parsed["verdict"] == "corrupted"
        cycles = [f for f in parsed["findings"] if f["category"] == "cycle"]
        assert len(cycles) >= 1
        assert cycles[0]["severity"] == "corrupted"
        assert cycles[0]["location"]["bin_index"] == bin_index


class TestClassifyAlignment:
    def test_unaligned_chunk_in_fastbin_flagged(self):
        corruption, gdb = build_corruption()
        misaligned = HEAP_BASE + 0x407  # 7 mod 16 -> misaligned
        fastbins = [misaligned] + [0] * 9
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(fastbinsY=fastbins))
        _place_chunk_header(gdb, misaligned, size_field=0x21)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        aligns = [f for f in parsed["findings"] if f["category"] == "alignment"]
        assert len(aligns) >= 1
        assert aligns[0]["severity"] == "corrupted"
        assert aligns[0]["location"]["chunk_address"] == misaligned
        assert aligns[0]["evidence"]["alignment"] == 16
        assert parsed["verdict"] == "corrupted"


class TestClassifyPointerOutOfMapping:
    def test_fastbin_head_outside_mapping_flagged(self):
        corruption, gdb = build_corruption()
        # Aligned but outside every GLIBC_MAPS mapping.
        out_of_map = 0x7FFF_1234_5000
        fastbins = [out_of_map] + [0] * 9
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(fastbinsY=fastbins))
        _place_chunk_header(gdb, out_of_map, size_field=0x21)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        outs = [
            f for f in parsed["findings"] if f["category"] == "pointer_out_of_mapping"
        ]
        assert len(outs) >= 1
        assert outs[0]["severity"] == "corrupted"
        assert outs[0]["location"]["chunk_address"] == out_of_map
        assert parsed["verdict"] == "corrupted"


class TestClassifyUnreadableChunk:
    def test_unreadable_chunk_produces_unreadable_severity(self):
        corruption, gdb = build_corruption()
        target = HEAP_BASE + 0x400
        fastbins = [target] + [0] * 9
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(fastbinsY=fastbins))
        # DELIBERATELY DO NOT place a chunk at ``target`` -> header read fails.
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        unreadables = [
            f for f in parsed["findings"] if f["category"] == "unreadable_chunk"
        ]
        assert len(unreadables) >= 1
        assert unreadables[0]["severity"] == "unreadable"
        # No corrupted-severity finding -> arena verdict downgrades to suspicious.
        assert parsed["verdict"] == "suspicious"


class TestClassifyDuplicate:
    def test_chunk_in_fastbin_and_unsorted_bin_flagged(self):
        corruption, gdb = build_corruption()
        chunk_addr = HEAP_BASE + 0x400
        bin_index = 1
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        fastbins = [chunk_addr] + [0] * 9
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(chunk_addr)
                bins_array.append(chunk_addr)
            else:
                oh = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(oh)
                bins_array.append(oh)
        gdb.place(
            MAIN_ARENA_ADDR,
            make_arena_bytes(fastbinsY=fastbins, bins=bins_array),
        )
        _place_free_chunk(gdb, chunk_addr, size_field=0x21, fd=head, bk=head)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        dups = [f for f in parsed["findings"] if f["category"] == "duplicate_entry"]
        assert len(dups) == 1
        assert dups[0]["location"]["chunk_address"] == chunk_addr
        occ_types = {occ["bin_type"] for occ in dups[0]["evidence"]["occurrences"]}
        assert "fastbin" in occ_types
        assert "unsorted" in occ_types
        assert parsed["verdict"] == "corrupted"


class TestClassifyFdBkInconsistency:
    def test_broken_bk_in_pair_flagged(self):
        corruption, gdb = build_corruption()
        bin_index = 4  # small
        head = _bin_head_address(MAIN_ARENA_ADDR, bin_index)
        a = HEAP_BASE + 0x400
        b = HEAP_BASE + 0x500
        bins_array: List[int] = []
        for i in range(1, NBINS):
            if i == bin_index:
                bins_array.append(a)
                bins_array.append(b)
            else:
                oh = _bin_head_address(MAIN_ARENA_ADDR, i)
                bins_array.append(oh)
                bins_array.append(oh)
        gdb.place(MAIN_ARENA_ADDR, make_arena_bytes(bins=bins_array))
        _place_free_chunk(gdb, a, size_field=0x41, fd=b, bk=head)
        # b.bk should be a; setting it to head deliberately breaks the pair.
        _place_free_chunk(gdb, b, size_field=0x41, fd=head, bk=head)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        fdbks = [
            f for f in parsed["findings"] if f["category"] == "fd_bk_inconsistency"
        ]
        assert len(fdbks) == 1
        assert fdbks[0]["severity"] == "corrupted"
        assert fdbks[0]["location"]["chunk_address"] == b
        assert fdbks[0]["evidence"]["expected_bk"] == a
        assert fdbks[0]["evidence"]["observed_bk"] == head
        assert parsed["verdict"] == "corrupted"


class TestClassifyExplicitArena:
    def test_non_main_arena_flag_and_address_reported(self):
        corruption, gdb = build_corruption()
        heads = list(range(1, NBINS))
        bins_array: List[int] = []
        for i in heads:
            oh = _bin_head_address(ARENA2_ADDR, i)
            bins_array.append(oh)
            bins_array.append(oh)
        gdb.place(ARENA2_ADDR, make_arena_bytes(bins=bins_array))
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = corruption.classify_heap_corruption(arena_address=f"0x{ARENA2_ADDR:x}")
        parsed = result["parsed"]
        assert parsed["arena_address"] == ARENA2_ADDR
        assert parsed["is_main_arena"] is False
        assert parsed["verdict"] == "valid"


# ---------------------------------------------------------------------------
# classify_heap_corruption -- additive technique hints (technique analysis)
# ---------------------------------------------------------------------------
TCACHE_MAX_BINS = 64


def _tcache(counts=None, entries=None):
    pad = lambda lst: (
        None if lst is None else list(lst) + [0] * (TCACHE_MAX_BINS - len(lst))
    )
    return make_tcache_bytes(counts=pad(counts), entries=pad(entries))


class TestClassifyTechniqueHints:
    def test_clean_heap_has_empty_hints_and_valid_verdict(self):
        corruption, gdb = build_corruption(debugger=TcacheAwareFakeDebugger())
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        # Verdicts are unchanged by the additive hints.
        assert parsed["verdict"] == "valid"
        assert parsed["findings"] == []
        hints = parsed["technique_hints"]
        assert isinstance(hints, list)
        assert len(hints) > 0
        assert all(entry["present"] is False for entry in hints)

    def test_tcache_double_free_surfaces_hint_without_changing_verdict(self):
        corruption, gdb = build_corruption(debugger=TcacheAwareFakeDebugger())
        a = HEAP_BASE + 0x300
        gdb.place(TCACHE_ADDR, _tcache(counts=[1, 1], entries=[a, a]))
        place_tcache_chain(gdb, [a])
        place_tcache_chain(gdb, [a])
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = corruption.classify_heap_corruption()
        parsed = result["parsed"]
        hints = {h["technique"]: h for h in parsed["technique_hints"]}
        assert hints["tcache_double_free"]["present"] is True
        # The corruption-classifier duplicate_entry finding is what drives the verdict; the
        # hints are purely additive.
        assert parsed["verdict"] == "corrupted"
        dups = [f for f in parsed["findings"] if f["category"] == "duplicate_entry"]
        assert len(dups) >= 1


# ---------------------------------------------------------------------------
# validate_free
# ---------------------------------------------------------------------------
class TestValidateFreeGuards:
    def test_musl_target_returns_unsupported(self):
        corruption, _ = build_corruption(mappings=MUSL_MAPS, libc=None)
        result = corruption.validate_free(address=hex(HEAP_BASE + 0x10))
        parsed = result["parsed"]
        assert parsed["schema"] == FREE_VALIDATION_SCHEMA
        assert parsed["verdict"] == "unsupported"
        assert parsed["failures"] == []

    def test_unreadable_chunk_yields_unreadable_verdict(self):
        corruption, gdb = build_corruption()
        # Place empty bins so tcache/fastbins queries succeed but no chunk
        # sits at the target address -> header read fails.
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = corruption.validate_free(address=hex(HEAP_BASE + 0x410))
        parsed = result["parsed"]
        assert parsed["verdict"] == "unreadable"
        assert "readable" in parsed["failures"]


class TestValidateFreeSafe:
    def test_valid_chunk_not_in_bins_yields_safe(self):
        # Both membership probes must be *checked* for a safe verdict:
        # resolve the tcache layout (TcacheAwareFakeDebugger + a placed
        # empty tcache struct) so tcache membership is verified, not merely
        # unanswerable.
        corruption, gdb = build_corruption(debugger=TcacheAwareFakeDebugger())
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        chunk_addr = HEAP_BASE + 0x400  # aligned to 16
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        result = corruption.validate_free(address=hex(user_addr))
        parsed = result["parsed"]
        assert parsed["verdict"] == "safe"
        assert parsed["failures"] == []
        assert parsed["chunk_summary"]["chunk_address"] == chunk_addr
        # Both membership probes ran (checked=True) and found nothing.
        assert parsed["membership"]["fastbins"] == {"found": False, "checked": True}
        assert parsed["membership"]["tcache"] == {"found": False, "checked": True}
        assert parsed["checks"]["not_in_tcache"]["ok"] is True
        assert parsed["checks"]["not_in_fastbins"]["ok"] is True


class TestValidateFreeUnverifiedMembership:
    def test_unresolvable_tcache_never_yields_safe(self):
        # Regression: when the tcache inspector cannot answer (no debug
        # symbols -> unresolvable), absent membership is not evidence of
        # absence.  The verdict must be bounded, never "safe".
        corruption, gdb = build_corruption()  # plain fake: tcache unresolved
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        chunk_addr = HEAP_BASE + 0x400
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        result = corruption.validate_free(address=hex(user_addr))
        parsed = result["parsed"]
        assert parsed["verdict"] == "suspicious"
        assert "not_in_tcache_unverified" in parsed["failures"]
        assert parsed["checks"]["not_in_tcache"]["ok"] is False
        assert "membership_unverified" in parsed["checks"]["not_in_tcache"]["reason"]
        assert parsed["membership"]["tcache"]["checked"] is False
        # fastbins still answered, so only the tcache probe is unverified.
        assert parsed["membership"]["fastbins"]["checked"] is True
        assert parsed["checks"]["not_in_fastbins"]["ok"] is True

    def test_unresolvable_fastbins_never_yields_safe(self):
        corruption, gdb = build_corruption(
            debugger=ArenaFakeDebugger(unresolved_fields=["fastbinsY"])
        )
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        chunk_addr = HEAP_BASE + 0x400
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        result = corruption.validate_free(address=hex(user_addr))
        parsed = result["parsed"]
        assert parsed["verdict"] == "suspicious"
        assert "not_in_fastbins_unverified" in parsed["failures"]
        assert parsed["membership"]["fastbins"]["checked"] is False

    def test_double_free_still_takes_precedence_over_unverified(self):
        # A found membership is still double_free even when the *other*
        # probe could not be answered.
        corruption, gdb = build_corruption()  # tcache unresolved
        chunk_addr = HEAP_BASE + 0x400
        fastbins = [chunk_addr] + [0] * 9
        heads = list(range(1, NBINS))
        bins_array: List[int] = []
        for i in heads:
            oh = _bin_head_address(MAIN_ARENA_ADDR, i)
            bins_array.append(oh)
            bins_array.append(oh)
        gdb.place(
            MAIN_ARENA_ADDR,
            make_arena_bytes(fastbinsY=fastbins, bins=bins_array),
        )
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        parsed = corruption.validate_free(address=hex(user_addr))["parsed"]
        assert parsed["verdict"] == "double_free"
        assert "not_in_fastbins" in parsed["failures"]


class TestValidateFreeCorrupted:
    def test_misaligned_chunk_yields_corrupted(self):
        corruption, gdb = build_corruption()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        misaligned_chunk = HEAP_BASE + 0x407  # not aligned to 16
        _place_chunk_header(gdb, misaligned_chunk, size_field=0x21)
        user_addr = misaligned_chunk + 16
        result = corruption.validate_free(address=hex(user_addr))
        parsed = result["parsed"]
        assert parsed["verdict"] == "corrupted"
        assert "alignment" in parsed["failures"]


class TestValidateFreeDoubleFree:
    def test_chunk_in_fastbins_yields_double_free(self):
        corruption, gdb = build_corruption()
        chunk_addr = HEAP_BASE + 0x400
        fastbins = [chunk_addr] + [0] * 9
        # Build a self-referential bins block alongside a populated fastbin.
        heads = list(range(1, NBINS))
        bins_array: List[int] = []
        for i in heads:
            oh = _bin_head_address(MAIN_ARENA_ADDR, i)
            bins_array.append(oh)
            bins_array.append(oh)
        gdb.place(
            MAIN_ARENA_ADDR,
            make_arena_bytes(fastbinsY=fastbins, bins=bins_array),
        )
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        result = corruption.validate_free(address=hex(user_addr))
        parsed = result["parsed"]
        assert parsed["verdict"] == "double_free"
        assert "not_in_fastbins" in parsed["failures"]
        assert parsed["membership"]["fastbins"]["found"] is True
        assert parsed["membership"]["fastbins"]["bin_index"] == 0


class TestValidateFreeExplicitChunkKind:
    def test_address_kind_chunk_still_verifies_membership(self):
        corruption, gdb = build_corruption(debugger=TcacheAwareFakeDebugger())
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        chunk_addr = HEAP_BASE + 0x400
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        # Pass the chunk address directly rather than the user pointer.
        result = corruption.validate_free(address=hex(chunk_addr), address_kind="chunk")
        parsed = result["parsed"]
        assert parsed["verdict"] == "safe"
        assert parsed["membership"]["tcache"]["checked"] is True


# ---------------------------------------------------------------------------
# HeapTools facade proxy
# ---------------------------------------------------------------------------
class TestHeapToolsFacadeCorruption:
    def _make_facade(self):
        gdb = ArenaFakeGdb(live=True)
        inspection = ArenaFakeInspection(
            gdb,
            mappings=_extend_maps_with_heap(),
            libc=_libc_version("2.35"),
        )
        debugger = TcacheAwareFakeDebugger()
        facade = HeapTools.__new__(HeapTools)
        facade.discovery = AllocatorDiscovery(
            gdb,
            inspection,
            debugger,
            "sess",
            binary_path_provider=lambda: "/bin/chal",
        )
        facade.glibc = GlibcHeap(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        facade.arenas = GlibcArenas(
            gdb, inspection, debugger, "sess", discovery=facade.discovery
        )
        facade.bins = GlibcBins(
            gdb,
            inspection,
            debugger,
            "sess",
            discovery=facade.discovery,
            arenas=facade.arenas,
        )
        facade.corruption = GlibcHeapCorruption(
            gdb,
            inspection,
            debugger,
            "sess",
            discovery=facade.discovery,
            arenas=facade.arenas,
            bins=facade.bins,
            glibc=facade.glibc,
        )
        return facade, gdb

    def test_facade_routes_classify_heap_corruption(self):
        facade, gdb = self._make_facade()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        result = facade.classify_heap_corruption()
        assert result["parsed"]["schema"] == HEAP_CORRUPTION_SCHEMA
        assert result["parsed"]["verdict"] == "valid"

    def test_facade_routes_validate_free(self):
        facade, gdb = self._make_facade()
        _place_self_referential_bins(gdb, MAIN_ARENA_ADDR)
        gdb.place(TCACHE_ADDR, make_tcache_bytes())
        chunk_addr = HEAP_BASE + 0x400
        _place_chunk_header(gdb, chunk_addr, size_field=0x21)
        user_addr = chunk_addr + 16
        result = facade.validate_free(address=hex(user_addr))
        assert result["parsed"]["schema"] == FREE_VALIDATION_SCHEMA
        assert result["parsed"]["verdict"] == "safe"
