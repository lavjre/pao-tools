"""Tests for Phase 0: the AllocatorAnalyzer seam + family dispatch.

Covers the pure dispatch (``analyzer_for_family``), the Protocol surface, the
facade's per-call resolution (``HeapTools._analyzer``), the TTL family cache,
and the critical guarantee that forwarding reads facade component attributes
at call time -- so the test pattern of swapping ``facade.glibc`` / friends
after construction keeps working through the seam.
"""

from typing import Any, Dict, Optional

import pytest

from paomcp.tools.backends.heap import CHUNK_INSPECTION_SCHEMA, HeapTools
from paomcp.tools.backends.heap_analyzers import (
    AllocatorAnalyzer,
    GlibcAnalyzer,
    UnsupportedAnalyzer,
    analyzer_for_family,
)

from tests.test_detect_uaf import build_uaf
from tests.test_glibc_chunks import MUSL_MAPS

# The twelve allocator-layout-dependent facade methods.  These are the
# contract every analyzer must expose; everything else on the facade is
# family-agnostic and stays direct.
ANALYZER_METHODS = (
    "inspect_chunk",
    "validate_chunk",
    "list_arenas",
    "inspect_arena",
    "inspect_fastbins",
    "inspect_tcache",
    "inspect_bins",
    "classify_heap_corruption",
    "validate_free",
    "analyze_heap_technique",
    "detect_uaf",
    "capture_heap_snapshot",
)


class StubDiscovery:
    """Stand-in for AllocatorDiscovery that only classifies the family."""

    def __init__(self, family: Optional[str]) -> None:
        self._family = family

    def detect_allocator(self, *, cancellation_check=None) -> Dict[str, Any]:
        return {"parsed": {"family": self._family}}


class StubGlibcComponent:
    """Proves the analyzer forwards to the *current* facade attribute."""

    def __init__(self) -> None:
        self.calls = []

    def inspect_chunk(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"stub": True}


# ---------------------------------------------------------------------------
# Pure dispatch.
# ---------------------------------------------------------------------------
class TestAnalyzerForFamily:
    def test_glibc_resolves_to_glibc_analyzer(self):
        analyzer = analyzer_for_family("glibc", facade=None)
        assert isinstance(analyzer, GlibcAnalyzer)
        assert analyzer.family == "glibc"

    @pytest.mark.parametrize("family", ["musl", "jemalloc", "scudo", "unknown", None])
    def test_non_glibc_families_resolve_to_unsupported(self, family):
        analyzer = analyzer_for_family(family, facade=None)
        assert isinstance(analyzer, UnsupportedAnalyzer)

    def test_both_analyzers_implement_the_full_surface(self):
        glibc = analyzer_for_family("glibc", facade=None)
        unsupported = analyzer_for_family("musl", facade=None)
        for analyzer in (glibc, unsupported):
            assert isinstance(analyzer, AllocatorAnalyzer), type(analyzer)
            for name in ANALYZER_METHODS:
                assert callable(getattr(analyzer, name)), name


# ---------------------------------------------------------------------------
# Facade resolution + cache.
# ---------------------------------------------------------------------------
class TestFacadeDispatch:
    def _facade(self, family: Optional[str]) -> HeapTools:
        heap, _ = build_uaf()
        heap.discovery = StubDiscovery(family)  # type: ignore[assignment]
        heap._family_cache = None  # ensure a fresh detection
        return heap

    def test_analyzer_resolves_from_detected_family(self):
        assert isinstance(self._facade("glibc")._analyzer(), GlibcAnalyzer)
        assert isinstance(self._facade("musl")._analyzer(), UnsupportedAnalyzer)

    def test_detected_family_is_ttl_cached(self):
        heap, _ = build_uaf()
        heap.discovery = StubDiscovery("glibc")  # type: ignore[assignment]
        heap._family_cache = None
        assert heap._detected_family() == "glibc"

        # Detection now disagrees, but the cache still serves glibc.
        heap.discovery = StubDiscovery("musl")  # type: ignore[assignment]
        assert heap._detected_family() == "glibc"

        # An expired cache re-detects.
        heap._family_cache = (0.0, "glibc")
        assert heap._detected_family() == "musl"

    def test_unresolvable_family_is_cached_as_none(self):
        heap, _ = build_uaf()
        heap.discovery = StubDiscovery(None)  # type: ignore[assignment]
        heap._family_cache = None
        assert heap._detected_family() is None
        assert heap._family_cache[1] is None


# ---------------------------------------------------------------------------
# Forwarding semantics.
# ---------------------------------------------------------------------------
class TestForwarding:
    def test_forwarding_reads_live_facade_attribute(self):
        """A component swapped onto the facade after construction is used."""
        heap, _ = build_uaf()
        heap.discovery = StubDiscovery("glibc")  # type: ignore[assignment]
        heap._family_cache = None
        stub = StubGlibcComponent()
        heap.glibc = stub  # type: ignore[assignment]

        result = heap.inspect_chunk(address="0x1234")
        assert result == {"stub": True}
        assert stub.calls, "analyzer must reach the live facade.glibc"
        assert stub.calls[0]["address"] == "0x1234"

    def test_unsupported_family_parity_through_facade(self):
        """A musl target returns the honest unsupported verdict end-to-end."""
        heap, _ = build_uaf(mappings=MUSL_MAPS, libc=None)
        result = heap.inspect_chunk(address="0x7f0000000000", address_kind="chunk")
        assert result["success"] is True
        parsed = result["parsed"]
        assert parsed["schema"] == CHUNK_INSPECTION_SCHEMA
        assert parsed["supported"] is False
        assert parsed["verdict"] == "unsupported"
        assert "glibc only" in parsed["reason"]
