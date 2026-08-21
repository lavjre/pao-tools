"""Tests for the static heap-technique catalog (no debugger required)."""

from paomcp.tools.backends.heap import HEAP_TECHNIQUE_CATALOG_SCHEMA, HeapTools

from tests.test_heap_techniques import build_techniques

REQUIRED_KEYS = {
    "technique",
    "era",
    "family",
    "primitive",
    "evidence_to_look_for",
    "confirm_tools",
    "prerequisites",
    "detectable",
}

# A representative slice of the user-supplied technique list across all three
# eras — the full catalog is 26 house-of techniques plus 8 detector techniques.
EXPECTED_TECHNIQUES = {
    # Classical Malloc Maleficarum (2005)
    "house_of_prime",
    "house_of_mind",
    "house_of_force",
    "house_of_lore",
    "house_of_spirit",
    # Mid-era / FSOP foundations (2.23–2.31)
    "house_of_einherjar",
    "house_of_orange",
    "house_of_rabbit",
    "house_of_roman",
    "house_of_storm",
    "house_of_botcake",
    "house_of_gods",
    "house_of_husk",
    "house_of_corrosion",
    "house_of_pig",
    "house_of_banana",
    # Modern & post-hook (2.34–2.40+)
    "house_of_kiwi",
    "house_of_emma",
    "house_of_apple",
    "house_of_cat",
    "house_of_tangerine",
    "house_of_water",
    "house_of_rust",
    "house_of_muney",
    "house_of_error",
    "house_of_blindness",
    # Detector-surface techniques
    "tcache_poisoning",
    "tcache_double_free",
    "fastbin_dup",
    "unsafe_unlink",
    "overlapping_chunks",
    "mmap_chunk_scan",
    "large_bin_attack",
    "top_chunk_exhaustion",
}


def _entries(parsed: dict):
    return parsed["techniques"]


def _catalog(technique=None):
    """Exercise the HeapTools facade passthrough to the static catalog."""
    heap, _ = build_techniques()
    return heap.heap_technique_catalog(technique=technique)


class TestHeapTechniqueCatalog:
    def test_full_catalog_has_expected_entries(self):
        result = _catalog()
        parsed = result["parsed"]
        assert parsed["schema"] == HEAP_TECHNIQUE_CATALOG_SCHEMA
        assert parsed["technique_filter"] is None
        techniques = {entry["technique"] for entry in _entries(parsed)}
        assert EXPECTED_TECHNIQUES <= techniques
        assert len(_entries(parsed)) >= 34

    def test_every_entry_has_required_keys(self):
        result = _catalog()
        parsed = result["parsed"]
        for entry in _entries(parsed):
            assert REQUIRED_KEYS <= set(entry), entry.get("technique")
            assert isinstance(entry["evidence_to_look_for"], str)
            assert isinstance(entry["confirm_tools"], list)

    def test_filter_returns_single_entry(self):
        result = _catalog(technique="house_of_force")
        parsed = result["parsed"]
        assert parsed["count"] == 1
        assert parsed["technique_filter"] == "house_of_force"
        assert _entries(parsed)[0]["technique"] == "house_of_force"

    def test_unknown_technique_returns_empty(self):
        result = _catalog(technique="not_a_real_technique")
        parsed = result["parsed"]
        assert parsed["count"] == 0
        assert parsed["techniques"] == []

    def test_detectable_flag_flags_detector_surface(self):
        result = _catalog(technique="house_of_force")
        assert _entries(result["parsed"])[0]["detectable"] is True
        result = _catalog(technique="house_of_prime")
        assert _entries(result["parsed"])[0]["detectable"] is False
