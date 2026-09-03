"""Static glibc heap-technique catalog + additive hint runner.

Split out of :mod:`paomcp.tools.backends.heap_techniques`: the technique
schema constants, the static ``_TECHNIQUE_CATALOG`` (classical / mid-era /
post-hook families), and :func:`technique_hints_for`, which runs the cheap
detectors over already-collected inspector output for
``GlibcHeapCorruption.classify_heap_corruption``.  No debugger needed except
the caller-supplied ``ReadFn``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.heap_schemas import PARSER_VERSION
from paomcp.tools.backends.heap_technique_context import (
    HINT_TECHNIQUES,
    ReadFn,
    TechniqueContext,
    _byteorder,
)
from paomcp.tools.backends.heap_technique_detectors import DETECTORS

HEAP_TECHNIQUE_SCHEMA = "pao.glibc-heap-technique.v1"
UAF_SCAN_SCHEMA = "pao.glibc-uaf-scan.v1"
HEAP_TECHNIQUE_CATALOG_SCHEMA = "pao.glibc-heap-technique-catalog.v1"


def _valid_detector_names(names: Sequence[str]) -> List[str]:
    unknown = [name for name in names if name not in DETECTORS]
    if unknown:
        raise ValueError(
            f"unknown heap technique(s) {unknown!r}; expected one of "
            f"{sorted(DETECTORS)}"
        )
    return list(names)


def technique_hints_for(
    *,
    fb_parsed: Mapping[str, Any],
    tc_parsed: Mapping[str, Any],
    bn_parsed: Mapping[str, Any],
    classification: Mapping[str, Any],
    mappings: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    top_chunk: Optional[Mapping[str, Any]],
    read: ReadFn,
) -> List[Dict[str, Any]]:
    """Run the additive hint detectors over already-collected inspector output.

    Used by ``GlibcHeapCorruption.classify_heap_corruption`` so its verdicts
    stay unchanged while a bounded set of technique hints rides along.  Only
    the cheap detectors run here (see :data:`HINT_TECHNIQUES`); no UAF scan,
    no overlap enumeration.
    """
    pointer_bits = profile.get("pointer_bits")
    if pointer_bits not in (32, 64):
        pointer_bits = 64
    size_sz = pointer_bits // 8
    byteorder = _byteorder(profile)
    ctx = TechniqueContext(
        classification=dict(classification),
        mappings=[dict(m) for m in mappings],
        profile=dict(profile),
        component_status={},
        fb_parsed=fb_parsed,
        tc_parsed=tc_parsed,
        bn_parsed=bn_parsed,
        pointer_bits=pointer_bits,
        size_sz=size_sz,
        byteorder=byteorder,
        alignment=2 * size_sz,
        safe_linking=classification.get("safe_linking", "unknown"),
        top_chunk=top_chunk,
    )
    return [DETECTORS[name](ctx, read) for name in HINT_TECHNIQUES if name in DETECTORS]


# ---------------------------------------------------------------------------
# Technique catalog (static)
# ---------------------------------------------------------------------------
_TECHNIQUE_CATALOG: Tuple[Dict[str, Any], ...] = (
    # --- Classical era: Malloc Maleficarum (2005) ------------------------
    {
        "technique": "house_of_prime",
        "era": "classical",
        "family": "fake_free",
        "primitive": "free() a forged chunk via an attacker-controlled chunk pointer",
        "evidence_to_look_for": "a free chunk whose header does not correspond to any real allocation",
        "confirm_tools": ["inspect_chunk", "validate_free"],
        "prerequisites": "glibc <= 2.4 (patched in 2.4)",
        "detectable": False,
    },
    {
        "technique": "house_of_mind",
        "era": "classical",
        "family": "nested_arena",
        "primitive": "free() into a fake arena's bins via the NON_MAIN_ARENA bit",
        "evidence_to_look_for": "NON_MAIN_ARENA chunks whose arena pointer lands in attacker memory",
        "confirm_tools": ["inspect_chunk", "list_arenas", "inspect_arena"],
        "prerequisites": "glibc 2.4 - 2.11 (patched in 2.11)",
        "detectable": False,
    },
    {
        "technique": "house_of_force",
        "era": "classical",
        "family": "top_chunk",
        "primitive": "overwrite the top chunk size so remaindering wraps and malloc returns arbitrary addresses",
        "evidence_to_look_for": "top chunk size field wildly large / beyond its mapping / misaligned",
        "confirm_tools": ["inspect_arena", "inspect_chunk"],
        "prerequisites": "glibc < 2.29 (top chunk size sanity check added in 2.29)",
        "detectable": True,
    },
    {
        "technique": "house_of_lore",
        "era": "classical",
        "family": "smallbin",
        "primitive": "forge a smallbin bk chain so a malloc pops a fake chunk from an arbitrary address",
        "evidence_to_look_for": "regular-bin bk (or fd) pointing outside the heap",
        "confirm_tools": ["inspect_bins", "inspect_chunk"],
        "prerequisites": "glibc < 2.26 (smallbin safe-unlink bck->fd != victim added in 2.26)",
        "detectable": True,
    },
    {
        "technique": "house_of_spirit",
        "era": "classical",
        "family": "fake_free",
        "primitive": "free() a fake chunk placed on the stack so the next malloc returns a stack address",
        "evidence_to_look_for": "a walked free chunk whose header lies outside every mapping",
        "confirm_tools": [
            "inspect_chunk",
            "inspect_fastbins",
            "inspect_tcache",
            "inspect_bins",
        ],
        "prerequisites": "still functional when alignment + size constraints are met",
        "detectable": True,
    },
    # --- Mid-era and FSOP foundations (2.23 - 2.31) -----------------------
    {
        "technique": "house_of_einherjar",
        "era": "mid",
        "family": "backward_consolidation",
        "primitive": "off-by-one null byte clears PREV_INUSE + forged prev_size -> backward consolidation overlaps a live allocation",
        "evidence_to_look_for": "free chunk with PREV_INUSE clear whose claimed prev_size exceeds its physical predecessor's size",
        "confirm_tools": ["inspect_chunk", "inspect_bins"],
        "prerequisites": "off-by-one / null-byte write primitive",
        "detectable": True,
    },
    {
        "technique": "house_of_orange",
        "era": "mid",
        "family": "fsop",
        "primitive": "shrink top chunk -> sysmalloc -> unsorted bin; FSOP via _IO_list_all / _IO_str_overflow",
        "evidence_to_look_for": "top size < requested, _IO_list_all pointing into the heap, a fake vtable",
        "confirm_tools": ["inspect_arena", "inspect_bins", "inspect_address"],
        "prerequisites": "glibc < 2.34 (hook era) or file-only FSOP with _IO_str_jumps",
        "detectable": False,
    },
    {
        "technique": "house_of_rabbit",
        "era": "mid",
        "family": "fastbin_bridging",
        "primitive": "fastbin -> largebin bridging for one-gadget / arbitrary write",
        "evidence_to_look_for": "fastbin chunk whose fd links into a largebin region",
        "confirm_tools": ["inspect_fastbins", "inspect_bins"],
        "prerequisites": "glibc 2.23 era",
        "detectable": False,
    },
    {
        "technique": "house_of_roman",
        "era": "mid",
        "family": "partial_overwrite",
        "primitive": "1-2 byte partial overwrites + unsorted remaindering for a stack pivot",
        "evidence_to_look_for": "unsorted bin chunks whose fd/bk differ from the intact header layout",
        "confirm_tools": ["inspect_bins", "inspect_chunk"],
        "prerequisites": "glibc 2.23 - 2.27",
        "detectable": False,
    },
    {
        "technique": "house_of_storm",
        "era": "mid",
        "family": "bin_write",
        "primitive": "combine an unsorted attack with a largebin sorting write for an arbitrary write",
        "evidence_to_look_for": "largebin chunk whose nextsize/size ordering is corrupted",
        "confirm_tools": ["inspect_bins"],
        "prerequisites": "glibc < 2.31 (safe-linking era boundary)",
        "detectable": False,
    },
    {
        "technique": "house_of_botcake",
        "era": "mid",
        "family": "tcache",
        "primitive": "fill tcache to capacity + unsorted backward consolidation so a chunk is live in tcache AND unsorted",
        "evidence_to_look_for": "same chunk present in tcache and in the unsorted/small bins",
        "confirm_tools": ["inspect_tcache", "inspect_bins", "validate_free"],
        "prerequisites": "glibc 2.26 - 2.34+ (tcache era)",
        "detectable": True,
    },
    {
        "technique": "house_of_gods",
        "era": "mid",
        "family": "arena",
        "primitive": "corrupt malloc_state.next to pivot arena traversal / consolidation",
        "evidence_to_look_for": "arena next pointer outside the known arena chain",
        "confirm_tools": ["list_arenas", "inspect_arena"],
        "prerequisites": "glibc with non-main arenas reachable",
        "detectable": False,
    },
    {
        "technique": "house_of_husk",
        "era": "mid",
        "family": "printf",
        "primitive": "unsorted attack overwrites __printf_arginfo_table / __printf_function_table for RIP control",
        "evidence_to_look_for": "printf hook tables pointing into the heap",
        "confirm_tools": ["inspect_address", "inspect_bins"],
        "prerequisites": "glibc 2.24 - 2.27",
        "detectable": False,
    },
    {
        "technique": "house_of_corrosion",
        "era": "mid",
        "family": "stderr",
        "primitive": "overwrite stderr buffer descriptors / locale translation tables for an FSOP write",
        "evidence_to_look_for": "_IO_stdfile buffers or locale tables redirected into the heap",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.23 - 2.26 era",
        "detectable": False,
    },
    {
        "technique": "house_of_pig",
        "era": "mid",
        "family": "fsop",
        "primitive": "largebin + tcache stashing unlink + _IO_str_jumps for file-only FSOP",
        "evidence_to_look_for": "largebin chunk with corrupted nextsize + a forged _IO_FILE",
        "confirm_tools": ["inspect_bins", "inspect_address"],
        "prerequisites": "glibc 2.30 - 2.33",
        "detectable": False,
    },
    {
        "technique": "house_of_banana",
        "era": "mid",
        "family": "ld",
        "primitive": "largebin attack on _rtld_global (DT_FINI_ARRAY) for exit-time RIP control",
        "evidence_to_look_for": "largebin chunk whose nextsize write targets _rtld_global",
        "confirm_tools": ["inspect_bins"],
        "prerequisites": "glibc 2.23 - 2.30 era",
        "detectable": False,
    },
    # --- Modern post-hook era (2.34 - 2.40+) ------------------------------
    {
        "technique": "house_of_kiwi",
        "era": "modern",
        "family": "assert",
        "primitive": "abuse __malloc_assert + _IO_file_jumps for a file-based write",
        "evidence_to_look_for": "a forced malloc assertion path with a forged _IO_FILE",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_emma",
        "era": "modern",
        "family": "wide_vtable",
        "primitive": "_IO_cookie_jumps / _IO_obstack_jumps wide-vtable hijack that bypasses pointer guard",
        "evidence_to_look_for": "a _IO_FILE whose vtable / wide_data points into the heap",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_apple",
        "era": "modern",
        "family": "wide_vtable",
        "primitive": "wide-vtable hijack via _IO_wfile_overflow / _IO_wfile_underflow / _IO_wfile_seekoff",
        "evidence_to_look_for": "a forged _IO_wide_data / wide vtable pointer on a FILE",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+ (variants 1 / 2 / 3)",
        "detectable": False,
    },
    {
        "technique": "house_of_cat",
        "era": "modern",
        "family": "wide_vtable",
        "primitive": "_IO_wfile_seekoff + forged _IO_wide_data for an FSOP write",
        "evidence_to_look_for": "a FILE whose _wide_data / vtable lands in attacker memory",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_tangerine",
        "era": "modern",
        "family": "top_chunk",
        "primitive": "modernized house of orange: shrink the top chunk to get a chunk into the unsorted bin",
        "evidence_to_look_for": "top chunk smaller than expected with an adjacent unsorted chunk",
        "confirm_tools": ["inspect_arena", "inspect_bins"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_water",
        "era": "modern",
        "family": "tcache_struct",
        "primitive": "forge metadata into the unsorted bin to capture tcache_perthread_struct, leakless",
        "evidence_to_look_for": "a free chunk in the unsorted bin that aliases the tcache struct region",
        "confirm_tools": ["inspect_bins", "inspect_tcache"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_rust",
        "era": "modern",
        "family": "tcache",
        "primitive": "Tcache Stashing Unlink+ combined with a largebin attack, leakless against full PIE/ASLR",
        "evidence_to_look_for": "tcache stash victim + corrupted largebin ordering",
        "confirm_tools": ["inspect_tcache", "inspect_bins"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_muney",
        "era": "modern",
        "family": "mmap",
        "primitive": "corrupt an mmap chunk header + the arena mmap threshold for an arbitrary munmap",
        "evidence_to_look_for": "IS_MMAPPED chunk headers / arena threshold corruption",
        "confirm_tools": ["inspect_chunk", "inspect_arena"],
        "prerequisites": "glibc 2.34+",
        "detectable": True,
    },
    {
        "technique": "house_of_error",
        "era": "modern",
        "family": "exit_handlers",
        "primitive": "_IO_mem_sync overwrites TLS exit handlers for exit-time RIP control",
        "evidence_to_look_for": "a forged _IO_FILE on the mem_sync path",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    {
        "technique": "house_of_blindness",
        "era": "modern",
        "family": "blind",
        "primitive": "blind layout alignment on file pointers for a fully-blind FSOP chain",
        "evidence_to_look_for": "FILE pointers aligned into attacker-controlled memory",
        "confirm_tools": ["inspect_address"],
        "prerequisites": "glibc 2.34+",
        "detectable": False,
    },
    # --- Analyzer-detectable primitives (detector-backed) ----------------
    {
        "technique": "tcache_poisoning",
        "era": "modern",
        "family": "tcache",
        "primitive": "overwrite a tcache entry's next pointer so a malloc returns an arbitrary address",
        "evidence_to_look_for": "a tcache next pointer targeting a writable non-heap address",
        "confirm_tools": ["inspect_tcache", "inspect_chunk"],
        "prerequisites": "tcache-enabled glibc (2.26+)",
        "detectable": True,
    },
    {
        "technique": "tcache_double_free",
        "era": "modern",
        "family": "tcache",
        "primitive": "free the same chunk twice into tcache so two mallocs return one region",
        "evidence_to_look_for": "same chunk twice in one tcache chain / in two bins / walked length > count",
        "confirm_tools": ["inspect_tcache", "validate_free"],
        "prerequisites": "tcache-enabled glibc (2.26+)",
        "detectable": True,
    },
    {
        "technique": "fastbin_dup",
        "era": "classical",
        "family": "fastbin",
        "primitive": "double-free a fastbin chunk (or corrupt its fd) so one region is returned twice",
        "evidence_to_look_for": "a fastbin chain cycle or a chunk in two fastbin bins",
        "confirm_tools": ["inspect_fastbins", "validate_free"],
        "prerequisites": "any glibc with fastbins",
        "detectable": True,
    },
    {
        "technique": "unsafe_unlink",
        "era": "classical",
        "family": "bin_write",
        "primitive": "forge fd/bk so the unlink write lands a 4/8-byte write at an arbitrary address",
        "evidence_to_look_for": "a free chunk's fd/bk targeting a writable non-heap mapping",
        "confirm_tools": ["inspect_bins", "inspect_chunk"],
        "prerequisites": "glibc with unlink still writable (pre-2.26 hardening or a bypass)",
        "detectable": True,
    },
    {
        "technique": "overlapping_chunks",
        "era": "modern",
        "family": "overlap",
        "primitive": "overlap two live allocations (OOB write / shrink / split abuse) so editing one edits the other",
        "evidence_to_look_for": "free ranges that intersect, or a prev_size field set while PREV_INUSE is set",
        "confirm_tools": ["inspect_bins", "inspect_chunk", "validate_free"],
        "prerequisites": "any glibc; requires an OOB write or size spoof",
        "detectable": True,
    },
    {
        "technique": "top_chunk_exhaustion",
        "era": "modern",
        "family": "top_chunk",
        "primitive": "heap near exhaustion so a large malloc triggers sysmalloc / mmap that can be redirected",
        "evidence_to_look_for": "top chunk with little room left in its mapping, or an unusually small top",
        "confirm_tools": ["inspect_arena", "inspect_chunk"],
        "prerequisites": "any glibc",
        "detectable": True,
    },
    {
        "technique": "large_bin_attack",
        "era": "mid",
        "family": "bin_write",
        "primitive": "forge a largebin chunk's fd_nextsize/bk_nextsize so the size-sorted insert writes an arbitrary value",
        "evidence_to_look_for": "largebin chunks whose fd_nextsize/bk_nextsize differ from sorted order",
        "confirm_tools": ["inspect_bins"],
        "prerequisites": "glibc with largebin sorting (any)",
        "detectable": True,
    },
    {
        "technique": "mmap_chunk_scan",
        "era": "modern",
        "family": "mmap",
        "primitive": "identify IS_MMAPPED allocations; arena-threshold corruption can free them back into the heap",
        "evidence_to_look_for": "IS_MMAPPED chunk headers in writable anon mappings",
        "confirm_tools": ["inspect_chunk", "memory_maps"],
        "prerequisites": "any glibc",
        "detectable": True,
    },
)

_CATALOG_BY_TECHNIQUE: Dict[str, Dict[str, Any]] = {
    entry["technique"]: dict(entry) for entry in _TECHNIQUE_CATALOG
}


def _catalog_response(
    *,
    entries: Sequence[Mapping[str, Any]],
    technique: Optional[str],
    request: str = "heap_technique_catalog",
) -> Dict[str, Any]:
    return build_backend_response(
        request,
        parsed={
            "schema": HEAP_TECHNIQUE_CATALOG_SCHEMA,
            "technique_filter": technique,
            "count": len(entries),
            "techniques": [dict(entry) for entry in entries],
        },
        raw_responses=[],
        state="idle",
        parser_version=PARSER_VERSION,
        warnings=None,
    )
