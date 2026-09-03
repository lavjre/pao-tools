"""Heap exploitation technique analyzer.

Three related surfaces, all heuristic and evidence-tagged:

* :meth:`GlibcTechniqueAnalyzer.analyze_heap_technique` runs the
  :data:`DETECTORS` registry over a single live snapshot of fastbins +
  tcache + regular bins and reports which House-of-* primitives and heap
  exploitation techniques have observable artifacts in memory right now.
  Each finding carries raw evidence, a confidence label, the existing MCP
  tools that can confirm it, and a one-line playbook.
* :meth:`GlibcTechniqueAnalyzer.detect_uaf` scans writable mappings for
  live 8-byte pointers whose value lands inside the user range of a
  currently-freed chunk -- the dangling-reference signature of a use-after-
  free.  The scan is bounded (``max_bytes`` / ``max_mappings`` /
  ``max_chunks``) and excludes the allocator's own bookkeeping (bin link
  slots, the arena struct, the tcache struct) so its own metadata does not
  read as user data.
* :meth:`GlibcTechniqueAnalyzer.heap_technique_catalog` is static, needs no
  debugger, and documents the classical / mid-era / post-hook technique
  families (Malloc Maleficarum through the modern _IO wide-vtable era).

Nothing here promotes a hypothesis to a fact: a detector only reports what
it observed in live memory, and every finding names the tools that can
verify it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import LiveInferiorGuard, mapping_for
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.heap import (
    IS_MMAPPED,
    PARSER_VERSION,
    PREV_INUSE,
    SIZE_BITS_MASK,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    GlibcHeapCorruption,
    build_backend_response,
)
from paomcp.tools.backends.inspection import PwndbgInspection

HEAP_TECHNIQUE_SCHEMA = "pao.glibc-heap-technique.v1"
UAF_SCAN_SCHEMA = "pao.glibc-uaf-scan.v1"
HEAP_TECHNIQUE_CATALOG_SCHEMA = "pao.glibc-heap-technique-catalog.v1"

# A detector may perform at most one bounded memory read (e.g. the
# physically-previous chunk header for einherjar); everything else must come
# from the walked inspector outputs already collected.
ReadFn = Callable[[int, int], Tuple[bytes, Optional[str]]]


def _byteorder(profile: Mapping[str, Any]) -> Literal["little", "big"]:
    """Resolve the inferior byteorder to a from_bytes-compatible literal."""
    value = profile.get("byteorder")
    if value in ("little", "big"):
        return value  # type: ignore[return-value]
    return "little"


def resolve_top_chunk(
    bins: "GlibcBins",
    arenas: "GlibcArenas",
    mappings: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any],
    cancellation_check: Optional[Callable[[], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve and read the arena's top chunk header (one bounded read).

    The arena's ``top`` field is a *pointer* (malloc_state.top), so the top
    chunk address is an 8-byte read at ``arena + top_offset`` followed by a
    16-byte header read at that address.  Returns ``None`` when the layout,
    the pointer, or the header cannot be resolved -- detectors treat that as
    "no evidence".
    """
    pointer_bits = profile.get("pointer_bits")
    if pointer_bits not in (32, 64):
        return None
    size_sz = pointer_bits // 8
    byteorder = _byteorder(profile)
    if cancellation_check is not None:
        cancellation_check()
    layout = arenas.resolver.resolve()
    arena_address = layout.get("arena_address")
    top_offset = layout["offsets"].get("top")
    if arena_address is None or top_offset is None:
        return None
    raw, err = bins._read_bytes(arena_address + top_offset, size_sz)
    if err is not None or len(raw) < size_sz:
        return None
    top_address = int.from_bytes(raw, byteorder)
    if top_address == 0:
        return None
    header, header_err = bins._read_bytes(top_address, 16)
    if header_err is not None or len(header) < 16:
        return {"address": top_address, "readable": False}
    size_field = int.from_bytes(header[8:16], byteorder)
    size = size_field & ~SIZE_BITS_MASK
    m = mapping_for(top_address, mappings)
    mapping_end: Optional[int] = None
    if m is not None:
        try:
            mapping_end = int(str(m.get("end")), 16)
        except (TypeError, ValueError):
            mapping_end = None
    return {
        "address": top_address,
        "size_field": size_field,
        "size": size,
        "mapping_end": mapping_end,
        "readable": True,
    }


_MMAP_SCAN_PAGE = 0x1000
_MMAP_SCAN_DEFAULT_MAX_MAPPINGS = 4
_MMAP_SCAN_DEFAULT_MAX_BYTES = 1 << 20
_UAF_DEFAULT_MAX_BYTES = 1 << 22  # 4 MiB total across all scanned mappings
_UAF_DEFAULT_MAX_MAPPINGS = 8
_UAF_DEFAULT_MAX_CHUNKS = 512
_UAF_MAX_CANDIDATES = 64
_OVERLAP_DEFAULT_MAX_CHUNKS = 256

# Techniques that are statically detectable by this analyzer (i.e. they have
# a live detector registered).  Everything else in the catalog is documented
# but can only be confirmed at runtime.
DETECTABLE_TECHNIQUES: Tuple[str, ...] = (
    "house_of_spirit",
    "house_of_einherjar",
    "house_of_force",
    "house_of_lore",
    "tcache_poisoning",
    "tcache_double_free",
    "house_of_botcake",
    "fastbin_dup",
    "unsafe_unlink",
    "overlapping_chunks",
    "top_chunk_exhaustion",
    "large_bin_attack",
    "mmap_chunk_scan",
)

# Techniques folded into ``GlibcHeapCorruption.classify_heap_corruption`` as
# additive ``technique_hints``.  Chosen because they need no UAF scan and no
# overlap enumeration -- pure over already-collected inspector outputs plus
# at most two bounded extra reads (top header, previous-chunk header).
HINT_TECHNIQUES: Tuple[str, ...] = (
    "house_of_einherjar",
    "house_of_force",
    "tcache_double_free",
    "fastbin_dup",
)


@dataclass(frozen=True)
class TechniqueContext:
    """Everything a detector may read; frozen so detectors stay pure-ish."""

    classification: Mapping[str, Any]
    mappings: Sequence[Mapping[str, Any]]
    profile: Mapping[str, Any]
    component_status: Mapping[str, Any]
    fb_parsed: Mapping[str, Any]
    tc_parsed: Mapping[str, Any]
    bn_parsed: Mapping[str, Any]
    pointer_bits: int
    size_sz: int
    byteorder: Literal["little", "big"]
    alignment: int
    safe_linking: str
    # ``house_of_force`` / ``top_chunk_exhaustion`` input (computed once by
    # the analyzer): the resolved top chunk's header + mapping context.
    top_chunk: Optional[Mapping[str, Any]] = None
    # Bounded scan configuration for ``mmap_chunk_scan``.
    mmap_scan: Mapping[str, Any] = field(default_factory=dict)
    # Cap for pairwise free-chunk overlap enumeration.
    max_overlap_chunks: int = _OVERLAP_DEFAULT_MAX_CHUNKS


# ---------------------------------------------------------------------------
# Finding builder
# ---------------------------------------------------------------------------
def _detector_result(
    technique: str,
    present: bool,
    *,
    confidence: str,
    severity: str,
    evidence: Dict[str, Any],
    confirm_tools: Sequence[str],
    playbook: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical technique finding with evidence + playbook."""
    result: Dict[str, Any] = {
        "technique": technique,
        "present": present,
        "confidence": confidence,
        "severity": severity,
        "evidence": evidence,
        "confirm_tools": list(confirm_tools),
        "playbook": playbook,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _not_present(
    technique: str,
    *,
    confirm_tools: Sequence[str],
    playbook: str,
    reason: str,
) -> Dict[str, Any]:
    return _detector_result(
        technique,
        False,
        confidence="low",
        severity="info",
        evidence={},
        confirm_tools=confirm_tools,
        playbook=playbook,
        reason=reason,
    )


def _is_heap_mapping(m: Optional[Mapping[str, Any]]) -> bool:
    """True when a mapping is one of glibc's heap regions."""
    if m is None:
        return False
    return str(m.get("file", "")).startswith("[heap]")


def _pointer_location(
    ptr: int, mappings: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Describe where ``ptr`` lands: mapping file, writable, heap flag."""
    m = mapping_for(ptr, mappings)
    if m is None:
        return {"in_mapping": False}
    return {
        "in_mapping": True,
        "mapping": m.get("file"),
        "writable": bool(m.get("writable")),
        "heap": _is_heap_mapping(m),
    }


# ---------------------------------------------------------------------------
# Iterators over walked bin outputs (fastbins / tcache / regular bins)
# ---------------------------------------------------------------------------
def _iter_fastbin_chunks(
    ctx: TechniqueContext,
) -> Sequence[Tuple[int, str, Dict[str, Any]]]:
    chunks: List[Tuple[int, str, Dict[str, Any]]] = []
    for bin_ in ctx.fb_parsed.get("fastbins", []):
        index = bin_.get("index")
        for chunk in bin_.get("chain", []):
            if isinstance(chunk, dict):
                chunks.append((index, "fastbin", chunk))
    return chunks


def _iter_tcache_chunks(
    ctx: TechniqueContext,
) -> Sequence[Tuple[int, str, Dict[str, Any]]]:
    chunks: List[Tuple[int, str, Dict[str, Any]]] = []
    for bin_ in ctx.tc_parsed.get("bins", []):
        index = bin_.get("index")
        for chunk in bin_.get("chain", []):
            if isinstance(chunk, dict):
                chunks.append((index, "tcache", chunk))
    return chunks


def _iter_regular_chunks(
    ctx: TechniqueContext,
) -> Sequence[Tuple[int, str, Dict[str, Any]]]:
    chunks: List[Tuple[int, str, Dict[str, Any]]] = []
    for bin_ in ctx.bn_parsed.get("bins", []):
        index = bin_.get("index")
        category = bin_.get("category")
        for chunk in bin_.get("chain", []):
            if isinstance(chunk, dict):
                chunks.append((index, category, chunk))
    return chunks


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _detect_house_of_spirit(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Free chunk whose header lies outside every mapping (a fake free)."""
    out_of_mapping: List[Dict[str, Any]] = []
    for bin_index, bin_type, chunk in _iter_fastbin_chunks(ctx):
        addr = chunk.get("chunk_address")
        if addr and chunk.get("in_mapping") is False:
            out_of_mapping.append(
                {
                    "chunk_address": addr,
                    "bin_type": bin_type,
                    "bin_index": bin_index,
                    "location": _pointer_location(addr, ctx.mappings),
                }
            )
    for bin_index, bin_type, chunk in _iter_tcache_chunks(ctx):
        addr = chunk.get("chunk_address")
        if addr and chunk.get("in_mapping") is False:
            out_of_mapping.append(
                {
                    "chunk_address": addr,
                    "bin_type": bin_type,
                    "bin_index": bin_index,
                    "location": _pointer_location(addr, ctx.mappings),
                }
            )
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if addr and chunk.get("in_mapping") is False:
            out_of_mapping.append(
                {
                    "chunk_address": addr,
                    "bin_type": category,
                    "bin_index": bin_index,
                    "location": _pointer_location(addr, ctx.mappings),
                }
            )
    if not out_of_mapping:
        return _not_present(
            "house_of_spirit",
            confirm_tools=[
                "inspect_chunk",
                "inspect_fastbins",
                "inspect_tcache",
                "inspect_bins",
            ],
            playbook="free() a fake chunk whose header lies outside the heap; "
            "the next malloc of that size class returns the fake address",
            reason="every walked free chunk lies inside a mapping",
        )
    return _detector_result(
        "house_of_spirit",
        True,
        confidence="high",
        severity="critical",
        evidence={
            "freed_chunks_outside_mappings": out_of_mapping[:8],
            "count": len(out_of_mapping),
        },
        confirm_tools=[
            "inspect_chunk",
            "inspect_fastbins",
            "inspect_tcache",
            "inspect_bins",
        ],
        playbook="a free chunk outside every mapping is a fake chunk -- "
        "likely freed by an attacker.  Verify the chunk header with "
        "inspect_chunk before treating the next malloc as attacker-controlled",
    )


def _detect_house_of_einherjar(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """PREV_INUSE cleared + forged prev_size vs the physical predecessor."""
    mismatches: List[Dict[str, Any]] = []
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if not addr or not chunk.get("chunk_readable"):
            continue
        if chunk.get("prev_inuse") is not False:
            continue  # only a chunk whose predecessor looks free matters
        prev_size = chunk.get("prev_size")
        size = chunk.get("size")
        if not prev_size or not size:
            continue
        physical_prev = addr - prev_size
        if physical_prev <= 0:
            continue
        header, err = read(physical_prev, 16)
        if err is not None or len(header) < 16:
            # Cannot read the real predecessor -- no evidence, stay silent.
            continue
        actual_prev_size = int.from_bytes(header[8:16], ctx.byteorder) & ~SIZE_BITS_MASK
        if actual_prev_size != prev_size:
            mismatches.append(
                {
                    "chunk_address": addr,
                    "claimed_prev_size": prev_size,
                    "physical_prev": physical_prev,
                    "actual_prev_chunk_size": actual_prev_size,
                    "bin_index": bin_index,
                    "category": category,
                }
            )
    if not mismatches:
        return _not_present(
            "house_of_einherjar",
            confirm_tools=["inspect_chunk", "inspect_bins"],
            playbook="null-byte off-by-one clears PREV_INUSE; forge prev_size "
            "so backward consolidation overlaps an allocation",
            reason="no regular-bin free chunk claims a prev_size that "
            "contradicts its physical predecessor",
        )
    return _detector_result(
        "house_of_einherjar",
        True,
        confidence="medium",
        severity="high",
        evidence={"prev_size_mismatches": mismatches[:8]},
        confirm_tools=["inspect_chunk", "inspect_bins"],
        playbook="a free chunk whose claimed prev_size is larger than the real "
        "predecessor chunk lets free() consolidate backwards across a live "
        "allocation -- the overlap is the primitive.  Confirm the two "
        "headers with inspect_chunk",
    )


def _detect_house_of_force(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Top chunk size wild / beyond its mapping / misaligned."""
    top = ctx.top_chunk
    if not top or top.get("readable") is not True:
        return _not_present(
            "house_of_force",
            confirm_tools=["inspect_arena", "inspect_chunk"],
            playbook="overwrite the top chunk size so malloc wraps the "
            "remaindering arithmetic and returns an arbitrary address",
            reason="top chunk could not be resolved or read",
        )
    size = int(top.get("size") or 0)
    address = int(top.get("address") or 0)
    mapping_end = top.get("mapping_end")
    huge = size > (1 << 40)
    beyond = mapping_end is not None and address + size > int(mapping_end)
    misaligned = size > 0 and (size % ctx.alignment) != 0
    present = huge or bool(beyond) or misaligned
    if not present:
        return _not_present(
            "house_of_force",
            confirm_tools=["inspect_arena", "inspect_chunk"],
            playbook="overwrite the top chunk size so malloc wraps the "
            "remaindering arithmetic and returns an arbitrary address",
            reason="top chunk size is sane relative to its mapping",
        )
    return _detector_result(
        "house_of_force",
        True,
        confidence="high",
        severity="critical",
        evidence={
            "top_address": address,
            "top_size_field": top.get("size_field"),
            "top_size": size,
            "mapping_end": mapping_end,
            "size_beyond_mapping": bool(beyond),
            "wildly_huge": huge,
            "misaligned": misaligned,
        },
        confirm_tools=["inspect_arena", "inspect_chunk"],
        playbook="an oversized / misaligned top chunk lets malloc return "
        "pointers outside the heap (house of force, patched in 2.29).  "
        "Confirm the top header with inspect_arena / inspect_chunk",
    )


def _detect_house_of_lore(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Regular-bin fd/bk pointing outside the heap (a fake smallbin)."""
    bad: List[Dict[str, Any]] = []
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if not addr or not chunk.get("chunk_readable"):
            continue
        for field_name in ("fd", "bk"):
            ptr = chunk.get(field_name)
            if not ptr:
                continue
            location = _pointer_location(ptr, ctx.mappings)
            if not location.get("in_mapping"):
                bad.append(
                    {
                        "chunk_address": addr,
                        "field": field_name,
                        "pointer": ptr,
                        "bin_index": bin_index,
                        "category": category,
                    }
                )
    if not bad:
        return _not_present(
            "house_of_lore",
            confirm_tools=["inspect_bins", "inspect_chunk"],
            playbook="fake a smallbin bk chain so malloc pops a fake chunk "
            "from an arbitrary address",
            reason="every regular-bin fd/bk pointer lands inside a mapping",
        )
    return _detector_result(
        "house_of_lore",
        True,
        confidence="medium",
        severity="high",
        evidence={"pointers_outside_mappings": bad[:8]},
        confirm_tools=["inspect_bins", "inspect_chunk"],
        playbook="a regular-bin fd/bk escaping the heap suggests a forged "
        "chain (house of lore, patched in 2.26).  Confirm with inspect_bins "
        "then inspect_chunk on the escaping pointer",
    )


def _detect_tcache_poisoning(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Tcache next pointer targeting a writable non-heap address."""
    poisoned: List[Dict[str, Any]] = []
    for bin_index, _bin_type, chunk in _iter_tcache_chunks(ctx):
        next_decoded = chunk.get("next_decoded")
        if not next_decoded:
            continue
        m = mapping_for(next_decoded, ctx.mappings)
        if m is None or (bool(m.get("writable")) and not _is_heap_mapping(m)):
            poisoned.append(
                {
                    "bin_index": bin_index,
                    "chunk_address": chunk.get("chunk_address"),
                    "user_address": chunk.get("user_address"),
                    "next_decoded": next_decoded,
                    "target": _pointer_location(next_decoded, ctx.mappings),
                }
            )
    if not poisoned:
        return _not_present(
            "tcache_poisoning",
            confirm_tools=["inspect_tcache", "inspect_chunk"],
            playbook="overwrite a tcache entry's next pointer; malloc pops "
            "the target address as the next chunk",
            reason="every tcache next pointer lands inside the heap",
        )
    return _detector_result(
        "tcache_poisoning",
        True,
        confidence="high",
        severity="critical",
        evidence={"poisoned_links": poisoned[:8]},
        confirm_tools=["inspect_tcache", "inspect_chunk"],
        playbook="a tcache next pointer escaping the heap is a poisoned link -- "
        "the next malloc of that size class returns the target.  Confirm the "
        "chain with inspect_tcache and the target header with inspect_chunk",
    )


def _detect_tcache_double_free(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Same chunk twice in a tcache chain / in two bins / count < walked."""
    seen: Dict[int, List[Dict[str, Any]]] = {}
    count_mismatches: List[Dict[str, Any]] = []
    for bin_ in ctx.tc_parsed.get("bins", []):
        index = bin_.get("index")
        chain = bin_.get("chain", [])
        count = bin_.get("count")
        if isinstance(count, int) and count >= 0 and isinstance(chain, list):
            if len(chain) > count:
                count_mismatches.append(
                    {
                        "bin_index": index,
                        "count": count,
                        "walked_length": len(chain),
                    }
                )
        for chunk in chain:
            addr = chunk.get("chunk_address") if isinstance(chunk, dict) else None
            if addr:
                seen.setdefault(addr, []).append({"bin_index": index})
    duplicates = {addr: entries for addr, entries in seen.items() if len(entries) > 1}
    if not duplicates and not count_mismatches:
        return _not_present(
            "tcache_double_free",
            confirm_tools=["inspect_tcache", "validate_free"],
            playbook="free the same chunk twice into tcache; the dup lets two "
            "mallocs return the same region",
            reason="no chunk repeats across the tcache and every count >= "
            "walked chain length",
        )
    return _detector_result(
        "tcache_double_free",
        True,
        confidence="high",
        severity="critical",
        evidence={
            "duplicate_chunks": [
                {"chunk_address": addr, "bins": entries}
                for addr, entries in duplicates.items()
            ][:8],
            "count_mismatches": count_mismatches[:8],
        },
        confirm_tools=["inspect_tcache", "validate_free"],
        playbook="a chunk appearing twice in tcache (or more walked entries "
        "than the count) is a double-free signature.  validate_free can "
        "confirm the freed state of the repeated chunk",
    )


def _detect_house_of_botcake(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Same chunk in tcache AND in the unsorted/small bins."""
    tcache_addrs = {
        chunk.get("chunk_address")
        for _idx, _bin_type, chunk in _iter_tcache_chunks(ctx)
        if chunk.get("chunk_address")
    }
    overlap: List[Dict[str, Any]] = []
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if addr in tcache_addrs and category in ("unsorted", "small"):
            overlap.append(
                {
                    "chunk_address": addr,
                    "regular_bin_index": bin_index,
                    "regular_bin_category": category,
                }
            )
    if not overlap:
        return _not_present(
            "house_of_botcake",
            confirm_tools=["inspect_tcache", "inspect_bins", "validate_free"],
            playbook="fill tcache to capacity, free twice so a chunk lands in "
            "tcache AND unsorted; a later malloc splits the unsorted chunk "
            "into the tcache entry",
            reason="no chunk appears in both tcache and unsorted/small bins",
        )
    return _detector_result(
        "house_of_botcake",
        True,
        confidence="high",
        severity="critical",
        evidence={"chunks_in_tcache_and_bins": overlap[:8]},
        confirm_tools=["inspect_tcache", "inspect_bins", "validate_free"],
        playbook="a chunk resident in both tcache and unsorted/small is the "
        "botcake setup -- the unsorted chunk can be split to overlap the "
        "still-live tcache entry.  Confirm both chains with inspect_tcache "
        "and inspect_bins",
    )


def _detect_fastbin_dup(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Fastbin chain cycle / chunk present in two fastbin bins."""
    cycle_bins: List[Dict[str, Any]] = []
    seen: Dict[int, List[Dict[str, Any]]] = {}
    for bin_ in ctx.fb_parsed.get("fastbins", []):
        index = bin_.get("index")
        if bin_.get("cycle_detected"):
            cycle_bins.append(
                {
                    "bin_index": index,
                    "size_class": bin_.get("size_class"),
                    "head": bin_.get("head"),
                    "walked_length": bin_.get("length"),
                }
            )
        for chunk in bin_.get("chain", []):
            addr = chunk.get("chunk_address") if isinstance(chunk, dict) else None
            if addr:
                seen.setdefault(addr, []).append({"bin_index": index})
    cross_bin = {addr: entries for addr, entries in seen.items() if len(entries) > 1}
    if not cycle_bins and not cross_bin:
        return _not_present(
            "fastbin_dup",
            confirm_tools=["inspect_fastbins", "validate_free"],
            playbook="double-free a fastbin chunk (or corrupt a fastbin fd) so "
            "the same chunk is returned twice",
            reason="no fastbin chain cycles and no chunk repeats across bins",
        )
    return _detector_result(
        "fastbin_dup",
        True,
        confidence="high",
        severity="critical",
        evidence={
            "cycle_bins": cycle_bins[:8],
            "cross_bin_duplicates": [
                {"chunk_address": addr, "bins": entries}
                for addr, entries in cross_bin.items()
            ][:8],
        },
        confirm_tools=["inspect_fastbins", "validate_free"],
        playbook="a cycling fastbin chain or a chunk in two fastbin bins is a "
        "double-free signature.  validate_free on the repeated chunk "
        "confirms its state",
    )


def _detect_unsafe_unlink(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Free chunk fd/bk targeting a writable non-heap mapping."""
    bad: List[Dict[str, Any]] = []
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if not addr or not chunk.get("chunk_readable"):
            continue
        for field_name in ("fd", "bk"):
            ptr = chunk.get(field_name)
            if not ptr:
                continue
            m = mapping_for(ptr, ctx.mappings)
            if m is None:
                continue  # out-of-mapping is house_of_lore's signal
            if bool(m.get("writable")) and not _is_heap_mapping(m):
                bad.append(
                    {
                        "chunk_address": addr,
                        "field": field_name,
                        "pointer": ptr,
                        "target_mapping": m.get("file"),
                        "bin_index": bin_index,
                        "category": category,
                    }
                )
    if not bad:
        return _not_present(
            "unsafe_unlink",
            confirm_tools=["inspect_bins", "inspect_chunk"],
            playbook="forge fd/bk so the unlink write (P->fd->bk = P->bk, "
            "P->bk->fd = P->fd) lands a write at an arbitrary address",
            reason="every regular-bin fd/bk targets the heap or unmapped space",
        )
    return _detector_result(
        "unsafe_unlink",
        True,
        confidence="medium",
        severity="high",
        evidence={"writable_targets": bad[:8]},
        confirm_tools=["inspect_bins", "inspect_chunk"],
        playbook="a free chunk's fd/bk pointing at a writable non-heap region "
        "is the unsafe-unlink write target.  Verify P->fd->bk and P->bk->fd "
        "both resolve to P before treating the write as exploitable",
    )


def _detect_overlapping_chunks(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """prev_size-with-PREV_INUSE mismatch, or pairwise free-range overlap."""
    # (a) A chunk claims PREV_INUSE (predecessor in use) yet the physical
    # predecessor at addr - prev_size decodes as a genuinely free chunk
    # (matching size, its own PREV_INUSE clear).  glibc clears PREV_INUSE when
    # it frees a predecessor, so this pairing is a shrink/consolidation
    # residue -- the free range hides behind a live chunk.
    min_chunk_size = 4 * ctx.size_sz
    prev_mismatches: List[Dict[str, Any]] = []
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        if not addr or not chunk.get("chunk_readable"):
            continue
        if chunk.get("prev_inuse") is not True:
            continue
        prev_size = chunk.get("prev_size") or 0
        if prev_size < min_chunk_size:
            continue
        prev_addr = addr - prev_size
        if prev_addr <= 0:
            continue
        header, err = read(prev_addr, 16)
        if err is not None or len(header) < 16:
            continue
        prev_size_field = int.from_bytes(header[8:16], ctx.byteorder)
        prev_real_size = prev_size_field & ~SIZE_BITS_MASK
        prev_prev_inuse = bool(prev_size_field & PREV_INUSE)
        if prev_real_size == prev_size and prev_prev_inuse is False:
            prev_mismatches.append(
                {
                    "chunk_address": addr,
                    "claimed_prev_inuse": True,
                    "prev_size": prev_size,
                    "hidden_free_predecessor": prev_addr,
                    "bin_index": bin_index,
                    "category": category,
                }
            )

    ranges: List[Dict[str, Any]] = []
    seen_ranges: set = set()
    for bin_index, _bin_type, chunk in _iter_fastbin_chunks(ctx):
        addr = chunk.get("chunk_address")
        size = chunk.get("size")
        if addr and size:
            key = (addr, size)
            if key not in seen_ranges:
                seen_ranges.add(key)
                ranges.append(
                    {"chunk_address": addr, "size": size, "source": "fastbin"}
                )
    for bin_index, _bin_type, chunk in _iter_tcache_chunks(ctx):
        addr = chunk.get("chunk_address")
        size = chunk.get("size")
        if addr and size:
            key = (addr, size)
            if key not in seen_ranges:
                seen_ranges.add(key)
                ranges.append({"chunk_address": addr, "size": size, "source": "tcache"})
    for bin_index, category, chunk in _iter_regular_chunks(ctx):
        addr = chunk.get("chunk_address")
        size = chunk.get("size")
        if addr and size:
            key = (addr, size)
            if key not in seen_ranges:
                seen_ranges.add(key)
                ranges.append({"chunk_address": addr, "size": size, "source": category})

    cap = ctx.max_overlap_chunks
    bounded = len(ranges) > cap
    if bounded:
        ranges = ranges[:cap]
    overlaps: List[Dict[str, Any]] = []
    for i in range(len(ranges)):
        a = ranges[i]
        a_start = a["chunk_address"]
        a_end = a_start + a["size"]
        for j in range(i + 1, len(ranges)):
            b = ranges[j]
            b_start = b["chunk_address"]
            b_end = b_start + b["size"]
            if (
                a_end > b_start
                and b_end > a_start
                and (a_start != b_start or a_end != b_end)
            ):
                overlaps.append(
                    {
                        "a": {
                            "chunk_address": a_start,
                            "size": a["size"],
                            "source": a["source"],
                        },
                        "b": {
                            "chunk_address": b_start,
                            "size": b["size"],
                            "source": b["source"],
                        },
                        "overlap_start": max(a_start, b_start),
                        "overlap_end": min(a_end, b_end),
                    }
                )
                if len(overlaps) >= 8:
                    break
        if len(overlaps) >= 8:
            break

    if not prev_mismatches and not overlaps:
        return _not_present(
            "overlapping_chunks",
            confirm_tools=["inspect_bins", "inspect_chunk", "validate_free"],
            playbook="overlap two live allocations (OOB write, shrink+free, "
            "split abuse) so editing one edits the other",
            reason="no prev_size/PREV_INUSE mismatch and no overlapping free "
            "chunk ranges",
        )
    return _detector_result(
        "overlapping_chunks",
        True,
        confidence="medium",
        severity="high",
        evidence={
            "prev_size_with_inuse": prev_mismatches[:8],
            "overlapping_ranges": overlaps[:8],
            "ranges_considered": len(ranges) if not bounded else cap,
            "range_cap_exceeded": bounded,
        },
        confirm_tools=["inspect_bins", "inspect_chunk", "validate_free"],
        playbook="two free chunks whose ranges intersect means a free/shrink "
        "overlapped an allocation -- the classic OOB/overlap primitive.  "
        "Confirm each header with inspect_chunk before writing through it",
    )


def _detect_top_chunk_exhaustion(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Top chunk close to its mapping end / unusually small."""
    top = ctx.top_chunk
    if not top or top.get("readable") is not True:
        return _not_present(
            "top_chunk_exhaustion",
            confirm_tools=["inspect_arena", "inspect_chunk"],
            playbook="exhaust the heap so malloc must sysmalloc a new top / "
            "mmap, or shrink the top to force a re-merge",
            reason="top chunk could not be resolved or read",
        )
    size = int(top.get("size") or 0)
    address = int(top.get("address") or 0)
    mapping_end = top.get("mapping_end")
    remaining: Optional[int] = None
    if mapping_end is not None and address <= int(mapping_end):
        remaining = int(mapping_end) - address
    near_end = remaining is not None and remaining < 0x1000
    tiny = 0 < size < 0x100
    present = bool(near_end or tiny)
    if not present:
        return _not_present(
            "top_chunk_exhaustion",
            confirm_tools=["inspect_arena", "inspect_chunk"],
            playbook="exhaust the heap so malloc must sysmalloc a new top / "
            "mmap, or shrink the top to force a re-merge",
            reason="top chunk has ample room in its mapping",
        )
    return _detector_result(
        "top_chunk_exhaustion",
        True,
        confidence="low",
        severity="low",
        evidence={
            "top_address": address,
            "top_size": size,
            "mapping_end": mapping_end,
            "remaining_in_mapping": remaining,
            "near_mapping_end": bool(near_end),
            "tiny_top": bool(tiny),
        },
        confirm_tools=["inspect_arena", "inspect_chunk"],
        playbook="a top chunk with little or no room left is a pressure point "
        "-- the next large malloc forces a sysmalloc / mmap that can be "
        "redirected.  Usually context, not a bug on its own",
    )


def _detect_large_bin_attack(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Catalog-only technique; not statically detectable today."""
    return _not_present(
        "large_bin_attack",
        confirm_tools=["inspect_bins"],
        playbook="forge a largebin chunk's fd_nextsize/bk_nextsize so the "
        "size-sorted insert writes an arbitrary value",
        reason="not statically detectable; inspect fd_nextsize/bk_nextsize of "
        "large-bin chunks at runtime",
    )


def _detect_mmap_chunk_scan(ctx: TechniqueContext, read: ReadFn) -> Dict[str, Any]:
    """Bounded scan of writable anon mappings for IS_MMAPPED chunk headers."""
    config = ctx.mmap_scan or {}
    if config.get("enabled") is False:
        return _not_present(
            "mmap_chunk_scan",
            confirm_tools=["inspect_chunk", "memory_maps"],
            playbook="IS_MMAPPED chunks are allocator metadata for large / "
            "mmap'd allocations; an mmap chunk inside a heap arena is a "
            "corruption signal",
            reason="mmap chunk scan disabled",
        )
    max_mappings = int(config.get("max_mappings", _MMAP_SCAN_DEFAULT_MAX_MAPPINGS))
    max_bytes = int(config.get("max_bytes", _MMAP_SCAN_DEFAULT_MAX_BYTES))
    candidates: List[Dict[str, Any]] = []
    mappings_scanned = 0
    bytes_scanned = 0
    for m in ctx.mappings:
        if not m.get("writable"):
            continue
        if _is_heap_mapping(m):
            continue  # IS_MMAPPED chunks never live in the heap arena
        try:
            start = int(str(m.get("start")), 16)
            end = int(str(m.get("end")), 16)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        mappings_scanned += 1
        if mappings_scanned > max_mappings:
            break
        region_end = min(end, start + max_bytes)
        cursor = start
        while cursor < region_end and bytes_scanned < max_bytes:
            if len(candidates) >= 8:
                break
            page = min(0x1000, region_end - cursor)
            raw, err = read(cursor, page)
            if err is None:
                bytes_scanned += len(raw)
                for off in range(0, len(raw) - ctx.size_sz, ctx.alignment):
                    size_field = int.from_bytes(
                        raw[off + ctx.size_sz : off + 2 * ctx.size_sz],
                        ctx.byteorder,
                    )
                    if not (size_field & IS_MMAPPED):
                        continue
                    chunk_address = cursor + off
                    size = size_field & ~SIZE_BITS_MASK
                    if size < 0x1000:
                        continue
                    candidates.append(
                        {
                            "chunk_address": chunk_address,
                            "size_field": size_field,
                            "size": size,
                            "mapping": m.get("file"),
                        }
                    )
                    break
            cursor += page
    if not candidates:
        return _not_present(
            "mmap_chunk_scan",
            confirm_tools=["inspect_chunk", "memory_maps"],
            playbook="IS_MMAPPED chunks are allocator metadata for large / "
            "mmap'd allocations; an mmap chunk inside a heap arena is a "
            "corruption signal",
            reason=(
                f"no IS_MMAPPED chunk headers found in {mappings_scanned} "
                f"writable mappings ({bytes_scanned} bytes scanned)"
            ),
        )
    return _detector_result(
        "mmap_chunk_scan",
        True,
        confidence="medium",
        severity="low",
        evidence={
            "mmap_chunks": candidates,
            "mappings_scanned": mappings_scanned,
            "bytes_scanned": bytes_scanned,
        },
        confirm_tools=["inspect_chunk", "memory_maps"],
        playbook="IS_MMAPPED chunk headers live in dedicated anon mappings; "
        "their presence (and their size/address layout) is the footprint of "
        "large allocations and of house_of_muney-style arena-threshold "
        "attacks",
    )


# ---------------------------------------------------------------------------
# Detector registry
# ---------------------------------------------------------------------------
DETECTORS: Dict[str, Callable[[TechniqueContext, ReadFn], Dict[str, Any]]] = {
    "house_of_spirit": _detect_house_of_spirit,
    "house_of_einherjar": _detect_house_of_einherjar,
    "house_of_force": _detect_house_of_force,
    "house_of_lore": _detect_house_of_lore,
    "tcache_poisoning": _detect_tcache_poisoning,
    "tcache_double_free": _detect_tcache_double_free,
    "house_of_botcake": _detect_house_of_botcake,
    "fastbin_dup": _detect_fastbin_dup,
    "unsafe_unlink": _detect_unsafe_unlink,
    "overlapping_chunks": _detect_overlapping_chunks,
    "top_chunk_exhaustion": _detect_top_chunk_exhaustion,
    "large_bin_attack": _detect_large_bin_attack,
    "mmap_chunk_scan": _detect_mmap_chunk_scan,
}


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


# ---------------------------------------------------------------------------
# The analyzer
# ---------------------------------------------------------------------------
class GlibcTechniqueAnalyzer:
    """Run heap-technique detectors over one live heap snapshot.

    Mirrors the ``GlibcHeapCorruption`` wiring: live-inferior guard,
    ``_prepare_context`` (classification + mappings + profile + component
    status), then delegation to the typed bin inspectors.  Every detector is
    heuristic and returns evidence + a confirmation path; nothing is promoted
    to a fact.
    """

    _CHECKS_RUN: Tuple[str, ...] = (
        "fastbin_walk",
        "tcache_walk",
        "regular_bin_walk",
        "top_chunk_read",
        "detector_registry",
    )

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        discovery: "AllocatorDiscovery",
        arenas: "GlibcArenas",
        bins: "GlibcBins",
        glibc: "GlibcHeap",
        corruption: "GlibcHeapCorruption",
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas
        self.bins = bins
        self.glibc = glibc
        self.corruption = corruption

    # ------------------------------------------------------------------
    # Live-target helpers (mirror GlibcHeapCorruption)
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Technique analysis requires a live, stopped inferior; run "
                "the target to a stop before inspecting the heap"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = AllocatorDiscovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    @staticmethod
    def _allocator_summary(classification: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

    def _unsupported_response(
        self,
        request: str,
        schema: str,
        classification: Mapping[str, Any],
        profile: Mapping[str, Any],
        component_status: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return build_backend_response(
            request,
            parsed={
                "schema": schema,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    "heap technique analysis supports glibc only; detected "
                    f"family={classification.get('family')}"
                ),
                "allocator": self._allocator_summary(classification),
                "target_profile": profile,
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def _read_top_chunk(
        self,
        mappings: Sequence[Mapping[str, Any]],
        profile: Mapping[str, Any],
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve and read the arena's top chunk header (one bounded read)."""
        return resolve_top_chunk(
            bins=self.bins,
            arenas=self.arenas,
            mappings=mappings,
            profile=profile,
            cancellation_check=cancellation_check,
        )

    # ------------------------------------------------------------------
    # analyze_heap_technique
    # ------------------------------------------------------------------
    def analyze_heap_technique(
        self,
        *,
        arena_address: Optional[str] = None,
        techniques: Optional[Sequence[str]] = None,
        scan_mmap_mappings: bool = True,
        mmap_max_mappings: int = _MMAP_SCAN_DEFAULT_MAX_MAPPINGS,
        mmap_max_bytes: int = _MMAP_SCAN_DEFAULT_MAX_BYTES,
        max_overlap_chunks: int = _OVERLAP_DEFAULT_MAX_CHUNKS,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Run the detector registry over one live heap snapshot."""
        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self._unsupported_response(
                "analyze_heap_technique",
                HEAP_TECHNIQUE_SCHEMA,
                classification,
                profile,
                component_status,
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot analyze heap techniques without a resolved pointer "
                f"width; target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = _byteorder(profile)
        alignment = 2 * size_sz

        names = (
            list(DETECTORS) if techniques is None else _valid_detector_names(techniques)
        )

        fastbins_response = self.bins.inspect_fastbins(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )
        tcache_response = self.bins.inspect_tcache(
            cancellation_check=cancellation_check,
        )
        bins_response = self.bins.inspect_bins(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )
        top_chunk = self._read_top_chunk(mappings, profile, cancellation_check)

        fb_parsed = fastbins_response.get("parsed", {}) or {}
        tc_parsed = tcache_response.get("parsed", {}) or {}
        bn_parsed = bins_response.get("parsed", {}) or {}

        warnings: List[str] = []
        for source, response in (
            ("fastbins", fastbins_response),
            ("tcache", tcache_response),
            ("bins", bins_response),
        ):
            for entry in response.get("warnings") or []:
                warnings.append(f"{source}: {entry}")

        if "unresolvable" in {
            fb_parsed.get("verdict"),
            bn_parsed.get("verdict"),
        }:
            return build_backend_response(
                "analyze_heap_technique",
                parsed={
                    "schema": HEAP_TECHNIQUE_SCHEMA,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": (
                        "arena bin layout could not be resolved via debug "
                        "symbols; technique analysis skipped"
                    ),
                    "allocator": self._allocator_summary(classification),
                    "target_profile": profile,
                    "techniques_requested": names,
                    "techniques_evaluated": 0,
                    "findings": [],
                    "summary": {"detected": 0, "suspected": 0, "not_detected": 0},
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        if top_chunk is None:
            warnings.append(
                "top chunk could not be resolved or read; house_of_force and "
                "top_chunk_exhaustion report no evidence"
            )

        ctx = TechniqueContext(
            classification=classification,
            mappings=mappings,
            profile=profile,
            component_status=component_status,
            fb_parsed=fb_parsed,
            tc_parsed=tc_parsed,
            bn_parsed=bn_parsed,
            pointer_bits=pointer_bits,
            size_sz=size_sz,
            byteorder=byteorder,
            alignment=alignment,
            safe_linking=classification.get("safe_linking", "unknown"),
            top_chunk=top_chunk,
            mmap_scan={
                "enabled": scan_mmap_mappings,
                "max_mappings": mmap_max_mappings,
                "max_bytes": mmap_max_bytes,
            },
            max_overlap_chunks=max_overlap_chunks,
        )

        read: ReadFn = self.bins._read_bytes
        findings: List[Dict[str, Any]] = []
        for name in names:
            if cancellation_check is not None:
                cancellation_check()
            findings.append(DETECTORS[name](ctx, read))

        detected = [
            f
            for f in findings
            if f["present"] and f.get("severity") in ("critical", "high")
        ]
        suspected = [
            f
            for f in findings
            if f["present"] and f.get("severity") not in ("critical", "high")
        ]
        not_detected = [f for f in findings if not f["present"]]

        if detected:
            verdict = "exploitable"
        elif suspected:
            verdict = "suspicious"
        else:
            verdict = "clean"

        parsed = {
            "schema": HEAP_TECHNIQUE_SCHEMA,
            "supported": True,
            "verdict": verdict,
            "allocator": self._allocator_summary(classification),
            "target_profile": profile,
            "techniques_requested": names,
            "techniques_evaluated": len(names),
            "findings": findings,
            "summary": {
                "detected": len(detected),
                "suspected": len(suspected),
                "not_detected": len(not_detected),
            },
            "component_status": component_status,
        }
        return build_backend_response(
            "analyze_heap_technique",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    # ------------------------------------------------------------------
    # detect_uaf
    # ------------------------------------------------------------------
    def detect_uaf(
        self,
        *,
        max_bytes: int = _UAF_DEFAULT_MAX_BYTES,
        max_mappings: int = _UAF_DEFAULT_MAX_MAPPINGS,
        max_chunks: int = _UAF_DEFAULT_MAX_CHUNKS,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan for live pointers into the user ranges of freed chunks."""
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes!r}")
        if max_mappings <= 0:
            raise ValueError(f"max_mappings must be positive, got {max_mappings!r}")
        if max_chunks <= 0:
            raise ValueError(f"max_chunks must be positive, got {max_chunks!r}")

        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self._unsupported_response(
                "detect_uaf",
                UAF_SCAN_SCHEMA,
                classification,
                profile,
                component_status,
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot scan for UAF without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = _byteorder(profile)
        assert byteorder is not None

        fastbins_response = self.bins.inspect_fastbins(
            cancellation_check=cancellation_check,
        )
        tcache_response = self.bins.inspect_tcache(
            cancellation_check=cancellation_check,
        )
        bins_response = self.bins.inspect_bins(
            cancellation_check=cancellation_check,
        )
        fb_parsed = fastbins_response.get("parsed", {}) or {}
        tc_parsed = tcache_response.get("parsed", {}) or {}
        bn_parsed = bins_response.get("parsed", {}) or {}

        warnings: List[str] = []
        for source, response in (
            ("fastbins", fastbins_response),
            ("tcache", tcache_response),
            ("bins", bins_response),
        ):
            for entry in response.get("warnings") or []:
                warnings.append(f"{source}: {entry}")

        if "unresolvable" in {
            fb_parsed.get("verdict"),
            bn_parsed.get("verdict"),
        }:
            return build_backend_response(
                "detect_uaf",
                parsed={
                    "schema": UAF_SCAN_SCHEMA,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": (
                        "arena bin layout could not be resolved via debug "
                        "symbols; UAF scan skipped"
                    ),
                    "allocator": self._allocator_summary(classification),
                    "target_profile": profile,
                    "freed_chunks": {"count": 0, "truncated": False},
                    "scan": {
                        "mappings_scanned": 0,
                        "bytes_scanned": 0,
                        "max_mappings": max_mappings,
                        "max_bytes": max_bytes,
                        "truncated": False,
                    },
                    "candidates": [],
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        # -- 1. collect freed user ranges --------------------------------
        ctx = TechniqueContext(
            classification=classification,
            mappings=mappings,
            profile=profile,
            component_status=component_status,
            fb_parsed=fb_parsed,
            tc_parsed=tc_parsed,
            bn_parsed=bn_parsed,
            pointer_bits=pointer_bits,
            size_sz=size_sz,
            byteorder=byteorder,
            alignment=2 * size_sz,
            safe_linking=classification.get("safe_linking", "unknown"),
        )
        freed: List[Dict[str, Any]] = []
        freed_truncated = False
        header_size = 2 * size_sz

        def _collect(addr: Optional[int], size: Optional[int], source: str) -> None:
            nonlocal freed_truncated
            if freed_truncated:
                return
            if not addr or not size or size <= header_size:
                return
            if len(freed) >= max_chunks:
                freed_truncated = True
                return
            freed.append(
                {
                    "chunk_address": addr,
                    "user_start": addr + header_size,
                    "user_end": addr + size,
                    "size": size,
                    "source": source,
                }
            )

        for bin_index, _bin_type, chunk in _iter_fastbin_chunks(ctx):
            _collect(
                chunk.get("chunk_address"),
                chunk.get("size"),
                f"fastbin[{bin_index}]",
            )
        for bin_index, _bin_type, chunk in _iter_tcache_chunks(ctx):
            _collect(
                chunk.get("chunk_address"),
                chunk.get("size"),
                f"tcache[{bin_index}]",
            )
        for bin_index, category, chunk in _iter_regular_chunks(ctx):
            _collect(
                chunk.get("chunk_address"),
                chunk.get("size"),
                f"{category}[{bin_index}]",
            )

        # -- 2. pointer-location exclusion ranges (allocator bookkeeping) --
        excluded: List[Tuple[int, int]] = []
        tcache_address = tc_parsed.get("tcache_address")
        if tcache_address:
            struct_size = (tc_parsed.get("layout") or {}).get("struct_size")
            struct_size = struct_size if isinstance(struct_size, int) else 0x400
            excluded.append(
                (int(tcache_address), int(tcache_address) + max(struct_size, 0x100))
            )
        arena_address = fb_parsed.get("arena_address") or bn_parsed.get("arena_address")
        if arena_address:
            excluded.append((int(arena_address), int(arena_address) + 0x2000))

        # -- 3. scan writable mappings -----------------------------------
        candidates: List[Dict[str, Any]] = []
        mappings_scanned = 0
        bytes_scanned = 0
        scan_truncated = False
        read_fn: ReadFn = self.bins._read_bytes
        for m in mappings:
            if not m.get("writable"):
                continue
            try:
                start = int(str(m.get("start")), 16)
                end = int(str(m.get("end")), 16)
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            mappings_scanned += 1
            if mappings_scanned > max_mappings:
                scan_truncated = True
                break
            region_end = min(end, start + (max_bytes - bytes_scanned))
            cursor = start
            while cursor < region_end:
                if len(candidates) >= _UAF_MAX_CANDIDATES:
                    break
                page = min(_MMAP_SCAN_PAGE, region_end - cursor)
                raw, err = read_fn(cursor, page)
                if err is not None:
                    cursor += page
                    bytes_scanned += page
                    continue
                bytes_scanned += len(raw)
                for off in range(0, len(raw) - size_sz + 1, size_sz):
                    qword = int.from_bytes(
                        raw[off : off + size_sz], byteorder  # type: ignore[arg-type]
                    )
                    if qword == 0:
                        continue
                    ptr_addr = cursor + off
                    # skip the allocator's own bookkeeping locations
                    if any(a <= ptr_addr < b for a, b in excluded):
                        continue
                    # skip the fd/bk/next link slots of freed chunks
                    link_slot = False
                    for f in freed:
                        if f["user_start"] <= ptr_addr < f["user_start"] + header_size:
                            link_slot = True
                            break
                    if link_slot:
                        continue
                    for f in freed:
                        if f["user_start"] <= qword < f["user_end"]:
                            candidates.append(
                                {
                                    "address_of_pointer": ptr_addr,
                                    "pointer_value": qword,
                                    "freed_chunk_address": f["chunk_address"],
                                    "freed_chunk_size": f["size"],
                                    "source_bin": f["source"],
                                    "offset_into_chunk": qword - f["user_start"],
                                    "mapping": m.get("file"),
                                }
                            )
                            break
                    if len(candidates) >= _UAF_MAX_CANDIDATES:
                        break
                cursor += page
            if bytes_scanned >= max_bytes:
                break

        verdict = (
            "dangling_references_found" if candidates else "no_dangling_references"
        )
        parsed = {
            "schema": UAF_SCAN_SCHEMA,
            "supported": True,
            "verdict": verdict,
            "allocator": self._allocator_summary(classification),
            "target_profile": profile,
            "freed_chunks": {
                "count": len(freed),
                "truncated": freed_truncated,
            },
            "scan": {
                "mappings_scanned": mappings_scanned,
                "bytes_scanned": bytes_scanned,
                "max_mappings": max_mappings,
                "max_bytes": max_bytes,
                "truncated": scan_truncated,
            },
            "candidates": candidates,
            "component_status": component_status,
        }
        return build_backend_response(
            "detect_uaf",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    # ------------------------------------------------------------------
    # heap_technique_catalog (static, no debugger)
    # ------------------------------------------------------------------
    @staticmethod
    def heap_technique_catalog(
        *,
        technique: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the static technique catalog, optionally filtered."""
        if technique is None:
            return _catalog_response(entries=_TECHNIQUE_CATALOG, technique=None)
        if technique not in _CATALOG_BY_TECHNIQUE:
            return _catalog_response(
                entries=[],
                technique=technique,
            )
        return _catalog_response(
            entries=[_CATALOG_BY_TECHNIQUE[technique]],
            technique=technique,
        )


__all__ = [
    "HEAP_TECHNIQUE_SCHEMA",
    "UAF_SCAN_SCHEMA",
    "HEAP_TECHNIQUE_CATALOG_SCHEMA",
    "DETECTABLE_TECHNIQUES",
    "HINT_TECHNIQUES",
    "TechniqueContext",
    "DETECTORS",
    "resolve_top_chunk",
    "technique_hints_for",
    "GlibcTechniqueAnalyzer",
]
