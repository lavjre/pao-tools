"""Live-memory detectors for glibc heap exploitation techniques.

Split out of :mod:`paomcp.tools.backends.heap_techniques`: the thirteen
``_detect_*`` heuristics plus the :data:`DETECTORS` registry.  Each detector
reads a frozen :class:`TechniqueContext` and may perform at most one bounded
memory read; everything else comes from already-walked inspector output.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from paomcp.tools.backends.foundation import mapping_for
from paomcp.tools.backends.heap_glibc_heap import (
    IS_MMAPPED,
    PREV_INUSE,
    SIZE_BITS_MASK,
)
from paomcp.tools.backends.heap_technique_context import (
    ReadFn,
    TechniqueContext,
    _MMAP_SCAN_DEFAULT_MAX_BYTES,
    _MMAP_SCAN_DEFAULT_MAX_MAPPINGS,
    _detector_result,
    _is_heap_mapping,
    _iter_fastbin_chunks,
    _iter_regular_chunks,
    _iter_tcache_chunks,
    _not_present,
    _pointer_location,
    chunk_header_len,
    size_field_from_header,
)


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
        header_len = chunk_header_len(ctx.size_sz)
        header, err = read(physical_prev, header_len)
        if err is not None or len(header) < header_len:
            # Cannot read the real predecessor -- no evidence, stay silent.
            continue
        actual_prev_size = (
            size_field_from_header(header, ctx.size_sz, ctx.byteorder) & ~SIZE_BITS_MASK
        )
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
        header_len = chunk_header_len(ctx.size_sz)
        header, err = read(prev_addr, header_len)
        if err is not None or len(header) < header_len:
            continue
        prev_size_field = size_field_from_header(header, ctx.size_sz, ctx.byteorder)
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
