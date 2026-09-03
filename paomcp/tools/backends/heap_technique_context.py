"""Shared context for glibc heap-technique detectors.

Split out of :mod:`paomcp.tools.backends.heap_techniques`: the frozen
:class:`TechniqueContext` every detector reads, the one-bounded-read
``ReadFn`` contract, the top-chunk resolver, the scan budgets, and the
finding/iterator helpers.  Pure data + helpers -- no detector logic, no
catalog, no orchestration.
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
    TYPE_CHECKING,
)

from paomcp.tools.backends.foundation import mapping_for
from paomcp.tools.backends.heap_glibc_heap import SIZE_BITS_MASK

if TYPE_CHECKING:  # annotations only; heap.py wires instances at runtime
    from paomcp.tools.backends.heap_glibc_arenas import GlibcArenas
    from paomcp.tools.backends.heap_glibc_bins import GlibcBins

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
