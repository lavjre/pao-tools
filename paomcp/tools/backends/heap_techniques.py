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

The detector context, detectors, and catalog live in
:mod:`paomcp.tools.backends.heap_technique_context`,
:mod:`paomcp.tools.backends.heap_technique_detectors`, and
:mod:`paomcp.tools.backends.heap_technique_catalog`; this module keeps the
:class:`GlibcTechniqueAnalyzer` orchestration and re-exports the public
names so existing import spells keep working.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import LiveInferiorGuard
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.heap import (
    PARSER_VERSION,
    AllocatorDiscovery,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    GlibcHeapCorruption,
    build_backend_response,
)
from paomcp.tools.backends.heap_technique_catalog import (
    HEAP_TECHNIQUE_CATALOG_SCHEMA,
    HEAP_TECHNIQUE_SCHEMA,
    UAF_SCAN_SCHEMA,
    _CATALOG_BY_TECHNIQUE,
    _TECHNIQUE_CATALOG,
    _catalog_response,
    _valid_detector_names,
    technique_hints_for,
)
from paomcp.tools.backends.heap_technique_context import (
    DETECTABLE_TECHNIQUES,
    HINT_TECHNIQUES,
    ReadFn,
    TechniqueContext,
    _MMAP_SCAN_DEFAULT_MAX_BYTES,
    _MMAP_SCAN_DEFAULT_MAX_MAPPINGS,
    _MMAP_SCAN_PAGE,
    _OVERLAP_DEFAULT_MAX_CHUNKS,
    _UAF_DEFAULT_MAX_BYTES,
    _UAF_DEFAULT_MAX_CHUNKS,
    _UAF_DEFAULT_MAX_MAPPINGS,
    _UAF_MAX_CANDIDATES,
    _byteorder,
    _iter_fastbin_chunks,
    _iter_regular_chunks,
    _iter_tcache_chunks,
    budgeted_read,
    resolve_top_chunk,
)
from paomcp.tools.backends.heap_technique_detectors import DETECTORS
from paomcp.tools.backends.inspection import PwndbgInspection


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
        if mmap_max_mappings <= 0 or mmap_max_mappings > 64:
            raise ValueError(
                "mmap_max_mappings must be between 1 and 64, "
                f"got {mmap_max_mappings!r}"
            )
        if mmap_max_bytes <= 0 or mmap_max_bytes > 64 * 1024 * 1024:
            raise ValueError(
                f"mmap_max_bytes must be between 1 and 64MiB, got {mmap_max_bytes!r}"
            )
        if max_overlap_chunks <= 0 or max_overlap_chunks > 2048:
            raise ValueError(
                "max_overlap_chunks must be between 1 and 2048, "
                f"got {max_overlap_chunks!r}"
            )
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

        read: ReadFn = budgeted_read(self.bins._read_bytes)
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
        if max_bytes <= 0 or max_bytes > 64 * 1024 * 1024:
            raise ValueError(
                f"max_bytes must be between 1 and 64MiB, got {max_bytes!r}"
            )
        if max_mappings <= 0 or max_mappings > 64:
            raise ValueError(
                f"max_mappings must be between 1 and 64, got {max_mappings!r}"
            )
        if max_chunks <= 0 or max_chunks > 8192:
            raise ValueError(
                f"max_chunks must be between 1 and 8192, got {max_chunks!r}"
            )

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
