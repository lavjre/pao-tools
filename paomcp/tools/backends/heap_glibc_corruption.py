"""glibc heap corruption checks (split from heap_glibc)."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

if TYPE_CHECKING:  # annotations only; heap.py wires instances at runtime
    from paomcp.tools.backends.heap_glibc_arenas import GlibcArenas
    from paomcp.tools.backends.heap_glibc_bins import GlibcBins
    from paomcp.tools.backends.heap_glibc_heap import GlibcHeap

from paomcp.tools.backends.backend import BackendSpec
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection
from paomcp.tools.backends.heap_discovery import (
    ALLOCATOR_RUNTIME_COMMANDS,
    MUSL_MALLOCNG_MIN_VERSION,
    TYPED_INSPECTION_FAMILIES,
    AllocatorDiscovery,
    encode_safe_linked,
)
from paomcp.tools.backends.heap_schemas import (
    PARSER_VERSION,
    ALLOCATOR_DETECTION_SCHEMA,
    ALLOCATOR_CAPABILITIES_SCHEMA,
    SAFE_LINKING_MIN_VERSION,
    DEFAULT_MAX_CHUNKS_PER_FASTBIN,
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
    DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
    HEAP_SNAPSHOT_SCHEMA,
    HEAP_SNAPSHOT_LIST_SCHEMA,
    HEAP_SNAPSHOT_DIFF_SCHEMA,
    HEAP_SNAPSHOT_LINK_SCHEMA,
    DEFAULT_MAX_HEAP_SNAPSHOTS,
    DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
)
from paomcp.tools.backends.heap_snapshots import (
    HeapHistory,
    HeapSnapshotError,
)

from paomcp.tools.backends.heap_discovery import AllocatorDiscovery
from paomcp.tools.backends.heap_technique_catalog import technique_hints_for
from paomcp.tools.backends.heap_technique_context import (
    budgeted_read,
    resolve_top_chunk,
)

# ----------------------------------------------------------------------
# glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).


# ----------------------------------------------------------------------
# glibc heap corruption classifier
# ----------------------------------------------------------------------
HEAP_CORRUPTION_SCHEMA = "pao.glibc-heap-corruption.v1"
FREE_VALIDATION_SCHEMA = "pao.glibc-free-validation.v1"


class GlibcHeapCorruption:
    """Classify glibc heap corruption from live bin state.

    Runs the typed bin inspectors (fastbins, tcache, regular bins) and
    synthesizes their per-chain signals -- ``cycle_detected``, ``in_mapping``,
    ``chunk_readable``, and fd/bk pairs -- into a bounded, evidence-tagged list
    of corruption findings.  Each finding carries its category, severity, and
    the raw location that triggered it; the arena-level ``verdict`` is a
    conservative aggregate of the per-finding severities.  A heuristic is
    never promoted to a fact: everything reported has been observed in live
    memory this call.

    Categories (from the ``s6begin`` vocabulary):

    * ``cycle``                   -- a bin's fd chain loops
    * ``duplicate_entry``         -- one chunk address appears in >= 2 bin chains
    * ``fd_bk_inconsistency``     -- ``Y.bk != X`` in a regular-bin (X, Y) pair
    * ``alignment``               -- chunk address not aligned to ``2 * size_sz``
    * ``pointer_out_of_mapping``  -- chunk address lies outside every mapping
    * ``unreadable_chunk``        -- chunk header could not be read
    * ``unreadable_bin``          -- bin head read failed

    Severities:

    * ``corrupted``   -- cycle, duplicate, fd/bk, alignment, out_of_mapping
    * ``unreadable``  -- read failures (finding-level; conservative aggregate)

    Arena-level verdicts:

    * ``valid``        -- no findings
    * ``suspicious``   -- only unreadable findings
    * ``corrupted``    -- any corrupted-severity finding
    * ``unsupported``  -- target is not glibc
    * ``unresolvable`` -- bin layout could not be resolved via debug symbols
    """

    _CHECKS_RUN: Tuple[str, ...] = (
        "bin_cycle_detection",
        "cross_bin_duplicate_detection",
        "chunk_alignment",
        "chunk_in_mapping",
        "chunk_readable",
        "regular_bin_fd_bk_consistency",
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
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas
        self.bins = bins
        self.glibc = glibc

    # ------------------------------------------------------------------
    # Live-target helpers (mirror GlibcBins pattern)
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Corruption classification requires a live, stopped inferior; "
                "run the target to a stop before inspecting the heap"
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

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finding(
        *,
        category: str,
        severity: str,
        location: Dict[str, Any],
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        finding: Dict[str, Any] = {
            "category": category,
            "severity": severity,
            "location": location,
            "reason": reason,
        }
        if evidence is not None:
            finding["evidence"] = evidence
        return finding

    @staticmethod
    def _aggregate_verdict(findings: Sequence[Mapping[str, Any]]) -> str:
        if not findings:
            return "valid"
        severities = {str(f.get("severity")) for f in findings}
        if "corrupted" in severities:
            return "corrupted"
        # Read failures alone do not prove corruption -- surface as suspicious
        # so the agent knows to investigate rather than treating the heap as
        # invalid on the strength of a memory-read miss.
        return "suspicious"

    @staticmethod
    def _allocator_summary(classification: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

    # ------------------------------------------------------------------
    # classify_heap_corruption
    # ------------------------------------------------------------------
    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Classify corruption across an arena's fastbins, tcache and regular bins.

        Delegates to the typed bin inspectors so cycle detection, safe-linking
        decode, and per-chain traversal caps are reused.  The tcache is
        thread-local (not arena-scoped) so its findings always reflect the
        current thread regardless of ``arena_address``.
        """
        classification, _mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        allocator_out = self._allocator_summary(classification)

        if classification["family"] != "glibc":
            return build_backend_response(
                "classify_heap_corruption",
                parsed={
                    "schema": HEAP_CORRUPTION_SCHEMA,
                    "supported": False,
                    "verdict": "unsupported",
                    "reason": (
                        "heap corruption classification supports glibc only; "
                        f"detected family={classification['family']}"
                    ),
                    "arena_address": None,
                    "is_main_arena": None,
                    "allocator": allocator_out,
                    "target_profile": profile,
                    "findings": [],
                    "chunks_inspected": 0,
                    "bins_scanned": {"fastbins": 0, "tcache": 0, "regular": 0},
                    "checks_run": [],
                    "sources": {
                        "fastbins_verdict": None,
                        "tcache_verdict": None,
                        "bins_verdict": None,
                    },
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot classify corruption without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        alignment = 2 * (pointer_bits // 8)

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

        arena_addr_out: Optional[int] = fb_parsed.get("arena_address") or bn_parsed.get(
            "arena_address"
        )
        is_main_arena_out: Optional[bool]
        if fb_parsed.get("is_main_arena") is not None:
            is_main_arena_out = fb_parsed.get("is_main_arena")
        else:
            is_main_arena_out = bn_parsed.get("is_main_arena")
        sources = {
            "fastbins_verdict": fb_parsed.get("verdict"),
            "tcache_verdict": tc_parsed.get("verdict"),
            "bins_verdict": bn_parsed.get("verdict"),
        }

        if "unresolvable" in {sources["fastbins_verdict"], sources["bins_verdict"]}:
            return build_backend_response(
                "classify_heap_corruption",
                parsed={
                    "schema": HEAP_CORRUPTION_SCHEMA,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": (
                        "arena bin layout could not be resolved via debug "
                        "symbols; classification skipped"
                    ),
                    "arena_address": arena_addr_out,
                    "is_main_arena": is_main_arena_out,
                    "allocator": allocator_out,
                    "target_profile": profile,
                    "findings": [],
                    "chunks_inspected": 0,
                    "bins_scanned": {"fastbins": 0, "tcache": 0, "regular": 0},
                    "checks_run": [],
                    "sources": sources,
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        findings: List[Dict[str, Any]] = []
        chunks_by_address: Dict[int, List[Dict[str, Any]]] = {}
        chunks_inspected = 0

        # Fastbins
        fastbins_scanned = 0
        for bin_ in fb_parsed.get("fastbins", []):
            index = bin_.get("index")
            if bin_.get("read_error"):
                findings.append(
                    self._finding(
                        category="unreadable_bin",
                        severity="unreadable",
                        location={"bin_type": "fastbin", "bin_index": index},
                        reason=str(bin_.get("read_error")),
                    )
                )
                continue
            head = bin_.get("head")
            if head is None or head == 0:
                continue
            fastbins_scanned += 1
            if bin_.get("cycle_detected"):
                findings.append(
                    self._finding(
                        category="cycle",
                        severity="corrupted",
                        location={"bin_type": "fastbin", "bin_index": index},
                        reason=(
                            "fastbin fd chain cycles back to a previously-seen " "chunk"
                        ),
                        evidence={"chain_length": bin_.get("length")},
                    )
                )
            for entry in bin_.get("chain", []):
                chunks_inspected += 1
                self._analyze_entry(
                    entry=entry,
                    bin_type="fastbin",
                    bin_index=index,
                    alignment=alignment,
                    findings=findings,
                    chunks_by_address=chunks_by_address,
                )

        # Tcache (thread-local; only sweep when the inspector parsed successfully)
        tcache_scanned = 0
        if tc_parsed.get("verdict") == "parsed":
            for bin_ in tc_parsed.get("bins", []):
                index = bin_.get("index")
                head = bin_.get("head")
                if head is None or head == 0:
                    continue
                tcache_scanned += 1
                if bin_.get("cycle_detected"):
                    findings.append(
                        self._finding(
                            category="cycle",
                            severity="corrupted",
                            location={"bin_type": "tcache", "bin_index": index},
                            reason=(
                                "tcache chain cycles back to a previously-seen " "entry"
                            ),
                            evidence={"chain_length": bin_.get("length")},
                        )
                    )
                for entry in bin_.get("chain", []):
                    chunks_inspected += 1
                    self._analyze_entry(
                        entry=entry,
                        bin_type="tcache",
                        bin_index=index,
                        alignment=alignment,
                        findings=findings,
                        chunks_by_address=chunks_by_address,
                    )

        # Regular bins (unsorted / small / large)
        regular_scanned = 0
        for bin_ in bn_parsed.get("bins", []):
            if bin_.get("empty"):
                continue
            regular_scanned += 1
            index = bin_.get("index")
            category = bin_.get("category", "regular")
            if bin_.get("cycle_detected"):
                findings.append(
                    self._finding(
                        category="cycle",
                        severity="corrupted",
                        location={"bin_type": category, "bin_index": index},
                        reason=f"{category} bin fd chain cycles",
                        evidence={"chain_length": bin_.get("length")},
                    )
                )
            chain = bin_.get("chain", [])
            for entry in chain:
                chunks_inspected += 1
                self._analyze_entry(
                    entry=entry,
                    bin_type=category,
                    bin_index=index,
                    alignment=alignment,
                    findings=findings,
                    chunks_by_address=chunks_by_address,
                )
            self._check_fd_bk_consistency(chain=chain, bin_=bin_, findings=findings)

        # Cross-bin duplicate detection.  Each chain's own walker guarantees no
        # intra-chain repeats via cycle detection, so a repeated address here
        # is genuinely a cross-bin appearance -- a strong double-free signal.
        for chunk_addr, occurrences in chunks_by_address.items():
            if len(occurrences) < 2:
                continue
            findings.append(
                self._finding(
                    category="duplicate_entry",
                    severity="corrupted",
                    location={"chunk_address": chunk_addr},
                    reason=(
                        f"chunk 0x{chunk_addr:x} appears in {len(occurrences)} "
                        "distinct bin chains; possible double-free"
                    ),
                    evidence={"occurrences": occurrences},
                )
            )

        verdict = self._aggregate_verdict(findings)

        # Additive technique-analysis hints over the already-collected inspector
        # outputs.  Purely additive: never mutates findings, verdicts, or the
        # check list.
        top_chunk = resolve_top_chunk(
            bins=self.bins,
            arenas=self.arenas,
            mappings=_mappings,
            profile=profile,
            cancellation_check=cancellation_check,
        )
        technique_hints = technique_hints_for(
            fb_parsed=fb_parsed,
            tc_parsed=tc_parsed,
            bn_parsed=bn_parsed,
            classification=classification,
            mappings=_mappings,
            profile=profile,
            top_chunk=top_chunk,
            read=budgeted_read(self.bins._read_bytes),
        )

        return build_backend_response(
            "classify_heap_corruption",
            parsed={
                "schema": HEAP_CORRUPTION_SCHEMA,
                "supported": True,
                "verdict": verdict,
                "arena_address": arena_addr_out,
                "is_main_arena": is_main_arena_out,
                "allocator": allocator_out,
                "target_profile": profile,
                "findings": findings,
                "technique_hints": technique_hints,
                "chunks_inspected": chunks_inspected,
                "bins_scanned": {
                    "fastbins": fastbins_scanned,
                    "tcache": tcache_scanned,
                    "regular": regular_scanned,
                },
                "checks_run": list(self._CHECKS_RUN),
                "sources": sources,
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def _analyze_entry(
        self,
        *,
        entry: Mapping[str, Any],
        bin_type: str,
        bin_index: Optional[int],
        alignment: int,
        findings: List[Dict[str, Any]],
        chunks_by_address: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        chunk_addr = entry.get("chunk_address")
        if chunk_addr is None:
            return
        location = {
            "bin_type": bin_type,
            "bin_index": bin_index,
            "chunk_address": chunk_addr,
        }
        # The address itself is corrupt data even when the read failed, so
        # alignment / mapping are evaluated regardless of readability.
        if chunk_addr % alignment != 0:
            findings.append(
                self._finding(
                    category="alignment",
                    severity="corrupted",
                    location=location,
                    reason=(
                        f"chunk address 0x{chunk_addr:x} is not aligned to "
                        f"{alignment} bytes"
                    ),
                    evidence={"alignment": alignment},
                )
            )
        if entry.get("in_mapping") is False:
            findings.append(
                self._finding(
                    category="pointer_out_of_mapping",
                    severity="corrupted",
                    location=location,
                    reason=(
                        f"chunk address 0x{chunk_addr:x} lies outside every "
                        "known memory mapping"
                    ),
                )
            )
        # Different inspectors use different field names for the read flag:
        #   fastbins:      entry["readable"]
        #   tcache / bins: entry["chunk_readable"]
        readable = entry.get("chunk_readable")
        if readable is None:
            readable = entry.get("readable")
        if readable is False:
            findings.append(
                self._finding(
                    category="unreadable_chunk",
                    severity="unreadable",
                    location=location,
                    reason=str(
                        entry.get("chunk_read_error")
                        or entry.get("read_error")
                        or "chunk header could not be read"
                    ),
                )
            )
        chunks_by_address.setdefault(chunk_addr, []).append(
            {"bin_type": bin_type, "bin_index": bin_index}
        )

    def _check_fd_bk_consistency(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        bin_: Mapping[str, Any],
        findings: List[Dict[str, Any]],
    ) -> None:
        """Verify Y.bk == X for every consecutive (X, Y) pair in a regular bin.

        A truncated or cycle-detected walk is skipped -- the walker already
        surfaced those signals, and a pair check on a cut-off chain would just
        add noise.  Entries whose header read failed are skipped (bk is None).
        """
        if bin_.get("truncated") or bin_.get("cycle_detected"):
            return
        for i in range(len(chain) - 1):
            x = chain[i]
            y = chain[i + 1]
            if x.get("chunk_readable") is False or y.get("chunk_readable") is False:
                continue
            x_addr = x.get("chunk_address")
            y_addr = y.get("chunk_address")
            bk = y.get("bk")
            if x_addr is None or y_addr is None or bk is None:
                continue
            if bk != x_addr:
                findings.append(
                    self._finding(
                        category="fd_bk_inconsistency",
                        severity="corrupted",
                        location={
                            "bin_type": bin_.get("category"),
                            "bin_index": bin_.get("index"),
                            "chunk_address": y_addr,
                        },
                        reason=(
                            f"chunk 0x{y_addr:x} bk pointer (0x{bk:x}) does "
                            f"not point back to previous chain entry "
                            f"0x{x_addr:x}"
                        ),
                        evidence={"expected_bk": x_addr, "observed_bk": bk},
                    )
                )

    # ------------------------------------------------------------------
    # validate_free
    # ------------------------------------------------------------------
    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return a safe-to-free verdict for a glibc chunk.

        Runs the chunk parse + containment checks and cross-references the
        chunk against the current thread's tcache and main_arena's fastbins to
        catch a double-free before ``free()`` fires.  A chunk already present
        in tcache or fastbins is a double-free candidate; a chunk that fails
        alignment / mapping / min-size checks is broken by construction.

        Verdict vocabulary:

        * ``unsupported`` -- target is not glibc
        * ``unreadable``  -- chunk header could not be read
        * ``double_free`` -- chunk already present in tcache or fastbins
        * ``corrupted``   -- alignment / mapping / min-size check failed
        * ``suspicious`` -- non-fatal check failed (e.g. payload overflow)
        * ``safe``        -- every check passed
        """
        inspection = self.glibc.inspect_chunk(
            address=address,
            address_kind=address_kind,
            include_next=True,
            cancellation_check=cancellation_check,
        )
        parsed_inspect = inspection.get("parsed", {}) or {}
        allocator_out = parsed_inspect.get("allocator")
        target_profile = parsed_inspect.get("target_profile")

        if not parsed_inspect.get("supported"):
            return build_backend_response(
                "validate_free",
                parsed={
                    "schema": FREE_VALIDATION_SCHEMA,
                    "address": parsed_inspect.get("address"),
                    "address_kind": parsed_inspect.get("address_kind"),
                    "verdict": "unsupported",
                    "failures": [],
                    "checks": {},
                    "reason": parsed_inspect.get("reason"),
                    "chunk_summary": None,
                    "allocator": allocator_out,
                    "target_profile": target_profile,
                    "membership": {"tcache": None, "fastbins": None},
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=inspection.get("warnings"),
            )

        chunk = parsed_inspect.get("chunk") or {}
        containment = parsed_inspect.get("mapping_containment") or {}
        warnings: List[str] = list(inspection.get("warnings") or [])
        failures: List[str] = []
        checks: Dict[str, Dict[str, Any]] = {}
        summary: Optional[Dict[str, Any]] = None
        verdict: str

        if not chunk.get("readable"):
            checks["readable"] = {
                "ok": False,
                "reason": chunk.get("reason") or "chunk_header_unreadable",
            }
            failures.append("readable")
            verdict = "unreadable"
        else:
            checks["readable"] = {"ok": True, "reason": None}
            structural = chunk.get("structural_checks", {}) or {}
            alignment_ok = bool(structural.get("alignment_ok"))
            min_size_ok = bool(structural.get("min_size_ok"))
            chunk_in_mapping = bool(containment.get("chunk_in_mapping"))
            header_in_mapping = bool(containment.get("header_in_mapping"))
            payload_in_mapping = containment.get("payload_in_mapping")
            payload_ok = payload_in_mapping is None or bool(payload_in_mapping)

            checks["alignment"] = {
                "ok": alignment_ok,
                "reason": (
                    None if alignment_ok else "chunk_address_or_size_not_aligned"
                ),
            }
            checks["min_size"] = {
                "ok": min_size_ok,
                "reason": None if min_size_ok else "size_below_min_chunk_size",
            }
            checks["chunk_in_mapping"] = {
                "ok": chunk_in_mapping,
                "reason": (
                    None if chunk_in_mapping else "chunk_address_outside_any_mapping"
                ),
            }
            checks["header_in_mapping"] = {
                "ok": header_in_mapping,
                "reason": (
                    None if header_in_mapping else "chunk_header_crosses_mapping_end"
                ),
            }
            checks["payload_in_mapping"] = {
                "ok": payload_ok,
                "reason": (
                    None if payload_ok else "chunk_payload_extends_past_mapping_end"
                ),
            }
            if not alignment_ok:
                failures.append("alignment")
            if not min_size_ok:
                failures.append("min_size")
            if not chunk_in_mapping:
                failures.append("chunk_in_mapping")
            if not header_in_mapping:
                failures.append("header_in_mapping")
            if payload_in_mapping is False:
                failures.append("payload_in_mapping")

            summary = {
                "chunk_address": chunk.get("chunk_address"),
                "user_address": chunk.get("user_address"),
                "size": chunk.get("fields", {}).get("size"),
                "flags": chunk.get("fields", {}).get("flags"),
                "allocation_status": (parsed_inspect.get("allocation") or {}).get(
                    "status"
                ),
            }

            hard_failures = {
                "alignment",
                "min_size",
                "chunk_in_mapping",
                "header_in_mapping",
            }
            if any(name in hard_failures for name in failures):
                verdict = "corrupted"
            elif "payload_in_mapping" in failures:
                verdict = "suspicious"
            else:
                verdict = "safe"

        tcache_membership: Optional[Dict[str, Any]] = None
        fastbins_membership: Optional[Dict[str, Any]] = None
        if chunk.get("readable"):
            tcache_membership = self._locate_in_tcache(
                user_address=chunk.get("user_address"),
                cancellation_check=cancellation_check,
            )
            fastbins_membership = self._locate_in_fastbins(
                chunk_address=chunk.get("chunk_address"),
                cancellation_check=cancellation_check,
            )
            in_tcache = bool((tcache_membership or {}).get("found"))
            in_fastbins = bool((fastbins_membership or {}).get("found"))
            checks["not_in_tcache"] = {
                "ok": not in_tcache,
                "reason": (
                    None if not in_tcache else "chunk_already_present_in_tcache"
                ),
            }
            checks["not_in_fastbins"] = {
                "ok": not in_fastbins,
                "reason": (
                    None if not in_fastbins else "chunk_already_present_in_fastbins"
                ),
            }
            if in_tcache:
                failures.append("not_in_tcache")
            if in_fastbins:
                failures.append("not_in_fastbins")
            # Double-free takes precedence over any earlier structural verdict.
            if in_tcache or in_fastbins:
                verdict = "double_free"

        return build_backend_response(
            "validate_free",
            parsed={
                "schema": FREE_VALIDATION_SCHEMA,
                "address": parsed_inspect.get("address"),
                "address_kind": parsed_inspect.get("address_kind"),
                "verdict": verdict,
                "failures": failures,
                "checks": checks,
                "chunk_summary": summary,
                "allocator": allocator_out,
                "target_profile": target_profile,
                "mapping_containment": containment,
                "allocation": parsed_inspect.get("allocation"),
                "membership": {
                    "tcache": tcache_membership,
                    "fastbins": fastbins_membership,
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def _locate_in_tcache(
        self,
        *,
        user_address: Optional[int],
        cancellation_check: Optional[Callable[[], None]],
    ) -> Optional[Dict[str, Any]]:
        if user_address is None:
            return None
        response = self.bins.inspect_tcache(cancellation_check=cancellation_check)
        parsed = response.get("parsed", {}) or {}
        inspector_verdict = parsed.get("verdict")
        # ``uninitialized`` (NULL tcache) is a well-defined empty state -- the
        # chunk is trivially not in tcache.  ``unresolvable`` / ``unreadable``
        # cannot answer the membership question at all.
        if inspector_verdict == "uninitialized":
            return {"found": False, "checked": True}
        if inspector_verdict != "parsed":
            return {
                "found": False,
                "checked": False,
                "reason": parsed.get("reason") or inspector_verdict,
            }
        for bin_ in parsed.get("bins", []):
            for entry in bin_.get("chain", []):
                if entry.get("user_address") == user_address:
                    return {
                        "found": True,
                        "checked": True,
                        "bin_index": bin_.get("index"),
                        "chunk_address": entry.get("chunk_address"),
                    }
        return {"found": False, "checked": True}

    def _locate_in_fastbins(
        self,
        *,
        chunk_address: Optional[int],
        cancellation_check: Optional[Callable[[], None]],
    ) -> Optional[Dict[str, Any]]:
        if chunk_address is None:
            return None
        response = self.bins.inspect_fastbins(cancellation_check=cancellation_check)
        parsed = response.get("parsed", {}) or {}
        if parsed.get("verdict") != "parsed":
            return {
                "found": False,
                "checked": False,
                "reason": parsed.get("reason") or parsed.get("verdict"),
            }
        for bin_ in parsed.get("fastbins", []):
            for entry in bin_.get("chain", []):
                if entry.get("chunk_address") == chunk_address:
                    return {
                        "found": True,
                        "checked": True,
                        "bin_index": bin_.get("index"),
                    }
        return {"found": False, "checked": True}
