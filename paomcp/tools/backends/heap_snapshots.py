"""Heap snapshot history and diff (split from heap.py)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
import logging
from typing import TYPE_CHECKING
from paomcp.tools.backends.heap_discovery import AllocatorDiscovery
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

if TYPE_CHECKING:  # annotations only; heap.py owns the classes
    from paomcp.tools.backends.heap import GlibcArenas, GlibcBins

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# heap snapshot history and diff
# ----------------------------------------------------------------------
# Snapshot schemas, entry count and byte budgets live in heap_schemas (the
# single source re-exported above); only snapshot-local caps stay here.
DEFAULT_MAX_SNAPSHOT_LEDGER_ENTRIES = 100_000
DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = 300.0
MAX_SNAPSHOT_ARENAS = 256
MAX_SNAPSHOT_CHUNKS_PER_BIN = 16384
MAX_SNAPSHOT_LEDGER_ENTRIES = 1_000_000


class HeapSnapshotError(RuntimeError):
    """Raised when a heap snapshot cannot be resolved or is over budget."""


@dataclass
class _HeapSnapshotEntry:
    snapshot_id: str
    name: Optional[str]
    captured_at: str
    payload_json: bytes
    linked_crash_ids: List[str] = field(default_factory=list)

    @property
    def payload_bytes(self) -> int:
        return len(self.payload_json)


class HeapHistory:
    """Bounded, named-snapshot registry for the current session's heap state.

    Captures a typed heap snapshot by driving the sibling discovery / arena
    inspectors (allocator discovery, arena / bin / tcache walkers) into a
    single reproducible payload.  Snapshots are stored session-locally
    under the same bounded-history budget the crash triage uses (16
    entries and 64 MiB total by default; oldest-first eviction).

    A pure ``diff_ledgers`` classifier computes ``new`` / ``removed`` /
    ``moved`` chunk sets between two snapshots and lives as a
    ``@staticmethod`` so it stays unit-testable without a debugger.  A
    sibling ``diff_arena_summaries`` reports per-arena top-address deltas.
    Snapshots may be tagged with one or more crash IDs via
    :meth:`link_snapshot_to_crash`; the association is annotation-only and
    does not verify the crash exists on the crash-history side.

    Live per-allocation tracing (breakpoint-based malloc / free hooks) is
    intentionally deferred here: it needs breakpoint orchestration and
    symbol availability that must be an opt-in feature, and the
    snapshot-diff already covers the practical investigation surface.
    """

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
        max_snapshots: int = DEFAULT_MAX_HEAP_SNAPSHOTS,
        max_snapshot_bytes: int = DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
    ) -> None:
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive")
        if max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be positive")
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas
        self.bins = bins
        self._history: "OrderedDict[str, _HeapSnapshotEntry]" = OrderedDict()
        self._name_index: Dict[str, str] = {}
        self._history_bytes = 0
        self._history_lock = threading.RLock()
        self._max_snapshots = max_snapshots
        self._max_snapshot_bytes = max_snapshot_bytes

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Heap snapshot capture requires a live, stopped inferior; "
                "run the target to a stop before capturing the heap"
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
    # Pure helpers (unit-testable without a live inferior)
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_ledger_from_arena_bins(
        arena_address: Optional[int],
        fastbins_parsed: Optional[Mapping[str, Any]],
        bins_parsed: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Flatten one arena's fastbins + regular bins into ledger entries."""
        ledger: List[Dict[str, Any]] = []
        if fastbins_parsed and fastbins_parsed.get("verdict") == "parsed":
            for bin_ in fastbins_parsed.get("fastbins", []) or []:
                bin_index = bin_.get("index")
                for entry in bin_.get("chain", []) or []:
                    addr = entry.get("chunk_address")
                    if addr is None:
                        continue
                    ledger.append(
                        {
                            "chunk_address": int(addr),
                            "size": entry.get("size"),
                            "arena_address": arena_address,
                            "location": {"kind": "fastbin", "index": bin_index},
                        }
                    )
        if bins_parsed and bins_parsed.get("verdict") == "parsed":
            for bin_ in bins_parsed.get("bins", []) or []:
                bin_index = bin_.get("index")
                bin_category = bin_.get("category")
                for entry in bin_.get("chain", []) or []:
                    addr = entry.get("chunk_address")
                    if addr is None:
                        continue
                    ledger.append(
                        {
                            "chunk_address": int(addr),
                            "size": entry.get("size"),
                            "arena_address": arena_address,
                            "location": {
                                "kind": bin_category or "regular",
                                "index": bin_index,
                            },
                        }
                    )
        return ledger

    @staticmethod
    def _chunk_ledger_from_tcache(
        tcache_parsed: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        if not tcache_parsed or tcache_parsed.get("verdict") != "parsed":
            return ledger
        for bin_ in tcache_parsed.get("bins", []) or []:
            bin_index = bin_.get("index")
            for entry in bin_.get("chain", []) or []:
                addr = entry.get("chunk_address")
                if addr is None:
                    continue
                ledger.append(
                    {
                        "chunk_address": int(addr),
                        "size": entry.get("size"),
                        "arena_address": None,
                        "location": {"kind": "tcache", "index": bin_index},
                    }
                )
        return ledger

    @staticmethod
    def diff_ledgers(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Compute new / removed / moved chunks between two ledgers.

        Address is the identity key.  A chunk with the same address in
        both ledgers whose location set differs is reported as ``moved``.
        A single address may legitimately appear multiple times per
        snapshot (cross-bin double-free), and each occurrence is
        preserved in the ``locations`` list.
        """

        def _index(
            items: Sequence[Mapping[str, Any]],
        ) -> Dict[int, Dict[str, Any]]:
            out: Dict[int, Dict[str, Any]] = {}
            for entry in items:
                addr = entry.get("chunk_address")
                if addr is None:
                    continue
                bucket = out.setdefault(
                    int(addr),
                    {
                        "chunk_address": int(addr),
                        "size": entry.get("size"),
                        "locations": [],
                    },
                )
                bucket["locations"].append(entry.get("location"))
            return out

        before_index = _index(before)
        after_index = _index(after)

        new_chunks: List[Dict[str, Any]] = []
        removed_chunks: List[Dict[str, Any]] = []
        moved_chunks: List[Dict[str, Any]] = []

        for addr in sorted(after_index.keys() - before_index.keys()):
            entry = after_index[addr]
            new_chunks.append(
                {
                    "chunk_address": addr,
                    "size": entry["size"],
                    "locations_after": entry["locations"],
                }
            )
        for addr in sorted(before_index.keys() - after_index.keys()):
            entry = before_index[addr]
            removed_chunks.append(
                {
                    "chunk_address": addr,
                    "size": entry["size"],
                    "locations_before": entry["locations"],
                }
            )
        for addr in sorted(after_index.keys() & before_index.keys()):
            before_entry = before_index[addr]
            after_entry = after_index[addr]
            if before_entry["locations"] != after_entry["locations"]:
                moved_chunks.append(
                    {
                        "chunk_address": addr,
                        "size": after_entry["size"],
                        "locations_before": before_entry["locations"],
                        "locations_after": after_entry["locations"],
                    }
                )

        return {
            "new_chunks": new_chunks,
            "removed_chunks": removed_chunks,
            "moved_chunks": moved_chunks,
            "summary": {
                "new_count": len(new_chunks),
                "removed_count": len(removed_chunks),
                "moved_count": len(moved_chunks),
                "before_chunk_count": sum(
                    len(v["locations"]) for v in before_index.values()
                ),
                "after_chunk_count": sum(
                    len(v["locations"]) for v in after_index.values()
                ),
            },
        }

    @staticmethod
    def diff_arena_summaries(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Per-arena presence + top / last_remainder delta between snapshots."""

        def _by_addr(
            items: Sequence[Mapping[str, Any]],
        ) -> Dict[int, Mapping[str, Any]]:
            out: Dict[int, Mapping[str, Any]] = {}
            for arena in items:
                addr = arena.get("arena_address")
                if addr is None:
                    continue
                out[int(addr)] = arena
            return out

        before_map = _by_addr(before)
        after_map = _by_addr(after)
        addresses = sorted(set(before_map.keys()) | set(after_map.keys()))
        deltas: List[Dict[str, Any]] = []
        for addr in addresses:
            b = before_map.get(addr)
            a = after_map.get(addr)
            entry: Dict[str, Any] = {
                "arena_address": addr,
                "present_before": b is not None,
                "present_after": a is not None,
                "top_before": b.get("top") if b else None,
                "top_after": a.get("top") if a else None,
                "last_remainder_before": b.get("last_remainder") if b else None,
                "last_remainder_after": a.get("last_remainder") if a else None,
            }
            top_b = entry["top_before"]
            top_a = entry["top_after"]
            if isinstance(top_b, int) and isinstance(top_a, int):
                entry["top_delta"] = top_a - top_b
            else:
                entry["top_delta"] = None
            deltas.append(entry)
        return deltas

    # ------------------------------------------------------------------
    # Snapshot store (bounded, thread-safe)
    # ------------------------------------------------------------------
    def _remember(
        self,
        *,
        name: Optional[str],
        payload: Dict[str, Any],
    ) -> _HeapSnapshotEntry:
        snapshot_id = payload["snapshot_id"]
        payload_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        if len(payload_json) > self._max_snapshot_bytes:
            raise HeapSnapshotError(
                "Heap snapshot exceeds the bounded history byte budget"
            )
        entry = _HeapSnapshotEntry(
            snapshot_id=snapshot_id,
            name=name,
            captured_at=payload["captured_at"],
            payload_json=payload_json,
        )
        with self._history_lock:
            if name is not None:
                existing_id = self._name_index.pop(name, None)
                if existing_id is not None:
                    previous = self._history.pop(existing_id, None)
                    if previous is not None:
                        self._history_bytes -= previous.payload_bytes
            while self._history and (
                len(self._history) >= self._max_snapshots
                or self._history_bytes + entry.payload_bytes > self._max_snapshot_bytes
            ):
                _, removed = self._history.popitem(last=False)
                self._history_bytes -= removed.payload_bytes
                if removed.name is not None:
                    self._name_index.pop(removed.name, None)
            self._history[snapshot_id] = entry
            self._history_bytes += entry.payload_bytes
            if name is not None:
                self._name_index[name] = snapshot_id
        return entry

    def _resolve_snapshot(self, identifier: str) -> _HeapSnapshotEntry:
        if not identifier or not isinstance(identifier, str):
            raise HeapSnapshotError("snapshot identifier must be a non-empty string")
        with self._history_lock:
            entry = self._history.get(identifier)
            if entry is None:
                mapped_id = self._name_index.get(identifier)
                if mapped_id is not None:
                    entry = self._history.get(mapped_id)
            if entry is None:
                raise HeapSnapshotError(
                    f"heap snapshot not found for identifier '{identifier}'"
                )
            return entry

    def _clone(self, entry: _HeapSnapshotEntry) -> Dict[str, Any]:
        payload = json.loads(entry.payload_json)
        payload["linked_crash_ids"] = list(entry.linked_crash_ids)
        return payload

    def _descriptor(self, entry: _HeapSnapshotEntry) -> Dict[str, Any]:
        payload = json.loads(entry.payload_json)
        summary = payload.get("summary") or {}
        allocator = payload.get("allocator") or {}
        return {
            "snapshot_id": entry.snapshot_id,
            "name": entry.name,
            "captured_at": entry.captured_at,
            "payload_bytes": entry.payload_bytes,
            "allocator_family": allocator.get("family"),
            "verdict": payload.get("verdict"),
            "chunk_count": summary.get("chunk_count"),
            "arena_count": summary.get("arena_count"),
            "linked_crash_ids": list(entry.linked_crash_ids),
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        max_arenas: int = 16,
        max_ledger_entries: int = DEFAULT_MAX_SNAPSHOT_LEDGER_ENTRIES,
        timeout_seconds: Optional[float] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture and store a typed heap snapshot for this session."""
        if name is not None:
            if not isinstance(name, str):
                raise ValueError(
                    f"snapshot name must be a string, got {type(name).__name__}"
                )
            if not name.strip():
                raise ValueError("snapshot name may not be empty or whitespace-only")
        if max_chunks_per_bin <= 0 or max_chunks_per_bin > MAX_SNAPSHOT_CHUNKS_PER_BIN:
            raise ValueError(
                "max_chunks_per_bin must be between 1 and "
                f"{MAX_SNAPSHOT_CHUNKS_PER_BIN}, got {max_chunks_per_bin!r}"
            )
        if max_arenas <= 0 or max_arenas > MAX_SNAPSHOT_ARENAS:
            raise ValueError(
                f"max_arenas must be between 1 and {MAX_SNAPSHOT_ARENAS}, "
                f"got {max_arenas!r}"
            )
        if max_ledger_entries <= 0 or max_ledger_entries > MAX_SNAPSHOT_LEDGER_ENTRIES:
            raise ValueError(
                "max_ledger_entries must be between 1 and "
                f"{MAX_SNAPSHOT_LEDGER_ENTRIES}, got {max_ledger_entries!r}"
            )
        if timeout_seconds is None:
            timeout_seconds = float(
                os.environ.get(
                    "PAO_SNAPSHOT_TIMEOUT_SECONDS",
                    str(DEFAULT_SNAPSHOT_TIMEOUT_SECONDS),
                )
            )
        if timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive, got {timeout_seconds!r}"
            )
        deadline = time.monotonic() + timeout_seconds

        classification, _mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        pid = self.gdb.get_inferior_pid()
        epoch = self.gdb.get_inferior_epoch()
        allocator_summary = {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

        snapshot_id = uuid.uuid4().hex[:16]
        captured_at = datetime.now(timezone.utc).isoformat()

        if classification["family"] != "glibc":
            payload: Dict[str, Any] = {
                "schema": HEAP_SNAPSHOT_SCHEMA,
                "snapshot_id": snapshot_id,
                "name": name,
                "captured_at": captured_at,
                "session_id": self.session_id,
                "inferior": {"pid": pid, "epoch": epoch},
                "allocator": allocator_summary,
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    "heap snapshot supports glibc only; detected "
                    f"family={classification.get('family')}"
                ),
                "arenas": [],
                "chunk_ledger": [],
                "tcache_summary": None,
                "component_status": component_status,
                "summary": {"chunk_count": 0, "arena_count": 0},
            }
            entry = self._remember(name=name, payload=payload)
            return build_backend_response(
                "capture_heap_snapshot",
                parsed=self._clone(entry),
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        warnings: List[str] = []
        arena_list_response = self.arenas.list_arenas(
            cancellation_check=cancellation_check
        )
        arena_list_parsed = arena_list_response.get("parsed", {}) or {}
        for w in arena_list_response.get("warnings") or []:
            warnings.append(f"list_arenas: {w}")

        raw_arenas = arena_list_parsed.get("arenas", []) or []
        arenas_truncated = len(raw_arenas) > max_arenas
        if arenas_truncated:
            raw_arenas = raw_arenas[:max_arenas]
            warnings.append(
                f"Arena walk capped at {max_arenas} of "
                f"{len(arena_list_parsed.get('arenas', []) or [])} arenas; "
                "pass a larger max_arenas for full coverage"
            )
        arena_summaries: List[Dict[str, Any]] = []
        chunk_ledger: List[Dict[str, Any]] = []
        ledger_truncated = False

        def _check_deadline() -> None:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"heap snapshot exceeded its {timeout_seconds:g}s budget; "
                    "retry scoped with smaller max_arenas/max_chunks_per_bin"
                )

        _check_deadline()
        for arena_entry in raw_arenas:
            if cancellation_check is not None:
                cancellation_check()
            _check_deadline()
            addr = arena_entry.get("address")
            if addr is None:
                continue
            addr_int = int(addr)
            addr_str = str(addr_int)
            arena_inspect = self.arenas.inspect_arena(
                arena_address=addr_str, cancellation_check=cancellation_check
            )
            arena_parsed = arena_inspect.get("parsed", {}) or {}
            fields = arena_parsed.get("fields") or {}
            arena_summaries.append(
                {
                    "arena_address": addr_int,
                    "is_main_arena": arena_parsed.get("is_main"),
                    "top": fields.get("top"),
                    "last_remainder": fields.get("last_remainder"),
                    "system_mem": fields.get("system_mem"),
                    "next": fields.get("next"),
                    "verdict": arena_parsed.get("verdict"),
                }
            )
            for w in arena_inspect.get("warnings") or []:
                warnings.append(f"inspect_arena[{addr_int:#x}]: {w}")

            fastbins_response = self.bins.inspect_fastbins(
                arena_address=addr_str,
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=cancellation_check,
            )
            for w in fastbins_response.get("warnings") or []:
                warnings.append(f"inspect_fastbins[{addr_int:#x}]: {w}")
            bins_response = self.bins.inspect_bins(
                arena_address=addr_str,
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=cancellation_check,
            )
            for w in bins_response.get("warnings") or []:
                warnings.append(f"inspect_bins[{addr_int:#x}]: {w}")

            for ledger_entry in self._chunk_ledger_from_arena_bins(
                arena_address=addr_int,
                fastbins_parsed=fastbins_response.get("parsed"),
                bins_parsed=bins_response.get("parsed"),
            ):
                if len(chunk_ledger) >= max_ledger_entries:
                    ledger_truncated = True
                    break
                chunk_ledger.append(ledger_entry)
            if ledger_truncated:
                warnings.append(
                    f"Chunk ledger capped at {max_ledger_entries} entries; "
                    "pass a larger max_ledger_entries for full coverage"
                )
                break
            _check_deadline()

        tcache_response = self.bins.inspect_tcache(
            cancellation_check=cancellation_check
        )
        tcache_parsed = tcache_response.get("parsed", {}) or {}
        for w in tcache_response.get("warnings") or []:
            warnings.append(f"inspect_tcache: {w}")
        tcache_summary = {
            "verdict": tcache_parsed.get("verdict"),
            "bin_count": len(tcache_parsed.get("bins", []) or []),
        }
        for ledger_entry in self._chunk_ledger_from_tcache(tcache_parsed):
            if len(chunk_ledger) >= max_ledger_entries:
                if not ledger_truncated:
                    ledger_truncated = True
                    warnings.append(
                        f"Chunk ledger capped at {max_ledger_entries} entries; "
                        "pass a larger max_ledger_entries for full coverage"
                    )
                break
            chunk_ledger.append(ledger_entry)

        payload = {
            "schema": HEAP_SNAPSHOT_SCHEMA,
            "snapshot_id": snapshot_id,
            "name": name,
            "captured_at": captured_at,
            "session_id": self.session_id,
            "inferior": {"pid": pid, "epoch": epoch},
            "allocator": allocator_summary,
            "target_profile": profile,
            "supported": True,
            "verdict": "captured",
            "arenas": arena_summaries,
            "chunk_ledger": chunk_ledger,
            "tcache_summary": tcache_summary,
            "component_status": component_status,
            "summary": {
                "chunk_count": len(chunk_ledger),
                "arena_count": len(arena_summaries),
                "arenas_truncated": arenas_truncated,
                "ledger_truncated": ledger_truncated,
            },
        }
        entry = self._remember(name=name, payload=payload)
        return build_backend_response(
            "capture_heap_snapshot",
            parsed=self._clone(entry),
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def list_heap_snapshots(self) -> Dict[str, Any]:
        with self._history_lock:
            descriptors = [self._descriptor(entry) for entry in self._history.values()]
            total_bytes = self._history_bytes
        return build_backend_response(
            "list_heap_snapshots",
            parsed={
                "schema": HEAP_SNAPSHOT_LIST_SCHEMA,
                "snapshots": descriptors,
                "budget": {
                    "count": len(descriptors),
                    "max_count": self._max_snapshots,
                    "bytes": total_bytes,
                    "max_bytes": self._max_snapshot_bytes,
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def get_heap_snapshot(self, *, identifier: str) -> Dict[str, Any]:
        entry = self._resolve_snapshot(identifier)
        return build_backend_response(
            "get_heap_snapshot",
            parsed=self._clone(entry),
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def diff_heap_snapshots(self, *, before: str, after: str) -> Dict[str, Any]:
        before_entry = self._resolve_snapshot(before)
        after_entry = self._resolve_snapshot(after)
        if before_entry.snapshot_id == after_entry.snapshot_id:
            raise HeapSnapshotError(
                "cannot diff a snapshot against itself "
                f"('{before_entry.snapshot_id}')"
            )
        before_payload = json.loads(before_entry.payload_json)
        after_payload = json.loads(after_entry.payload_json)
        header_before = {
            "snapshot_id": before_entry.snapshot_id,
            "name": before_entry.name,
            "captured_at": before_entry.captured_at,
            "allocator_family": (before_payload.get("allocator") or {}).get("family"),
        }
        header_after = {
            "snapshot_id": after_entry.snapshot_id,
            "name": after_entry.name,
            "captured_at": after_entry.captured_at,
            "allocator_family": (after_payload.get("allocator") or {}).get("family"),
        }

        if not (
            before_payload.get("supported") is True
            and after_payload.get("supported") is True
        ):
            return build_backend_response(
                "diff_heap_snapshots",
                parsed={
                    "schema": HEAP_SNAPSHOT_DIFF_SCHEMA,
                    "before": header_before,
                    "after": header_after,
                    "supported": False,
                    "verdict": "unsupported",
                    "reason": (
                        "diff supports snapshots captured on glibc only; "
                        "one or both snapshots were unsupported"
                    ),
                    "chunk_diff": None,
                    "arena_deltas": [],
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        chunk_diff = self.diff_ledgers(
            before_payload.get("chunk_ledger") or [],
            after_payload.get("chunk_ledger") or [],
        )
        arena_deltas = self.diff_arena_summaries(
            before_payload.get("arenas") or [],
            after_payload.get("arenas") or [],
        )
        return build_backend_response(
            "diff_heap_snapshots",
            parsed={
                "schema": HEAP_SNAPSHOT_DIFF_SCHEMA,
                "before": header_before,
                "after": header_after,
                "supported": True,
                "verdict": "diffed",
                "chunk_diff": chunk_diff,
                "arena_deltas": arena_deltas,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def link_snapshot_to_crash(
        self, *, identifier: str, crash_id: str
    ) -> Dict[str, Any]:
        if not crash_id or not isinstance(crash_id, str):
            raise HeapSnapshotError("crash_id must be a non-empty string")
        with self._history_lock:
            entry = self._resolve_snapshot(identifier)
            if crash_id not in entry.linked_crash_ids:
                entry.linked_crash_ids.append(crash_id)
            linked = list(entry.linked_crash_ids)
        return build_backend_response(
            "link_snapshot_to_crash",
            parsed={
                "schema": HEAP_SNAPSHOT_LINK_SCHEMA,
                "snapshot_id": entry.snapshot_id,
                "name": entry.name,
                "linked_crash_ids": linked,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = ["HeapHistory", "HeapSnapshotError"]
