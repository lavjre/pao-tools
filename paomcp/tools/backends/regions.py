"""JIT / memory-region tracker -- allocator-independent vmmap classification.

JIT memory-region tracking: the Chrome piece.  V8's JIT heap is
anonymous + executable -- RWX during lazy compilation, W^X-flipped to plain
executable after code generation -- so "JIT memory surface" is findable purely
from the process's memory mappings, with no allocator parser and no V8
internals.  PartitionAlloc itself is deliberately NOT walked: its freelists are
per-PartitionPage and encoded ``((slot ^ random) << 16)``, its metadata changes
every Chrome version, and release builds are stripped -- there is no stable
structure to parse.

This tracker is therefore generic and allocator-agnostic: it consumes the same
``memory_maps()`` vmmap surface every other backend uses, classifies each
region on executable / writable / anonymous / file-backed axes, and answers
two questions any Chrome-pwn workflow needs:

* ``scan_jit_regions`` -- where is the JIT spray space?  (anonymous
  executable regions, RWX included, sorted by size)
* ``classify_memory_region`` -- does this pointer land in a mapped region,
  and what are its flags?  (feeds write-what-where planning)

Everything is read-only.  Unlike the libmalloc analyzer it never touches
``AllocatorDiscovery.collect_evidence``: that would drag in ``libc_info()`` and
family classification, both irrelevant to region tracking.  There is no family
gate -- a heap of any (or unknown) family can be region-tracked.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.heap import PARSER_VERSION

JIT_REGIONS_SCHEMA = "pao.jit-regions.v1"
MEMORY_REGION_SCHEMA = "pao.memory-region.v1"
REGION_WATCH_SCHEMA = "pao.region-watch.v1"
MAX_WATCH_BYTES = 64 * 1024 * 1024
MAX_WATCH_BASELINE_BYTES = 64 * 1024 * 1024
MAX_WATCHES = 32
MAX_REPORTED_RANGES = 64

# Region normalization constants.
_PERMISSION_CHARS = frozenset("rwxps-")

# Kernel-mapped special regions: executable but never JIT spray space.
_NON_JIT_SPECIAL = frozenset({"[vdso]", "[vsyscall]", "[vvar]", "[vectors]"})


def _permission_flags(
    permissions: Any,
) -> Tuple[Optional[Dict[str, bool]], Optional[str]]:
    """Decode a ``rwx?-style`` permission string; ``(None, warning)`` when unusable.

    Non-string or <3 chars -> skip (warning).  Unknown characters -> skip
    (warning).  Exactly 3 chars -> r/w/x decoded with private/shared unknown
    (warning).  4+ chars -> decode r/w/x plus the private/shared flag.
    """
    if not isinstance(permissions, str):
        return None, f"noncanonical permission string {permissions!r}; region skipped"
    perm = permissions.lower()
    if len(perm) < 3:
        return None, f"noncanonical permission string {permissions!r}; region skipped"
    if any(char not in _PERMISSION_CHARS for char in perm):
        return None, f"noncanonical permission string {permissions!r}; region skipped"
    flags = {
        "readable": perm[0] == "r",
        "writable": perm[1] == "w",
        "executable": perm[2] == "x",
    }
    if len(perm) < 4:
        return flags, "permission string is only 3 chars; private/shared unknown"
    flags["private"] = perm[3] == "p"
    flags["shared"] = perm[3] == "s"
    return flags, None


class JitRegionTracker:
    """JIT/memory-region tracker from vmmap entries (allocator-agnostic).

    Reads ``facade.discovery.inspection.memory_maps()`` directly and classifies
    each mapping on executable / writable / anonymous / file-backed axes.
    ``jit_candidate`` marks anonymous executable regions (the V8 JIT-heap
    signature); ``rwx`` marks writable-executable regions regardless of
    anonymity.
    """

    def __init__(self, facade: Any) -> None:
        self.facade = facade
        self._watches: Dict[str, Dict[str, Any]] = {}

    # -- context -----------------------------------------------------
    def _require_live(self) -> None:
        """Raise a typed error unless the target is live and stopped."""
        LiveInferiorGuard.require_live(
            self.facade.gdb.get_last_stop_record(),
            self.facade.gdb.get_inferior_pid(),
            detail=(
                "JIT/memory-region tracking requires a live, stopped inferior; "
                "run the target to a stop before scanning its memory mappings"
            ),
        )

    # -- target access ------------------------------------------------
    @staticmethod
    def _is_anonymous(file_field: Any) -> bool:
        """Return whether a vmmap ``file`` field denotes an anonymous mapping."""
        file_display = "" if file_field is None else str(file_field).strip()
        if not file_display:
            return True
        return file_display.startswith("[") and file_display.endswith("]")

    @classmethod
    def _normalize_region(
        cls, item: Dict[str, Any], warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Normalize one vmmap entry into a canonical region; ``None`` to skip.

        ``start``/``end`` are coerced with ``int(str(x), 16)`` inside
        try/except (the crash.py ``estimate_core_size`` pattern): a garbage
        value skips the entry silently, exactly like the rest of the
        vmmap-consuming backends.  Permission strings that cannot decode skip
        with a warning; short strings (3 chars) decode r/w/x but warn.
        """
        try:
            start = int(str(item.get("start")), 16)
            end = int(str(item.get("end")), 16)
        except (TypeError, ValueError):
            return None
        if end <= start:
            return None
        flags, warn = _permission_flags(item.get("permissions"))
        if flags is None:
            if warn:
                warnings.append(warn)
            return None
        if warn:
            warnings.append(warn)
        anonymous = cls._is_anonymous(item.get("file"))
        file_display = "" if item.get("file") is None else str(item.get("file")).strip()
        return {
            "start": start,
            "end": end,
            "size": end - start,
            "permissions": str(item.get("permissions")).lower(),
            "executable": flags["executable"],
            "writable": flags["writable"],
            "anonymous": anonymous,
            "jit_candidate": bool(
                flags["executable"]
                and anonymous
                and file_display not in _NON_JIT_SPECIAL
            ),
            "rwx": flags["writable"] and flags["executable"],
        }

    def _read_maps(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], bool]:
        """Return ``(regions, raw_mappings, warnings, available)``.

        ``available=False`` when ``memory_maps()`` failed or produced no usable
        parsed list (verdict ``unavailable``).  Cancellation is checked before
        the backend call, mirroring ``AllocatorDiscovery.collect_evidence``.
        """
        if cancellation_check is not None:
            cancellation_check()
        result = self.facade.discovery.inspection.memory_maps()
        if not result.get("success"):
            return [], [], [], False
        parsed = result.get("parsed")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("mappings"), list):
            return [], [], [], False
        # A live process always has at least one mapping; an empty set means
        # the region evidence itself is missing, not that no JIT region
        # exists -- surface that as ``unavailable`` rather than ``none_found``.
        if not parsed["mappings"]:
            return [], [], [], False
        warnings: List[str] = []
        raw_mappings: List[Dict[str, Any]] = []
        regions: List[Dict[str, Any]] = []
        for item in parsed["mappings"]:
            if not isinstance(item, dict):
                continue
            raw_mappings.append(item)
            region = self._normalize_region(item, warnings)
            if region is not None:
                regions.append(region)
        return regions, raw_mappings, warnings, True

    # -- facade surface ----------------------------------------------
    def scan_jit_regions(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """List anonymous executable mappings (V8 JIT-heap candidates)."""
        self._require_live()
        regions, _raw, warnings, available = self._read_maps(cancellation_check)
        if not available:
            verdict = "unavailable"
            reason = "memory maps unavailable or unparseable"
            candidates: List[Dict[str, Any]] = []
        else:
            candidates = sorted(
                (r for r in regions if r["jit_candidate"]),
                key=lambda r: r["size"],
                reverse=True,
            )
            if not candidates:
                verdict = "none_found"
                reason = None
            else:
                verdict = "listed"
                reason = None
        parsed: Dict[str, Any] = {
            "schema": JIT_REGIONS_SCHEMA,
            "verdict": verdict,
            "supported": verdict in ("listed", "none_found"),
            "regions": candidates,
            "summary": {
                "count": len(candidates),
                "total_bytes": sum(r["size"] for r in candidates),
                "largest_region": candidates[0] if candidates else None,
            },
            "component_status": {"mappings": available},
        }
        if reason is not None:
            parsed["reason"] = reason
        return build_backend_response(
            "scan_jit_regions",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings,
        )

    def classify_memory_region(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Resolve the memory mapping containing ``address``."""
        self._require_live()
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        regions, raw_mappings, warnings, available = self._read_maps(cancellation_check)
        if not available:
            parsed: Dict[str, Any] = {
                "schema": MEMORY_REGION_SCHEMA,
                "address": address_int,
                "address_kind": address_kind,
                "verdict": "unavailable",
                "supported": False,
                "region": None,
                "component_status": {"mappings": False},
                "reason": "memory maps unavailable or unparseable",
            }
        else:
            raw = mapping_for(address_int, raw_mappings)
            region: Optional[Dict[str, Any]] = None
            if raw is None:
                verdict = "unmapped"
            else:
                region = self._normalize_region(raw, warnings)
                if region is None:
                    # The address IS inside a mapping but the entry cannot be
                    # normalized (garbage start/end or undecodable permission
                    # string).  That is not "unmapped" -- call it what it is
                    # so a caller never mistakes a parse failure for an
                    # unmapped address (which would change planning).
                    verdict = "unparseable"
                else:
                    verdict = "mapped"
            parsed = {
                "schema": MEMORY_REGION_SCHEMA,
                "address": address_int,
                "address_kind": address_kind,
                "verdict": verdict,
                "supported": True,
                "region": region,
                "component_status": {"mappings": True},
            }
        return build_backend_response(
            "classify_memory_region",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings,
        )

    def classify_pa_regions(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Flag PartitionAlloc-like super pages (region classifier, no freelist walk)."""
        self._require_live()
        _regions, raw_mappings, warnings, available = self._read_maps(
            cancellation_check
        )
        if not available:
            parsed: Dict[str, Any] = {
                "schema": PA_REGIONS_SCHEMA,
                "verdict": "unavailable",
                "supported": False,
                "candidates": [],
                "reason": "memory maps unavailable or unparseable",
            }
        else:
            candidates = classify_pa_candidates(raw_mappings)
            parsed = {
                "schema": PA_REGIONS_SCHEMA,
                "verdict": "listed" if candidates else "none_found",
                "supported": True,
                "candidates": candidates,
                "candidate_count": len(candidates),
                "total_bytes": sum(int(c["size"]) for c in candidates),
                "limit_note": (
                    "super-page shape heuristic only; the encoded "
                    "((slot ^ random) << 16) freelist is never decoded here"
                ),
            }
        return build_backend_response(
            "classify_pa_regions",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings,
        )

    # -- write-watch (JIT code-emission detection) -------------------
    def _read_exact(self, address: int, length: int) -> Optional[bytes]:
        """Read exactly ``length`` bytes or return None (never partial)."""
        result = self.facade.gdb.read_memory_bytes(f"0x{address:x}", length)
        if not result.get("success"):
            return None
        contents = DebuggerOperations._memory_contents(result)
        if contents is None:
            return None
        try:
            raw = bytes.fromhex(str(contents))
        except ValueError:
            return None
        return raw if len(raw) == length else None

    @staticmethod
    def _changed_ranges(before: bytes, after: bytes) -> List[List[int]]:
        """Coalesce differing offsets into [start, end) ranges (capped)."""
        ranges: List[List[int]] = []
        start: Optional[int] = None
        for offset, (old, new) in enumerate(zip(before, after)):
            if old != new:
                if start is None:
                    start = offset
            elif start is not None:
                ranges.append([start, offset])
                start = None
                if len(ranges) >= MAX_REPORTED_RANGES:
                    return ranges
        if start is not None and len(ranges) < MAX_REPORTED_RANGES:
            ranges.append([start, len(before)])
        return ranges

    def arm_region_watch(
        self,
        *,
        address: str,
        length: int,
        label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Snapshot a region hash; later checks report JIT writes as diffs."""
        self._require_live()
        base = parse_int(address)
        if base is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        if length <= 0 or length > MAX_WATCH_BYTES:
            raise ValueError(f"length must be between 1 and {MAX_WATCH_BYTES}")
        if len(self._watches) >= MAX_WATCHES:
            raise ValueError(f"at most {MAX_WATCHES} region watches per session")
        retained = sum(len(w["snapshot"]) for w in self._watches.values())
        if retained + length > MAX_WATCH_BASELINE_BYTES:
            raise ValueError(
                "region-watch baseline budget exhausted "
                f"({retained} of {MAX_WATCH_BASELINE_BYTES} bytes retained); "
                "disarm a watch or watch a smaller range"
            )
        snapshot = self._read_exact(base, length)
        if snapshot is None:
            return build_backend_response(
                "arm_region_watch",
                parsed={
                    "schema": REGION_WATCH_SCHEMA,
                    "supported": True,
                    "verdict": "unreadable",
                    "reason": "region bytes could not be read at arm time",
                    "watch_id": None,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        watch_id = uuid.uuid4().hex[:12]
        self._watches[watch_id] = {
            "watch_id": watch_id,
            "address": base,
            "length": length,
            "label": label,
            "sha256": hashlib.sha256(snapshot).hexdigest(),
            "snapshot": snapshot,
        }
        return build_backend_response(
            "arm_region_watch",
            parsed={
                "schema": REGION_WATCH_SCHEMA,
                "supported": True,
                "verdict": "armed",
                "watch_id": watch_id,
                "address": base,
                "length": length,
                "label": label,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    @staticmethod
    def _public_watch(watch: Dict[str, Any]) -> Dict[str, Any]:
        """Watch descriptor without the retained baseline bytes.

        Baselines live server-side for diffing; echoing them would return
        up to 64 MiB of raw bytes per watch inside the response itself.
        """
        return {k: v for k, v in watch.items() if k != "snapshot"}

    def check_region_watches(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-hash every armed region; report changed byte ranges."""
        self._require_live()
        reports: List[Dict[str, Any]] = []
        for watch in list(self._watches.values()):
            if cancellation_check is not None:
                cancellation_check()
            current = self._read_exact(watch["address"], watch["length"])
            if current is None:
                reports.append(
                    {
                        **self._public_watch(watch),
                        "verdict": "unreadable",
                        "changed": None,
                    }
                )
                continue
            digest = hashlib.sha256(current).hexdigest()
            if digest == watch["sha256"]:
                reports.append(
                    {
                        **self._public_watch(watch),
                        "verdict": "unchanged",
                        "changed": False,
                    }
                )
                continue
            # Re-read the baseline is impossible (it lived only as a hash),
            # so changed ranges need both buffers: keep the last hash AND
            # the last full bytes for armed watches.
            baseline = watch.get("snapshot")
            if baseline is None or len(baseline) != watch["length"]:
                reports.append(
                    {
                        **self._public_watch(watch),
                        "verdict": "changed",
                        "changed": True,
                        "changed_ranges": [],
                        "changed_range_count": 0,
                        "note": "baseline bytes not retained; re-arm for ranges",
                    }
                )
                continue
            ranges = self._changed_ranges(baseline, current)
            # Rebaseline to the observed bytes so the next check reports
            # only *new* writes instead of staying changed forever.
            watch["sha256"] = digest
            watch["snapshot"] = current
            reports.append(
                {
                    **self._public_watch(watch),
                    "verdict": "changed",
                    "changed": True,
                    "changed_ranges": ranges,
                    "changed_range_count": len(ranges),
                    "changed_bytes": sum(e - s for s, e in ranges),
                    "rebaselined": True,
                }
            )
        return build_backend_response(
            "check_region_watches",
            parsed={
                "schema": REGION_WATCH_SCHEMA,
                "watch_count": len(reports),
                "changed_count": sum(1 for r in reports if r.get("changed")),
                "watches": reports,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def disarm_region_watch(self, *, watch_id: str) -> Dict[str, Any]:
        """Drop one armed region watch."""
        removed = self._watches.pop(watch_id, None)
        return build_backend_response(
            "disarm_region_watch",
            parsed={
                "schema": REGION_WATCH_SCHEMA,
                "watch_id": watch_id,
                "removed": removed is not None,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )


PA_REGIONS_SCHEMA = "pao.pa-regions.v1"
_PA_SUPER_PAGE = 2 * 1024 * 1024
_PA_MIN_SIZE = 1 * 1024 * 1024
_PA_SLOT_HINTS = (64, 128, 256, 512, 1024, 2048, 4096)


def classify_pa_candidates(
    mappings: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Flag PartitionAlloc-like super pages from mappings (pure, no freelist walk).

    A candidate is an anonymous writable non-executable region >= 1 MiB whose
    size is a multiple of 1 MiB (super-page shaped). Confidence is ``medium``
    when the file marker names partition_alloc/chrome or the start is 2 MiB
    aligned, else ``low``. Slot sizes are hints, never facts: the encoded
    ``((slot ^ random) << 16)`` freelist is deliberately never decoded here.
    """
    import re

    out: List[Dict[str, Any]] = []
    for mp in mappings:
        try:
            start = int(str(mp.get("start")), 16)
            end = int(str(mp.get("end")), 16)
        except (TypeError, ValueError):
            continue
        size = end - start
        if size < _PA_MIN_SIZE:
            continue
        perms = str(mp.get("permissions") or "").lower()
        writable = bool(mp.get("writable")) or (len(perms) > 1 and perms[1] == "w")
        executable = bool(mp.get("executable")) or (len(perms) > 2 and perms[2] == "x")
        if not writable or executable:
            continue
        fname = str(mp.get("file") or "")
        if fname.startswith("[") and fname not in ("[heap]", "[anon]", "[anon:scudo]"):
            low = fname.lower()
            if "partition" not in low and "chrome" not in low and "anon" not in low:
                continue
        aligned = (start % _PA_SUPER_PAGE) == 0
        sized = (size % _PA_SUPER_PAGE) == 0
        marker = "partition" in fname.lower() or "chrome" in fname.lower()
        if not (aligned or sized or marker):
            continue
        confidence = "medium" if (marker or (aligned and sized)) else "low"
        out.append(
            {
                "start": start,
                "end": end,
                "size": size,
                "super_pages": max(1, (size + _PA_SUPER_PAGE - 1) // _PA_SUPER_PAGE),
                "aligned_2mib": aligned,
                "file": fname,
                "confidence": confidence,
                "slot_size_hints": list(_PA_SLOT_HINTS),
            }
        )
    out.sort(key=lambda r: int(r["size"]), reverse=True)
    return out


__all__ = [
    "JitRegionTracker",
    "JIT_REGIONS_SCHEMA",
    "MEMORY_REGION_SCHEMA",
    "REGION_WATCH_SCHEMA",
    "PA_REGIONS_SCHEMA",
    "classify_pa_candidates",
]
