"""JIT / memory-region tracker -- allocator-independent vmmap classification.

Stage 6.12 (Phase 4 of the roadmap): the Chrome piece.  V8's JIT heap is
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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.heap import PARSER_VERSION

JIT_REGIONS_SCHEMA = "pao.jit-regions.v1"
MEMORY_REGION_SCHEMA = "pao.memory-region.v1"

# Region normalization constants.
_PERMISSION_CHARS = frozenset("rwxps-")


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
        return {
            "start": start,
            "end": end,
            "size": end - start,
            "permissions": str(item.get("permissions")).lower(),
            "executable": flags["executable"],
            "writable": flags["writable"],
            "anonymous": anonymous,
            "jit_candidate": flags["executable"] and anonymous,
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
                    verdict = "unmapped"
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


__all__ = [
    "JitRegionTracker",
    "JIT_REGIONS_SCHEMA",
    "MEMORY_REGION_SCHEMA",
]
