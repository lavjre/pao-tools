"""jemalloc extent/rtree inspection (Phase 1: extent/rtree core only).

Runs pwndbg's ``jemalloc`` command surface (``heap``, ``find-extent``,
``extent-info``) on the gdb backend and parses the printed text into the
typed envelope.  pwndbg's plugin reads jemalloc's extent rtree from the
target at runtime (``rtree_levels`` + struct types via GDB expressions, with
a 48-bit fallback geometry), so it needs a jemalloc inferior with usable
type/symbol information and is officially tested against jemalloc 5.3.0.

Scope (user-confirmed): extent/rtree core only.  ``inspect_chunk`` and
``validate_chunk`` wrap ``find-extent`` (pointer -> allocation lookup),
``list_arenas`` wraps ``jemalloc heap`` (extent inventory; jemalloc arenas
are not exposed by the plugin, so this reports the tracked extents), and
``inspect_arena`` wraps ``extent-info`` when given an extent address.  The
remaining eight facade methods -- fastbins (jemalloc has none), tcache/bin
walking, corruption/free/technique/UAF analysis, and glibc-shaped snapshot
capture -- return an honest typed ``unsupported`` verdict until a later
phase builds those readers.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    parse_int,
)
from paomcp.tools.backends.heap import (
    PARSER_VERSION,
    ARENA_INSPECTION_SCHEMA,
    ARENA_LIST_SCHEMA,
    BIN_INSPECTION_SCHEMA,
    CHUNK_INSPECTION_SCHEMA,
    CHUNK_VALIDATION_SCHEMA,
    DEFAULT_MAX_CHUNKS_PER_FASTBIN,
    DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
    DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
    FASTBIN_INSPECTION_SCHEMA,
    FREE_VALIDATION_SCHEMA,
    HEAP_CORRUPTION_SCHEMA,
    HEAP_SNAPSHOT_SCHEMA,
    HEAP_TECHNIQUE_SCHEMA,
    TCACHE_INSPECTION_SCHEMA,
    UAF_SCAN_SCHEMA,
)

JEMALLOC_EXTENT_SCHEMA = "pao.jemalloc-extent.v1"
JEMALLOC_EXTENT_LIST_SCHEMA = "pao.jemalloc-extent-list.v1"

# pwndbg's jemalloc plugin prints a fixed compatibility warning on every
# invocation; it is surfaced once per response instead of littering the
# parsed output.
_JEMALLOC_WARNING = (
    "pwndbg jemalloc support is tested only for jemalloc 5.3.0; "
    "older/newer layouts may misparse"
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_KEY_VALUE_RE = re.compile(r"^([A-Za-z][A-Za-z ]*): (.*)$")


def _console_text(result: Dict[str, Any]) -> str:
    """Join console-stream records from an execute_command result, ANSI-free."""
    parts: List[str] = []
    for response in result.get("responses", []):
        if not isinstance(response, dict) or response.get("type") != "console":
            continue
        payload = response.get("payload")
        if isinstance(payload, str):
            parts.append(payload)
    return _ANSI_RE.sub("", "".join(parts))


def _coerce(value: Optional[str]) -> Any:
    """Coerce a printed ``Key: value`` string to int/bool when possible."""
    if value is None:
        return None
    text = value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return int(text, 16) if text.startswith(("0x", "0X")) else int(text)
    except ValueError:
        return text


def _parse_key_values(text: str) -> Dict[str, Any]:
    """Parse ``Key: value`` lines from a single extent-info block."""
    parsed: Dict[str, Any] = {}
    for line in text.splitlines():
        match = _KEY_VALUE_RE.match(line.strip())
        if match:
            parsed[match.group(1).strip()] = _coerce(match.group(2).strip())
    return parsed


def _parse_extent_blocks(text: str) -> List[Dict[str, Any]]:
    """Parse the repeated extent-info blocks printed by ``jemalloc heap``."""
    blocks: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = {}
            continue
        match = _KEY_VALUE_RE.match(line)
        if match:
            current[match.group(1).strip()] = _coerce(match.group(2).strip())
    if current:
        blocks.append(current)
    return blocks


def _extent_from_keyvalues(values: Dict[str, Any]) -> Dict[str, Any]:
    """Map parsed key-values to the canonical extent field names."""
    return {
        "pointer_address": values.get("Pointer Address"),
        "extent_address": values.get("Extent Address"),
        "allocated_address": values.get("Allocated Address"),
        "size": values.get("Size"),
        "has_slab": values.get("Small class"),
        "state": values.get("State"),
    }


class JemallocAnalyzer:
    """Analyzer for the jemalloc family via pwndbg's rtree-backed plugin."""

    family: Optional[str] = "jemalloc"

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    def _require_live(self) -> None:
        """Raise a typed error unless the target is live and stopped."""
        LiveInferiorGuard.require_live(
            self.facade.gdb.get_last_stop_record(),
            self.facade.gdb.get_inferior_pid(),
            detail=(
                "jemalloc inspection requires a live, stopped inferior; run "
                "the target to a stop before inspecting its extents"
            ),
        )

    # -- context -----------------------------------------------------
    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Live-check, classify, and resolve the target profile."""
        self._require_live()
        mappings, libcinfo, component_status = self.facade.discovery.collect_evidence(
            cancellation_check
        )
        classification = self.facade.discovery.classify(mappings, libcinfo)
        profile = self.facade.discovery.target_profile(mappings)
        return classification, profile, component_status

    def _allocator(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification["family"],
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
        }

    def _command_result(self, command: str) -> Tuple[Dict[str, Any], str]:
        result = self.facade.gdb.execute_command(command)
        return result, _console_text(result)

    # -- response builders -------------------------------------------
    def _unsupported(
        self,
        command: str,
        schema: str,
        *,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
        reason: str,
        address: Optional[int] = None,
        address_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": False,
            "verdict": "unsupported",
            "reason": reason,
            "component_status": component_status,
        }
        if address is not None:
            parsed["address"] = address
        if address_kind is not None:
            parsed["address_kind"] = address_kind
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
        )

    def _extent_response(
        self,
        command: str,
        *,
        schema: str,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
        address: int,
        address_kind: str,
        verdict: str,
        extent: Optional[Dict[str, Any]],
        text: str,
        raw_responses: List[Dict[str, Any]],
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
            "address": address,
            "address_kind": address_kind,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": True,
            "verdict": verdict,
            "extent": extent,
            "component_status": component_status,
        }
        if reason is not None:
            parsed["reason"] = reason
        warnings = [_JEMALLOC_WARNING]
        if "tested only for jemalloc" in text:
            warnings.append("target version below 5.3.0 is unsupported by pwndbg")
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=raw_responses,
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings,
        )

    # -- extent lookups ----------------------------------------------
    def _find_extent(
        self,
        command: str,
        address_int: int,
        address_kind: str,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
    ) -> Dict[str, Any]:
        result, text = self._command_result(command)
        raw = result.get("responses", [])
        if "Extent not found" in text:
            return self._extent_response(
                command,
                schema=JEMALLOC_EXTENT_SCHEMA,
                classification=classification,
                profile=profile,
                component_status=component_status,
                address=address_int,
                address_kind=address_kind,
                verdict="not_allocated",
                extent=None,
                text=text,
                raw_responses=raw,
                reason="the address is not tracked by any extent in the rtree",
            )
        error = _first_error(text)
        if error is not None:
            return self._extent_response(
                command,
                schema=JEMALLOC_EXTENT_SCHEMA,
                classification=classification,
                profile=profile,
                component_status=component_status,
                address=address_int,
                address_kind=address_kind,
                verdict="unresolvable",
                extent=None,
                text=text,
                raw_responses=raw,
                reason=error,
            )
        return self._extent_response(
            command,
            schema=JEMALLOC_EXTENT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            address=address_int,
            address_kind=address_kind,
            verdict="allocated",
            extent=_extent_from_keyvalues(_parse_key_values(text)),
            text=text,
            raw_responses=raw,
        )

    # -- AllocatorAnalyzer surface -----------------------------------
    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Resolve a pointer to its jemalloc extent (``jemalloc find-extent``)."""
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = f"jemalloc find-extent {address_int}"
        return self._find_extent(
            command,
            address_int,
            address_kind,
            classification,
            profile,
            component_status,
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Check whether an address is a live jemalloc allocation."""
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = f"jemalloc find-extent {address_int}"
        response = self._find_extent(
            command,
            address_int,
            address_kind,
            classification,
            profile,
            component_status,
        )
        verdict = response["parsed"]["verdict"]
        if verdict == "allocated":
            response["parsed"]["verdict"] = "valid"
        elif verdict == "not_allocated":
            response["parsed"]["verdict"] = "invalid"
        return response

    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """List the tracked jemalloc extents (``jemalloc heap``).

        jemalloc arenas are not exposed by pwndbg's plugin, so this reports
        the extent inventory the rtree walks -- the allocator's top-level
        view of what it currently manages.
        """
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = "jemalloc heap"
        result, text = self._command_result(command)
        raw = result.get("responses", [])
        if "No extents found" in text:
            parsed: Dict[str, Any] = {
                "schema": JEMALLOC_EXTENT_LIST_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "no_extents",
                "extent_count": 0,
                "extents": [],
                "component_status": component_status,
            }
            return build_backend_response(
                command,
                parsed=parsed,
                raw_responses=raw,
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        error = _first_error(text)
        if error is not None:
            parsed = {
                "schema": JEMALLOC_EXTENT_LIST_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "unresolvable",
                "extent_count": 0,
                "extents": [],
                "component_status": component_status,
                "reason": error,
            }
            return build_backend_response(
                command,
                parsed=parsed,
                raw_responses=raw,
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        blocks = _parse_extent_blocks(text)
        extents = [
            _extent_from_keyvalues(block)
            for block in blocks
            if block.get("Extent Address") is not None
        ]
        parsed = {
            "schema": JEMALLOC_EXTENT_LIST_SCHEMA,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": True,
            "verdict": "listed",
            "extent_count": len(extents),
            "extents": extents,
            "component_status": component_status,
        }
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=raw,
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
        )

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Inspect a jemalloc extent by address (``jemalloc extent-info``).

        jemalloc has no exposed arena address; pass an extent address.
        """
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        target = parse_int(arena_address) if arena_address is not None else None
        if target is None:
            return self._unsupported(
                "inspect_arena",
                ARENA_INSPECTION_SCHEMA,
                classification=classification,
                profile=profile,
                component_status=component_status,
                reason=(
                    "jemalloc arenas are not exposed by the pwndbg plugin; "
                    "pass an extent address to inspect it"
                ),
            )
        command = f"jemalloc extent-info {target}"
        result, text = self._command_result(command)
        raw = result.get("responses", [])
        error = _first_error(text)
        if error is not None:
            parsed: Dict[str, Any] = {
                "schema": JEMALLOC_EXTENT_SCHEMA,
                "address": target,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "unresolvable",
                "extent": None,
                "component_status": component_status,
                "reason": error,
            }
            return build_backend_response(
                command,
                parsed=parsed,
                raw_responses=raw,
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        values = _parse_key_values(text)
        extent = _extent_from_keyvalues(values)
        parsed = {
            "schema": JEMALLOC_EXTENT_SCHEMA,
            "address": target,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": True,
            "verdict": "parsed",
            "extent": extent,
            "component_status": component_status,
        }
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=raw,
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
        )

    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "inspect_fastbins",
            FASTBIN_INSPECTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "jemalloc has no fastbins (run-based slab bins); fastbin "
                "inspection is glibc-only"
            ),
        )

    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "inspect_tcache",
            TCACHE_INSPECTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "jemalloc tcache (TLS LIFO stack) inspection is not "
                "implemented in Phase 1; scope is extent/rtree core only"
            ),
        )

    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "inspect_bins",
            BIN_INSPECTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason="jemalloc bin inspection is not implemented in Phase 1",
        )

    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "classify_heap_corruption",
            HEAP_CORRUPTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "the glibc heap-corruption classifier is family-specific; "
                "jemalloc corruption analysis is not in Phase 1"
            ),
        )

    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        address_int = parse_int(address)
        return self._unsupported(
            "validate_free",
            FREE_VALIDATION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason="free validation is glibc-specific; not implemented for jemalloc",
            address=address_int,
            address_kind=address_kind,
        )

    def analyze_heap_technique(
        self,
        *,
        arena_address: Optional[str] = None,
        techniques: Optional[Sequence[str]] = None,
        scan_mmap_mappings: bool = True,
        mmap_max_mappings: int = 4,
        mmap_max_bytes: int = 1 << 20,
        max_overlap_chunks: int = 256,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "analyze_heap_technique",
            HEAP_TECHNIQUE_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "House-of-* / exploit-primitive detectors are glibc-specific; "
                "jemalloc technique analysis is not in Phase 1"
            ),
        )

    def detect_uaf(
        self,
        *,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "detect_uaf",
            UAF_SCAN_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "the UAF scan targets glibc freed-chunk bins; not implemented "
                "for jemalloc"
            ),
        )

    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._unsupported(
            "capture_heap_snapshot",
            HEAP_SNAPSHOT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                "snapshot capture is glibc bin-history based; not implemented "
                "for jemalloc"
            ),
        )


def _first_error(text: str) -> Optional[str]:
    """Return the first ``ERROR:`` message from a pwndbg jemalloc response."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ERROR:"):
            return line[len("ERROR:") :].strip()
    return None


__all__ = [
    "JemallocAnalyzer",
    "JEMALLOC_EXTENT_SCHEMA",
    "JEMALLOC_EXTENT_LIST_SCHEMA",
]
