"""jemalloc extent/rtree inspection (Phase 2: + corruption/UAF/snapshot).

Runs pwndbg's ``jemalloc`` command surface (``heap``, ``find-extent``,
``extent-info``) on the gdb backend and parses the printed text into the
typed envelope.  pwndbg's plugin reads jemalloc's extent rtree from the
target at runtime (``rtree_levels`` + struct types via GDB expressions, with
a 48-bit fallback geometry), so it needs a jemalloc inferior with usable
type/symbol information and is officially tested against jemalloc 5.3.0.

Scope: ``inspect_chunk`` and ``validate_chunk`` wrap ``find-extent``
(pointer -> allocation lookup), ``list_arenas`` wraps ``jemalloc heap``
(extent inventory; jemalloc arenas are not exposed by the plugin, so this
reports the tracked extents), ``inspect_arena`` wraps ``extent-info`` when
given an extent address, and ``classify_heap_corruption`` /
``detect_uaf`` / ``capture_heap_snapshot`` work over the extent inventory
(overlap/duplicate/size checks, freed-state pointer scan, extent ledger).
Still ``unsupported``: fastbins (jemalloc has none), tcache/bin walking,
free validation, and House-of-* technique analysis.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, cast

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
JEMALLOC_CORRUPTION_SCHEMA = "pao.jemalloc-corruption.v1"
JEMALLOC_UAF_SCHEMA = "pao.jemalloc-uaf.v1"

_JEMALLOC_KNOWN_STATES = frozenset({"active", "inactive", "dirty", "muzzy", "retained"})
_JEMALLOC_FREED_STATES = frozenset({"inactive", "dirty", "muzzy", "retained"})


def _extent_range(extent: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    try:
        base = extent.get("extent_address")
        size = extent.get("size")
        if base is None or size is None:
            return None
        base_int = int(base)
        size_int = int(size)
        if size_int <= 0:
            return None
        return (base_int, base_int + size_int)
    except (TypeError, ValueError):
        return None


def _check_extent_overlaps(extents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranged: List[Tuple[int, int, int]] = []
    for idx, ext in enumerate(extents):
        rng = _extent_range(ext)
        if rng is not None:
            ranged.append((rng[0], rng[1], idx))
    ranged.sort()
    issues: List[Dict[str, Any]] = []
    for i in range(1, len(ranged)):
        prev_start, prev_end, prev_idx = ranged[i - 1]
        cur_start, cur_end, cur_idx = ranged[i]
        if cur_start < prev_end:
            issues.append(
                {
                    "check": "extent_overlap",
                    "severity": "corrupted",
                    "extent_a": extents[prev_idx].get("extent_address"),
                    "extent_b": extents[cur_idx].get("extent_address"),
                    "detail": f"extent {hex(int(extents[cur_idx].get('extent_address') or 0))} overlaps prior extent ending at {hex(prev_end)}",
                }
            )
    return issues


def _ledger_from_extents(extents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ledger: List[Dict[str, Any]] = []
    for ext in extents:
        addr = ext.get("extent_address")
        if addr is None:
            continue
        ledger.append(
            {
                "chunk_address": int(addr),
                "size": ext.get("size"),
                "arena_address": None,
                "location": {"kind": "jemalloc-extent", "state": ext.get("state")},
            }
        )
    return ledger


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

    def _prepare_with_mappings(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo, component_status = self.facade.discovery.collect_evidence(
            cancellation_check
        )
        classification = self.facade.discovery.classify(mappings, libcinfo)
        profile = self.facade.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    def _collect_extents(
        self,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        result, text = self._command_result("jemalloc heap")
        if "No extents found" in text:
            return [], None
        error = _first_error(text)
        if error is not None:
            return [], error
        blocks = _parse_extent_blocks(text)
        if not text.strip() or not blocks:
            return [], (
                "the jemalloc command produced no recognizable extent blocks; "
                "the pwndbg jemalloc plugin may be missing or the format is unknown"
            )
        extents = [
            _extent_from_keyvalues(block)
            for block in blocks
            if block.get("Extent Address") is not None
        ]
        return extents, None

    def _read_bytes(self, address: int, byte_count: int) -> Tuple[bytes, Optional[str]]:
        result = self.facade.gdb.read_memory_bytes(f"0x{address:x}", byte_count)
        if not isinstance(result, dict) or not result.get("success"):
            err = None
            if isinstance(result, dict):
                err = result.get("error")
            return b"", str(err or "memory read failed")
        contents = None
        try:
            from paomcp.tools.backends.debugger import DebuggerOperations

            contents = DebuggerOperations._memory_contents(result)
        except Exception:
            contents = None
        if contents is None:
            return b"", "memory read returned no contents"
        try:
            raw = bytes.fromhex(str(contents))
        except ValueError as exc:
            return b"", f"invalid hex contents: {exc}"
        return raw, None

    def _verify_family(
        self,
        command: str,
        schema: str,
        *,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
        address: Optional[int] = None,
        address_kind: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the typed ``unsupported`` envelope when the freshly-
        classified family is not jemalloc, else ``None``.

        The facade dispatches on a short TTL-cached family (``_FAMILY_CACHE_
        TTL_SEC``); this per-call re-verify makes that cache harmless -- a
        target that switched allocators inside the TTL window is never parsed
        with jemalloc commands, and the re-classified family from
        ``_prepare_context`` always wins over the dispatch-time guess.
        """
        if classification.get("family") == self.family:
            return None
        return self._unsupported(
            command,
            schema,
            classification=classification,
            profile=profile,
            component_status=component_status,
            reason=(
                f"the current target is not classified as {self.family}; "
                "jemalloc extent reads are not meaningful here"
            ),
            address=address,
            address_kind=address_kind,
        )

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
        # ``_JEMALLOC_WARNING`` (the static "tested only for jemalloc 5.3.0"
        # banner) is emitted once per response.  It must NOT be re-read from
        # the command text: pwndbg prints the banner on every jemalloc command
        # regardless of the actual target version, so keying a per-call
        # version warning off it would fire on every 5.3.0+ inspection.
        warnings = [_JEMALLOC_WARNING]
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
        values = _parse_key_values(text)
        if not text.strip() or not values:
            # The command produced no recognizable extent block: either the
            # plugin is missing (undefined command / empty output) or the
            # response format changed.  Report that honestly instead of
            # claiming the address is a live jemalloc allocation.
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
                reason=(
                    "the jemalloc command produced no recognizable extent "
                    "output; the pwndbg jemalloc plugin may be missing "
                    "(undefined command) or the response format is unknown"
                ),
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
            extent=_extent_from_keyvalues(values),
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
        guard = self._verify_family(
            command,
            JEMALLOC_EXTENT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            address=address_int,
            address_kind=address_kind,
        )
        if guard is not None:
            return guard
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
        """Check whether an address is a live jemalloc allocation.

        Verdict vocabulary: ``valid`` / ``invalid`` (the rtree lookup
        resolved the address to a live extent or not), or ``unresolvable``
        (the find-extent command could not run or produced no recognizable
        output).
        """
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = f"jemalloc find-extent {address_int}"
        guard = self._verify_family(
            command,
            JEMALLOC_EXTENT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            address=address_int,
            address_kind=address_kind,
        )
        if guard is not None:
            return guard
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
        guard = self._verify_family(
            command,
            JEMALLOC_EXTENT_LIST_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
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
        if not text.strip() or not blocks:
            parsed = {
                "schema": JEMALLOC_EXTENT_LIST_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "unresolvable",
                "extent_count": 0,
                "extents": [],
                "component_status": component_status,
                "reason": (
                    "the jemalloc command produced no recognizable extent "
                    "blocks; the pwndbg jemalloc plugin may be missing "
                    "(undefined command) or the response format is unknown"
                ),
            }
            return build_backend_response(
                command,
                parsed=parsed,
                raw_responses=raw,
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
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
        guard = self._verify_family(
            "inspect_arena",
            ARENA_INSPECTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            address=target,
        )
        if guard is not None:
            return guard
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
        if not values:
            return self._extent_response(
                command,
                schema=JEMALLOC_EXTENT_SCHEMA,
                classification=classification,
                profile=profile,
                component_status=component_status,
                address=target,
                address_kind="extent",
                verdict="unresolvable",
                extent=None,
                text=text,
                raw_responses=raw,
                reason=(
                    "extent-info produced no recognizable fields; nothing " "was parsed"
                ),
            )
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
                "implemented; scope is extent/rtree core only"
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
            reason="jemalloc bin inspection is not implemented (extent/rtree core only)",
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
        guard = self._verify_family(
            "classify_heap_corruption",
            JEMALLOC_CORRUPTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        extents, error = self._collect_extents(
            classification, profile, component_status
        )
        if error is not None:
            return build_backend_response(
                "classify_heap_corruption",
                parsed={
                    "schema": JEMALLOC_CORRUPTION_SCHEMA,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": error,
                    "checks": [],
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        checks: List[Dict[str, Any]] = []
        seen: Dict[int, int] = {}
        for ext in extents:
            addr = ext.get("extent_address")
            size = ext.get("size")
            state = ext.get("state")
            if addr is None:
                checks.append(
                    {
                        "check": "missing_address",
                        "severity": "suspicious",
                        "extent": ext,
                    }
                )
                continue
            addr_int = int(addr)
            if addr_int in seen:
                checks.append(
                    {
                        "check": "duplicate_extent",
                        "severity": "corrupted",
                        "extent_address": addr_int,
                        "detail": "same extent address listed twice",
                    }
                )
            seen[addr_int] = seen.get(addr_int, 0) + 1
            if not isinstance(size, int) or size <= 0:
                checks.append(
                    {
                        "check": "bad_size",
                        "severity": "corrupted",
                        "extent_address": addr_int,
                        "size": size,
                    }
                )
            if addr_int % 0x1000 != 0:
                checks.append(
                    {
                        "check": "misaligned_extent",
                        "severity": "suspicious",
                        "extent_address": addr_int,
                    }
                )
            if state is not None and state not in _JEMALLOC_KNOWN_STATES:
                checks.append(
                    {
                        "check": "unknown_state",
                        "severity": "suspicious",
                        "extent_address": addr_int,
                        "state": state,
                    }
                )
        checks.extend(_check_extent_overlaps(extents))
        if any(c.get("severity") == "corrupted" for c in checks):
            verdict = "corrupted"
        elif checks:
            verdict = "suspicious"
        else:
            verdict = "valid"
        return build_backend_response(
            "classify_heap_corruption",
            parsed={
                "schema": JEMALLOC_CORRUPTION_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": verdict,
                "extent_count": len(extents),
                "checks": checks,
                "failed_checks": [c["check"] for c in checks],
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
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
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        guard = self._verify_family(
            "detect_uaf",
            JEMALLOC_UAF_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        extents, error = self._collect_extents(
            classification, profile, component_status
        )
        if error is not None:
            return build_backend_response(
                "detect_uaf",
                parsed={
                    "schema": JEMALLOC_UAF_SCHEMA,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": error,
                    "candidates": [],
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        freed: List[Tuple[int, int]] = []
        for ext in extents[: max(0, max_chunks)]:
            state = ext.get("state")
            rng = _extent_range(ext)
            if rng is None:
                continue
            if state in _JEMALLOC_FREED_STATES:
                freed.append(rng)
        if not freed:
            return build_backend_response(
                "detect_uaf",
                parsed={
                    "schema": JEMALLOC_UAF_SCHEMA,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "supported": True,
                    "verdict": "clean",
                    "scanned_mappings": 0,
                    "scanned_bytes": 0,
                    "freed_extent_count": 0,
                    "candidates": [],
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        freed.sort()
        ptr_size = int(profile.get("pointer_bits") or 64) // 8
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        cands: List[Dict[str, Any]] = []
        scanned_mappings = 0
        scanned_bytes = 0
        writable = [m for m in mappings if m.get("writable") and m.get("readable")]
        for mp in writable[: max(0, max_mappings)]:
            if cancellation_check is not None:
                cancellation_check()
            try:
                start = int(str(mp.get("start")), 16)
                end = int(str(mp.get("end")), 16)
            except (TypeError, ValueError):
                continue
            size = max(0, end - start)
            if size <= 0:
                continue
            budget_left = max_bytes - scanned_bytes
            if budget_left <= 0:
                break
            want = min(size, budget_left, 65536)
            raw, _ = self._read_bytes(start, want)
            if not raw:
                continue
            scanned_mappings += 1
            scanned_bytes += len(raw)
            for off in range(0, len(raw) - ptr_size + 1, ptr_size):
                chunk = raw[off : off + ptr_size]
                if len(chunk) < ptr_size:
                    break
                try:
                    val = int.from_bytes(chunk, byteorder)
                except Exception:
                    continue
                for fstart, fend in freed:
                    if fstart <= val < fend:
                        cands.append(
                            {
                                "pointer_location": start + off,
                                "pointer_value": val,
                                "freed_range": [fstart, fend],
                            }
                        )
                        if len(cands) >= 128:
                            break
                    if len(cands) >= 128:
                        break
                if len(cands) >= 128:
                    break
            if len(cands) >= 128:
                break
        verdict = "suspicious" if cands else "clean"
        return build_backend_response(
            "detect_uaf",
            parsed={
                "schema": JEMALLOC_UAF_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": verdict,
                "scanned_mappings": scanned_mappings,
                "scanned_bytes": scanned_bytes,
                "freed_extent_count": len(freed),
                "candidates": cands,
                "candidate_count": len(cands),
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
        )

    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        max_arenas: int = 16,
        max_ledger_entries: int = 100_000,
        timeout_seconds: Optional[float] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        guard = self._verify_family(
            "capture_heap_snapshot",
            HEAP_SNAPSHOT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        deadline = time.monotonic() + float(timeout_seconds or 300.0)
        extents, error = self._collect_extents(
            classification, profile, component_status
        )
        if error is not None:
            return build_backend_response(
                "capture_heap_snapshot",
                parsed={
                    "schema": HEAP_SNAPSHOT_SCHEMA,
                    "supported": False,
                    "verdict": "unresolvable",
                    "reason": error,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_JEMALLOC_WARNING],
            )
        if time.monotonic() > deadline:
            raise TimeoutError("jemalloc snapshot exceeded its time budget")
        ledger_full = _ledger_from_extents(extents)
        truncated = len(ledger_full) > max_ledger_entries
        ledger = ledger_full[:max_ledger_entries]
        payload: Dict[str, Any] = {
            "schema": HEAP_SNAPSHOT_SCHEMA,
            "snapshot_id": uuid.uuid4().hex[:16],
            "name": name,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.facade.session_id,
            "inferior": {
                "pid": self.facade.gdb.get_inferior_pid(),
                "epoch": self.facade.gdb.get_inferior_epoch(),
            },
            "allocator": {
                "family": classification.get("family"),
                "variant": classification.get("variant"),
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
                "safe_linking": classification.get("safe_linking"),
            },
            "target_profile": profile,
            "supported": True,
            "verdict": "captured",
            "arenas": [],
            "chunk_ledger": ledger,
            "tcache_summary": None,
            "jemalloc_extents": extents[:max_ledger_entries],
            "component_status": component_status,
            "summary": {
                "chunk_count": len(ledger),
                "arena_count": 0,
                "extent_count": len(extents),
                "ledger_truncated": truncated,
            },
        }
        try:
            history = getattr(self.facade, "history", None)
            if history is not None and hasattr(history, "_remember"):
                entry = history._remember(name=name, payload=payload)
                stored = history._clone(entry)
                return build_backend_response(
                    "capture_heap_snapshot",
                    parsed=stored,
                    raw_responses=[],
                    state=self.facade.gdb.get_state(),
                    parser_version=PARSER_VERSION,
                    warnings=[_JEMALLOC_WARNING],
                )
        except Exception:
            pass
        return build_backend_response(
            "capture_heap_snapshot",
            parsed=payload,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_JEMALLOC_WARNING],
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
    "JEMALLOC_CORRUPTION_SCHEMA",
    "JEMALLOC_UAF_SCHEMA",
]
