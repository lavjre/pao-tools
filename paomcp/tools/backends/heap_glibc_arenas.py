"""glibc arena discovery (split from heap_glibc)."""

from __future__ import annotations

import copy
import json
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

from paomcp.tools.backends.heap_glibc_heap import glibc_arch_layout

from paomcp.tools.backends.heap_discovery import AllocatorDiscovery

# ----------------------------------------------------------------------
# glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).


# ----------------------------------------------------------------------
# glibc arenas and malloc_state parsing
# ----------------------------------------------------------------------

# Compile-time glibc constants (stable since glibc 2.14).  The parser reads
# them from source rather than probing gdb because they participate in the
# expected malloc_state layout: querying them wouldn't tell us anything the
# per-field offset lookups already do.
#
# ``NFASTBINS`` is architecture-dependent (see GLIBC_ARCH_LAYOUTS in
# heap_glibc_heap): 10 on x86_64/aarch64/arm32 but 11 on i386, where glibc's
# SIZE_SZ-based fastbin index step plus the 16-byte MALLOC_ALIGNMENT rounding
# in request2size() add a slot (the 0x60 fastbin).  This constant is the
# 64-bit default kept for callers that predate the per-arch table; code that
# knows the target profile must use :func:`nfastbins_for` /
# :func:`malloc_state_fields`.
NFASTBINS = 10
NBINS = 128
# glibc's bin array is ``mchunkptr bins[NBINS * 2 - 2]`` -- 254 entries on
# every ABI.  Each real bin occupies two slots (fd/bk), except the unsorted
# bin which shares its layout with the first small bin, hence ``- 2``.
BINS_ARRAY_LEN = NBINS * 2 - 2
BINMAPSIZE = 4

# Arena chain traversal cap.  Real programs cap arenas at roughly
# ``8 * cpu_count``; 256 leaves enormous headroom while still bounding a
# corrupted ring that never returns to main_arena.
MAX_ARENAS_TRAVERSAL = 256

ARENA_LIST_SCHEMA = "pao.glibc-arena-list.v1"
ARENA_INSPECTION_SCHEMA = "pao.glibc-arena.v1"

# ``(name, kind, count)`` for each malloc_state field parsed by the arena parser.
# ``kind`` is one of ``int32`` (4-byte int), ``int32_array`` (``count`` * 4
# bytes), ``ptr`` (single pointer of the target's pointer width), ``ptr_array``
# (``count`` pointers) or ``size_t`` (a single INTERNAL_SIZE_T, which glibc
# typedefs to size_t and therefore matches the pointer width).
_MALLOC_STATE_FIELDS: Tuple[Tuple[str, str, int], ...] = (
    ("mutex", "int32", 1),
    ("flags", "int32", 1),
    ("have_fastchunks", "int32", 1),
    ("fastbinsY", "ptr_array", NFASTBINS),
    ("top", "ptr", 1),
    ("last_remainder", "ptr", 1),
    ("bins", "ptr_array", BINS_ARRAY_LEN),
    ("binmap", "int32_array", BINMAPSIZE),
    ("next", "ptr", 1),
    ("next_free", "ptr", 1),
    ("attached_threads", "size_t", 1),
    ("system_mem", "size_t", 1),
    ("max_system_mem", "size_t", 1),
)


def nfastbins_for(pointer_bits: int, architecture: Optional[str] = None) -> int:
    """Length of ``malloc_state.fastbinsY`` for the target (10, or 11 on i386)."""
    return glibc_arch_layout(pointer_bits, architecture).nfastbins


def malloc_state_fields(
    pointer_bits: int, architecture: Optional[str] = None
) -> Tuple[Tuple[str, str, int], ...]:
    """``_MALLOC_STATE_FIELDS`` with the target's ``fastbinsY`` count."""
    nfastbins = nfastbins_for(pointer_bits, architecture)
    if nfastbins == NFASTBINS:
        return _MALLOC_STATE_FIELDS
    return tuple(
        (name, kind, nfastbins if name == "fastbinsY" else count)
        for name, kind, count in _MALLOC_STATE_FIELDS
    )


def arena_layout_cache_key(
    classification: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    family: str = "glibc",
    epoch: Optional[int] = None,
) -> Optional[tuple]:
    """Key a resolved arena layout by libc identity (no GDB calls).

    Field offsets depend only on the loaded libc -- never on the ASLR slide
    or inferior state -- so one ``resolve()`` per (family, version, libc
    file, pointer width, inferior epoch) serves every later inspection in
    the session. The epoch (bumped on every run/attach) retires entries
    across inferior restarts; pass ``None`` when the controller cannot
    report one. Returns ``None`` when no matching mapping is visible
    (static binaries): callers must then resolve fresh rather than risk a
    stale hit across different binaries.
    """
    if not isinstance(classification, Mapping) or not isinstance(profile, Mapping):
        return None
    if classification.get("family") != family:
        return None
    libc_file = None
    evidence = classification.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if (
                isinstance(item, Mapping)
                and item.get("family") == family
                and item.get("file")
            ):
                libc_file = str(item.get("file"))
                break
    if libc_file is None:
        libc = classification.get("libc")
        if isinstance(libc, Mapping) and libc.get("object"):
            libc_file = str(libc.get("object"))
    if libc_file is None:
        return None
    return (
        str(classification.get("family")),
        str(classification.get("version")),
        libc_file,
        profile.get("pointer_bits"),
        epoch,
    )


def inferior_epoch(gdb: Any) -> Optional[int]:
    """Read the inferior epoch counter when the controller reports one."""
    getter = getattr(gdb, "get_inferior_epoch", None)
    if not callable(getter):
        return None
    try:
        epoch = getter()
    except Exception:
        return None
    return epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None


class ArenaLayoutResolver:
    """Resolve main_arena's address and malloc_state field offsets via GDB.

    Uses GDB's ``-data-evaluate-expression`` to interrogate the running
    inferior for symbol addresses and struct field offsets.  The queries
    require libc built with debug symbols (``libc6-dbg`` on Debian /
    ``glibc-debuginfo`` on Fedora); with a stripped libc the resolver
    reports each unresolved field explicitly rather than guessing.  A
    version-keyed offset table for stripped libcs is out of scope for
    arena discovery and would live in a later component.
    """

    FIELD_NAMES: Tuple[str, ...] = tuple(name for name, _, _ in _MALLOC_STATE_FIELDS)

    def __init__(self, debugger: DebuggerOperations) -> None:
        self.debugger = debugger

    def _eval_int(self, expression: str) -> Optional[int]:
        result = self.debugger.evaluate_expression(expression)
        if not result.get("success"):
            return None
        parsed = result.get("parsed") or {}
        return parse_int(parsed.get("value"))

    def read_arena_address(self) -> Optional[int]:
        """Read the live ``main_arena`` address (one eval; ASLR-fresh)."""
        return self._eval_int("(void*)&main_arena")

    @staticmethod
    def _offset_expression(field_name: str) -> str:
        # Cast to (char*) so pointer arithmetic returns a byte offset rather
        # than a scaled index; the outer (unsigned long) yields a plain
        # decimal integer that ``parse_int`` handles uniformly.
        return f"(unsigned long)((char*)&main_arena.{field_name}" "-(char*)&main_arena)"

    def resolve(self) -> Dict[str, Any]:
        """Return the resolved layout or explicit unresolved fields."""
        arena_address = self._eval_int("(void*)&main_arena")
        struct_size = self._eval_int("(unsigned long)sizeof(struct malloc_state)")

        offsets: Dict[str, Optional[int]] = {}
        unresolved_fields: List[str] = []
        for name in self.FIELD_NAMES:
            value = self._eval_int(self._offset_expression(name))
            if value is None:
                unresolved_fields.append(name)
            offsets[name] = value

        return {
            "arena_address": arena_address,
            "struct_size": struct_size,
            "offsets": offsets,
            "unresolved_fields": unresolved_fields,
            "resolution_source": "gdb_debug_symbols",
        }


class GlibcArenas:
    """Enumerate glibc arenas and parse the malloc_state struct.

    Two operations:

    * ``list_arenas`` walks the circular ``main_arena.next`` chain with
      cycle detection and a bounded step count, so a corrupted heap can
      never hang the tool.
    * ``inspect_arena`` reads a full malloc_state struct and decodes its
      integer, pointer and array fields with the raw byte slice each field
      was derived from.

    Both refuse to run against a non-glibc target and refuse to guess when
    debug symbols cannot resolve the struct layout.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        discovery: "AllocatorDiscovery",
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.resolver = ArenaLayoutResolver(debugger)
        # Layout cache: offsets are libc-stable, so one resolve() per libc
        # identity serves the whole session (see arena_layout_cache_key).
        # Only resolvable layouts are cached; stripped-libc misses keep the
        # legacy always-resolve behavior instead of freezing "unresolvable".
        self._layout_cache: Optional[Tuple[tuple, Dict[str, Any]]] = None

    def cached_arena_layout(
        self,
        classification: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return the arena layout, resolving once per libc identity.

        The cached copy carries fresh ``arena_address``: ASLR slides the
        address every run while offsets stay put, so the address is always
        re-read (one eval) and spliced into the cached offsets.
        """
        key = arena_layout_cache_key(
            classification, profile, epoch=inferior_epoch(self.gdb)
        )
        if (
            key is not None
            and self._layout_cache is not None
            and self._layout_cache[0] == key
        ):
            layout = copy.deepcopy(self._layout_cache[1])
            # ASLR slides the address every run while offsets stay put, so
            # the address is always re-read (one eval). Failures are never
            # stored: the next call retries from the intact cached offsets.
            layout["arena_address"] = self.resolver.read_arena_address()
            return layout
        layout = self.resolver.resolve()
        if key is not None and layout.get("offsets", {}).get("next") is not None:
            stored = {k: v for k, v in layout.items() if k != "arena_address"}
            self._layout_cache = (key, copy.deepcopy(stored))
        return layout

    # ------------------------------------------------------------------
    # Pure parser core (no debugger required)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_arena_struct(
        raw_bytes: bytes,
        offsets: Mapping[str, Optional[int]],
        pointer_bits: int,
        byteorder: Optional[str],
        architecture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Decode a malloc_state from raw bytes plus resolved field offsets.

        Fields whose offset is ``None`` (unresolved) or whose byte slice
        falls outside ``raw_bytes`` are reported as ``None`` and listed
        under ``unreadable_fields``.  The caller can distinguish a truly
        zero value from an unreadable field via this list.
        ``architecture`` selects the per-arch ``fastbinsY`` array length
        (11 on i386, 10 elsewhere).
        """
        if pointer_bits not in (32, 64):
            raise ValueError(f"pointer_bits must be 32 or 64, got {pointer_bits!r}")
        size_sz = pointer_bits // 8
        order = byteorder if byteorder in ("little", "big") else "little"
        length = len(raw_bytes)

        fields: Dict[str, Any] = {}
        unreadable: List[str] = []

        def _read_int(offset: int, byte_count: int) -> Optional[int]:
            end = offset + byte_count
            if offset < 0 or end > length:
                return None
            return int.from_bytes(raw_bytes[offset:end], order)  # type: ignore[arg-type]

        def _read_ptr(offset: int) -> Optional[int]:
            return _read_int(offset, size_sz)

        for name, kind, count in malloc_state_fields(pointer_bits, architecture):
            offset = offsets.get(name)
            if offset is None:
                fields[name] = None
                unreadable.append(name)
                continue
            value: Any
            if kind == "int32":
                value = _read_int(offset, 4)
            elif kind == "int32_array":
                items = [_read_int(offset + i * 4, 4) for i in range(count)]
                value = None if any(v is None for v in items) else items
            elif kind == "ptr":
                value = _read_ptr(offset)
            elif kind == "ptr_array":
                items = [_read_ptr(offset + i * size_sz) for i in range(count)]
                value = None if any(v is None for v in items) else items
            elif kind == "size_t":
                value = _read_int(offset, size_sz)
            else:  # pragma: no cover -- constant table under our control
                value = None
            if value is None:
                unreadable.append(name)
            fields[name] = value

        return {
            "fields": fields,
            "unreadable_fields": unreadable,
        }

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Arena inspection requires a live, stopped inferior; run the "
                "target to a stop before inspecting the heap"
            ),
        )

    def _read_bytes(self, address: int, byte_count: int) -> Tuple[bytes, Optional[str]]:
        result = self.gdb.read_memory_bytes(f"0x{address:x}", byte_count)
        if not result.get("success"):
            return b"", "memory read failed"
        contents = DebuggerOperations._memory_contents(result)
        if contents is None:
            return b"", "memory read returned no contents"
        try:
            raw = bytes.fromhex(contents)
        except ValueError as exc:
            return b"", f"invalid hex contents: {exc}"
        if len(raw) < byte_count:
            return raw, f"short read: expected {byte_count} bytes, got {len(raw)}"
        return raw, None

    def _prepare_context(
        self,
        cancellation_check: Optional[Callable[[], None]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = AllocatorDiscovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    def _unsupported_response(
        self,
        command: str,
        schema: str,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
            },
            "target_profile": profile,
            "supported": False,
            "verdict": "unsupported",
            "reason": (
                f"arena parsing supports glibc only; detected "
                f"family={classification['family']!r}"
            ),
        }
        if extra:
            parsed.update(extra)
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=["Arena inspection refused to run against a non-glibc target"],
        )

    def _unresolvable_response(
        self,
        command: str,
        schema: str,
        layout: Dict[str, Any],
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
            },
            "target_profile": profile,
            "layout": layout,
            "supported": True,
            "verdict": "unresolvable",
            "reason": reason,
        }
        if extra:
            parsed.update(extra)
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[
                "libc debug symbols are required to locate main_arena and its "
                "field offsets; install libc6-dbg / glibc-debuginfo or use a "
                "libc built with -g"
            ],
        )

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Walk the circular main_arena.next chain, returning every arena."""
        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self._unsupported_response(
                "list_arenas", ARENA_LIST_SCHEMA, classification, profile
            )

        layout = self.cached_arena_layout(classification, profile)
        if layout["arena_address"] is None or layout["offsets"].get("next") is None:
            return self._unresolvable_response(
                "list_arenas",
                ARENA_LIST_SCHEMA,
                layout,
                classification,
                profile,
                reason=(
                    "main_arena address or malloc_state.next offset could "
                    "not be resolved via debug symbols"
                ),
                extra={
                    "arenas": [],
                    "count": 0,
                    "component_status": component_status,
                },
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot walk arena chain without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = (
            profile.get("byteorder")
            if profile.get("byteorder")
            in (
                "little",
                "big",
            )
            else "little"
        )
        assert byteorder is not None  # narrows for mypy

        start_address = layout["arena_address"]
        next_offset = layout["offsets"]["next"]
        arenas: List[Dict[str, Any]] = []
        visited: List[int] = []
        cycle_detected = False
        truncated = False
        current = start_address
        # The chain naturally returns to the start (circular) or hits NULL on
        # a partially-initialized libc; both are non-error terminations.
        while len(visited) < MAX_ARENAS_TRAVERSAL:
            if cancellation_check is not None:
                cancellation_check()
            arena_entry: Dict[str, Any] = {
                "address": current,
                "index": len(visited),
                "is_main": current == start_address,
                "in_mapping": mapping_for(current, mappings) is not None,
            }
            raw, err = self._read_bytes(current + next_offset, size_sz)
            if err is not None or len(raw) < size_sz:
                arena_entry["next"] = None
                arena_entry["read_error"] = err or "short read"
                arenas.append(arena_entry)
                visited.append(current)
                break
            next_address = int.from_bytes(raw, byteorder)  # type: ignore[arg-type]
            arena_entry["next"] = next_address
            arenas.append(arena_entry)
            visited.append(current)
            if next_address == 0:
                break
            if next_address == start_address:
                break
            if next_address in visited:
                cycle_detected = True
                break
            current = next_address
        else:
            truncated = True

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            # Same diagnostic inspect_arena / inspect_chunk emit: a silently
            # defaulted byteorder would decode .next pointers with wrong endian
            # on a big-endian glibc and produce garbage arena addresses.
            warnings.append(
                "byteorder could not be resolved from the target; parsed "
                "with little-endian default"
            )
        if truncated:
            warnings.append(
                f"arena chain traversal capped at {MAX_ARENAS_TRAVERSAL} "
                "arenas; the chain may not have returned to main_arena"
            )
        if cycle_detected:
            warnings.append(
                "arena chain contained a non-terminal cycle; heap may be " "corrupted"
            )

        parsed = {
            "schema": ARENA_LIST_SCHEMA,
            "arenas": arenas,
            "count": len(arenas),
            "layout": {
                "arena_address": layout["arena_address"],
                "next_offset": next_offset,
                "resolution_source": layout["resolution_source"],
                "unresolved_fields": layout["unresolved_fields"],
            },
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
            },
            "target_profile": profile,
            "traversal": {
                "max_arenas": MAX_ARENAS_TRAVERSAL,
                "truncated": truncated,
                "cycle_detected": cycle_detected,
            },
            "supported": True,
            "verdict": "parsed",
            "component_status": component_status,
        }
        return build_backend_response(
            "list_arenas",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Read + parse a full malloc_state struct at ``arena_address``.

        ``arena_address`` may be ``None`` (default) to inspect main_arena,
        or a hex/decimal string identifying a non-main arena discovered
        via ``list_arenas``.
        """
        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self._unsupported_response(
                "inspect_arena", ARENA_INSPECTION_SCHEMA, classification, profile
            )

        layout = self.cached_arena_layout(classification, profile)
        struct_size = layout.get("struct_size")

        target_address: Optional[int]
        if arena_address is None:
            target_address = layout.get("arena_address")
        else:
            target_address = parse_int(arena_address)
            if target_address is None:
                raise ValueError(
                    f"arena_address must be an integer, got {arena_address!r}"
                )

        if target_address is None or struct_size is None or struct_size <= 0:
            return self._unresolvable_response(
                "inspect_arena",
                ARENA_INSPECTION_SCHEMA,
                layout,
                classification,
                profile,
                reason=(
                    "arena address or sizeof(struct malloc_state) could not "
                    "be resolved via debug symbols"
                ),
                extra={
                    "arena_address": target_address,
                    "fields": None,
                    "component_status": component_status,
                },
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot parse arena without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        byteorder = (
            profile.get("byteorder")
            if profile.get("byteorder")
            in (
                "little",
                "big",
            )
            else "little"
        )

        arena_mapping = mapping_for(target_address, mappings)
        if cancellation_check is not None:
            cancellation_check()
        raw, err = self._read_bytes(target_address, struct_size)

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            warnings.append(
                "byteorder could not be resolved from the target; parsed "
                "with little-endian default"
            )
        if err is not None:
            warnings.append(f"arena struct read: {err}")
        if arena_mapping is None:
            warnings.append("arena address falls outside every known memory mapping")

        if not raw or len(raw) < struct_size:
            parsed = {
                "schema": ARENA_INSPECTION_SCHEMA,
                "arena_address": target_address,
                "is_main": target_address == layout.get("arena_address"),
                "layout": {
                    "arena_address": layout.get("arena_address"),
                    "struct_size": struct_size,
                    "offsets": layout.get("offsets"),
                    "unresolved_fields": layout.get("unresolved_fields", []),
                    "resolution_source": layout.get("resolution_source"),
                },
                "fields": None,
                "unreadable_fields": list(ArenaLayoutResolver.FIELD_NAMES),
                "raw_hex": raw.hex(),
                "mapping": arena_mapping,
                "allocator": {
                    "family": classification["family"],
                    "version": classification.get("version"),
                    "confidence": classification.get("confidence"),
                },
                "target_profile": profile,
                "component_status": component_status,
                "supported": True,
                "verdict": "unreadable",
                "reason": err or "short read",
            }
            return build_backend_response(
                "inspect_arena",
                parsed=parsed,
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        parsed_fields = self.parse_arena_struct(
            raw,
            layout["offsets"],
            pointer_bits,
            byteorder,
            architecture=profile.get("architecture"),
        )

        parsed = {
            "schema": ARENA_INSPECTION_SCHEMA,
            "arena_address": target_address,
            "is_main": target_address == layout.get("arena_address"),
            "layout": {
                "arena_address": layout.get("arena_address"),
                "struct_size": struct_size,
                "offsets": layout.get("offsets"),
                "unresolved_fields": layout.get("unresolved_fields", []),
                "resolution_source": layout.get("resolution_source"),
            },
            "fields": parsed_fields["fields"],
            "unreadable_fields": parsed_fields["unreadable_fields"],
            "raw_hex": raw.hex(),
            "mapping": arena_mapping,
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
                "safe_linking": classification.get("safe_linking"),
            },
            "target_profile": profile,
            "component_status": component_status,
            "supported": True,
            "verdict": "parsed",
        }
        return build_backend_response(
            "inspect_arena",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )
