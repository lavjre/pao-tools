"""Apple libmalloc analyzer -- malloc_zone introspection.

libmalloc is macOS/iOS's default userspace allocator.  Unlike glibc's malloc
(one arena state reachable through ``main_arena``) it is zone-based: the
process holds a registry of ``malloc_zone_t`` objects -- the default zone
first -- each an independent allocator with its own function table and an
optional ``malloc_introspection_t`` table for debugger tooling.

This analyzer reads those structures straight out of the live target:

* ``malloc_zone_t`` -- the public custom-allocator contract.  Its layout is
  frozen (unchanged since macOS 10.2), so field offsets are hard-coded rather
  than expression-resolved, like ``_MALLOC_STATE_FIELDS`` but for an ABI that
  never moves.
* ``malloc_introspection_t`` -- the stable debugger/statistics prefix.
* zone discovery -- ``malloc_default_zone()`` (an inferior call, pure query)
  plus the ``_malloc_zones`` / ``_num_zones`` data symbols (best-effort).
* pointer ownership -- ``malloc_size()`` for the global answer and per-zone
  ``zone->size`` probes for the owning-zone hint.

Scope (user-confirmed): full zone surface.  ``list_arenas`` lists the zones,
``inspect_arena`` reads one zone deep (function + introspection tables),
``inspect_chunk`` / ``validate_chunk`` classify pointer ownership, and the
zone-native facade tools (``inspect_libmalloc_zones``, ``inspect_malloc_zone``,
``classify_malloc_pointer``) expose the same internals under their own
schemas.  The remaining eight protocol methods -- fastbins (libmalloc has
none), tcache/bin walking, corruption/free/technique/UAF analysis, and
glibc-shaped snapshot capture -- return an honest typed ``unsupported``
verdict until a later phase builds those readers.

Everything is read-only: memory reads and inferior-call expression evals
(``malloc_default_zone`` / ``malloc_size`` / ``malloc_good_size`` / a zone's
``size`` function) never mutate the target.  Expression failures map to typed
``zone_unavailable`` / ``unresolvable`` verdicts, never an exception.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from paomcp.tools.backends.debugger import DebuggerOperations
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

LIBMALLOC_ZONES_SCHEMA = "pao.libmalloc-zones.v1"
LIBMALLOC_ZONE_SCHEMA = "pao.libmalloc-zone.v1"
LIBMALLOC_POINTER_SCHEMA = "pao.libmalloc-pointer.v1"

# malloc_zone_t -- 0x98 bytes on 64-bit, frozen since macOS 10.2.
_MALLOC_ZONE_SIZE = 0x98
_MALLOC_ZONE_OFFSETS = {
    "reserved1": 0x00,
    "reserved2": 0x08,
    "size": 0x10,
    "malloc": 0x18,
    "calloc": 0x20,
    "valloc": 0x28,
    "free": 0x30,
    "realloc": 0x38,
    "destroy": 0x40,
    "zone_name": 0x48,
    "introspect": 0x50,
    "reserved3": 0x58,
    "reserved4": 0x60,
    "batch_malloc": 0x68,
    "batch_free": 0x70,
    "reserved5": 0x78,
    "memalign": 0x80,
    "free_definite_size": 0x88,
    "pressure_relief": 0x90,
}
_ZONE_FUNCTION_FIELDS = (
    "size",
    "malloc",
    "calloc",
    "valloc",
    "free",
    "realloc",
    "destroy",
    "batch_malloc",
    "batch_free",
    "memalign",
    "free_definite_size",
    "pressure_relief",
)

# malloc_introspection_t -- stable prefix, 0x78 bytes.
_INTROSPECT_SIZE = 0x78
_INTROSPECT_OFFSETS = {
    "enumerator": 0x00,
    "good_size": 0x08,
    "check": 0x10,
    "print": 0x18,
    "log": 0x20,
    "force_lock": 0x28,
    "force_unlock": 0x30,
    "statistics": 0x38,
    "zone_locked": 0x40,
    "enable_disable": 0x48,
    "logger": 0x50,
    "reinit_lock": 0x58,
    "reset_statistics": 0x60,
    "print_task": 0x68,
    "statistics_task": 0x70,
}

# malloc_good_size probe sizes (bytes) used to sketch a zone's size classes.
_GOOD_SIZE_PROBES = (16, 32, 64, 128, 256, 512, 1024, 4096)
# Bounds on the _malloc_zones registry walk so a corrupt count can't over-read.
_MAX_REGISTRY_ZONES = 64

_HEX_TOKEN = re.compile(r"0[xX][0-9a-fA-F]+")
_TRAILING_DECORATION = re.compile(r"<.*?>")
_CAST_PREFIX = re.compile(r"^\([^)]*\)\s*")


def _extract_unsigned(value: Optional[str]) -> Optional[int]:
    """Extract an unsigned integer from a debugger expression value.

    GDB renders expression results in many decorated shapes: ``(void *)
    0x7fff... <symbol>``, ``0x...``, bare ``64``, ``(size_t) 64``.  A hex token
    wins when present; otherwise the text is stripped of trailing ``<sym>``
    decorations and a leading C cast and parsed base-auto.
    """
    if value is None:
        return None
    text = str(value).strip()
    match = _HEX_TOKEN.search(text)
    if match is not None:
        return int(match.group(0), 16)
    text = _TRAILING_DECORATION.sub("", text)
    text = _CAST_PREFIX.sub("", text)
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        return None


class LibmallocAnalyzer:
    """Analyzer for the libmalloc family via malloc_zone introspection."""

    family: Optional[str] = "libmalloc"

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    # -- context -----------------------------------------------------
    def _require_live(self) -> None:
        """Raise a typed error unless the target is live and stopped."""
        LiveInferiorGuard.require_live(
            self.facade.gdb.get_last_stop_record(),
            self.facade.gdb.get_inferior_pid(),
            detail=(
                "libmalloc inspection requires a live, stopped inferior; run "
                "the target to a stop before inspecting its malloc zones"
            ),
        )

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

    # -- target access ------------------------------------------------
    def _eval_unsigned(self, expression: str) -> Optional[int]:
        """Evaluate an expression (inferior call or symbol) as an unsigned int."""
        result = self.facade.discovery.debugger.evaluate_expression(expression)
        if not result.get("success"):
            return None
        parsed = result.get("parsed") or {}
        return _extract_unsigned(parsed.get("value"))

    def _read_bytes(self, address: int, byte_count: int) -> Tuple[bytes, Optional[str]]:
        result = self.facade.gdb.read_memory_bytes(f"0x{address:x}", byte_count)
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

    def _read_cstring(self, address: int, max_len: int = 128) -> Optional[str]:
        raw, _err = self._read_bytes(address, max_len)
        if not raw:
            return None
        end = raw.find(b"\x00")
        if end == -1:
            end = len(raw)
        return raw[:end].decode("utf-8", errors="replace")

    # -- zone structure reads ----------------------------------------
    def _read_zone(self, zone_addr: int) -> Optional[Dict[str, Any]]:
        """Read one malloc_zone_t; ``None`` when the struct is unreadable."""
        raw, err = self._read_bytes(zone_addr, _MALLOC_ZONE_SIZE)
        if err is not None or len(raw) < _MALLOC_ZONE_SIZE:
            return None

        def ptr(offset: int) -> Optional[int]:
            return int.from_bytes(raw[offset : offset + 8], "little") or None

        name_ptr = ptr(_MALLOC_ZONE_OFFSETS["zone_name"])
        introspect_ptr = ptr(_MALLOC_ZONE_OFFSETS["introspect"])
        zone: Dict[str, Any] = {
            "address": zone_addr,
            "name": self._read_cstring(name_ptr) if name_ptr is not None else None,
            "function_table": {
                field: ptr(_MALLOC_ZONE_OFFSETS[field])
                for field in _ZONE_FUNCTION_FIELDS
            },
            "introspect_address": introspect_ptr,
            "introspect": None,
        }
        if introspect_ptr is not None:
            zone["introspect"] = self._read_introspect(introspect_ptr)
        return zone

    def _read_introspect(self, introspect_addr: int) -> Optional[Dict[str, Any]]:
        """Read the malloc_introspection_t table; ``None`` when unreadable."""
        raw, err = self._read_bytes(introspect_addr, _INTROSPECT_SIZE)
        if err is not None or len(raw) < _INTROSPECT_SIZE:
            return None
        return {
            name: (int.from_bytes(raw[offset : offset + 8], "little") or None)
            for name, offset in _INTROSPECT_OFFSETS.items()
        }

    # -- zone discovery ----------------------------------------------
    def _default_zone(self) -> Optional[int]:
        """Resolve the default zone address via ``malloc_default_zone()``."""
        return self._eval_unsigned("(void*)malloc_default_zone()")

    def _registry_zones(self) -> Optional[List[int]]:
        """Best-effort read of the ``_malloc_zones`` registry.

        Returns ``None`` when either data symbol is unavailable or the registry
        cannot be read -- the caller falls back to the default zone.
        """
        base = self._eval_unsigned("(void*)_malloc_zones")
        count = self._eval_unsigned("(unsigned long)_num_zones")
        if base is None or count is None:
            return None
        if count <= 0 or count > _MAX_REGISTRY_ZONES:
            return None
        zones: List[int] = []
        for index in range(count):
            raw, err = self._read_bytes(base + index * 8, 8)
            if err is not None or len(raw) < 8:
                return None
            addr = int.from_bytes(raw, "little")
            if addr:  # a zero slot is an empty registry entry, not a read failure
                zones.append(addr)
        return zones

    def _discover_zones(self) -> Tuple[List[Dict[str, Any]], str, List[str]]:
        """Discover and read every zone; returns (zones, source, warnings).

        ``source`` is ``"registry"``, ``"registry+default"``, ``"default_zone"``
        (registry symbols unavailable), or ``"none"`` (nothing resolvable).
        """
        default = self._default_zone()
        registry = self._registry_zones()
        warnings: List[str] = []
        if registry:
            addrs = list(registry)
            source = "registry"
            if default is not None and default not in addrs:
                addrs.append(default)
                source = "registry+default"
        elif default is not None:
            addrs = [default]
            source = "default_zone"
            warnings.append(
                "_malloc_zones/_num_zones symbols unavailable; "
                "listing the default zone only"
            )
        else:
            return [], "none", warnings

        zones: List[Dict[str, Any]] = []
        unreadable = 0
        for addr in addrs:
            zone = self._read_zone(addr)
            if zone is None:
                unreadable += 1
                continue
            zones.append(zone)
        if unreadable:
            warnings.append(f"{unreadable} zone struct(s) unreadable and skipped")
        return zones, source, warnings

    # -- pointer ownership -------------------------------------------
    def _malloc_size(self, address: int) -> Optional[int]:
        """``malloc_size()`` -- the pointer's allocation size, or 0."""
        return self._eval_unsigned(f"(unsigned long)malloc_size((void*)0x{address:x})")

    def _zone_hint(self, address: int, zones: List[Dict[str, Any]]) -> Optional[str]:
        """Ask each zone's ``size`` function which one owns the pointer."""
        for zone in zones:
            fn = (zone.get("function_table") or {}).get("size")
            zone_addr = zone.get("address")
            if not fn or not zone_addr:
                continue
            claimed = self._eval_unsigned(
                f"((unsigned long (*)(void *, void *)) 0x{fn:x})"
                f"(0x{zone_addr:x}, 0x{address:x})"
            )
            if claimed:
                name = zone.get("name")
                return name or f"zone@0x{zone_addr:x}"
        return None

    def _classify_pointer(
        self, address: int
    ) -> Tuple[str, Optional[int], Optional[str]]:
        size = self._malloc_size(address)
        if size is None:
            return "unresolvable", None, None
        if size > 0:
            zones, _source, _warnings = self._discover_zones()
            return "allocated", size, self._zone_hint(address, zones)
        return "not_allocated", 0, None

    def _good_size_probes(self) -> List[Dict[str, Any]]:
        """Probe ``malloc_good_size`` over common size classes (default zone)."""
        probes: List[Dict[str, Any]] = []
        for requested in _GOOD_SIZE_PROBES:
            good = self._eval_unsigned(f"(unsigned long)malloc_good_size({requested})")
            if good is None:
                continue
            probes.append({"requested": requested, "good_size": good})
        return probes

    # -- Mach-O binary context ---------------------------------------
    def _macho_binary(self) -> Dict[str, Any]:
        """Best-effort Mach-O metadata for the inferior's main binary."""
        provider = getattr(self.facade.discovery, "_binary_path_provider", None)
        if provider is None:
            return {"is_macho": False, "reason": "no binary path provider wired"}
        try:
            path = provider()
        except Exception:  # noqa: BLE001 -- best-effort context
            return {"is_macho": False, "reason": "binary path provider raised"}
        if not path:
            return {"is_macho": False, "reason": "binary path unavailable"}
        from .macho import MachOError, load_macho  # noqa: PLC0415

        try:
            macho = load_macho(path)
        except MachOError as exc:
            return {"is_macho": False, "reason": str(exc)}
        except OSError as exc:
            return {"is_macho": False, "reason": f"cannot read binary: {exc}"}
        return {
            "is_macho": True,
            "path": path,
            "is_64": macho.is_64,
            "filetype": macho.filetype,
            "filetype_name": macho.filetype_name,
            "cputype": macho.cputype,
            "cpu_name": macho.cpu_name,
            "entry": macho.entry,
            "uuid": macho.uuid,
            "dylibs": macho.dylibs,
            "segments": [
                {
                    "name": segment.name,
                    "vmaddr": segment.vmaddr,
                    "vmsize": segment.vmsize,
                    "initprot": segment.initprot,
                }
                for segment in macho.segments
            ],
            "links_libsystem": macho.links_libsystem,
            "links_libsystem_malloc": macho.links_libsystem_malloc,
        }

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
            warnings=[],
        )

    def _zones_envelope(
        self,
        command: str,
        schema: str,
        *,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
        default: Optional[int],
        extra: Optional[Dict[str, Any]] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        zones, source, discovered_warnings = self._discover_zones()
        if not zones and source == "none":
            verdict = "zone_unavailable"
            reason = "could not resolve the default zone or the _malloc_zones registry"
        elif not zones:
            verdict = "unreadable"
            reason = "zone addresses resolved but no zone struct could be read"
        else:
            verdict = "listed"
            reason = None
        parsed: Dict[str, Any] = {
            "schema": schema,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": verdict == "listed",
            "verdict": verdict,
            "default_zone": default,
            "source": source,
            "zone_count": len(zones),
            "zones": zones,
            "component_status": component_status,
        }
        if reason is not None:
            parsed["reason"] = reason
        if extra:
            parsed.update(extra)
        merged_warnings = list(discovered_warnings)
        if warnings:
            merged_warnings.extend(warnings)
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=merged_warnings,
        )

    # -- zone-native facade surface ----------------------------------
    def inspect_zones(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """List every malloc zone with its function + introspection tables."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        return self._zones_envelope(
            "inspect_libmalloc_zones",
            LIBMALLOC_ZONES_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            default=self._default_zone(),
            extra={"binary": self._macho_binary()},
        )

    def inspect_zone(
        self,
        *,
        zone_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Inspect one malloc zone deep (default zone when address omitted)."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        default = self._default_zone()
        target = parse_int(zone_address) if zone_address is not None else None
        if target is None:
            target = default
        warnings: List[str] = []
        size_classes: List[Dict[str, Any]] = []
        size_class_note: Optional[str] = None
        if target is None:
            verdict = "zone_unavailable"
            reason = "no zone address given and the default zone is unresolvable"
            zone = None
        else:
            zone = self._read_zone(target)
            if zone is None:
                verdict = "unreadable"
                reason = f"could not read malloc_zone_t at 0x{target:x}"
            else:
                verdict = "inspected"
                reason = None
                if target == default:
                    size_classes = self._good_size_probes()
                else:
                    size_class_note = (
                        "probes omitted: malloc_good_size reflects the default "
                        "zone, not the inspected zone"
                    )
                    warnings.append(size_class_note)
        parsed: Dict[str, Any] = {
            "schema": LIBMALLOC_ZONE_SCHEMA,
            "address": target,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": verdict == "inspected",
            "verdict": verdict,
            "zone": zone,
            "size_classes": size_classes,
            "component_status": component_status,
        }
        if reason is not None:
            parsed["reason"] = reason
        return build_backend_response(
            "inspect_malloc_zone",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings,
        )

    def classify_pointer(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Classify whether an address is a live libmalloc allocation."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        verdict, size, zone_hint = self._classify_pointer(address_int)
        parsed: Dict[str, Any] = {
            "schema": LIBMALLOC_POINTER_SCHEMA,
            "address": address_int,
            "address_kind": address_kind,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": verdict in ("allocated", "not_allocated"),
            "verdict": verdict,
            "size": size,
            "zone_hint": zone_hint,
            "component_status": component_status,
        }
        if verdict == "unresolvable":
            parsed["reason"] = (
                "malloc_size() failed or the expression was unavailable; "
                "ownership is unresolved"
            )
        return build_backend_response(
            "classify_malloc_pointer",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[],
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
        """Classify a pointer's libmalloc ownership via ``malloc_size``."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        verdict, size, zone_hint = self._classify_pointer(address_int)
        parsed: Dict[str, Any] = {
            "schema": CHUNK_INSPECTION_SCHEMA,
            "address": address_int,
            "address_kind": address_kind,
            "include_next": include_next,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": verdict in ("allocated", "not_allocated"),
            "verdict": verdict,
            "size": size,
            "zone_hint": zone_hint,
            "component_status": component_status,
        }
        if verdict == "unresolvable":
            parsed["reason"] = (
                "malloc_size() failed or the expression was unavailable; "
                "ownership is unresolved"
            )
        return build_backend_response(
            "inspect_chunk",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[],
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Check whether an address is a live libmalloc allocation."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        verdict, size, zone_hint = self._classify_pointer(address_int)
        mapped = {"allocated": "valid", "not_allocated": "invalid"}.get(
            verdict, verdict
        )
        parsed: Dict[str, Any] = {
            "schema": CHUNK_VALIDATION_SCHEMA,
            "address": address_int,
            "address_kind": address_kind,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": mapped in ("valid", "invalid"),
            "verdict": mapped,
            "size": size,
            "zone_hint": zone_hint,
            "component_status": component_status,
        }
        if verdict == "unresolvable":
            parsed["reason"] = (
                "malloc_size() failed or the expression was unavailable; "
                "validity is unresolved"
            )
        return build_backend_response(
            "validate_chunk",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[],
        )

    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """List the malloc zones (libmalloc's arena equivalent)."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification.get("family") != "libmalloc":
            return self._unsupported(
                "list_arenas",
                ARENA_LIST_SCHEMA,
                classification=classification,
                profile=profile,
                component_status=component_status,
                reason=(
                    "the current target is not classified as libmalloc; "
                    "malloc zones are not meaningful here"
                ),
            )
        return self._zones_envelope(
            "list_arenas",
            ARENA_LIST_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
            default=self._default_zone(),
        )

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Inspect a malloc zone (default zone when no address given)."""
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        default = self._default_zone()
        target = parse_int(arena_address) if arena_address is not None else None
        if target is None:
            target = default
        if target is None:
            verdict = "zone_unavailable"
            reason = "no zone address given and the default zone is unresolvable"
            zone = None
        else:
            zone = self._read_zone(target)
            if zone is None:
                verdict = "unreadable"
                reason = f"could not read malloc_zone_t at 0x{target:x}"
            else:
                verdict = "inspected"
                reason = None
        parsed: Dict[str, Any] = {
            "schema": ARENA_INSPECTION_SCHEMA,
            "address": target,
            "allocator": self._allocator(classification),
            "target_profile": profile,
            "supported": verdict == "inspected",
            "verdict": verdict,
            "zone": zone,
            "component_status": component_status,
        }
        if reason is not None:
            parsed["reason"] = reason
        return build_backend_response(
            "inspect_arena",
            parsed=parsed,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[],
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
                "libmalloc has no fastbins (its zones use size classes and "
                "magazines); fastbin inspection is glibc-only"
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
                "libmalloc's per-thread free lists are not exposed as a "
                "tcache; tcache inspection is glibc-only"
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
            reason=(
                "libmalloc bin inspection (nano/magazine internals) is not "
                "implemented in this phase; scope is the zone surface only"
            ),
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
                "libmalloc corruption analysis is not implemented"
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
            reason="free validation is glibc-specific; not implemented for libmalloc",
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
                "libmalloc technique analysis is not implemented"
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
                "for libmalloc"
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
                "for libmalloc"
            ),
        )


__all__ = [
    "LibmallocAnalyzer",
    "LIBMALLOC_ZONES_SCHEMA",
    "LIBMALLOC_ZONE_SCHEMA",
    "LIBMALLOC_POINTER_SCHEMA",
]
