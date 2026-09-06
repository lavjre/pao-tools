"""Typed Scudo inspection (P1 read-only minimal).

Scudo (LLVM compiler-rt, Android bionic) has no pwndbg plugin surface, so this
analyzer reads the primary chunk headers directly out of the live inferior:

* header is 16 bytes at ``chunk_address``; user pointer is ``chunk + 16``.
* ``+0 u64 checksum``, ``+8 u16 class_id``, ``+10 u8 state``, ``+11 u8 origin``,
  ``+12 u32 size_or_unused`` (documented Scudo primary layout v1).

Offsets are version-dependent, so every response tags
``layout_source: documented-scudo-primary-v1`` plus a warning when symbols are
missing. ChunkState: 0=available, 1=quarantine, 2=allocated. Anything else is
``suspicious``. Without readable memory every entry point reports
``unresolvable`` instead of guessing.
"""

from __future__ import annotations

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

SCUDO_CHUNK_SCHEMA = "pao.scudo-chunk.v1"
SCUDO_REGION_LIST_SCHEMA = "pao.scudo-region-list.v1"
SCUDO_CORRUPTION_SCHEMA = "pao.scudo-corruption.v1"
SCUDO_UAF_SCHEMA = "pao.scudo-uaf.v1"

SCUDO_HEADER_SIZE = 16
SCUDO_STATE_AVAILABLE = 0
SCUDO_STATE_QUARANTINE = 1
SCUDO_STATE_ALLOCATED = 2
_SCUDO_KNOWN_STATES = frozenset({0, 1, 2})
_SCUDO_MAX_CLASS_ID = 128
_SCUDO_SCAN_PAGE = 4096
_SCUDO_SCAN_MAX_REGIONS = 8
_SCUDO_SCAN_MAX_BYTES = 1 << 20

_LAYOUT_WARNING = (
    "scudo header offsets follow documented primary layout v1; "
    "vendor builds may shift fields — header hex is authoritative"
)


class ScudoAnalyzer:
    """Read-only Scudo inspector over live GDB memory."""

    family: Optional[str] = "scudo"

    def __init__(self, facade: Any) -> None:
        self.facade = facade

    def _require_live(self) -> None:
        LiveInferiorGuard.require_live(
            self.facade.gdb.get_last_stop_record(),
            self.facade.gdb.get_inferior_pid(),
            detail=(
                "scudo inspection requires a live, stopped inferior; run "
                "the target to a stop before inspecting its chunks"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo, component_status = self.facade.discovery.collect_evidence(
            cancellation_check
        )
        classification = self.facade.discovery.classify(mappings, libcinfo)
        profile = self.facade.discovery.target_profile(mappings)
        return classification, profile, component_status

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

    def _allocator(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification["family"],
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
        }

    def _verify_family(
        self,
        command: str,
        schema: str,
        *,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
        component_status: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if classification.get("family") == self.family:
            return None
        return build_backend_response(
            command,
            parsed={
                "schema": schema,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    f"the current target is not classified as {self.family}; "
                    "scudo header reads are not meaningful here"
                ),
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )

    def _read_bytes(self, address: int, byte_count: int) -> Tuple[bytes, Optional[str]]:
        result = self.facade.gdb.read_memory_bytes(f"0x{address:x}", byte_count)
        if not isinstance(result, dict) or not result.get("success"):
            err = result.get("error") if isinstance(result, dict) else None
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

    @staticmethod
    def _decode_header(
        raw: bytes, byteorder: Literal["little", "big"] = "little"
    ) -> Dict[str, Any]:
        if len(raw) < SCUDO_HEADER_SIZE:
            return {"readable": False}
        checksum = int.from_bytes(raw[0:8], byteorder)
        class_id = int.from_bytes(raw[8:10], byteorder)
        state = raw[10]
        origin = raw[11]
        size_field = int.from_bytes(raw[12:16], byteorder)
        return {
            "readable": True,
            "checksum": checksum,
            "class_id": class_id,
            "state": state,
            "origin": origin,
            "size_field": size_field,
            "header_hex": raw[:SCUDO_HEADER_SIZE].hex(),
        }

    def _chunk_from_user(
        self, address: int, address_kind: str, header_size: int = SCUDO_HEADER_SIZE
    ) -> int:
        if address_kind == "chunk":
            return address
        return address - header_size

    def _scudo_regions(
        self, mappings: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        regions: List[Dict[str, Any]] = []
        for mp in mappings:
            if not mp.get("readable") or not mp.get("writable"):
                continue
            fname = str(mp.get("file") or "")
            if fname.startswith("[") and fname not in ("[heap]", "[anon]"):
                if "scudo" not in fname.lower() and "anon" not in fname.lower():
                    continue
            try:
                start = int(str(mp.get("start")), 16)
                end = int(str(mp.get("end")), 16)
            except (TypeError, ValueError):
                continue
            if end - start <= 0:
                continue
            regions.append({"mapping": mp, "start": start, "end": end})
        return regions[:_SCUDO_SCAN_MAX_REGIONS]

    def _scan_headers(
        self,
        mappings: Sequence[Dict[str, Any]],
        byteorder: Literal["little", "big"],
        max_bytes: int = _SCUDO_SCAN_MAX_BYTES,
    ) -> Tuple[List[Dict[str, Any]], int]:
        chunks: List[Dict[str, Any]] = []
        scanned = 0
        for region in self._scudo_regions(mappings):
            if scanned >= max_bytes:
                break
            start = int(region["start"])
            end = int(region["end"])
            want = min(end - start, max_bytes - scanned, 65536)
            if want <= 0:
                continue
            raw, err = self._read_bytes(start, want)
            if err is not None or not raw:
                continue
            scanned += len(raw)
            for off in range(0, len(raw) - SCUDO_HEADER_SIZE + 1, 16):
                window = raw[off : off + SCUDO_HEADER_SIZE]
                if len(window) < SCUDO_HEADER_SIZE:
                    break
                class_id = int.from_bytes(window[8:10], byteorder)
                state = window[10]
                checksum = int.from_bytes(window[0:8], byteorder)
                if checksum == 0:
                    continue
                if class_id > _SCUDO_MAX_CLASS_ID:
                    continue
                if state not in _SCUDO_KNOWN_STATES:
                    continue
                chunks.append(
                    {
                        "chunk_address": start + off,
                        "user_address": start + off + SCUDO_HEADER_SIZE,
                        "class_id": class_id,
                        "state": state,
                        "checksum": checksum,
                        "header_hex": window.hex(),
                    }
                )
                if len(chunks) >= 512:
                    break
            if len(chunks) >= 512:
                break
        return chunks, scanned

    # -- AllocatorAnalyzer surface -----------------------------------
    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = f"scudo inspect_chunk {address_int}"
        guard = self._verify_family(
            command,
            SCUDO_CHUNK_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        chunk = self._chunk_from_user(address_int, address_kind)
        raw, err = self._read_bytes(chunk, SCUDO_HEADER_SIZE)
        if err is not None:
            return build_backend_response(
                command,
                parsed={
                    "schema": SCUDO_CHUNK_SCHEMA,
                    "address": address_int,
                    "address_kind": address_kind,
                    "chunk_address": chunk,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": err,
                    "layout_source": "documented-scudo-primary-v1",
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_LAYOUT_WARNING],
            )
        header = self._decode_header(raw, byteorder)
        state = int(header.get("state") or 0)
        class_id = int(header.get("class_id") or 0)
        if state not in _SCUDO_KNOWN_STATES:
            verdict = "suspicious"
        elif class_id > _SCUDO_MAX_CLASS_ID:
            verdict = "corrupted"
        elif state == SCUDO_STATE_ALLOCATED:
            verdict = "allocated"
        elif state == SCUDO_STATE_QUARANTINE:
            verdict = "quarantined"
        else:
            verdict = "available"
        return build_backend_response(
            command,
            parsed={
                "schema": SCUDO_CHUNK_SCHEMA,
                "address": address_int,
                "address_kind": address_kind,
                "chunk_address": chunk,
                "user_address": chunk + SCUDO_HEADER_SIZE,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": verdict,
                "header": header,
                "layout_source": "documented-scudo-primary-v1",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, profile, component_status = self._prepare_context(
            cancellation_check
        )
        command = f"scudo validate_chunk {address_int}"
        guard = self._verify_family(
            command,
            SCUDO_CHUNK_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        response = self.inspect_chunk(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )
        parsed = dict(response.get("parsed") or {})
        verdict = parsed.get("verdict")
        if verdict == "allocated":
            parsed["verdict"] = "valid"
        elif verdict in ("quarantined", "available"):
            parsed["verdict"] = "invalid"
        elif verdict not in ("valid", "invalid"):
            parsed["verdict"] = (
                "unresolvable" if verdict == "unresolvable" else "suspicious"
            )
        response["parsed"] = parsed
        response["command"] = command
        return response

    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        command = "scudo list_regions"
        guard = self._verify_family(
            command,
            SCUDO_REGION_LIST_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        regions = self._scudo_regions(mappings)
        return build_backend_response(
            command,
            parsed={
                "schema": SCUDO_REGION_LIST_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "listed",
                "region_count": len(regions),
                "regions": [
                    {
                        "start": r["start"],
                        "end": r["end"],
                        "size": r["end"] - r["start"],
                        "file": r["mapping"].get("file"),
                        "permissions": r["mapping"].get("permissions"),
                    }
                    for r in regions
                ],
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        command = "scudo inspect_region"
        guard = self._verify_family(
            command,
            ARENA_INSPECTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        target = parse_int(arena_address) if arena_address is not None else None
        regions = self._scudo_regions(mappings)
        if target is None:
            if not regions:
                verdict: str = "unresolvable"
                region = None
            else:
                region = regions[0]
                verdict = "parsed"
                target = int(region["start"])
        else:
            region = next((r for r in regions if r["start"] <= target < r["end"]), None)
            verdict = "parsed" if region is not None else "unresolvable"
        return build_backend_response(
            command,
            parsed={
                "schema": ARENA_INSPECTION_SCHEMA,
                "address": target,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": verdict,
                "region": (
                    {
                        "start": int(region["start"]),
                        "end": int(region["end"]),
                        "file": region["mapping"].get("file"),
                    }
                    if region is not None
                    else None
                ),
                "layout_source": "documented-scudo-primary-v1",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
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
        return build_backend_response(
            "scudo inspect_fastbins",
            parsed={
                "schema": FASTBIN_INSPECTION_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": "scudo has no fastbins; freelist is per-size-class with checksums",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
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
        return build_backend_response(
            "scudo inspect_tcache",
            parsed={
                "schema": TCACHE_INSPECTION_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": "scudo has no tcache; per-thread cache is quarantine-based",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
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
        return build_backend_response(
            "scudo inspect_bins",
            parsed={
                "schema": BIN_INSPECTION_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": "scudo size classes are not glibc bins; use scudo scan",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )

    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        command = "scudo classify_corruption"
        guard = self._verify_family(
            command,
            SCUDO_CORRUPTION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        chunks, scanned = self._scan_headers(mappings, byteorder)
        checks: List[Dict[str, Any]] = []
        seen: Dict[int, int] = {}
        for ch in chunks:
            addr = int(ch["chunk_address"])
            seen[addr] = seen.get(addr, 0) + 1
            if seen[addr] > 1:
                checks.append(
                    {
                        "check": "duplicate_chunk",
                        "severity": "corrupted",
                        "chunk_address": addr,
                    }
                )
            if int(ch["class_id"]) > _SCUDO_MAX_CLASS_ID:
                checks.append(
                    {
                        "check": "bad_class_id",
                        "severity": "corrupted",
                        "chunk_address": addr,
                        "class_id": ch["class_id"],
                    }
                )
        for mp in mappings:
            if mp.get("writable") and mp.get("executable"):
                checks.append(
                    {
                        "check": "writable_executable_scudo_mapping",
                        "severity": "suspicious",
                        "file": mp.get("file"),
                    }
                )
        if any(c.get("severity") == "corrupted" for c in checks):
            verdict = "corrupted"
        elif checks:
            verdict = "suspicious"
        else:
            verdict = "valid"
        return build_backend_response(
            command,
            parsed={
                "schema": SCUDO_CORRUPTION_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": verdict,
                "scanned_bytes": scanned,
                "chunk_count": len(chunks),
                "checks": checks,
                "failed_checks": [c["check"] for c in checks],
                "layout_source": "documented-scudo-primary-v1",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
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
        command = f"scudo validate_free {address}"
        guard = self._verify_family(
            command,
            FREE_VALIDATION_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        response = self.inspect_chunk(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )
        parsed = dict(response.get("parsed") or {})
        verdict = parsed.get("verdict")
        if verdict == "allocated":
            parsed["verdict"] = "unsafe"
            parsed["reason"] = (
                "chunk is still allocated; freeing twice risks double-free"
            )
        elif verdict in ("quarantined", "available"):
            parsed["verdict"] = "safe"
        else:
            parsed["verdict"] = "unresolvable"
        parsed["schema"] = FREE_VALIDATION_SCHEMA
        response["parsed"] = parsed
        response["command"] = command
        return response

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
        return build_backend_response(
            "scudo analyze_technique",
            parsed={
                "schema": HEAP_TECHNIQUE_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": "House-of-* detectors are glibc-specific; scudo technique analysis is quarantine/UAF only",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )

    def detect_uaf(
        self,
        *,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        command = "scudo detect_uaf"
        guard = self._verify_family(
            command,
            SCUDO_UAF_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        chunks, _ = self._scan_headers(
            mappings, byteorder, max_bytes=min(max_bytes, 1 << 20)
        )
        quarantined = [
            (int(c["chunk_address"]), int(c["chunk_address"]) + 64)
            for c in chunks[: max(0, max_chunks)]
            if int(c["state"]) == SCUDO_STATE_QUARANTINE
        ]
        if not quarantined:
            return build_backend_response(
                command,
                parsed={
                    "schema": SCUDO_UAF_SCHEMA,
                    "allocator": self._allocator(classification),
                    "target_profile": profile,
                    "supported": True,
                    "verdict": "clean",
                    "scanned_mappings": 0,
                    "scanned_bytes": 0,
                    "quarantined_count": 0,
                    "candidates": [],
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.facade.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[_LAYOUT_WARNING],
            )
        ptr_size = int(profile.get("pointer_bits") or 64) // 8
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
            want = min(end - start, max_bytes - scanned_bytes, 65536)
            if want <= 0:
                break
            raw, _ = self._read_bytes(start, want)
            if not raw:
                continue
            scanned_mappings += 1
            scanned_bytes += len(raw)
            for off in range(0, len(raw) - ptr_size + 1, ptr_size):
                try:
                    val = int.from_bytes(raw[off : off + ptr_size], byteorder)
                except Exception:
                    continue
                for qstart, qend in quarantined:
                    if qstart <= val < qend:
                        cands.append(
                            {
                                "pointer_location": start + off,
                                "pointer_value": val,
                                "quarantined_range": [qstart, qend],
                            }
                        )
                        if len(cands) >= 128:
                            break
                if len(cands) >= 128:
                    break
            if len(cands) >= 128:
                break
        return build_backend_response(
            command,
            parsed={
                "schema": SCUDO_UAF_SCHEMA,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": True,
                "verdict": "suspicious" if cands else "clean",
                "scanned_mappings": scanned_mappings,
                "scanned_bytes": scanned_bytes,
                "quarantined_count": len(quarantined),
                "candidates": cands,
                "candidate_count": len(cands),
                "layout_source": "documented-scudo-primary-v1",
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
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
        classification, mappings, profile, component_status = (
            self._prepare_with_mappings(cancellation_check)
        )
        command = "scudo capture_snapshot"
        guard = self._verify_family(
            command,
            HEAP_SNAPSHOT_SCHEMA,
            classification=classification,
            profile=profile,
            component_status=component_status,
        )
        if guard is not None:
            return guard
        deadline = time.monotonic() + float(timeout_seconds or 300.0)
        byteorder = cast(Literal["little", "big"], profile.get("byteorder") or "little")
        chunks, _ = self._scan_headers(mappings, byteorder)
        if time.monotonic() > deadline:
            raise TimeoutError("scudo snapshot exceeded its time budget")
        ledger = [
            {
                "chunk_address": int(c["chunk_address"]),
                "size": None,
                "arena_address": None,
                "location": {"kind": "scudo-chunk", "state": int(c["state"])},
            }
            for c in chunks[:max_ledger_entries]
        ]
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
            "scudo_chunks": chunks[:max_ledger_entries],
            "layout_source": "documented-scudo-primary-v1",
            "component_status": component_status,
            "summary": {
                "chunk_count": len(ledger),
                "arena_count": 0,
                "ledger_truncated": len(chunks) > max_ledger_entries,
            },
        }
        try:
            history = getattr(self.facade, "history", None)
            if history is not None and hasattr(history, "_remember"):
                entry = history._remember(name=name, payload=payload)
                stored = history._clone(entry)
                return build_backend_response(
                    command,
                    parsed=stored,
                    raw_responses=[],
                    state=self.facade.gdb.get_state(),
                    parser_version=PARSER_VERSION,
                    warnings=[_LAYOUT_WARNING],
                )
        except Exception:
            pass
        return build_backend_response(
            command,
            parsed=payload,
            raw_responses=[],
            state=self.facade.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=[_LAYOUT_WARNING],
        )


__all__ = [
    "ScudoAnalyzer",
    "SCUDO_CHUNK_SCHEMA",
    "SCUDO_REGION_LIST_SCHEMA",
    "SCUDO_CORRUPTION_SCHEMA",
    "SCUDO_UAF_SCHEMA",
]
