"""glibc malloc_chunk parsing (split from heap_glibc)."""

from __future__ import annotations

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

from paomcp.tools.backends.heap_discovery import AllocatorDiscovery

# ----------------------------------------------------------------------
# glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).


# ----------------------------------------------------------------------
# glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).
PREV_INUSE = 0x1
IS_MMAPPED = 0x2
NON_MAIN_ARENA = 0x4
SIZE_BITS_MASK = 0x7

CHUNK_INSPECTION_SCHEMA = "pao.glibc-heap-chunk.v1"
CHUNK_VALIDATION_SCHEMA = "pao.glibc-heap-chunk-validation.v1"


class GlibcHeap:
    """Parse glibc malloc_chunk headers and physical chunk boundaries.

    Reads the ``2 * SIZE_SZ`` header directly from the live inferior, decodes
    ``prev_size`` / ``size`` / flags with per-field raw evidence, computes the
    chunk vs user address and physical next chunk, and -- when possible --
    infers allocated / free status from the physically next chunk's
    ``PREV_INUSE`` bit.  No pwndbg terminal output is parsed; every conclusion
    is grounded in bytes actually read.

    Refuses to run against a non-glibc target: safe-linking, chunk layout and
    bin membership all diverge across allocators, and pretending otherwise
    would produce plausible-looking garbage.
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

    # ------------------------------------------------------------------
    # Pure parser core (no debugger required)
    # ------------------------------------------------------------------
    @staticmethod
    def _size_sz_from_bits(pointer_bits: int) -> int:
        if pointer_bits not in (32, 64):
            raise ValueError(f"pointer_bits must be 32 or 64, got {pointer_bits!r}")
        return pointer_bits // 8

    @staticmethod
    def _decode_byteorder(byteorder: Optional[str]) -> str:
        """Return a concrete byteorder for parsing, defaulting to little-endian.

        Virtually every glibc target the platform targets today (x86, x86_64,
        AArch64 LE, ppc64le, riscv) is little-endian.  The caller records an
        explicit warning when the byteorder could not be resolved so this
        fallback is never silent.
        """
        if byteorder in ("little", "big"):
            return byteorder
        return "little"

    @classmethod
    def parse_chunk_header(
        cls,
        raw_bytes: bytes,
        chunk_address: int,
        pointer_bits: int,
        byteorder: Optional[str],
    ) -> Dict[str, Any]:
        """Decode a malloc_chunk header from raw bytes with per-field evidence.

        Returns either ``{"readable": False, ...}`` when the byte string is
        shorter than the two-pointer header, or a full parse dict with the
        integer fields, decoded flags, physical navigation, structural checks
        (alignment / minimum-size) and the raw byte slice each field was
        derived from.
        """
        size_sz = cls._size_sz_from_bits(pointer_bits)
        header_size = 2 * size_sz
        alignment = 2 * size_sz
        min_chunk_size = 4 * size_sz
        order = cls._decode_byteorder(byteorder)

        if len(raw_bytes) < header_size:
            return {
                "readable": False,
                "reason": (
                    f"insufficient bytes for header: need {header_size}, "
                    f"got {len(raw_bytes)}"
                ),
                "size_sz": size_sz,
                "alignment": alignment,
                "min_chunk_size": min_chunk_size,
                "chunk_address": chunk_address,
                "raw_hex": raw_bytes.hex(),
            }

        prev_size_bytes = raw_bytes[0:size_sz]
        size_bytes = raw_bytes[size_sz : 2 * size_sz]
        prev_size = int.from_bytes(prev_size_bytes, order)  # type: ignore[arg-type]
        size_field = int.from_bytes(size_bytes, order)  # type: ignore[arg-type]

        flags_raw = size_field & SIZE_BITS_MASK
        size = size_field & ~SIZE_BITS_MASK
        prev_inuse = bool(flags_raw & PREV_INUSE)
        is_mmapped = bool(flags_raw & IS_MMAPPED)
        non_main_arena = bool(flags_raw & NON_MAIN_ARENA)

        user_address = chunk_address + 2 * size_sz
        physical_next = chunk_address + size if size >= header_size else None
        physical_prev = (
            chunk_address - prev_size
            if (not prev_inuse and prev_size >= min_chunk_size)
            else None
        )
        prev_reason: Optional[str]
        if physical_prev is not None:
            prev_reason = None
        elif prev_inuse:
            prev_reason = "prev_inuse_set"
        elif prev_size == 0:
            prev_reason = "prev_size_zero"
        else:
            prev_reason = "prev_size_below_min_chunk"

        alignment_ok = (chunk_address % alignment == 0) and (
            size % alignment == 0 if size > 0 else True
        )
        min_size_ok = size >= min_chunk_size

        return {
            "readable": True,
            "size_sz": size_sz,
            "alignment": alignment,
            "min_chunk_size": min_chunk_size,
            "chunk_address": chunk_address,
            "user_address": user_address,
            "raw_hex": raw_bytes[:header_size].hex(),
            "fields": {
                "prev_size": prev_size,
                "size_field": size_field,
                "size": size,
                "flags": {
                    "raw": flags_raw,
                    "prev_inuse": prev_inuse,
                    "is_mmapped": is_mmapped,
                    "non_main_arena": non_main_arena,
                },
            },
            "raw_evidence": {
                "prev_size": {
                    "offset": 0,
                    "length": size_sz,
                    "hex": prev_size_bytes.hex(),
                },
                "size_field": {
                    "offset": size_sz,
                    "length": size_sz,
                    "hex": size_bytes.hex(),
                },
            },
            "physical_next": {
                "address": physical_next,
                "known": physical_next is not None,
                "reason": None if physical_next is not None else "size_below_header",
            },
            "physical_prev": {
                "address": physical_prev,
                "known": physical_prev is not None,
                "reason": prev_reason,
            },
            "structural_checks": {
                "alignment_ok": alignment_ok,
                "min_size_ok": min_size_ok,
            },
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
                "Heap chunk inspection requires a live, stopped inferior; run "
                "the target to a stop before inspecting the heap"
            ),
        )

    def _read_header(
        self, chunk_address: int, header_size: int
    ) -> Tuple[bytes, Optional[str]]:
        """Read ``header_size`` bytes at ``chunk_address``.

        Returns ``(bytes, error)``; ``error`` is ``None`` on success.  Short
        reads still return whatever bytes came back so the caller can surface
        them as evidence.
        """
        result = self.gdb.read_memory_bytes(f"0x{chunk_address:x}", header_size)
        if not result.get("success"):
            return b"", "memory read failed"
        contents = DebuggerOperations._memory_contents(result)
        if contents is None:
            return b"", "memory read returned no contents"
        try:
            raw = bytes.fromhex(contents)
        except ValueError as exc:
            return b"", f"invalid hex contents: {exc}"
        if len(raw) < header_size:
            return raw, f"short read: expected {header_size} bytes, got {len(raw)}"
        return raw, None

    @staticmethod
    def _chunk_from_user(address_int: int, address_kind: str, size_sz: int) -> int:
        if address_kind == "chunk":
            return address_int
        if address_kind == "user":
            return address_int - 2 * size_sz
        raise ValueError(
            f"address_kind must be 'user' or 'chunk', got {address_kind!r}"
        )

    def _mapping_containment(
        self,
        chunk_address: int,
        size: int,
        header_size: int,
        mappings: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Locate the mapping holding the chunk and boundary-check header/payload."""
        containing = mapping_for(chunk_address, mappings)
        header_end = chunk_address + header_size
        payload_end = chunk_address + size if size > 0 else header_end
        mapping_end: Optional[int] = None
        if containing is not None:
            try:
                mapping_end = int(str(containing.get("end")), 16)
            except (TypeError, ValueError):
                mapping_end = None
        header_in_mapping = mapping_end is not None and header_end <= mapping_end
        payload_in_mapping = mapping_end is not None and payload_end <= mapping_end
        return {
            "chunk_in_mapping": containing is not None,
            "header_in_mapping": bool(header_in_mapping),
            "payload_in_mapping": bool(payload_in_mapping) if size > 0 else None,
            "mapping": containing,
            "mapping_end": mapping_end,
        }

    def _lookup_next_status(
        self,
        parsed: Dict[str, Any],
        pointer_bits: int,
        byteorder: str,
        mappings: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Infer allocated/free from the next chunk's PREV_INUSE bit.

        A glibc chunk's own ``PREV_INUSE`` bit describes the *previous* chunk,
        never itself; allocation status must be read from the physically next
        chunk's size field.  ``IS_MMAPPED`` chunks stand alone (not part of a
        heap arena) and are always allocated by construction.
        """
        fields = parsed.get("fields", {})
        flags = fields.get("flags", {})
        size = fields.get("size", 0)
        chunk_address = parsed["chunk_address"]

        if flags.get("is_mmapped"):
            return {
                "status": "allocated",
                "reason": "is_mmapped",
                "next_chunk": None,
            }
        if size <= 0:
            return {
                "status": "unknown",
                "reason": "size_field_zero_or_negative",
                "next_chunk": None,
            }
        if not parsed.get("structural_checks", {}).get("alignment_ok"):
            return {
                "status": "unknown",
                "reason": "size_not_aligned",
                "next_chunk": None,
            }
        # A size below MIN_CHUNK_SIZE means chunk_address + size lands inside
        # this chunk's own user payload (on 64-bit, size == 2 * SIZE_SZ hits
        # user_address exactly).  Reading there and interpreting it as a next
        # chunk's header would derive allocation status from arbitrary user
        # data, so refuse the lookup the same way the alignment gate does.
        if not parsed.get("structural_checks", {}).get("min_size_ok"):
            return {
                "status": "unknown",
                "reason": "size_below_min_chunk",
                "next_chunk": None,
            }

        size_sz = pointer_bits // 8
        header_size = 2 * size_sz
        next_address = chunk_address + size

        if mapping_for(next_address, mappings) is None:
            return {
                "status": "unknown",
                "reason": "next_out_of_mapping",
                "next_chunk": {"address": next_address, "readable": False},
            }

        raw_bytes, error = self._read_header(next_address, header_size)
        if error is not None or len(raw_bytes) < header_size:
            return {
                "status": "unknown",
                "reason": error or "next_read_short",
                "next_chunk": {"address": next_address, "readable": False},
            }

        next_size_bytes = raw_bytes[size_sz : 2 * size_sz]
        next_size_field = int.from_bytes(next_size_bytes, byteorder)  # type: ignore[arg-type]
        next_prev_inuse = bool(next_size_field & PREV_INUSE)
        return {
            "status": "allocated" if next_prev_inuse else "free",
            "reason": (
                "next_prev_inuse_set" if next_prev_inuse else "next_prev_inuse_clear"
            ),
            "next_chunk": {
                "address": next_address,
                "readable": True,
                "size_field": next_size_field,
                "prev_inuse": next_prev_inuse,
                "raw_hex": raw_bytes.hex(),
            },
        }

    def _prepare_context(
        self,
        cancellation_check: Optional[Callable[[], None]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Live-check, collect vmmap+libcinfo, classify family, resolve profile."""
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
        address_int: int,
        address_kind: str,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        parsed = {
            "schema": schema,
            "address": address_int,
            "address_kind": address_kind,
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
            },
            "target_profile": profile,
            "chunk": None,
            "supported": False,
            "verdict": "unsupported",
            "reason": (
                f"chunk parsing supports glibc only; detected "
                f"family={classification['family']!r}"
            ),
        }
        return build_backend_response(
            command,
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=["Glibc chunk parser refused to run against a non-glibc target"],
        )

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Read + parse a glibc malloc_chunk with allocation-status inference."""
        address_int = parse_int(address)
        if address_int is None:
            raise ValueError(f"address must be an integer, got {address!r}")

        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self._unsupported_response(
                "inspect_chunk",
                CHUNK_INSPECTION_SCHEMA,
                address_int,
                address_kind,
                classification,
                profile,
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot parse chunk without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        header_size = 2 * size_sz
        byteorder = self._decode_byteorder(profile.get("byteorder"))
        chunk_address = self._chunk_from_user(address_int, address_kind, size_sz)

        if cancellation_check is not None:
            cancellation_check()
        raw_bytes, read_error = self._read_header(chunk_address, header_size)
        chunk_parsed = self.parse_chunk_header(
            raw_bytes, chunk_address, pointer_bits, byteorder
        )

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            warnings.append(
                "byteorder could not be resolved from the target; parsed with "
                "little-endian default"
            )
        if read_error is not None:
            warnings.append(f"chunk header read: {read_error}")

        if not chunk_parsed.get("readable"):
            parsed = {
                "schema": CHUNK_INSPECTION_SCHEMA,
                "address": address_int,
                "address_kind": address_kind,
                "allocator": {
                    "family": classification["family"],
                    "version": classification.get("version"),
                    "confidence": classification.get("confidence"),
                },
                "target_profile": profile,
                "chunk": chunk_parsed,
                "mapping_containment": None,
                "allocation": {
                    "status": "unknown",
                    "reason": "header_unreadable",
                    "next_chunk": None,
                },
                "supported": True,
                "verdict": "unreadable",
                "reason": chunk_parsed.get("reason"),
                "component_status": component_status,
            }
            return build_backend_response(
                "inspect_chunk",
                parsed=parsed,
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        containment = self._mapping_containment(
            chunk_address,
            chunk_parsed["fields"]["size"],
            header_size,
            mappings,
        )

        if include_next:
            if cancellation_check is not None:
                cancellation_check()
            allocation = self._lookup_next_status(
                chunk_parsed, pointer_bits, byteorder, mappings
            )
        else:
            allocation = {
                "status": "unknown",
                "reason": "next_lookup_disabled",
                "next_chunk": None,
            }

        parsed = {
            "schema": CHUNK_INSPECTION_SCHEMA,
            "address": address_int,
            "address_kind": address_kind,
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
                "safe_linking": classification.get("safe_linking"),
            },
            "target_profile": profile,
            "chunk": chunk_parsed,
            "mapping_containment": containment,
            "allocation": allocation,
            "supported": True,
            "verdict": "parsed",
            "component_status": component_status,
        }
        return build_backend_response(
            "inspect_chunk",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return a structural verdict for a glibc chunk.

        Verdict vocabulary (conservative on purpose):

        * ``unsupported`` -- target is not glibc
        * ``unreadable``  -- the chunk header could not be read
        * ``corrupted``   -- alignment or minimum-size checks failed, or the
                             chunk header sits outside every known mapping
        * ``suspicious``  -- header sits in a mapping but the declared size
                             extends past that mapping's end
        * ``valid``       -- every structural check passed
        """
        inspection = self.inspect_chunk(
            address=address,
            address_kind=address_kind,
            include_next=True,
            cancellation_check=cancellation_check,
        )
        parsed = inspection.get("parsed", {}) or {}

        if not parsed.get("supported"):
            return build_backend_response(
                "validate_chunk",
                parsed={
                    "schema": CHUNK_VALIDATION_SCHEMA,
                    "address": parsed.get("address"),
                    "address_kind": parsed.get("address_kind"),
                    "verdict": "unsupported",
                    "failures": [],
                    "checks": {},
                    "reason": parsed.get("reason"),
                    "chunk_summary": None,
                    "allocator": parsed.get("allocator"),
                    "target_profile": parsed.get("target_profile"),
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=inspection.get("warnings"),
            )

        chunk = parsed.get("chunk") or {}
        containment = parsed.get("mapping_containment") or {}
        failures: List[str] = []
        checks: Dict[str, Dict[str, Any]] = {}
        summary: Optional[Dict[str, Any]] = None

        if not chunk.get("readable"):
            verdict = "unreadable"
            checks["readable"] = {
                "ok": False,
                "reason": chunk.get("reason") or "header_unreadable",
            }
            failures.append("readable")
        else:
            checks["readable"] = {"ok": True, "reason": None}
            structural = chunk.get("structural_checks", {}) or {}
            alignment_ok = bool(structural.get("alignment_ok"))
            min_size_ok = bool(structural.get("min_size_ok"))
            checks["alignment"] = {
                "ok": alignment_ok,
                "reason": None if alignment_ok else "chunk_address_or_size_not_aligned",
            }
            checks["min_size"] = {
                "ok": min_size_ok,
                "reason": None if min_size_ok else "size_below_min_chunk_size",
            }
            chunk_in_mapping = bool(containment.get("chunk_in_mapping"))
            header_in_mapping = bool(containment.get("header_in_mapping"))
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
            payload_in_mapping = containment.get("payload_in_mapping")
            payload_ok = payload_in_mapping is None or bool(payload_in_mapping)
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
                verdict = "valid"

            summary = {
                "chunk_address": chunk.get("chunk_address"),
                "user_address": chunk.get("user_address"),
                "size": chunk.get("fields", {}).get("size"),
                "flags": chunk.get("fields", {}).get("flags"),
                "allocation_status": (parsed.get("allocation") or {}).get("status"),
            }

        return build_backend_response(
            "validate_chunk",
            parsed={
                "schema": CHUNK_VALIDATION_SCHEMA,
                "address": parsed.get("address"),
                "address_kind": parsed.get("address_kind"),
                "verdict": verdict,
                "failures": failures,
                "checks": checks,
                "chunk_summary": summary,
                "allocator": parsed.get("allocator"),
                "target_profile": parsed.get("target_profile"),
                "mapping_containment": containment,
                "allocation": parsed.get("allocation"),
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=inspection.get("warnings"),
        )
