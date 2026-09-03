"""glibc bin/tcache walking, S6.4 (split from heap_glibc)."""

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
# Stage 6.2 -- glibc malloc_chunk parsing
# ----------------------------------------------------------------------

# glibc malloc_chunk flag bits (low three bits of the ``size`` field).
from paomcp.tools.backends.heap_glibc_heap import (
    PREV_INUSE,
    SIZE_BITS_MASK,
)
from paomcp.tools.backends.heap_glibc_arenas import (
    BINS_ARRAY_LEN,
    NBINS,
    NFASTBINS,
)

# ----------------------------------------------------------------------
# Stage 6.3b -- glibc fastbin traversal
# ----------------------------------------------------------------------

FASTBIN_INSPECTION_SCHEMA = "pao.glibc-fastbins.v1"

# Default per-fastbin chain length cap.  Real fastbins hold ~7 chunks at
# most in normal operation but corruption can create arbitrarily long
# chains, so a bound is necessary to keep the tool safe against a
# malicious/broken heap.  Sized generously so real workloads never hit it.


# glibc's per-thread tcache is capped at TCACHE_MAX_BINS singly-linked
# chains, one per aligned size class up to ~1 KiB (64-bit) / ~0.5 KiB
# (32-bit).  The value is compile-time in glibc and has been stable at 64
# across every version that ships tcache (2.26+).
TCACHE_MAX_BINS = 64

TCACHE_INSPECTION_SCHEMA = "pao.glibc-tcache.v1"

# Per-bin traversal cap for tcache chains.  Real tcache bins hold at most
# tcache_count (default 7) chunks in normal operation but corruption can
# lengthen a chain arbitrarily, so a bound is required.  Sized generously
# above the real limit.


# Stage 6.3d -- regular bins (unsorted / small / large).  Bin index 1 is
# the unsorted bin, 2..63 are the small bins (one fixed size class each),
# 64..127 are the large bins (each covers a size range with a secondary
# fd_nextsize/bk_nextsize list, not walked here -- deferred).  The value
# 127 = NBINS - 1 (bin 0 does not exist) and has been stable in glibc
# since the introduction of size bins.
BIN_INSPECTION_SCHEMA = "pao.glibc-bins.v1"

# Per-bin cap for regular-bin traversal.  Large bins can legitimately
# accumulate thousands of chunks between coalescing passes; 1024 is well
# above typical steady-state depths while still bounding a corrupted
# chain that never returns to its bin head.


# Stage 6.10 -- kernel SLUB.  Default system page size for slab geometry
# (slab_bytes = PAGE_SIZE << order).  Holds for x86_64/arm64 kernels and is
# the value a bare gdbstub guest rarely overrides; callers can pass their own
# when the kernel uses a different base page.
DEFAULT_SLUB_PAGE_SIZE = 0x1000


class GlibcBins:
    """Walk glibc bin chains (fastbins today; tcache/regular bins later).

    Reads chunk headers and fd pointers directly from the live inferior and
    decodes ``PROTECT_PTR``-encoded fd pointers when safe-linking is active
    (glibc >= 2.32).  Every chain walk is bounded and cycle-detected so a
    corrupted allocator cannot hang the tool.
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
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas

    # ------------------------------------------------------------------
    # Pure helpers (no debugger required)
    # ------------------------------------------------------------------
    @staticmethod
    def fastbin_size_for_index(index: int, pointer_bits: int) -> int:
        """Expected chunk size for fastbinsY[index] on the target.

        Matches glibc's ``fastbin_index`` inverse: the first slot holds
        chunks of ``2 * MALLOC_ALIGNMENT`` bytes and each subsequent slot
        adds ``MALLOC_ALIGNMENT``.  On 64-bit that yields the classic
        0x20/0x30/.../0xb0 size ladder; on 32-bit it is 0x10/0x18/.../0x58.
        """
        if pointer_bits not in (32, 64):
            raise ValueError(f"pointer_bits must be 32 or 64, got {pointer_bits!r}")
        if index < 0 or index >= NFASTBINS:
            raise ValueError(f"fastbin index out of range [0, {NFASTBINS}): {index!r}")
        malloc_alignment = 2 * (pointer_bits // 8)  # 16 on 64-bit, 8 on 32-bit
        return (2 + index) * malloc_alignment

    @staticmethod
    def decode_safe_linked(encoded: int, position: int) -> int:
        """glibc ``PROTECT_PTR``: decoded = encoded XOR (position >> 12).

        ``position`` is the address of the fd/next field itself (i.e. where
        the encoded pointer is *stored*), not the address of the chunk that
        holds it -- match the macro's ``pos`` argument exactly.
        """
        return encoded ^ (position >> 12)

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Bin inspection requires a live, stopped inferior; run the "
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

    def _walk_fastbin_chain(
        self,
        head_address: int,
        mappings: Sequence[Mapping[str, Any]],
        pointer_bits: int,
        byteorder: str,
        safe_linking: str,
        max_chunks: int,
        cancellation_check: Optional[Callable[[], None]],
    ) -> Dict[str, Any]:
        """Walk a singly-linked fastbin chain starting at ``head_address``."""
        size_sz = pointer_bits // 8
        header_size = 2 * size_sz
        chain: List[Dict[str, Any]] = []
        visited: Dict[int, int] = {}
        cycle_detected = False
        truncated = False
        current: int = head_address

        while current != 0:
            if cancellation_check is not None:
                cancellation_check()
            if current in visited:
                cycle_detected = True
                break
            if len(chain) >= max_chunks:
                truncated = True
                break
            visited[current] = len(chain)
            entry: Dict[str, Any] = {
                "index": len(chain),
                "chunk_address": current,
                "in_mapping": mapping_for(current, mappings) is not None,
            }
            header, err = self._read_bytes(current, header_size)
            if err is not None or len(header) < header_size:
                entry["readable"] = False
                entry["read_error"] = err or "short read"
                chain.append(entry)
                break
            entry["readable"] = True
            size_field = int.from_bytes(
                header[size_sz : 2 * size_sz], byteorder  # type: ignore[arg-type]
            )
            entry["size_field"] = size_field
            entry["size"] = size_field & ~SIZE_BITS_MASK
            entry["prev_inuse"] = bool(size_field & PREV_INUSE)

            fd_address = current + 2 * size_sz
            fd_bytes, fd_err = self._read_bytes(fd_address, size_sz)
            entry["fd_position"] = fd_address
            if fd_err is not None or len(fd_bytes) < size_sz:
                entry["fd_raw"] = None
                entry["fd_decoded"] = None
                entry["fd_read_error"] = fd_err or "short read"
                entry["safe_linking_applied"] = False
                chain.append(entry)
                break

            fd_raw = int.from_bytes(fd_bytes, byteorder)  # type: ignore[arg-type]
            entry["fd_raw"] = fd_raw
            if safe_linking == "enabled":
                entry["fd_decoded"] = self.decode_safe_linked(fd_raw, fd_address)
                entry["safe_linking_applied"] = True
            else:
                entry["fd_decoded"] = fd_raw
                entry["safe_linking_applied"] = False
            chain.append(entry)
            current = entry["fd_decoded"]

        return {
            "chunks": chain,
            "length": len(chain),
            "truncated": truncated,
            "cycle_detected": cycle_detected,
        }

    # ------------------------------------------------------------------
    # Context preparation (mirrors GlibcArenas._prepare_context)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Walk the 10 fastbin chains rooted in an arena's ``fastbinsY`` array.

        ``arena_address`` may be ``None`` (inspect main_arena) or the address
        of a non-main arena obtained via ``list_arenas``.  Each fastbin's
        chain is walked to completion or ``max_chunks_per_bin`` chunks --
        whichever comes first -- with cycle detection.  When the target's
        glibc has safe-linking enabled (>= 2.32) fd pointers are decoded via
        ``PROTECT_PTR`` so the chain addresses are usable directly.
        """
        if max_chunks_per_bin <= 0:
            raise ValueError(
                f"max_chunks_per_bin must be positive, got {max_chunks_per_bin!r}"
            )

        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self.arenas._unsupported_response(
                "inspect_fastbins",
                FASTBIN_INSPECTION_SCHEMA,
                classification,
                profile,
                extra={"fastbins": [], "component_status": component_status},
            )

        layout = self.arenas.resolver.resolve()
        fastbins_offset = layout["offsets"].get("fastbinsY")
        target_address: Optional[int]
        if arena_address is None:
            target_address = layout.get("arena_address")
        else:
            target_address = parse_int(arena_address)
            if target_address is None:
                raise ValueError(
                    f"arena_address must be an integer, got {arena_address!r}"
                )

        if target_address is None or fastbins_offset is None:
            return self.arenas._unresolvable_response(
                "inspect_fastbins",
                FASTBIN_INSPECTION_SCHEMA,
                layout,
                classification,
                profile,
                reason=(
                    "arena address or fastbinsY field offset could not be "
                    "resolved via debug symbols"
                ),
                extra={
                    "arena_address": target_address,
                    "fastbins": [],
                    "component_status": component_status,
                },
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot walk fastbins without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = (
            profile.get("byteorder")
            if profile.get("byteorder") in ("little", "big")
            else "little"
        )
        assert byteorder is not None
        safe_linking = classification.get("safe_linking", "unknown")

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            warnings.append(
                "byteorder could not be resolved from the target; parsed "
                "with little-endian default"
            )
        if safe_linking == "unknown":
            warnings.append(
                "safe-linking applicability is unknown (libc version was "
                "not resolved); fd pointers reported as raw with no decode"
            )

        # Read the whole fastbinsY block in a single 10*size_sz gulp so the
        # chain walks see a consistent snapshot of the heads.
        if cancellation_check is not None:
            cancellation_check()
        heads_block, heads_err = self._read_bytes(
            target_address + fastbins_offset, NFASTBINS * size_sz
        )
        if heads_err is not None or len(heads_block) < NFASTBINS * size_sz:
            warnings.append(f"fastbinsY read: {heads_err or 'short read'}")
            fastbins_out: List[Dict[str, Any]] = []
            for i in range(NFASTBINS):
                fastbins_out.append(
                    {
                        "index": i,
                        "size_class": self.fastbin_size_for_index(i, pointer_bits),
                        "head": None,
                        "head_raw_hex": None,
                        "chain": [],
                        "length": 0,
                        "truncated": False,
                        "cycle_detected": False,
                        "read_error": heads_err or "short read",
                    }
                )
            parsed = {
                "schema": FASTBIN_INSPECTION_SCHEMA,
                "arena_address": target_address,
                "is_main_arena": target_address == layout.get("arena_address"),
                "layout": {
                    "arena_address": layout.get("arena_address"),
                    "fastbinsY_offset": fastbins_offset,
                    "resolution_source": layout.get("resolution_source"),
                    "unresolved_fields": layout.get("unresolved_fields", []),
                },
                "allocator": {
                    "family": classification["family"],
                    "version": classification.get("version"),
                    "confidence": classification.get("confidence"),
                    "safe_linking": safe_linking,
                },
                "safe_linking": {
                    "state": safe_linking,
                    "applied": safe_linking == "enabled",
                },
                "target_profile": profile,
                "fastbins": fastbins_out,
                "component_status": component_status,
                "supported": True,
                "verdict": "unreadable",
                "reason": heads_err or "short read",
            }
            return build_backend_response(
                "inspect_fastbins",
                parsed=parsed,
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        fastbins_out = []
        for i in range(NFASTBINS):
            slot_bytes = heads_block[i * size_sz : (i + 1) * size_sz]
            head_ptr = int.from_bytes(slot_bytes, byteorder)  # type: ignore[arg-type]
            bin_entry: Dict[str, Any] = {
                "index": i,
                "size_class": self.fastbin_size_for_index(i, pointer_bits),
                "head": head_ptr,
                "head_raw_hex": slot_bytes.hex(),
                "chain": [],
                "length": 0,
                "truncated": False,
                "cycle_detected": False,
            }
            if head_ptr == 0:
                fastbins_out.append(bin_entry)
                continue
            walk = self._walk_fastbin_chain(
                head_ptr,
                mappings,
                pointer_bits,
                byteorder,
                safe_linking,
                max_chunks_per_bin,
                cancellation_check,
            )
            bin_entry["chain"] = walk["chunks"]
            bin_entry["length"] = walk["length"]
            bin_entry["truncated"] = walk["truncated"]
            bin_entry["cycle_detected"] = walk["cycle_detected"]
            fastbins_out.append(bin_entry)

        parsed = {
            "schema": FASTBIN_INSPECTION_SCHEMA,
            "arena_address": target_address,
            "is_main_arena": target_address == layout.get("arena_address"),
            "layout": {
                "arena_address": layout.get("arena_address"),
                "fastbinsY_offset": fastbins_offset,
                "resolution_source": layout.get("resolution_source"),
                "unresolved_fields": layout.get("unresolved_fields", []),
            },
            "allocator": {
                "family": classification["family"],
                "version": classification.get("version"),
                "confidence": classification.get("confidence"),
                "safe_linking": safe_linking,
            },
            "safe_linking": {
                "state": safe_linking,
                "applied": safe_linking == "enabled",
            },
            "target_profile": profile,
            "fastbins": fastbins_out,
            "max_chunks_per_bin": max_chunks_per_bin,
            "component_status": component_status,
            "supported": True,
            "verdict": "parsed",
        }
        return build_backend_response(
            "inspect_fastbins",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    # ------------------------------------------------------------------
    # Stage 6.3c -- glibc tcache traversal
    # ------------------------------------------------------------------
    @staticmethod
    def tcache_size_for_index(index: int, pointer_bits: int) -> int:
        """Expected chunk size for tcache bin ``index`` on the target.

        Matches glibc's ``csize2tidx`` inverse: same size formula as
        fastbins (``(2 + i) * MALLOC_ALIGNMENT``) but with a wider index
        range (0..TCACHE_MAX_BINS-1 = 0..63 vs fastbins' 0..9).  On
        64-bit that yields 0x20/0x30/.../0x410; on 32-bit 0x10/.../0x208.
        """
        if pointer_bits not in (32, 64):
            raise ValueError(f"pointer_bits must be 32 or 64, got {pointer_bits!r}")
        if index < 0 or index >= TCACHE_MAX_BINS:
            raise ValueError(
                f"tcache index out of range [0, {TCACHE_MAX_BINS}): {index!r}"
            )
        malloc_alignment = 2 * (pointer_bits // 8)
        return (2 + index) * malloc_alignment

    def _walk_tcache_chain(
        self,
        head_user_address: int,
        mappings: Sequence[Mapping[str, Any]],
        pointer_bits: int,
        byteorder: str,
        safe_linking: str,
        max_chunks: int,
        cancellation_check: Optional[Callable[[], None]],
    ) -> Dict[str, Any]:
        """Walk a tcache chain from a head USER address.

        Unlike a fastbin chain (whose head is a chunk address), a tcache
        head is the address of a ``tcache_entry`` -- i.e. the user pointer
        of the containing chunk.  ``next`` is at offset 0 of the entry,
        which coincides with the chunk's ``fd`` slot.  Both the header
        (at user - 2*SIZE_SZ) and the next pointer (at user + 0) are read
        so the entry carries the chunk size alongside the chain link.
        """
        size_sz = pointer_bits // 8
        header_size = 2 * size_sz
        chain: List[Dict[str, Any]] = []
        visited: Dict[int, int] = {}
        cycle_detected = False
        truncated = False
        current: int = head_user_address

        while current != 0:
            if cancellation_check is not None:
                cancellation_check()
            if current in visited:
                cycle_detected = True
                break
            if len(chain) >= max_chunks:
                truncated = True
                break
            visited[current] = len(chain)
            chunk_address = current - header_size
            entry: Dict[str, Any] = {
                "index": len(chain),
                "user_address": current,
                "chunk_address": chunk_address,
                "in_mapping": mapping_for(current, mappings) is not None,
            }
            header, header_err = self._read_bytes(chunk_address, header_size)
            if header_err is not None or len(header) < header_size:
                entry["chunk_readable"] = False
                entry["chunk_read_error"] = header_err or "short read"
                entry["size_field"] = None
                entry["size"] = None
            else:
                entry["chunk_readable"] = True
                size_field = int.from_bytes(
                    header[size_sz : 2 * size_sz], byteorder  # type: ignore[arg-type]
                )
                entry["size_field"] = size_field
                entry["size"] = size_field & ~SIZE_BITS_MASK

            next_bytes, next_err = self._read_bytes(current, size_sz)
            if next_err is not None or len(next_bytes) < size_sz:
                entry["next_raw"] = None
                entry["next_decoded"] = None
                entry["next_read_error"] = next_err or "short read"
                entry["safe_linking_applied"] = False
                chain.append(entry)
                break
            next_raw = int.from_bytes(next_bytes, byteorder)  # type: ignore[arg-type]
            entry["next_raw"] = next_raw
            if safe_linking == "enabled":
                entry["next_decoded"] = self.decode_safe_linked(next_raw, current)
                entry["safe_linking_applied"] = True
            else:
                entry["next_decoded"] = next_raw
                entry["safe_linking_applied"] = False
            chain.append(entry)
            current = entry["next_decoded"]

        return {
            "chunks": chain,
            "length": len(chain),
            "truncated": truncated,
            "cycle_detected": cycle_detected,
        }

    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Walk the per-thread tcache bins and their singly-linked chains.

        Requires libc debug symbols to resolve the thread-local ``tcache``
        pointer, ``sizeof(struct tcache_perthread_struct)`` and the
        ``counts``/``entries`` field offsets (plus the size of a single
        counts element -- ``uint8_t`` before glibc 2.30, ``uint16_t``
        from 2.30 onwards).  With stripped libc the tool returns
        ``verdict = "unresolvable"`` rather than guessing.

        When the thread-local ``tcache`` is NULL (thread has not touched
        tcache yet) the tool returns ``verdict = "uninitialized"`` with
        empty bins -- a legitimate, not-error state.
        """
        if max_chunks_per_bin <= 0:
            raise ValueError(
                f"max_chunks_per_bin must be positive, got {max_chunks_per_bin!r}"
            )

        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self.arenas._unsupported_response(
                "inspect_tcache",
                TCACHE_INSPECTION_SCHEMA,
                classification,
                profile,
                extra={
                    "tcache_address": None,
                    "bins": [],
                    "component_status": component_status,
                },
            )

        resolver = TcacheLayoutResolver(self.debugger)
        layout = resolver.resolve()

        tcache_address = layout.get("tcache_address")
        struct_size = layout.get("struct_size")
        counts_offset = layout["offsets"].get("counts")
        entries_offset = layout["offsets"].get("entries")
        counts_entry_size = layout.get("counts_entry_size")

        if (
            tcache_address is None
            or struct_size is None
            or struct_size <= 0
            or counts_offset is None
            or entries_offset is None
            or counts_entry_size is None
            or counts_entry_size <= 0
        ):
            return build_backend_response(
                "inspect_tcache",
                parsed={
                    "schema": TCACHE_INSPECTION_SCHEMA,
                    "tcache_address": tcache_address,
                    "layout": layout,
                    "allocator": {
                        "family": classification["family"],
                        "version": classification.get("version"),
                        "confidence": classification.get("confidence"),
                    },
                    "target_profile": profile,
                    "bins": [],
                    "component_status": component_status,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": (
                        "tcache address, struct size or field offsets could "
                        "not be resolved via debug symbols"
                    ),
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=[
                    "libc debug symbols are required to locate the tcache "
                    "thread-local and its layout; install libc6-dbg / "
                    "glibc-debuginfo or use a libc built with -g"
                ],
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot walk tcache without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = (
            profile.get("byteorder")
            if profile.get("byteorder") in ("little", "big")
            else "little"
        )
        assert byteorder is not None
        safe_linking = classification.get("safe_linking", "unknown")

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            warnings.append(
                "byteorder could not be resolved from the target; parsed "
                "with little-endian default"
            )
        if safe_linking == "unknown":
            warnings.append(
                "safe-linking applicability is unknown (libc version was "
                "not resolved); next pointers reported as raw with no decode"
            )

        layout_out: Dict[str, Any] = {
            "tcache_address": tcache_address,
            "struct_size": struct_size,
            "counts_offset": counts_offset,
            "counts_entry_size": counts_entry_size,
            "entries_offset": entries_offset,
            "entries_entry_size": layout.get("entries_entry_size"),
            "unresolved_fields": layout.get("unresolved_fields", []),
            "resolution_source": layout.get("resolution_source"),
        }
        allocator_out = {
            "family": classification["family"],
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": safe_linking,
        }
        safe_linking_out = {
            "state": safe_linking,
            "applied": safe_linking == "enabled",
        }

        if tcache_address == 0:
            # Thread has never allocated a chunk in the tcache range yet.
            # This is a well-defined empty state, NOT an error.
            empty_bins: List[Dict[str, Any]] = [
                {
                    "index": i,
                    "size_class": self.tcache_size_for_index(i, pointer_bits),
                    "count": 0,
                    "head": None,
                    "chain": [],
                    "length": 0,
                    "truncated": False,
                    "cycle_detected": False,
                }
                for i in range(TCACHE_MAX_BINS)
            ]
            return build_backend_response(
                "inspect_tcache",
                parsed={
                    "schema": TCACHE_INSPECTION_SCHEMA,
                    "tcache_address": 0,
                    "layout": layout_out,
                    "allocator": allocator_out,
                    "safe_linking": safe_linking_out,
                    "target_profile": profile,
                    "bins": empty_bins,
                    "max_chunks_per_bin": max_chunks_per_bin,
                    "component_status": component_status,
                    "supported": True,
                    "verdict": "uninitialized",
                    "reason": (
                        "tcache thread-local is NULL; the thread has not "
                        "populated tcache yet"
                    ),
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        if cancellation_check is not None:
            cancellation_check()
        raw, err = self._read_bytes(tcache_address, struct_size)
        if err is not None or len(raw) < struct_size:
            warnings.append(f"tcache struct read: {err or 'short read'}")
            return build_backend_response(
                "inspect_tcache",
                parsed={
                    "schema": TCACHE_INSPECTION_SCHEMA,
                    "tcache_address": tcache_address,
                    "layout": layout_out,
                    "allocator": allocator_out,
                    "safe_linking": safe_linking_out,
                    "target_profile": profile,
                    "bins": [],
                    "raw_hex": raw.hex(),
                    "component_status": component_status,
                    "supported": True,
                    "verdict": "unreadable",
                    "reason": err or "short read",
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings,
            )

        # Decode counts[] and entries[] from the read bytes.
        counts: List[Optional[int]] = []
        for i in range(TCACHE_MAX_BINS):
            offset = counts_offset + i * counts_entry_size
            end = offset + counts_entry_size
            if end > len(raw):
                counts.append(None)
            else:
                counts.append(int.from_bytes(raw[offset:end], byteorder))  # type: ignore[arg-type]
        heads: List[Optional[int]] = []
        for i in range(TCACHE_MAX_BINS):
            offset = entries_offset + i * size_sz
            end = offset + size_sz
            if end > len(raw):
                heads.append(None)
            else:
                heads.append(int.from_bytes(raw[offset:end], byteorder))  # type: ignore[arg-type]

        bins_out: List[Dict[str, Any]] = []
        for i in range(TCACHE_MAX_BINS):
            head = heads[i]
            bin_entry: Dict[str, Any] = {
                "index": i,
                "size_class": self.tcache_size_for_index(i, pointer_bits),
                "count": counts[i],
                "head": head,
                "chain": [],
                "length": 0,
                "truncated": False,
                "cycle_detected": False,
            }
            if head is None or head == 0:
                bins_out.append(bin_entry)
                continue
            walk = self._walk_tcache_chain(
                head,
                mappings,
                pointer_bits,
                byteorder,
                safe_linking,
                max_chunks_per_bin,
                cancellation_check,
            )
            bin_entry["chain"] = walk["chunks"]
            bin_entry["length"] = walk["length"]
            bin_entry["truncated"] = walk["truncated"]
            bin_entry["cycle_detected"] = walk["cycle_detected"]
            bins_out.append(bin_entry)

        return build_backend_response(
            "inspect_tcache",
            parsed={
                "schema": TCACHE_INSPECTION_SCHEMA,
                "tcache_address": tcache_address,
                "layout": layout_out,
                "allocator": allocator_out,
                "safe_linking": safe_linking_out,
                "target_profile": profile,
                "bins": bins_out,
                "max_chunks_per_bin": max_chunks_per_bin,
                "raw_hex": raw.hex(),
                "component_status": component_status,
                "supported": True,
                "verdict": "parsed",
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    # ------------------------------------------------------------------
    # Stage 6.3d -- regular bins (unsorted / small / large)
    # ------------------------------------------------------------------
    @staticmethod
    def bin_category(index: int) -> str:
        """Return ``unsorted`` / ``small`` / ``large`` for a bin index.

        Bin 1 is the unsorted bin (recently-freed chunks awaiting
        classification).  Bins 2..63 are the small bins -- one fixed size
        class each, doubly-linked FIFO order.  Bins 64..127 are the large
        bins -- each covers a size range and additionally maintains a
        secondary size-sorted list via fd_nextsize/bk_nextsize (not
        walked here; deferred to a later stage).
        """
        if index == 1:
            return "unsorted"
        if 2 <= index <= 63:
            return "small"
        if 64 <= index <= NBINS - 1:
            return "large"
        raise ValueError(f"bin index out of range [1, {NBINS}): {index!r}")

    def _walk_regular_bin(
        self,
        bin_head_address: int,
        first_fd: int,
        mappings: Sequence[Mapping[str, Any]],
        pointer_bits: int,
        byteorder: str,
        max_chunks: int,
        cancellation_check: Optional[Callable[[], None]],
    ) -> Dict[str, Any]:
        """Walk a regular bin's doubly-linked chain from ``first_fd``.

        Termination is glibc-canonical: the chain circles back to
        ``bin_head_address`` (a synthetic mchunkptr whose fd/bk slots ARE
        the arena's ``bins[]`` entries).  A NULL fd is treated as an
        additional terminator so a partially-initialized bin cannot
        endless-loop; a repeated non-head address triggers cycle
        detection.  Every chunk is read in a single 4*SIZE_SZ gulp so
        header + fd + bk are all captured from one memory read.
        """
        size_sz = pointer_bits // 8
        header_and_ptrs_size = 4 * size_sz  # prev_size + size + fd + bk
        chain: List[Dict[str, Any]] = []
        visited: Dict[int, int] = {}
        cycle_detected = False
        truncated = False
        current: int = first_fd

        while current != bin_head_address:
            if cancellation_check is not None:
                cancellation_check()
            if current == 0:
                # NULL fd is unusual (bins are usually self-referential
                # when empty) but treat it as a benign end-of-chain rather
                # than a crash.
                break
            if current in visited:
                cycle_detected = True
                break
            if len(chain) >= max_chunks:
                truncated = True
                break
            visited[current] = len(chain)
            entry: Dict[str, Any] = {
                "index": len(chain),
                "chunk_address": current,
                "in_mapping": mapping_for(current, mappings) is not None,
            }
            buf, err = self._read_bytes(current, header_and_ptrs_size)
            if err is not None or len(buf) < header_and_ptrs_size:
                entry["chunk_readable"] = False
                entry["chunk_read_error"] = err or "short read"
                chain.append(entry)
                break
            entry["chunk_readable"] = True
            size_field = int.from_bytes(
                buf[size_sz : 2 * size_sz], byteorder  # type: ignore[arg-type]
            )
            entry["size_field"] = size_field
            entry["size"] = size_field & ~SIZE_BITS_MASK
            entry["prev_inuse"] = bool(size_field & PREV_INUSE)
            # prev_size is meaningful only when PREV_INUSE is clear (the
            # preceding chunk is free); while a chunk is in use this slot
            # aliases the predecessor's user data.  Stored raw; consumers
            # decide based on the PREV_INUSE flag.
            entry["prev_size"] = int.from_bytes(
                buf[0:size_sz], byteorder  # type: ignore[arg-type]
            )
            entry["fd"] = int.from_bytes(
                buf[2 * size_sz : 3 * size_sz], byteorder  # type: ignore[arg-type]
            )
            entry["bk"] = int.from_bytes(
                buf[3 * size_sz : 4 * size_sz], byteorder  # type: ignore[arg-type]
            )
            chain.append(entry)
            current = entry["fd"]

        return {
            "chunks": chain,
            "length": len(chain),
            "truncated": truncated,
            "cycle_detected": cycle_detected,
        }

    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Walk all 127 regular bins (unsorted / small / large).

        Reads the whole 254-pointer ``bins[]`` array in a single gulp for
        a consistent snapshot, then per bin: computes ``bin_at(m, i)``
        (the synthetic mchunkptr whose fd/bk slots are the arena entries),
        detects the empty case (fd == bin_head_addr), and walks the
        doubly-linked chain via ``fd`` with per-chain cycle detection and
        a per-bin cap.  Each chunk carries both fd and bk for evidence;
        bk consistency validation is deferred to a corruption classifier.
        """
        if max_chunks_per_bin <= 0:
            raise ValueError(
                f"max_chunks_per_bin must be positive, got {max_chunks_per_bin!r}"
            )

        classification, mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        if classification["family"] != "glibc":
            return self.arenas._unsupported_response(
                "inspect_bins",
                BIN_INSPECTION_SCHEMA,
                classification,
                profile,
                extra={"bins": [], "component_status": component_status},
            )

        layout = self.arenas.resolver.resolve()
        bins_offset = layout["offsets"].get("bins")
        target_address: Optional[int]
        if arena_address is None:
            target_address = layout.get("arena_address")
        else:
            target_address = parse_int(arena_address)
            if target_address is None:
                raise ValueError(
                    f"arena_address must be an integer, got {arena_address!r}"
                )

        if target_address is None or bins_offset is None:
            return self.arenas._unresolvable_response(
                "inspect_bins",
                BIN_INSPECTION_SCHEMA,
                layout,
                classification,
                profile,
                reason=(
                    "arena address or bins field offset could not be "
                    "resolved via debug symbols"
                ),
                extra={
                    "arena_address": target_address,
                    "bins": [],
                    "component_status": component_status,
                },
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot walk regular bins without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        size_sz = pointer_bits // 8
        byteorder = (
            profile.get("byteorder")
            if profile.get("byteorder") in ("little", "big")
            else "little"
        )
        assert byteorder is not None

        warnings: List[str] = []
        if profile.get("byteorder") not in ("little", "big"):
            warnings.append(
                "byteorder could not be resolved from the target; parsed "
                "with little-endian default"
            )

        layout_out: Dict[str, Any] = {
            "arena_address": layout.get("arena_address"),
            "bins_offset": bins_offset,
            "resolution_source": layout.get("resolution_source"),
            "unresolved_fields": layout.get("unresolved_fields", []),
        }
        allocator_out = {
            "family": classification["family"],
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
        }

        # Snapshot the whole bins[] array in one read so every bin sees a
        # consistent view of its fd/bk pair.
        bins_bytes_len = BINS_ARRAY_LEN * size_sz
        if cancellation_check is not None:
            cancellation_check()
        bins_block, err = self._read_bytes(target_address + bins_offset, bins_bytes_len)
        if err is not None or len(bins_block) < bins_bytes_len:
            warnings.append(f"bins[] block read: {err or 'short read'}")
            return build_backend_response(
                "inspect_bins",
                parsed={
                    "schema": BIN_INSPECTION_SCHEMA,
                    "arena_address": target_address,
                    "is_main_arena": target_address == layout.get("arena_address"),
                    "layout": layout_out,
                    "allocator": allocator_out,
                    "target_profile": profile,
                    "bins": [],
                    "max_chunks_per_bin": max_chunks_per_bin,
                    "component_status": component_status,
                    "supported": True,
                    "verdict": "unreadable",
                    "reason": err or "short read",
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings,
            )

        bins_out: List[Dict[str, Any]] = []
        for i in range(1, NBINS):
            fd_slot_offset = (i - 1) * 2 * size_sz
            bk_slot_offset = ((i - 1) * 2 + 1) * size_sz
            fd_slot = int.from_bytes(
                bins_block[fd_slot_offset : fd_slot_offset + size_sz],
                byteorder,  # type: ignore[arg-type]
            )
            bk_slot = int.from_bytes(
                bins_block[bk_slot_offset : bk_slot_offset + size_sz],
                byteorder,  # type: ignore[arg-type]
            )
            # bin_at(m, i) = &bins[(i-1)*2] - offsetof(chunk, fd)
            #             = arena + bins_offset + (i-1)*2*size_sz - 2*size_sz
            bin_head_address = (
                target_address + bins_offset + fd_slot_offset - 2 * size_sz
            )
            empty = fd_slot == bin_head_address
            bin_entry: Dict[str, Any] = {
                "index": i,
                "category": self.bin_category(i),
                "bin_head_address": bin_head_address,
                "fd_head": fd_slot,
                "bk_head": bk_slot,
                "empty": empty,
                "chain": [],
                "length": 0,
                "truncated": False,
                "cycle_detected": False,
            }
            if empty or fd_slot == 0:
                bins_out.append(bin_entry)
                continue
            walk = self._walk_regular_bin(
                bin_head_address,
                fd_slot,
                mappings,
                pointer_bits,
                byteorder,
                max_chunks_per_bin,
                cancellation_check,
            )
            bin_entry["chain"] = walk["chunks"]
            bin_entry["length"] = walk["length"]
            bin_entry["truncated"] = walk["truncated"]
            bin_entry["cycle_detected"] = walk["cycle_detected"]
            bins_out.append(bin_entry)

        return build_backend_response(
            "inspect_bins",
            parsed={
                "schema": BIN_INSPECTION_SCHEMA,
                "arena_address": target_address,
                "is_main_arena": target_address == layout.get("arena_address"),
                "layout": layout_out,
                "allocator": allocator_out,
                "target_profile": profile,
                "bins": bins_out,
                "max_chunks_per_bin": max_chunks_per_bin,
                "component_status": component_status,
                "supported": True,
                "verdict": "parsed",
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )


class TcacheLayoutResolver:
    """Resolve the per-thread ``tcache`` pointer and struct layout via GDB.

    Uses ``-data-evaluate-expression`` to interrogate the running inferior
    for the thread-local ``tcache`` symbol address, the size of
    ``struct tcache_perthread_struct``, the offsets of its ``counts`` and
    ``entries`` fields, and the size of a single counts element (glibc
    2.29- stores ``uint8_t counts[]`` while 2.30+ stores ``uint16_t
    counts[]``).  Requires libc debug symbols; with a stripped libc each
    unresolved value is reported explicitly rather than guessed.
    """

    FIELD_NAMES: Tuple[str, ...] = ("counts", "entries")

    def __init__(self, debugger: DebuggerOperations) -> None:
        self.debugger = debugger

    def _eval_int(self, expression: str) -> Optional[int]:
        result = self.debugger.evaluate_expression(expression)
        if not result.get("success"):
            return None
        parsed = result.get("parsed") or {}
        return parse_int(parsed.get("value"))

    @staticmethod
    def _offset_expression(field_name: str) -> str:
        # Use the ((T*)0)->field trick so offset resolution succeeds even
        # when tcache is NULL (thread has not yet allocated).
        return (
            "(unsigned long)((char*)&((struct tcache_perthread_struct*)0)->"
            f"{field_name}-(char*)0)"
        )

    def resolve(self) -> Dict[str, Any]:
        """Return the resolved layout or explicit unresolved fields."""
        tcache_address = self._eval_int("(void*)tcache")
        struct_size = self._eval_int(
            "(unsigned long)sizeof(struct tcache_perthread_struct)"
        )
        counts_entry_size = self._eval_int(
            "(unsigned long)sizeof(((struct tcache_perthread_struct*)0)->counts[0])"
        )
        entries_entry_size = self._eval_int(
            "(unsigned long)sizeof(((struct tcache_perthread_struct*)0)->entries[0])"
        )

        offsets: Dict[str, Optional[int]] = {}
        unresolved_fields: List[str] = []
        for name in self.FIELD_NAMES:
            value = self._eval_int(self._offset_expression(name))
            if value is None:
                unresolved_fields.append(name)
            offsets[name] = value

        # Fallback: if we know both offsets and struct is well-formed
        # (counts comes first), derive the counts element size from the
        # gap.  This lets us succeed even when the direct sizeof-of-element
        # query fails while the struct offsets resolved cleanly.
        counts_off = offsets.get("counts")
        entries_off = offsets.get("entries")
        if (
            counts_entry_size is None
            and counts_off is not None
            and entries_off is not None
            and entries_off > counts_off
        ):
            derived = (entries_off - counts_off) // TCACHE_MAX_BINS
            if derived > 0:
                counts_entry_size = derived

        return {
            "tcache_address": tcache_address,
            "struct_size": struct_size,
            "counts_entry_size": counts_entry_size,
            "entries_entry_size": entries_entry_size,
            "offsets": offsets,
            "unresolved_fields": unresolved_fields,
            "resolution_source": "gdb_debug_symbols",
        }
