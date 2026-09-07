"""Typed musl mallocng inspection.

musl >= 1.2 uses the mallocng allocator: 16-byte (``UNIT``) slots inside
4 KiB groups, each group described by a ``struct meta`` living in a
``struct meta_area``.  A user pointer resolves to its group with pure
arithmetic plus a few targeted reads (``get_meta`` in musl's
``mallocng/meta.h``), which this module mirrors check by check.  All
layouts come from GDB ``offsetof``/``sizeof`` queries against the *real*
symbol names (``__malloc_context``, ``__malloc_size_classes`` -- the
short names are macro-renamed away), so nothing is guessed from a
hardcoded table; without debug symbols every entry point reports
``unresolvable`` instead of fabricating offsets.
"""

from __future__ import annotations

import copy
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.heap_glibc_arenas import (
    arena_layout_cache_key,
    inferior_epoch,
)

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"
MUSL_CHUNK_SCHEMA = "pao.musl-chunk.v1"
MUSL_GROUPS_SCHEMA = "pao.musl-groups.v1"
MUSL_MAX_GROUPS = 256
MUSL_MAX_AREAS = 64
MUSL_MAX_NSLOTS = 4096
MUSL_UNIT = 16
MUSL_IB = 4
MUSL_SIZECLASS_COUNT = 48


class MuslAnalyzer:
    """Inspect musl mallocng chunks and groups on a live inferior."""

    def __init__(
        self,
        gdb: Any,
        inspection: Any,
        debugger: Any,
        session_id: str,
        *,
        discovery: Any,
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        # Layout cache: offsets/sizes are libc-stable, so one resolve per
        # libc identity serves the session. Volatile fields (ctx address,
        # secret, class-table address) are never cached: ASLR slides
        # addresses and the secret is per-exec random.
        self._layout_cache: Optional[Tuple[tuple, Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> None:
        LiveInferiorGuard.require_live(
            self.gdb.get_last_stop_record(),
            self.gdb.get_inferior_pid(),
            detail=(
                "musl inspection requires a live, stopped inferior; run "
                "the target to a stop before inspecting its allocations"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Live-check, classify, and resolve the target profile."""
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = self.discovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    def _eval_int(self, expression: str) -> Optional[int]:
        result = self.debugger.evaluate_expression(expression)
        if not result.get("success"):
            return None
        parsed = result.get("parsed") or {}
        return parse_int(parsed.get("value"))

    def _read_bytes(self, address: int, byte_count: int) -> Tuple[bytes, Optional[str]]:
        result = self.gdb.read_memory_bytes(f"0x{address:x}", byte_count)
        if not result.get("success"):
            return b"", "memory read failed"
        contents = DebuggerOperations._memory_contents(result)
        if contents is None:
            return b"", "memory read returned no contents"
        try:
            raw = bytes.fromhex(str(contents))
        except ValueError as exc:
            return b"", f"invalid hex contents: {exc}"
        if len(raw) < byte_count:
            return raw, f"short read: expected {byte_count} bytes, got {len(raw)}"
        return raw, None

    def _resolve_context(self) -> Dict[str, Any]:
        """Resolve mallocng symbols; every field is explicit or None."""
        return {
            "ctx_addr": self._eval_int("(void*)&__malloc_context"),
            "secret": self._eval_int("(unsigned long)__malloc_context.secret"),
            "area_head_off": self._eval_int(
                "(unsigned long)((char*)&((struct malloc_context*)0)"
                "->meta_area_head-(char*)0)"
            ),
            "meta_size": self._eval_int("(unsigned long)sizeof(struct meta)"),
            "mem_off": self._eval_int(
                "(unsigned long)((char*)&((struct meta*)0)->mem-(char*)0)"
            ),
            "classes_addr": self._eval_int("(void*)&__malloc_size_classes"),
        }

    def cached_musl_layout(
        self,
        classification: Mapping[str, Any],
        profile: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return the mallocng layout, resolving once per libc identity.

        Only stable fields (area-head/meta offsets, meta size) come from
        the cache; ``ctx_addr``, ``secret`` and ``classes_addr`` are always
        re-read (ASLR slide, per-exec secret). Nothing is cached when the
        layout is unresolvable, preserving the legacy always-resolve path.
        """
        key = arena_layout_cache_key(
            classification, profile, family="musl", epoch=inferior_epoch(self.gdb)
        )
        layout: Dict[str, Any]
        if (
            key is not None
            and self._layout_cache is not None
            and self._layout_cache[0] == key
        ):
            layout = copy.deepcopy(self._layout_cache[1])
        else:
            full = self._resolve_context()
            layout = {
                k: v
                for k, v in full.items()
                if k not in ("ctx_addr", "secret", "classes_addr")
            }
            if (
                key is not None
                and layout.get("area_head_off") is not None
                and layout.get("meta_size") in (40, 28)
            ):
                self._layout_cache = (key, copy.deepcopy(layout))
        layout["ctx_addr"] = self._eval_int("(void*)&__malloc_context")
        layout["secret"] = self._eval_int("(unsigned long)__malloc_context.secret")
        layout["classes_addr"] = self._eval_int("(void*)&__malloc_size_classes")
        return layout

    def _allocator(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification.get("family"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
        }

    def _unsupported(
        self,
        command: str,
        schema: str,
        classification: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        return build_backend_response(
            command,
            parsed={
                "schema": schema,
                "allocator": self._allocator(classification),
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    "musl mallocng parsing supports musl only; detected "
                    f"family={classification.get('family')!r}"
                ),
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=["musl inspection refused on a non-musl target"],
        )

    def _unresolvable(
        self,
        command: str,
        schema: str,
        reason: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
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
            warnings=["musl layout needs libc debug symbols; refusing to guess"],
        )

    @staticmethod
    def _decode_meta_bits(word: int, byteorder: Optional[str]) -> Dict[str, int]:
        """Decode the packed meta bitfield on either endianness."""
        if byteorder == "big":
            return {
                "last_idx": (word >> 59) & 31,
                "freeable": (word >> 58) & 1,
                "sizeclass": (word >> 52) & 63,
                "maplen": word & ((1 << 52) - 1),
            }
        return {
            "last_idx": word & 31,
            "freeable": (word >> 5) & 1,
            "sizeclass": (word >> 6) & 63,
            "maplen": word >> 12,
        }

    def _read_size_classes(
        self, classes_addr: Optional[int], byteorder: str = "little"
    ) -> Optional[List[int]]:
        """Read the 48-entry stride table (units of 16 bytes)."""
        if classes_addr is None:
            return None
        raw, err = self._read_bytes(classes_addr, 2 * MUSL_SIZECLASS_COUNT)
        if err is not None or len(raw) < 2 * MUSL_SIZECLASS_COUNT:
            return None
        order = byteorder if byteorder in ("little", "big") else "little"
        return [
            int.from_bytes(raw[i * 2 : i * 2 + 2], order)  # type: ignore[arg-type]
            for i in range(MUSL_SIZECLASS_COUNT)
        ]

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def inspect_musl_chunk(
        self,
        *,
        address: str,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Resolve a user pointer to its mallocng group and validate it."""
        target = parse_int(address)
        if target is None:
            raise ValueError(f"address must be an integer, got {address!r}")
        classification, mappings, profile, _ = self._prepare_context(cancellation_check)
        if classification.get("family") != "musl":
            return self._unsupported(
                "inspect_musl_chunk", MUSL_CHUNK_SCHEMA, classification, profile
            )
        checks: List[Dict[str, Any]] = []

        def check(name: str, ok: bool, detail: Any = None) -> bool:
            checks.append({"check": name, "passed": bool(ok), "detail": detail})
            return bool(ok)

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot parse musl chunks without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        raw_order = profile.get("byteorder")
        byteorder: Literal["little", "big"] = (
            cast(Literal["little", "big"], raw_order)
            if raw_order in ("little", "big")
            else "little"
        )
        size_sz = pointer_bits // 8

        if target % MUSL_UNIT != 0:
            check("alignment_16", False, f"{target:#x} is not 16-aligned")
            return self._chunk_verdict(
                target, None, checks, mappings, "misaligned pointer"
            )
        check("alignment_16", True, target)

        context = self.cached_musl_layout(classification, profile)
        missing = sorted(k for k, v in context.items() if v is None)
        if missing:
            return self._unresolvable(
                "inspect_musl_chunk",
                MUSL_CHUNK_SCHEMA,
                "musl debug symbols missing for: " + ", ".join(missing),
                extra={"address": target, "checks": checks},
            )
        secret = context["secret"]
        assert secret is not None

        # Header window: the 8 bytes musl itself reads (offset/index/words).
        header, header_err = self._read_bytes(target - 8, 8)
        if header_err is not None or len(header) < 8:
            check("header_readable", False, header_err or "short read")
            return self._chunk_verdict(
                target, None, checks, mappings, "unreadable header"
            )
        check("header_readable", True, 8)
        offset = int.from_bytes(header[6:8], byteorder)
        index = header[5] & 31
        if header[4]:
            offset32 = int.from_bytes(header[0:4], byteorder)
            if offset32 <= 0xFFFF:
                check("wide_offset_sane", False, offset32)
                return self._chunk_verdict(
                    target, None, checks, mappings, "bad wide offset"
                )
            offset = offset32
        check("wide_offset_sane", True, offset)

        group_base = target - MUSL_UNIT * offset - MUSL_UNIT
        group_meta_raw, group_err = self._read_bytes(group_base, size_sz)
        if group_err is not None or len(group_meta_raw) < size_sz:
            check("group_meta_readable", False, group_err or "short read")
            return self._chunk_verdict(
                target, None, checks, mappings, "unreadable group"
            )
        group_meta = int.from_bytes(group_meta_raw, byteorder)
        check("group_meta_readable", True, f"{group_meta:#x}")

        meta_size = context["meta_size"]
        meta_raw, meta_err = self._read_bytes(group_meta, meta_size)
        if meta_err is not None or len(meta_raw) < meta_size:
            check("meta_readable", False, meta_err or "short read")
            return self._chunk_verdict(
                target, None, checks, mappings, "unreadable meta"
            )
        check("meta_readable", True, meta_size)
        mem_off = context["mem_off"]
        meta_mem = int.from_bytes(meta_raw[mem_off : mem_off + size_sz], byteorder)
        if meta_mem != group_base:
            check(
                "meta_mem_matches_group", False, [f"{meta_mem:#x}", f"{group_base:#x}"]
            )
            return self._chunk_verdict(
                target, None, checks, mappings, "meta->mem mismatch"
            )
        check("meta_mem_matches_group", True, f"{group_base:#x}")

        bits_off = 3 * size_sz + 8
        bits = int.from_bytes(meta_raw[bits_off : bits_off + 8], byteorder)
        fields = self._decode_meta_bits(bits, byteorder)
        last_idx, sizeclass, maplen = (
            fields["last_idx"],
            fields["sizeclass"],
            fields["maplen"],
        )
        if index > last_idx:
            check("index_in_range", False, [index, last_idx])
            return self._chunk_verdict(
                target, None, checks, mappings, "slot index out of range"
            )
        check("index_in_range", True, [index, last_idx])

        avail = int.from_bytes(meta_raw[3 * size_sz : 3 * size_sz + 4], byteorder)
        freed = int.from_bytes(meta_raw[3 * size_sz + 4 : 3 * size_sz + 8], byteorder)
        bit = 1 << index
        if (avail & bit) and (freed & bit):
            check("masks_consistent", False, [f"{avail:#x}", f"{freed:#x}"])
            return self._chunk_verdict(
                target, None, checks, mappings, "slot in avail+freed"
            )
        check("masks_consistent", True, None)

        area_base = group_meta & -4096
        area_raw, area_err = self._read_bytes(area_base, 8)
        if area_err is None and len(area_raw) >= 8:
            area_check = int.from_bytes(area_raw[:8], byteorder)
            if area_check != secret:
                check("meta_area_secret", False, f"{area_check:#x}")
                return self._chunk_verdict(
                    target, None, checks, mappings, "meta-area secret mismatch"
                )
            check("meta_area_secret", True, None)
        else:
            check("meta_area_secret", False, area_err or "short read")
            return self._chunk_verdict(
                target, None, checks, mappings, "unreadable meta area"
            )

        classes = self._read_size_classes(context["classes_addr"], byteorder)
        if sizeclass < MUSL_SIZECLASS_COUNT:
            if classes is None:
                return self._unresolvable(
                    "inspect_musl_chunk",
                    MUSL_CHUNK_SCHEMA,
                    "size_classes table unreadable",
                    extra={"address": target, "checks": checks},
                )
            if not (
                classes[sizeclass] * index <= offset < classes[sizeclass] * (index + 1)
            ):
                check("offset_in_stride_range", False, [offset, sizeclass])
                return self._chunk_verdict(
                    target, None, checks, mappings, "offset outside stride"
                )
            check("offset_in_stride_range", True, None)
            stride: Optional[int] = MUSL_UNIT * classes[sizeclass]
        elif sizeclass == 63:
            check("offset_in_stride_range", True, "mmap path")
            stride = None
        else:
            check("sizeclass_known", False, sizeclass)
            return self._chunk_verdict(
                target, None, checks, mappings, "unknown sizeclass"
            )
        if maplen and not (offset <= maplen * 4096 // MUSL_UNIT - 1):
            check("maplen_bounds", False, [offset, maplen])
            return self._chunk_verdict(
                target, None, checks, mappings, "offset beyond maplen"
            )
        check("maplen_bounds", True, maplen or 0)

        # Nominal size (musl get_nominal_size): reserved tail bookkeeping.
        nominal: Optional[int] = None
        reserved = header[5] >> 5
        if stride is not None:
            slot_start = group_base + MUSL_UNIT + stride * index
            end = slot_start + stride - MUSL_IB
            if reserved >= 5:
                tail, tail_err = self._read_bytes(end - 4, 4)
                if tail_err is None and len(tail) == 4 and end - 5 >= target:
                    reserved = int.from_bytes(tail, byteorder)
            if reserved <= end - target:
                nominal = end - reserved - target
        state = "in_use"
        if freed & bit:
            state = "freed"
        elif avail & bit:
            state = "available"
        return self._chunk_verdict(
            target,
            {
                "group_base": group_base,
                "meta_address": group_meta,
                "slot_index": index,
                "group_offset": offset,
                "sizeclass": sizeclass,
                "stride": stride,
                "maplen": maplen,
                "nominal_size": nominal,
                "state": state,
                "avail_mask": avail,
                "freed_mask": freed,
                "last_idx": last_idx,
            },
            checks,
            mappings,
            None,
        )

    def _chunk_verdict(
        self,
        target: int,
        detail: Optional[Dict[str, Any]],
        checks: List[Dict[str, Any]],
        mappings: Sequence[Mapping[str, Any]],
        corrupted: Optional[str],
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": MUSL_CHUNK_SCHEMA,
            "address": target,
            "supported": True,
            "verdict": "corrupted" if corrupted else "valid",
            "checks": checks,
            "in_mapping": mapping_for(target, mappings) is not None,
        }
        if corrupted:
            parsed["corruption"] = corrupted
        if detail:
            parsed.update(detail)
        return build_backend_response(
            "inspect_musl_chunk",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def list_musl_groups(
        self,
        *,
        max_groups: int = 64,
        max_areas: int = 16,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Walk ``ctx.meta_area_head`` and inventory every allocation group."""
        if max_groups <= 0 or max_groups > 4096:
            raise ValueError("max_groups must be between 1 and 4096")
        if max_areas <= 0 or max_areas > 64:
            raise ValueError("max_areas must be between 1 and 64")
        classification, mappings, profile, _ = self._prepare_context(cancellation_check)
        if classification.get("family") != "musl":
            return self._unsupported(
                "list_musl_groups", MUSL_GROUPS_SCHEMA, classification, profile
            )
        context = self.cached_musl_layout(classification, profile)
        missing = sorted(k for k, v in context.items() if v is None)
        try:
            pointer_bits = profile.get("pointer_bits") or 64
        except Exception:  # noqa: BLE001 - default width on profile failure
            pointer_bits = 64
        if pointer_bits not in (32, 64):
            raise RuntimeError(f"unsupported pointer width {pointer_bits!r}")
        void_star = 8 if pointer_bits == 64 else 4
        expected_meta = 40 if pointer_bits == 64 else 28
        raw_order = profile.get("byteorder")
        byteorder: Literal["little", "big"] = (
            cast(Literal["little", "big"], raw_order)
            if raw_order in ("little", "big")
            else "little"
        )
        if missing or context.get("meta_size") != expected_meta:
            return self._unresolvable(
                "list_musl_groups",
                MUSL_GROUPS_SCHEMA,
                "musl debug symbols missing for: "
                + ", ".join(missing or ["meta_size"]),
            )
        if cancellation_check is not None:
            cancellation_check()
        secret = context["secret"]
        assert secret is not None
        area_head_raw, area_head_err = self._read_bytes(
            context["ctx_addr"] + context["area_head_off"], void_star
        )
        if area_head_err is not None:
            return self._unresolvable(
                "list_musl_groups",
                MUSL_GROUPS_SCHEMA,
                "meta_area_head unreadable",
            )
        head = int.from_bytes(area_head_raw, byteorder)
        groups: List[Dict[str, Any]] = []
        areas = 0
        corrupted: Optional[str] = None
        seen = {0}
        cursor: Optional[int] = head
        while cursor:
            if cancellation_check is not None:
                cancellation_check()
            if areas >= max_areas or cursor in seen:
                if cursor in seen and cursor != 0:
                    corrupted = f"meta_area cycle at {cursor:#x}"
                break
            seen.add(cursor)
            header, header_err = self._read_bytes(cursor, 16 + void_star)
            if header_err is not None or len(header) < 16 + void_star:
                corrupted = f"unreadable meta_area at {cursor:#x}"
                break
            if int.from_bytes(header[:8], byteorder) != secret:
                corrupted = f"meta_area secret mismatch at {cursor:#x}"
                break
            nxt = int.from_bytes(header[8 : 8 + void_star], byteorder)
            nslots = int.from_bytes(header[8 + void_star : 12 + void_star], byteorder)
            if nslots <= 0 or nslots > 4096:
                corrupted = f"implausible nslots {nslots} at {cursor:#x}"
                break
            areas += 1
            meta_size = context["meta_size"]
            blob, blob_err = self._read_bytes(
                cursor + 16 + void_star, nslots * meta_size
            )
            if blob_err is not None or len(blob) < nslots * meta_size:
                corrupted = f"unreadable meta slots at {cursor:#x}"
                break
            for i in range(nslots):
                if len(groups) >= max_groups:
                    break
                slot = blob[i * meta_size : (i + 1) * meta_size]
                mem = int.from_bytes(slot[2 * void_star : 3 * void_star], byteorder)
                if mem == 0:
                    continue
                bits = int.from_bytes(
                    slot[3 * void_star + 8 : 3 * void_star + 16], byteorder
                )
                fields = self._decode_meta_bits(bits, byteorder)
                groups.append(
                    {
                        "meta_address": cursor + 16 + void_star + i * meta_size,
                        "group_base": mem,
                        "sizeclass": fields["sizeclass"],
                        "last_idx": fields["last_idx"],
                        "freeable": fields["freeable"],
                        "maplen": fields["maplen"],
                        "avail_mask": int.from_bytes(
                            slot[3 * void_star : 3 * void_star + 4], byteorder
                        ),
                        "freed_mask": int.from_bytes(
                            slot[3 * void_star + 4 : 3 * void_star + 8], byteorder
                        ),
                    }
                )
            cursor = nxt or None
        parsed: Dict[str, Any] = {
            "schema": MUSL_GROUPS_SCHEMA,
            "supported": True,
            "verdict": "corrupted" if corrupted else "parsed",
            "group_count": len(groups),
            "area_count": areas,
            "groups": groups,
        }
        if corrupted:
            parsed["corruption"] = corrupted
        return build_backend_response(
            "list_musl_groups",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = [
    "MuslAnalyzer",
    "MUSL_CHUNK_SCHEMA",
    "MUSL_GROUPS_SCHEMA",
]
