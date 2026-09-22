"""Kernel SLUB slab analyzer -- BTF field offsets + obfuscated freelist decode.

SLUB (the default Linux slab allocator)
hardens its freelist with ``CONFIG_SLAB_FREELIST_HARDENED``: every stored
freepointer is XORed with a per-cache random value and the address it is
stored at,

    stored = ptr ^ s->random ^ swab(store_addr)      (kernels >= 5.7)
    stored = ptr ^ s->random ^  store_addr           (pre-5.7 "legacy")

Decoding needs ``s->random``, the freepointer offset ``s->offset`` inside
each object, and the slab geometry -- all resolved from BTF on any kernel
version without hard-coded layout tables.  Two subtleties:

* the *slab head* (``slab->freelist``) is not necessarily encoded: on 6.x
  ``allocate_slab()`` stores a raw decoded pointer there and only object
  slots are obfuscated, whereas 5.x-era ``__slab_free`` stored an encoded
  head.  The decoder ranges over ``head_mode x encoding`` and picks the
  chain whose objects land inside the slab window,
* the swab boundary is commit ``1ad53d9fa3f6`` ("slub: improve bit
  diffusion for freelist ptr obfuscation", v5.7), not the 6.2 ``freeptr_t``
  rename -- the rename did not change the formula.

This module holds the pure, memory-agnostic decode machinery and the
``KernelSlab`` component that feeds it live target memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    cast,
)

_Byteorder = Literal["little", "big"]

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    parse_int,
)
from paomcp.tools.backends.btf import BtfError, BtfFile, BtfMember, load_btf
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.heap import (
    PARSER_VERSION,
    AllocatorDiscovery,
    DEFAULT_SLUB_PAGE_SIZE,
)
from paomcp.tools.backends.inspection import PwndbgInspection

# ---------------------------------------------------------------------------
# Schemas + constants
# ---------------------------------------------------------------------------
SLUB_CACHE_SCHEMA = "pao.kernel.slub-cache.v1"
SLUB_FREELIST_SCHEMA = "pao.kernel.slub-freelist.v1"

SLUB_ENCODING_SWAB = "swab"
SLUB_ENCODING_LEGACY = "legacy"
SLUB_HEAD_RAW = "raw"
SLUB_HEAD_ENCODED = "encoded_at_field"

# Safety cap on decoded chain length: a corrupted freepointer can loop, and
# even a healthy chain rarely exceeds the object count of one slab.
DEFAULT_MAX_SLUB_OBJECTS = 4096

# Header sources (struct slab first, struct page fallback).
_SLAB_HEADER_STRUCT_SLAB = "slab"
_SLAB_HEADER_STRUCT_PAGE = "page"

# BTF source display name.
_DEFAULT_BTF_SOURCE = "/sys/kernel/btf/vmlinux"


# ---------------------------------------------------------------------------
# Pure decode helpers
# ---------------------------------------------------------------------------
def slub_swab(value: int, pointer_bits: int) -> int:
    """Byte-reverse a pointer-sized integer (the kernel's ``swab64``/``swab32``)."""
    byte_count = pointer_bits // 8
    result = 0
    for i in range(byte_count):
        result = (result << 8) | ((value >> (8 * i)) & 0xFF)
    return result


def slub_decode_ptr(
    encoded: int,
    random: int,
    store_addr: int,
    pointer_bits: int,
    *,
    hardened: bool,
    encoding: Optional[str],
) -> int:
    """Decode one SLUB freepointer.

    ``store_addr`` is the address the encoded value lives at: ``object +
    s->offset`` for per-object slots, ``&slab->freelist`` for the head.
    Returns the decoded pointer (0 == NULL terminator).  ``encoding`` is
    ignored (and may be ``None``) when ``hardened`` is False.
    """
    mask = (1 << pointer_bits) - 1
    if not hardened:
        return encoded & mask
    if encoding == SLUB_ENCODING_SWAB:
        return (encoded ^ random ^ slub_swab(store_addr, pointer_bits)) & mask
    if encoding == SLUB_ENCODING_LEGACY:
        return (encoded ^ random ^ store_addr) & mask
    raise ValueError(f"unknown SLUB encoding: {encoding!r}")


@dataclass(frozen=True)
class SlubFreeObject:
    """One decoded free object and the encoded link stored inside it."""

    object_address: int
    freepointer_address: int
    encoded_next: int
    decoded_next: int


@dataclass
class SlubChain:
    """A decoded SLUB freelist chain for one head interpretation."""

    head: int
    head_mode: str
    encoding: Optional[str]
    hardened: bool
    objects: List[int] = field(default_factory=list)
    chain: List[SlubFreeObject] = field(default_factory=list)
    termination: str = "null"
    note: Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.objects)


def slub_walk_chain(
    *,
    head: int,
    head_is_encoded: bool,
    head_store_addr: int,
    cache_offset: int,
    cache_random: int,
    hardened: bool,
    encoding: Optional[str],
    slab_mem: int,
    slab_bytes: int,
    max_objects: int,
    pointer_bits: int,
    reader: Callable[[int], Optional[int]],
) -> SlubChain:
    """Walk one freelist head interpretation, following per-object links.

    ``reader(addr)`` returns the raw encoded freepointer stored at ``addr``
    or ``None`` when the address is unreadable.  The walk is bounded by the
    slab window, a seen-set (cycle protection) and ``max_objects``.
    """
    mask = (1 << pointer_bits) - 1
    if head_is_encoded:
        current = slub_decode_ptr(
            head,
            cache_random,
            head_store_addr,
            pointer_bits,
            hardened=hardened,
            encoding=encoding,
        )
    else:
        current = head & mask

    chain = SlubChain(
        head=head,
        head_mode=SLUB_HEAD_ENCODED if head_is_encoded else SLUB_HEAD_RAW,
        encoding=encoding,
        hardened=hardened,
    )
    if current == 0:
        chain.termination = "null"
        return chain

    seen = set()
    slab_end = slab_mem + slab_bytes
    while True:
        if current in seen:
            chain.termination = "cycle"
            chain.note = f"freepointer loop detected at {hex(current)}"
            break
        seen.add(current)
        if not (slab_mem <= current < slab_end):
            chain.termination = "out_of_range"
            chain.note = (
                f"decoded object {hex(current)} outside slab window "
                f"[{hex(slab_mem)}, {hex(slab_end)})"
            )
            break
        if len(chain.objects) >= max_objects:
            chain.termination = "max_objects"
            chain.note = f"stopped at max_objects={max_objects}"
            break

        freepointer_address = current + cache_offset
        encoded_next = reader(freepointer_address)
        if encoded_next is None:
            chain.termination = "unreadable"
            chain.note = f"could not read freepointer at {hex(freepointer_address)}"
            break
        decoded_next = slub_decode_ptr(
            encoded_next,
            cache_random,
            freepointer_address,
            pointer_bits,
            hardened=hardened,
            encoding=encoding,
        )
        chain.objects.append(current)
        chain.chain.append(
            SlubFreeObject(
                object_address=current,
                freepointer_address=freepointer_address,
                encoded_next=encoded_next,
                decoded_next=decoded_next,
            )
        )
        if decoded_next == 0:
            chain.termination = "null"
            break
        current = decoded_next
    return chain


def slub_detect_chain(
    *,
    head: int,
    head_store_addr: int,
    cache_offset: int,
    cache_random: int,
    hardened: bool,
    slab_mem: int,
    slab_bytes: int,
    slab_objects: int,
    max_objects: int,
    pointer_bits: int,
    reader: Callable[[int], Optional[int]],
) -> Tuple[Optional[SlubChain], List[SlubChain]]:
    """Detect the correct head/encoding interpretation by window scoring.

    Returns ``(best_chain, attempts)``; ``best_chain`` is ``None`` when no
    candidate decoded at least one object inside the slab window (verdict:
    ``unresolved``).  ``attempts`` always carries every candidate for
    evidence.  Preference: most decoded objects, then raw head, then swab.
    """
    if not hardened:
        chain = slub_walk_chain(
            head=head,
            head_is_encoded=False,
            head_store_addr=head_store_addr,
            cache_offset=cache_offset,
            cache_random=cache_random,
            hardened=False,
            encoding=None,
            slab_mem=slab_mem,
            slab_bytes=slab_bytes,
            max_objects=max_objects,
            pointer_bits=pointer_bits,
            reader=reader,
        )
        return chain, [chain]

    mask = (1 << pointer_bits) - 1
    if head & mask == 0:
        # Empty freelist (``slab->freelist == NULL``).  A zero head is
        # unambiguous under both plain and hardened kernels: interpreting it
        # as encoded would XOR NULL with ``random``/store-address and invent a
        # pointer that walks out of the window.  Report the raw NULL-terminated
        # empty chain directly so hardened and plain kernels produce the same
        # "decoded, count 0" verdict instead of the hardened path falling
        # through the count>=1 candidate filter to ``None`` (unresolved).
        chain = slub_walk_chain(
            head=head,
            head_is_encoded=False,
            head_store_addr=head_store_addr,
            cache_offset=cache_offset,
            cache_random=cache_random,
            hardened=hardened,
            encoding=None,
            slab_mem=slab_mem,
            slab_bytes=slab_bytes,
            max_objects=max_objects,
            pointer_bits=pointer_bits,
            reader=reader,
        )
        return chain, [chain]

    attempts: List[SlubChain] = []
    candidates: List[SlubChain] = []
    for head_mode, head_is_encoded in (
        (SLUB_HEAD_RAW, False),
        (SLUB_HEAD_ENCODED, True),
    ):
        for encoding in (SLUB_ENCODING_SWAB, SLUB_ENCODING_LEGACY):
            chain = slub_walk_chain(
                head=head,
                head_is_encoded=head_is_encoded,
                head_store_addr=head_store_addr,
                cache_offset=cache_offset,
                cache_random=cache_random,
                hardened=True,
                encoding=encoding,
                slab_mem=slab_mem,
                slab_bytes=slab_bytes,
                max_objects=max_objects,
                pointer_bits=pointer_bits,
                reader=reader,
            )
            attempts.append(chain)
            if chain.count >= 1 and chain.termination != "out_of_range":
                candidates.append(chain)

    if not candidates:
        return None, attempts

    def _rank(candidate: SlubChain) -> Tuple[int, int, int]:
        # Most objects first; then raw head over encoded; then swab over legacy.
        return (
            candidate.count,
            1 if candidate.head_mode == SLUB_HEAD_RAW else 0,
            1 if candidate.encoding == SLUB_ENCODING_SWAB else 0,
        )

    best = max(candidates, key=_rank)
    return best, attempts


# ---------------------------------------------------------------------------
# KernelSlab component
# ---------------------------------------------------------------------------
class KernelSlab:
    """SLUB slab analyzer for a live kernel target (BTF-backed, read-only).

    A dedicated, self-gated component (like ``GlibcAttackHelper``): kernel
    targets are not classified by the userspace allocator protocol, so the
    facade never routes it through ``classify`` -- the caller names a cache
    address and the analyzer resolves the layout from BTF.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        discovery: AllocatorDiscovery,
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        # Target memory byte order.  Defaults little-endian; a caller on a
        # big-endian kernel (s390x, ppc64, mips64, arm64_be) must pass
        # byteorder="big" so struct reads decode correctly.  BTF itself is
        # always little-endian, so only *target memory* reads care.
        self._byteorder: _Byteorder = "little"

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "SLUB analysis requires a live, stopped inferior; run the "
                "target to a stop before inspecting the kernel heap"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, _, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        profile = self.discovery.target_profile(mappings)
        return profile, component_status

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

    # ------------------------------------------------------------------
    # BTF + field-offset resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _load_btf(btf_path: Optional[str]) -> Tuple[Optional[BtfFile], Optional[str]]:
        """Load + parse a BTF blob; returns (btf, error_reason)."""
        source = btf_path or _DEFAULT_BTF_SOURCE
        try:
            return load_btf(source), None
        except OSError as exc:
            return None, f"cannot read BTF blob {source}: {exc}"
        except BtfError as exc:
            return None, f"malformed BTF blob {source}: {exc}"

    @staticmethod
    def _btf_info(btf: BtfFile, source: str) -> Dict[str, Any]:
        return {
            "source": source,
            "version": btf.header.version,
            "type_count": btf.type_count,
        }

    def _read_unsigned(
        self, buf: bytes, offset: int, width: int, byteorder: _Byteorder
    ) -> int:
        return int.from_bytes(buf[offset : offset + width], byteorder)

    def _read_member(
        self,
        buf: bytes,
        member: BtfMember,
        pointer_bits: int,
        byteorder: _Byteorder,
    ) -> int:
        """Extract a member value from a struct buffer, bitfield-aware.

        Bitfields (``slab.inuse/objects/frozen`` live in the ``counters``
        word) are shifted out of the containing ``pointer_bits//8``-byte word
        by their intra-word bit offset and masked to their width.
        """
        word_bytes = pointer_bits // 8
        byte_offset = member.byte_offset
        if byte_offset + word_bytes > len(buf):
            raise ValueError(
                f"member {member.name!r} at byte {byte_offset} overruns the "
                f"{len(buf)}-byte buffer"
            )
        word = int.from_bytes(buf[byte_offset : byte_offset + word_bytes], byteorder)
        if not member.is_bitfield:
            return word
        shift = member.bit_offset % 8
        mask = (1 << member.bitfield_size) - 1
        return (word >> shift) & mask

    def _read_cstring(self, address: int, max_len: int = 128) -> Optional[str]:
        raw, _err = self._read_bytes(address, max_len)
        if not raw:
            return None
        end = raw.find(b"\x00")
        if end == -1:
            end = len(raw)
        return raw[:end].decode("utf-8", errors="replace")

    def _reader(
        self, pointer_bits: int, byteorder: _Byteorder
    ) -> Callable[[int], Optional[int]]:
        """A memory reader returning pointer-sized ints in target byte order."""
        size = pointer_bits // 8

        def read(address: int) -> Optional[int]:
            raw, err = self._read_bytes(address, size)
            if err is not None or len(raw) < size:
                return None
            return int.from_bytes(raw, byteorder)

        return read

    @staticmethod
    def _resolve_cache_offsets(
        btf: BtfFile,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Resolve kmem_cache field offsets; ``random`` is optional.

        Returns (offsets, error).  ``offsets`` maps field name -> byte offset
        and carries ``_size_bytes`` (the struct size) and ``_hardened`` (the
        ``random`` member exists and is readable -- nonzero value checked at
        read time).
        """
        struct = btf.struct("kmem_cache")
        if struct is None:
            return None, "struct kmem_cache not found in BTF"
        offsets: Dict[str, Any] = {"_size_bytes": struct.size_bytes}
        missing: List[str] = []
        for name in ("name", "object_size", "size", "offset", "oo", "random"):
            member = btf.member("kmem_cache", name)
            if member is None:
                if name == "random":
                    continue  # CONFIG_SLAB_FREELIST_HARDENED off => no field
                missing.append(name)
                continue
            offsets[name] = member.byte_offset
        if missing:
            return None, f"struct kmem_cache missing fields: {', '.join(missing)}"
        return offsets, None

    @staticmethod
    def _resolve_slab_header(
        btf: BtfFile,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Resolve the slab header (struct slab, falling back to struct page).

        Returns (resolved, error).  ``resolved`` holds ``_struct`` (source
        name), ``_size_bytes``, ``freelist`` (required byte offset) and
        optional ``objects``/``inuse``/``frozen``/``cpu`` members.  Member
        lookups recurse through anonymous struct/union members, so the
        ``struct page`` fallback finds ``freelist`` in its nested anonymous
        unions on kernels without ``struct slab``.
        """
        for struct_name in (_SLAB_HEADER_STRUCT_SLAB, _SLAB_HEADER_STRUCT_PAGE):
            struct = btf.struct(struct_name)
            freelist = btf.member(struct_name, "freelist")
            if struct is None or freelist is None:
                continue
            resolved: Dict[str, Any] = {
                "_struct": f"struct {struct_name}",
                "_size_bytes": struct.size_bytes,
                "freelist": freelist.byte_offset,
            }
            for optional in ("objects", "inuse", "frozen", "cpu"):
                member = btf.member(struct_name, optional)
                if member is not None:
                    resolved[optional] = member
            return resolved, None
        return None, (
            "neither struct slab nor struct page exposes a freelist field " "in BTF"
        )

    # ------------------------------------------------------------------
    # Shared verdict helpers
    # ------------------------------------------------------------------
    def _verdict_response(
        self,
        request: str,
        schema: str,
        profile: Mapping[str, Any],
        component_status: Mapping[str, Any],
        *,
        verdict: str,
        reason: str,
        btf: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {
            "schema": schema,
            "supported": verdict in ("decoded", "unresolved"),
            "verdict": verdict,
            "reason": reason,
            "allocator": {"family": "linux-kernel", "subsystem": "slub"},
            "target_profile": profile,
            "component_status": component_status,
        }
        if btf is not None:
            parsed["btf"] = btf
        if extra:
            parsed.update(extra)
        return build_backend_response(
            request,
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    @staticmethod
    def _serialize_attempts(attempts: List[SlubChain]) -> List[Dict[str, Any]]:
        return [
            {
                "head_mode": chain.head_mode,
                "encoding": chain.encoding,
                "count": chain.count,
                "termination": chain.termination,
                "objects": chain.objects,
                "note": chain.note,
            }
            for chain in attempts
        ]

    # ------------------------------------------------------------------
    # inspect_slub_cache
    # ------------------------------------------------------------------
    def inspect_slub_cache(
        self,
        *,
        cache_address: str,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        byteorder: str = "little",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode one ``struct kmem_cache``: geometry + hardening state."""
        request = "inspect_slub_cache"
        if byteorder not in ("little", "big"):
            raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
        bo = cast(_Byteorder, byteorder)
        profile, component_status = self._prepare_context(cancellation_check)
        pointer_bits = int(profile.get("pointer_bits") or 64)

        cache_addr = parse_int(cache_address)
        if cache_addr is None:
            raise ValueError(f"invalid cache_address: {cache_address!r}")
        source = btf_path or _DEFAULT_BTF_SOURCE

        btf, btf_err = self._load_btf(btf_path)
        if btf is None:
            return self._verdict_response(
                request,
                SLUB_CACHE_SCHEMA,
                profile,
                component_status,
                verdict="btf_unavailable",
                reason=btf_err or "BTF unavailable",
                btf={"source": source, "error": btf_err},
            )

        offsets, resolve_err = self._resolve_cache_offsets(btf)
        if offsets is None:
            return self._verdict_response(
                request,
                SLUB_CACHE_SCHEMA,
                profile,
                component_status,
                verdict="struct_offsets_unresolved",
                reason=resolve_err or "kmem_cache offsets unresolved",
                btf=self._btf_info(btf, source),
                extra={"missing_fields": resolve_err},
            )

        cache_size = int(offsets["_size_bytes"])
        raw, read_err = self._read_bytes(cache_addr, cache_size)
        if read_err is not None:
            return self._verdict_response(
                request,
                SLUB_CACHE_SCHEMA,
                profile,
                component_status,
                verdict="unreadable",
                reason=f"cannot read struct kmem_cache at {hex(cache_addr)}: {read_err}",
                btf=self._btf_info(btf, source),
            )

        name_ptr = self._read_unsigned(raw, offsets["name"], pointer_bits // 8, bo)
        # kmem_cache scalars are u32; random (unsigned long) is pointer-sized.
        object_size = self._read_unsigned(raw, offsets["object_size"], 4, bo)
        size = self._read_unsigned(raw, offsets["size"], 4, bo)
        freepointer_offset = self._read_unsigned(raw, offsets["offset"], 4, bo)
        oo_x = self._read_unsigned(raw, offsets["oo"], 4, bo)
        cache_order = oo_x >> 16
        objects_per_slab = oo_x & 0xFFFF
        slab_bytes = page_size << cache_order
        random = (
            self._read_unsigned(raw, offsets["random"], pointer_bits // 8, bo)
            if "random" in offsets
            else 0
        )
        hardened = bool(random)

        cache_name = self._read_cstring(name_ptr)
        warnings: List[str] = []
        if cache_name is None:
            warnings.append(f"could not read cache name string at {hex(name_ptr)}")

        return build_backend_response(
            request,
            parsed={
                "schema": SLUB_CACHE_SCHEMA,
                "supported": True,
                "verdict": "decoded",
                "allocator": {"family": "linux-kernel", "subsystem": "slub"},
                "target_profile": profile,
                "component_status": component_status,
                "cache_address": cache_addr,
                "cache": {
                    "name": cache_name,
                    "object_size": object_size,
                    "size": size,
                    "freepointer_offset": freepointer_offset,
                    "random": random,
                    "hardened": hardened,
                    "order": cache_order,
                    "objects_per_slab": objects_per_slab,
                    "slab_bytes": slab_bytes,
                },
                "btf": {
                    **self._btf_info(btf, source),
                    "struct_size_bytes": cache_size,
                    "resolved_fields": sorted(
                        k for k in offsets if not k.startswith("_")
                    ),
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    # ------------------------------------------------------------------
    # decode_slub_freelist
    # ------------------------------------------------------------------
    def decode_slub_freelist(
        self,
        *,
        cache_address: str,
        slab_address: str,
        slab_mem_address: Optional[str] = None,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        max_objects: Optional[int] = None,
        byteorder: str = "little",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode one slab's obfuscated freelist chain."""
        request = "decode_slub_freelist"
        if byteorder not in ("little", "big"):
            raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
        bo = cast(_Byteorder, byteorder)
        profile, component_status = self._prepare_context(cancellation_check)
        pointer_bits = int(profile.get("pointer_bits") or 64)

        cache_addr = parse_int(cache_address)
        if cache_addr is None:
            raise ValueError(f"invalid cache_address: {cache_address!r}")
        slab_addr = parse_int(slab_address)
        if slab_addr is None:
            raise ValueError(f"invalid slab_address: {slab_address!r}")
        slab_mem = parse_int(slab_mem_address) if slab_mem_address else None
        source = btf_path or _DEFAULT_BTF_SOURCE

        btf, btf_err = self._load_btf(btf_path)
        if btf is None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="btf_unavailable",
                reason=btf_err or "BTF unavailable",
                btf={"source": source, "error": btf_err},
            )
        btf_info = self._btf_info(btf, source)

        cache_offsets, cache_err = self._resolve_cache_offsets(btf)
        if cache_offsets is None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="struct_offsets_unresolved",
                reason=cache_err or "kmem_cache offsets unresolved",
                btf=btf_info,
                extra={"missing_fields": cache_err},
            )
        slab_header, slab_err = self._resolve_slab_header(btf)
        if slab_header is None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="unsupported",
                reason=slab_err or "no slab header layout in BTF",
                btf=btf_info,
            )

        cache_size = int(cache_offsets["_size_bytes"])
        cache_raw, cache_read_err = self._read_bytes(cache_addr, cache_size)
        if cache_read_err is not None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="unreadable",
                reason=f"cannot read struct kmem_cache at {hex(cache_addr)}: {cache_read_err}",
                btf=btf_info,
            )
        slab_size = int(slab_header["_size_bytes"])
        slab_raw, slab_read_err = self._read_bytes(slab_addr, slab_size)
        if slab_read_err is not None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="unreadable",
                reason=f"cannot read {slab_header['_struct']} at {hex(slab_addr)}: {slab_read_err}",
                btf=btf_info,
            )

        # -- cache fields ---------------------------------------------
        freepointer_offset = self._read_unsigned(
            cache_raw, cache_offsets["offset"], 4, bo
        )
        oo_x = self._read_unsigned(cache_raw, cache_offsets["oo"], 4, bo)
        order = oo_x >> 16
        objects_per_slab = oo_x & 0xFFFF
        slab_bytes = page_size << order
        random = (
            self._read_unsigned(
                cache_raw, cache_offsets["random"], pointer_bits // 8, bo
            )
            if "random" in cache_offsets
            else 0
        )
        hardened = bool(random)

        # -- slab header fields ----------------------------------------
        freelist_offset = int(slab_header["freelist"])
        head_raw = int.from_bytes(
            slab_raw[freelist_offset : freelist_offset + pointer_bits // 8],
            bo,
        )
        head_store_addr = slab_addr + freelist_offset
        header_source = str(slab_header["_struct"])

        def slab_field(name: str) -> Optional[int]:
            member = slab_header.get(name)
            if member is None:
                return None
            try:
                return self._read_member(slab_raw, member, pointer_bits, bo)
            except ValueError:
                return None

        slab_objects = slab_field("objects")
        inuse = slab_field("inuse")
        frozen = slab_field("frozen")
        cpu = slab_field("cpu")

        # -- derive the slab window ------------------------------------
        warnings: List[str] = []
        if slab_mem is None:
            # Align to the slab's *full extent* (page_size << order), not one
            # page: a higher-order slab spans multiple pages and its struct
            # address is the base page, so masking against the order-aware
            # extent keeps the window base correct for order > 0 too.
            slab_mem = slab_addr & ~(slab_bytes - 1)
            warnings.append(
                f"slab_mem_address not given; derived slab window base "
                f"{hex(slab_mem)} from the slab address (order-aware, "
                f"slab_bytes={hex(slab_bytes)})"
            )

        # -- walk + detect the chain -----------------------------------
        cap_objects = (
            max_objects or slab_objects or objects_per_slab or DEFAULT_MAX_SLUB_OBJECTS
        )
        cap_objects = min(cap_objects, DEFAULT_MAX_SLUB_OBJECTS)
        chain, attempts = slub_detect_chain(
            head=head_raw,
            head_store_addr=head_store_addr,
            cache_offset=freepointer_offset,
            cache_random=random,
            hardened=hardened,
            slab_mem=slab_mem,
            slab_bytes=slab_bytes,
            slab_objects=objects_per_slab,
            max_objects=cap_objects,
            pointer_bits=pointer_bits,
            reader=self._reader(pointer_bits, bo),
        )

        if chain is None:
            return self._verdict_response(
                request,
                SLUB_FREELIST_SCHEMA,
                profile,
                component_status,
                verdict="unresolved",
                reason=(
                    "no head/encoding interpretation decoded an object inside "
                    f"the slab window [{hex(slab_mem)}, {hex(slab_mem + slab_bytes)})"
                ),
                btf=btf_info,
                extra={
                    "slab": self._slab_summary(
                        header_source,
                        head_raw,
                        inuse,
                        slab_objects,
                        frozen,
                        cpu,
                        order,
                        objects_per_slab,
                        slab_bytes,
                    ),
                    "freelist": {
                        "hardened": hardened,
                        "head": head_raw,
                        "count": 0,
                        "object_addresses": [],
                        "chain": [],
                        "termination": None,
                        "note": None,
                    },
                    "attempts": self._serialize_attempts(attempts),
                },
            )

        return build_backend_response(
            request,
            parsed={
                "schema": SLUB_FREELIST_SCHEMA,
                "supported": True,
                "verdict": "decoded",
                "allocator": {"family": "linux-kernel", "subsystem": "slub"},
                "target_profile": profile,
                "component_status": component_status,
                "cache_address": cache_addr,
                "slab_address": slab_addr,
                "slab": self._slab_summary(
                    header_source,
                    head_raw,
                    inuse,
                    slab_objects,
                    frozen,
                    cpu,
                    order,
                    objects_per_slab,
                    slab_bytes,
                ),
                "freelist": {
                    "hardened": hardened,
                    "encoding": chain.encoding,
                    "head_mode": chain.head_mode,
                    "head": head_raw,
                    "count": chain.count,
                    "object_addresses": chain.objects,
                    "chain": [
                        {
                            "object_address": entry.object_address,
                            "freepointer_address": entry.freepointer_address,
                            "encoded_next": entry.encoded_next,
                            "decoded_next": entry.decoded_next,
                        }
                        for entry in chain.chain
                    ],
                    "termination": chain.termination,
                    "note": chain.note,
                },
                "attempts": self._serialize_attempts(attempts),
                "btf": {
                    **btf_info,
                    "cache_struct_size_bytes": cache_size,
                    "slab_header_struct": header_source,
                    "slab_header_size_bytes": slab_size,
                    "resolved_cache_fields": sorted(
                        k for k in cache_offsets if not k.startswith("_")
                    ),
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    @staticmethod
    def _slab_summary(
        header_source: str,
        head_raw: int,
        inuse: Optional[int],
        objects: Optional[int],
        frozen: Optional[int],
        cpu: Optional[int],
        order: int,
        objects_per_slab: int,
        slab_bytes: int,
    ) -> Dict[str, Any]:
        return {
            "header_source": header_source,
            "freelist_raw": head_raw,
            "inuse": inuse,
            "objects": objects,
            "frozen": frozen,
            "cpu": cpu,
            "order": order,
            "objects_per_slab": objects_per_slab,
            "slab_bytes": slab_bytes,
        }


__all__ = [
    "SLUB_CACHE_SCHEMA",
    "SLUB_FREELIST_SCHEMA",
    "SLUB_ENCODING_SWAB",
    "SLUB_ENCODING_LEGACY",
    "SLUB_HEAD_RAW",
    "SLUB_HEAD_ENCODED",
    "DEFAULT_MAX_SLUB_OBJECTS",
    "KernelSlab",
    "SlubChain",
    "SlubFreeObject",
    "slub_decode_ptr",
    "slub_detect_chain",
    "slub_swab",
    "slub_walk_chain",
]
