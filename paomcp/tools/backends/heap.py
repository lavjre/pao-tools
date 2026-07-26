"""Typed heap allocator discovery (Stage 6.1).

Stage 6 gives an agent a typed understanding of the target's heap without
forcing it to parse pwndbg's ANSI terminal output.  S6.1 is the foundation of
that work: before any chunk, bin, or tcache can be parsed, the platform must
answer three questions honestly and conservatively:

* Which allocator is this process actually using (glibc ptmalloc, musl
  mallocng, jemalloc, or something unknown)?
* What is the concrete runtime profile the later parsers must obey -- pointer
  width, byte order, and architecture?
* Which inspection capabilities are genuinely available right now, and which
  are merely planned or unsupported?

Every conclusion carries the raw evidence it was derived from, and a heuristic
is never promoted to a fact.  Detection needs a *running* inferior because a
heap only exists at runtime, so the live-inferior invariant from Stage 5 is
reused here rather than re-implemented.

The heavy lifting lives in :meth:`AllocatorDiscovery.classify`, a pure function
over already-collected mapping/libc evidence.  That keeps the interesting logic
unit-testable without a debugger, matching the design of
:mod:`paomcp.tools.backends.foundation`.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

PARSER_VERSION = "1"
ALLOCATOR_DETECTION_SCHEMA = "pao.allocator-detection.v1"
ALLOCATOR_CAPABILITIES_SCHEMA = "pao.allocator-capabilities.v1"

# glibc adopted tcache/fastbin pointer "safe-linking" in 2.32.  Detection of the
# libc version therefore decides whether a later S6.4 parser must decode masked
# forward pointers.  Below this version the fd pointers are stored raw.
SAFE_LINKING_MIN_VERSION: Tuple[int, int] = (2, 32)

# One allocator family maps to the set of pwndbg runtime commands that, when
# present, let later stages read its internal structures.  This mirrors the
# static readiness probe in ``paomcp.tools.backends.platform`` but is evaluated
# against the *detected* allocator instead of every family unconditionally.
ALLOCATOR_RUNTIME_COMMANDS: Dict[str, Tuple[str, ...]] = {
    "glibc": ("arena", "arenas", "bins", "heap", "malloc-chunk", "tcachebins"),
    "musl": ("mallocng",),
    "jemalloc": ("jemalloc",),
    "scudo": (),
    "unknown": (),
}

# Filename evidence.  Kept deliberately narrow: a match must look like a real
# shared object, not merely mention the word "libc" somewhere in a path.
_FAMILY_FILENAME_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "musl": (
        re.compile(r"(?:^|/)ld-musl-[^/]*\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libc\.musl-[^/]*\.so", re.IGNORECASE),
    ),
    "jemalloc": (re.compile(r"(?:^|/)libjemalloc\.so", re.IGNORECASE),),
    "glibc": (
        re.compile(r"(?:^|/)libc\.so\.6(?:$|\.)", re.IGNORECASE),
        re.compile(r"(?:^|/)libc-2\.\d+(?:\.\d+)?\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libc\.so(?:$|\.)", re.IGNORECASE),
    ),
}

_FILENAME_VERSION_PATTERN = re.compile(
    r"libc-(\d+)\.(\d+)(?:\.\d+)?\.so", re.IGNORECASE
)
_TEXT_VERSION_PATTERN = re.compile(r"\b(\d+)\.(\d+)(?:\.\d+)?\b")


class AllocatorDiscovery:
    """Conservatively identify the target allocator and its runtime profile.

    The class talks to GDB only through the same typed inspection/debugger
    backends the rest of the platform uses; it never issues raw MI itself.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        binary_path_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self._binary_path_provider = binary_path_provider

    # ------------------------------------------------------------------
    # Pure classification core (no debugger required)
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_mapping_family(file_name: Optional[str]) -> Optional[str]:
        """Return the allocator family a mapped object name implies, if any."""
        if not file_name:
            return None
        for family, patterns in _FAMILY_FILENAME_PATTERNS.items():
            if any(pattern.search(file_name) for pattern in patterns):
                return family
        return None

    @staticmethod
    def _version_tuple(text: Optional[str]) -> Optional[Tuple[int, int]]:
        """Parse a ``major.minor`` glibc version from arbitrary text."""
        if not text:
            return None
        match = _TEXT_VERSION_PATTERN.search(str(text))
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _version_from_filename(
        cls, file_name: Optional[str]
    ) -> Optional[Tuple[int, int]]:
        if not file_name:
            return None
        match = _FILENAME_VERSION_PATTERN.search(file_name)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _format_version(version: Optional[Tuple[int, int]]) -> Optional[str]:
        return None if version is None else f"{version[0]}.{version[1]}"

    @classmethod
    def _safe_linking_status(
        cls, family: str, version: Optional[Tuple[int, int]]
    ) -> str:
        """Return safe-linking applicability: enabled / disabled / unknown / n/a."""
        if family != "glibc":
            # Only glibc's tcache/fastbin safe-linking is modeled here.  Other
            # allocators are reported as not-applicable rather than guessed.
            return "not_applicable"
        if version is None:
            return "unknown"
        return "enabled" if version >= SAFE_LINKING_MIN_VERSION else "disabled"

    @classmethod
    def classify(
        cls,
        mappings: Sequence[Mapping[str, Any]],
        libcinfo: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Classify the allocator family from mapping and libcinfo evidence.

        This is intentionally a pure function so it can be exhaustively tested
        without a live debugger.  ``mappings`` are vmmap entries (each a dict
        with at least ``file``) and ``libcinfo`` is the parsed ``libcinfo``
        payload (may be ``None`` when unavailable).

        Returns a dict describing the detected ``family``, its ``version`` (when
        known), a ``confidence`` label, the concrete ``evidence`` list, and the
        ``safe_linking`` applicability.  Confidence is never higher than the
        evidence supports:

        * ``confirmed``  -- a matching mapping *and* a resolved version
        * ``likely``     -- a matching mapping with no version
        * ``ambiguous``  -- more than one distinct family matched
        * ``unknown``    -- no allocator-bearing mapping was found
        """
        evidence: List[Dict[str, Any]] = []
        families_seen: List[str] = []
        for mapping in mappings:
            if not isinstance(mapping, Mapping):
                continue
            file_name = mapping.get("file")
            family = cls._classify_mapping_family(
                str(file_name) if file_name is not None else None
            )
            if family is None:
                continue
            if family not in families_seen:
                families_seen.append(family)
            evidence.append(
                {
                    "source": "mapping",
                    "family": family,
                    "file": file_name,
                    "start": mapping.get("start"),
                    "end": mapping.get("end"),
                    "permissions": mapping.get("permissions"),
                }
            )

        libc_version_text: Optional[str] = None
        libc_object: Optional[str] = None
        if isinstance(libcinfo, Mapping):
            libc_version_text = (
                str(libcinfo.get("version"))
                if libcinfo.get("version") is not None
                else None
            )
            libc_object = (
                str(libcinfo.get("libc")) if libcinfo.get("libc") is not None else None
            )

        # Determine the primary family.  A libcinfo-reported glibc object is
        # corroborating evidence but never overrides an explicit musl/jemalloc
        # mapping match.
        distinct = list(dict.fromkeys(families_seen))
        if not distinct and libc_object and cls._classify_mapping_family(libc_object):
            corroborated = cls._classify_mapping_family(libc_object)
            if corroborated is not None:
                distinct = [corroborated]
                evidence.append(
                    {
                        "source": "libcinfo",
                        "family": corroborated,
                        "file": libc_object,
                    }
                )

        if not distinct:
            return {
                "family": "unknown",
                "version": None,
                "confidence": "unknown",
                "safe_linking": "unknown",
                "candidate_families": [],
                "evidence": evidence,
                "libc": {"object": libc_object, "reported_version": libc_version_text},
            }

        # Prefer a non-glibc match when families conflict, because a musl or
        # jemalloc object is a stronger, more specific signal than a generic
        # libc.so pattern that some musl builds also satisfy.
        primary = distinct[0]
        for candidate in distinct:
            if candidate != "glibc":
                primary = candidate
                break

        version = cls._version_tuple(libc_version_text)
        version_source: Optional[str] = "libcinfo" if version is not None else None
        if version is None:
            for item in evidence:
                filename_version = cls._version_from_filename(
                    str(item.get("file")) if item.get("file") is not None else None
                )
                if filename_version is not None:
                    version = filename_version
                    version_source = "mapping_filename"
                    break

        ambiguous = len(distinct) > 1
        if ambiguous:
            confidence = "ambiguous"
        elif version is not None:
            confidence = "confirmed"
        else:
            confidence = "likely"

        return {
            "family": primary,
            "version": cls._format_version(version),
            "version_source": version_source,
            "confidence": confidence,
            "safe_linking": cls._safe_linking_status(primary, version),
            "candidate_families": distinct,
            "evidence": evidence,
            "libc": {"object": libc_object, "reported_version": libc_version_text},
        }

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live_inferior(self) -> Dict[str, Any]:
        """Return the current live stop record or raise a typed error.

        A heap exists only while a process is running, so allocator discovery is
        meaningless against a dead or never-started inferior.
        """
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Allocator discovery requires a live, stopped inferior; run the "
                "target to a stop before inspecting the heap"
            ),
        )

    @staticmethod
    def _endianness_from_show(raw_output: str) -> Optional[str]:
        text = raw_output.lower()
        if "big endian" in text:
            return "big"
        if "little endian" in text:
            return "little"
        return None

    @staticmethod
    def _pointer_bits_from_mappings(
        mappings: Sequence[Mapping[str, Any]],
    ) -> Optional[int]:
        """Infer pointer width from the highest observed mapping address."""
        widest: Optional[int] = None
        for mapping in mappings:
            if not isinstance(mapping, Mapping):
                continue
            end = parse_int(mapping.get("end"))
            if end is None:
                continue
            widest = end if widest is None else max(widest, end)
        if widest is None:
            return None
        return 64 if widest > 0xFFFFFFFF else 32

    def target_profile(self, mappings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Resolve pointer width, byte order, and architecture with provenance.

        Each field records the source it was derived from so a caller can tell a
        runtime-verified value apart from an inferred fallback.  Nothing here is
        guessed silently: an unresolved field is reported as ``None`` with an
        explicit ``unknown`` source.
        """
        pointer_bits: Optional[int] = None
        pointer_source = "unknown"
        sizeof = self.debugger.evaluate_expression("sizeof(void*)")
        if sizeof.get("success"):
            size_value = parse_int(sizeof.get("parsed", {}).get("value"))
            if size_value in (4, 8):
                pointer_bits = size_value * 8
                pointer_source = "runtime_sizeof"
        if pointer_bits is None:
            inferred = self._pointer_bits_from_mappings(mappings)
            if inferred is not None:
                pointer_bits = inferred
                pointer_source = "mapping_width"

        byteorder: Optional[str] = None
        byteorder_source = "unknown"
        endian_result = self.inspection.gdb.execute_command(
            "show endian", timeout_sec=self.inspection.timeout_seconds
        )
        endian_text = PwndbgInspection._raw_output(endian_result)
        detected_endian = self._endianness_from_show(endian_text)
        if detected_endian is not None:
            byteorder = detected_endian
            byteorder_source = "gdb_show_endian"

        arch: Optional[str] = None
        arch_source = "unknown"
        binary_path = (
            self._binary_path_provider() if self._binary_path_provider else None
        )
        if binary_path:
            checksec = self.inspection.checksec(binary_path)
            if checksec.get("success"):
                arch = checksec.get("parsed", {}).get("arch")
                if arch:
                    arch_source = "checksec"

        return {
            "pointer_bits": pointer_bits,
            "pointer_bits_source": pointer_source,
            "byteorder": byteorder,
            "byteorder_source": byteorder_source,
            "architecture": arch,
            "architecture_source": arch_source,
        }

    def collect_evidence(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
        """Gather vmmap + libcinfo evidence, tolerating a missing libcinfo."""
        if cancellation_check is not None:
            cancellation_check()
        vmmap = self.inspection.memory_maps()
        mappings_raw = vmmap.get("parsed", {}).get("mappings", [])
        mappings = [item for item in mappings_raw if isinstance(item, dict)]

        if cancellation_check is not None:
            cancellation_check()
        libcinfo_result = self.inspection.libc_info()
        libcinfo_parsed: Optional[Dict[str, Any]] = None
        if libcinfo_result.get("success"):
            parsed = libcinfo_result.get("parsed")
            if isinstance(parsed, dict):
                libcinfo_parsed = parsed

        component_status = {
            "mappings": bool(vmmap.get("success")),
            "libcinfo": bool(libcinfo_result.get("success")),
        }
        return mappings, libcinfo_parsed, component_status

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def detect_allocator(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Detect the allocator family, version, and target runtime profile."""
        self._require_live_inferior()
        mappings, libcinfo_parsed, component_status = self.collect_evidence(
            cancellation_check
        )
        classification = self.classify(mappings, libcinfo_parsed)
        profile = self.target_profile(mappings)

        warnings: List[str] = []
        if not component_status["mappings"]:
            warnings.append(
                "Memory mappings were unavailable; allocator detection used "
                "only libc metadata and is less reliable"
            )
        if classification["family"] == "unknown":
            warnings.append(
                "No known allocator mapping was found; the target may be "
                "statically linked or use a custom allocator"
            )
        if classification["confidence"] == "ambiguous":
            warnings.append(
                "Multiple allocator families matched; review candidate_families "
                "and evidence before trusting the primary family"
            )

        parsed = {
            "schema": ALLOCATOR_DETECTION_SCHEMA,
            "family": classification["family"],
            "version": classification["version"],
            "version_source": classification.get("version_source"),
            "confidence": classification["confidence"],
            "safe_linking": classification["safe_linking"],
            "candidate_families": classification["candidate_families"],
            "target_profile": profile,
            "libc": classification["libc"],
            "evidence": classification["evidence"],
            "component_status": component_status,
        }
        return build_backend_response(
            "detect_allocator",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    @staticmethod
    def _capability_matrix(
        detected_family: str,
        version: Optional[str],
        safe_linking: str,
        command_names: Sequence[str],
    ) -> Dict[str, Any]:
        """Build the per-family supported/unsupported/unknown capability view."""
        available_commands = set(command_names)
        families: Dict[str, Any] = {}
        for family, expected in ALLOCATOR_RUNTIME_COMMANDS.items():
            present = [name for name in expected if name in available_commands]
            missing = [name for name in expected if name not in available_commands]
            detected = family == detected_family
            if expected:
                runtime_status = (
                    "supported"
                    if present and not missing
                    else "partial" if present else "unsupported"
                )
            else:
                # No pwndbg primitives are known for this family, so the runtime
                # cannot be probed -- report unknown rather than a false negative.
                runtime_status = "unknown"
            families[family] = {
                "detected": detected,
                "runtime_commands": {
                    "available": present,
                    "missing": missing,
                    "status": runtime_status,
                },
                # No typed heap parser ships in S6.1; later stages flip this on
                # per family.  It is reported honestly as planned/unavailable.
                "typed_inspection_api": {
                    "status": "planned" if family in {"glibc"} else "unavailable",
                    "available": False,
                },
            }
        return families

    def get_allocator_capabilities(
        self,
        *,
        debugger_command_names: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Report allocator inspection capabilities for the live target.

        ``debugger_command_names`` are the pwndbg command names known to the
        running debugger (supplied by the tool layer from the cached platform
        compatibility report).  When omitted, runtime-command availability is
        reported as ``unknown`` rather than assumed absent.
        """
        self._require_live_inferior()
        mappings, libcinfo_parsed, component_status = self.collect_evidence(
            cancellation_check
        )
        classification = self.classify(mappings, libcinfo_parsed)
        profile = self.target_profile(mappings)

        command_names = list(debugger_command_names or [])
        matrix = self._capability_matrix(
            classification["family"],
            classification["version"],
            classification["safe_linking"],
            command_names,
        )

        detected = classification["family"]
        detected_caps = matrix.get(detected, {})
        runtime_status = (
            detected_caps.get("runtime_commands", {}).get("status", "unknown")
            if detected != "unknown"
            else "unsupported"
        )
        # Overall verdict for the detected allocator: the discovery capability
        # itself is always supported once an inferior is live; the value here is
        # the honest state of *inspection* readiness for later stages.
        if detected == "unknown":
            overall = "unsupported"
        elif runtime_status in {"supported", "partial"}:
            overall = "supported" if runtime_status == "supported" else "partial"
        elif not command_names:
            overall = "unknown"
        else:
            overall = "unsupported"

        warnings: List[str] = []
        if not command_names:
            warnings.append(
                "No debugger command inventory was provided; runtime allocator "
                "primitive availability is reported as unknown"
            )

        parsed = {
            "schema": ALLOCATOR_CAPABILITIES_SCHEMA,
            "detected_family": detected,
            "detected_version": classification["version"],
            "detection_confidence": classification["confidence"],
            "safe_linking": classification["safe_linking"],
            "overall_status": overall,
            "discovery": {"status": "supported"},
            "target_profile": profile,
            "allocators": matrix,
            "command_inventory_size": len(command_names),
            "component_status": component_status,
        }
        return build_backend_response(
            "get_allocator_capabilities",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )


# ----------------------------------------------------------------------
# Stage 6.2 -- glibc malloc_chunk parsing
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


# ----------------------------------------------------------------------
# Stage 6.3 -- glibc arenas and malloc_state parsing
# ----------------------------------------------------------------------

# Compile-time glibc constants (stable since glibc 2.14).  The parser reads
# them from source rather than probing gdb because they participate in the
# expected malloc_state layout: querying them wouldn't tell us anything the
# per-field offset lookups already do.
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

# ``(name, kind, count)`` for each malloc_state field parsed by S6.3.
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


class ArenaLayoutResolver:
    """Resolve main_arena's address and malloc_state field offsets via GDB.

    Uses GDB's ``-data-evaluate-expression`` to interrogate the running
    inferior for symbol addresses and struct field offsets.  The queries
    require libc built with debug symbols (``libc6-dbg`` on Debian /
    ``glibc-debuginfo`` on Fedora); with a stripped libc the resolver
    reports each unresolved field explicitly rather than guessing.  A
    version-keyed offset table for stripped libcs is out of scope for
    S6.3 and would live in a later stage.
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

    # ------------------------------------------------------------------
    # Pure parser core (no debugger required)
    # ------------------------------------------------------------------
    @staticmethod
    def parse_arena_struct(
        raw_bytes: bytes,
        offsets: Mapping[str, Optional[int]],
        pointer_bits: int,
        byteorder: Optional[str],
    ) -> Dict[str, Any]:
        """Decode a malloc_state from raw bytes plus resolved field offsets.

        Fields whose offset is ``None`` (unresolved) or whose byte slice
        falls outside ``raw_bytes`` are reported as ``None`` and listed
        under ``unreadable_fields``.  The caller can distinguish a truly
        zero value from an unreadable field via this list.
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

        for name, kind, count in _MALLOC_STATE_FIELDS:
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

        layout = self.resolver.resolve()
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

        layout = self.resolver.resolve()
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
            raw, layout["offsets"], pointer_bits, byteorder
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


class HeapTools:
    """Facade that groups every heap-inspection component under one object.

    Later stages (S6.3 arenas/bins, S6.6 musl/jemalloc parsers) attach here so
    the tool layer keeps a single ``session.tools.heap`` entry point regardless
    of the detected allocator family.  Delegation is intentionally boring:
    each public method routes to the appropriate sub-component untouched.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        binary_path_provider: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        self.discovery = AllocatorDiscovery(
            gdb,
            inspection,
            debugger,
            session_id,
            binary_path_provider=binary_path_provider,
        )
        self.glibc = GlibcHeap(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )
        self.arenas = GlibcArenas(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )

    # ---- S6.1 allocator discovery ------------------------------------
    def detect_allocator(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.discovery.detect_allocator(cancellation_check=cancellation_check)

    def get_allocator_capabilities(
        self,
        *,
        debugger_command_names: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.discovery.get_allocator_capabilities(
            debugger_command_names=debugger_command_names,
            cancellation_check=cancellation_check,
        )

    # ---- S6.2 glibc chunks -------------------------------------------
    def inspect_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        include_next: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.glibc.inspect_chunk(
            address=address,
            address_kind=address_kind,
            include_next=include_next,
            cancellation_check=cancellation_check,
        )

    def validate_chunk(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.glibc.validate_chunk(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    # ---- S6.3 glibc arenas -------------------------------------------
    def list_arenas(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.arenas.list_arenas(cancellation_check=cancellation_check)

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        return self.arenas.inspect_arena(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )


__all__ = [
    "AllocatorDiscovery",
    "GlibcHeap",
    "GlibcArenas",
    "ArenaLayoutResolver",
    "HeapTools",
    "ALLOCATOR_DETECTION_SCHEMA",
    "ALLOCATOR_CAPABILITIES_SCHEMA",
    "CHUNK_INSPECTION_SCHEMA",
    "CHUNK_VALIDATION_SCHEMA",
    "ARENA_LIST_SCHEMA",
    "ARENA_INSPECTION_SCHEMA",
    "SAFE_LINKING_MIN_VERSION",
    "ALLOCATOR_RUNTIME_COMMANDS",
    "PREV_INUSE",
    "IS_MMAPPED",
    "NON_MAIN_ARENA",
    "SIZE_BITS_MASK",
    "NFASTBINS",
    "NBINS",
    "BINS_ARRAY_LEN",
    "BINMAPSIZE",
    "MAX_ARENAS_TRAVERSAL",
]
