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

PARSER_VERSION = "1"
ALLOCATOR_DETECTION_SCHEMA = "pao.allocator-detection.v1"
ALLOCATOR_CAPABILITIES_SCHEMA = "pao.allocator-capabilities.v1"

# glibc adopted tcache/fastbin pointer "safe-linking" in 2.32.  Detection of the
# libc version therefore decides whether a later S6.4 parser must decode masked
# forward pointers.  Below this version the fd pointers are stored raw.
SAFE_LINKING_MIN_VERSION: Tuple[int, int] = (2, 32)


def encode_safe_linked(plain: int, position: int) -> int:
    """glibc ``PROTECT_PTR`` encode: encoded = plain XOR (position >> 12).

    Inverse of :meth:`GlibcBins.decode_safe_linked`.  ``position`` is the
    address of the fd/next field itself (i.e. where the encoded pointer is
    *stored*), not the address of the chunk that holds it -- match the
    macro's ``pos`` argument exactly.
    """
    return plain ^ (position >> 12)


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

# The S6.2-S6.5 typed parsers operate on glibc heap structures directly
# through GDB memory/expression APIs.  Runtime pwndbg commands remain useful
# evidence, but their absence does not make the shipped typed API disappear.
TYPED_INSPECTION_FAMILIES = frozenset({"glibc"})

# Filename evidence.  Kept deliberately narrow: a match must look like a real
# shared object, not merely mention the word "libc" somewhere in a path.
_FAMILY_FILENAME_PATTERNS: Dict[str, Tuple[re.Pattern[str], ...]] = {
    "musl": (
        re.compile(r"(?:^|/)ld-musl-[^/]*\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libc\.musl-[^/]*\.so", re.IGNORECASE),
    ),
    "jemalloc": (
        re.compile(r"(?:^|/)libjemalloc\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libjemalloc-[^/]*\.so", re.IGNORECASE),
    ),
    "scudo": (
        # Standalone Scudo runtime (llvm compiler-rt) and the Android/Fuchsia
        # ``libc.so`` that folds Scudo in are the two shipping shapes.  A bare
        # ``libc.so`` alone is not a scudo signal -- the specific
        # ``libclang_rt.scudo*`` and Android's ``bionic/*/libc.so`` are.
        re.compile(r"(?:^|/)libclang_rt\.scudo[^/]*\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libscudo[^/]*\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)bionic/[^/]*/libc\.so", re.IGNORECASE),
    ),
    "glibc": (
        re.compile(r"(?:^|/)libc\.so\.6(?:$|\.)", re.IGNORECASE),
        re.compile(r"(?:^|/)libc-2\.\d+(?:\.\d+)?\.so", re.IGNORECASE),
        re.compile(r"(?:^|/)libc\.so(?:$|\.)", re.IGNORECASE),
    ),
}

# musl swapped from an oldmalloc allocator to mallocng in 1.2.0.  Only the
# variant name is version-gated here; safe-linking remains a glibc-only
# concept and is reported as ``not_applicable`` for every other family.
MUSL_MALLOCNG_MIN_VERSION: Tuple[int, int] = (1, 2)

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
    def _allocator_variant(
        cls, family: str, version: Optional[Tuple[int, int]]
    ) -> Optional[str]:
        """Return the concrete allocator variant when the family has more than one.

        * ``glibc``  -> always ``"ptmalloc2"``
        * ``musl``   -> ``"mallocng"`` (>= 1.2), ``"oldmalloc"`` (< 1.2), or
                        ``None`` when the version is unresolved
        * ``jemalloc`` / ``scudo`` / ``unknown`` -> ``None`` (single variant
          today, or the variant is unknowable from mapping evidence alone)
        """
        if family == "glibc":
            return "ptmalloc2"
        if family == "musl":
            if version is None:
                return None
            return "mallocng" if version >= MUSL_MALLOCNG_MIN_VERSION else "oldmalloc"
        return None

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

        Recognized families: ``glibc``, ``musl``, ``jemalloc``, ``scudo`` and
        the sentinel ``unknown``.  The returned ``variant`` disambiguates
        allocators that ship more than one implementation over their
        lifetime (``ptmalloc2`` for glibc, ``mallocng`` vs ``oldmalloc``
        for musl >= 1.2 / < 1.2); other families report ``None``.

        Confidence is never higher than the evidence supports:

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
                "variant": None,
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
            "variant": cls._allocator_variant(primary, version),
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
            typed_available = family in TYPED_INSPECTION_FAMILIES
            families[family] = {
                "detected": detected,
                "runtime_commands": {
                    "available": present,
                    "missing": missing,
                    "status": runtime_status,
                },
                "typed_inspection_api": {
                    "status": "supported" if typed_available else "unavailable",
                    "available": typed_available,
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
        typed_available = bool(
            detected_caps.get("typed_inspection_api", {}).get("available")
        )
        # Overall verdict for the detected allocator: the discovery capability
        # itself is always supported once an inferior is live.  A shipped typed
        # parser is sufficient for full inspection support; pwndbg primitives
        # are reported independently and provide a partial fallback for families
        # that do not have a typed parser.
        if detected == "unknown":
            overall = "unsupported"
        elif typed_available:
            overall = "supported"
        elif runtime_status in {"supported", "partial"}:
            overall = "partial"
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


# ----------------------------------------------------------------------
# Stage 6.3b -- glibc fastbin traversal
# ----------------------------------------------------------------------

FASTBIN_INSPECTION_SCHEMA = "pao.glibc-fastbins.v1"

# Default per-fastbin chain length cap.  Real fastbins hold ~7 chunks at
# most in normal operation but corruption can create arbitrarily long
# chains, so a bound is necessary to keep the tool safe against a
# malicious/broken heap.  Sized generously so real workloads never hit it.
DEFAULT_MAX_CHUNKS_PER_FASTBIN = 128

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
DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN = 64

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
DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN = 1024

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


# ----------------------------------------------------------------------
# Stage 6.4 -- glibc heap corruption classifier
# ----------------------------------------------------------------------
HEAP_CORRUPTION_SCHEMA = "pao.glibc-heap-corruption.v1"
FREE_VALIDATION_SCHEMA = "pao.glibc-free-validation.v1"


class GlibcHeapCorruption:
    """Classify glibc heap corruption from live bin state.

    Runs the typed bin inspectors (fastbins, tcache, regular bins) and
    synthesizes their per-chain signals -- ``cycle_detected``, ``in_mapping``,
    ``chunk_readable``, and fd/bk pairs -- into a bounded, evidence-tagged list
    of corruption findings.  Each finding carries its category, severity, and
    the raw location that triggered it; the arena-level ``verdict`` is a
    conservative aggregate of the per-finding severities.  A heuristic is
    never promoted to a fact: everything reported has been observed in live
    memory this call.

    Categories (from the ``s6begin`` vocabulary):

    * ``cycle``                   -- a bin's fd chain loops
    * ``duplicate_entry``         -- one chunk address appears in >= 2 bin chains
    * ``fd_bk_inconsistency``     -- ``Y.bk != X`` in a regular-bin (X, Y) pair
    * ``alignment``               -- chunk address not aligned to ``2 * size_sz``
    * ``pointer_out_of_mapping``  -- chunk address lies outside every mapping
    * ``unreadable_chunk``        -- chunk header could not be read
    * ``unreadable_bin``          -- bin head read failed

    Severities:

    * ``corrupted``   -- cycle, duplicate, fd/bk, alignment, out_of_mapping
    * ``unreadable``  -- read failures (finding-level; conservative aggregate)

    Arena-level verdicts:

    * ``valid``        -- no findings
    * ``suspicious``   -- only unreadable findings
    * ``corrupted``    -- any corrupted-severity finding
    * ``unsupported``  -- target is not glibc
    * ``unresolvable`` -- bin layout could not be resolved via debug symbols
    """

    _CHECKS_RUN: Tuple[str, ...] = (
        "bin_cycle_detection",
        "cross_bin_duplicate_detection",
        "chunk_alignment",
        "chunk_in_mapping",
        "chunk_readable",
        "regular_bin_fd_bk_consistency",
    )

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        discovery: "AllocatorDiscovery",
        arenas: "GlibcArenas",
        bins: "GlibcBins",
        glibc: "GlibcHeap",
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas
        self.bins = bins
        self.glibc = glibc

    # ------------------------------------------------------------------
    # Live-target helpers (mirror GlibcBins pattern)
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Corruption classification requires a live, stopped inferior; "
                "run the target to a stop before inspecting the heap"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = AllocatorDiscovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _finding(
        *,
        category: str,
        severity: str,
        location: Dict[str, Any],
        reason: str,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        finding: Dict[str, Any] = {
            "category": category,
            "severity": severity,
            "location": location,
            "reason": reason,
        }
        if evidence is not None:
            finding["evidence"] = evidence
        return finding

    @staticmethod
    def _aggregate_verdict(findings: Sequence[Mapping[str, Any]]) -> str:
        if not findings:
            return "valid"
        severities = {str(f.get("severity")) for f in findings}
        if "corrupted" in severities:
            return "corrupted"
        # Read failures alone do not prove corruption -- surface as suspicious
        # so the agent knows to investigate rather than treating the heap as
        # invalid on the strength of a memory-read miss.
        return "suspicious"

    @staticmethod
    def _allocator_summary(classification: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

    # ------------------------------------------------------------------
    # classify_heap_corruption
    # ------------------------------------------------------------------
    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Classify corruption across an arena's fastbins, tcache and regular bins.

        Delegates to the typed bin inspectors so cycle detection, safe-linking
        decode, and per-chain traversal caps are reused.  The tcache is
        thread-local (not arena-scoped) so its findings always reflect the
        current thread regardless of ``arena_address``.
        """
        classification, _mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        allocator_out = self._allocator_summary(classification)

        if classification["family"] != "glibc":
            return build_backend_response(
                "classify_heap_corruption",
                parsed={
                    "schema": HEAP_CORRUPTION_SCHEMA,
                    "supported": False,
                    "verdict": "unsupported",
                    "reason": (
                        "heap corruption classification supports glibc only; "
                        f"detected family={classification['family']}"
                    ),
                    "arena_address": None,
                    "is_main_arena": None,
                    "allocator": allocator_out,
                    "target_profile": profile,
                    "findings": [],
                    "chunks_inspected": 0,
                    "bins_scanned": {"fastbins": 0, "tcache": 0, "regular": 0},
                    "checks_run": [],
                    "sources": {
                        "fastbins_verdict": None,
                        "tcache_verdict": None,
                        "bins_verdict": None,
                    },
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        pointer_bits = profile.get("pointer_bits")
        if pointer_bits not in (32, 64):
            raise RuntimeError(
                "Cannot classify corruption without a resolved pointer width; "
                f"target_profile returned pointer_bits={pointer_bits!r}"
            )
        alignment = 2 * (pointer_bits // 8)

        fastbins_response = self.bins.inspect_fastbins(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )
        tcache_response = self.bins.inspect_tcache(
            cancellation_check=cancellation_check,
        )
        bins_response = self.bins.inspect_bins(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

        fb_parsed = fastbins_response.get("parsed", {}) or {}
        tc_parsed = tcache_response.get("parsed", {}) or {}
        bn_parsed = bins_response.get("parsed", {}) or {}

        warnings: List[str] = []
        for source, response in (
            ("fastbins", fastbins_response),
            ("tcache", tcache_response),
            ("bins", bins_response),
        ):
            for entry in response.get("warnings") or []:
                warnings.append(f"{source}: {entry}")

        arena_addr_out: Optional[int] = fb_parsed.get("arena_address") or bn_parsed.get(
            "arena_address"
        )
        is_main_arena_out: Optional[bool]
        if fb_parsed.get("is_main_arena") is not None:
            is_main_arena_out = fb_parsed.get("is_main_arena")
        else:
            is_main_arena_out = bn_parsed.get("is_main_arena")
        sources = {
            "fastbins_verdict": fb_parsed.get("verdict"),
            "tcache_verdict": tc_parsed.get("verdict"),
            "bins_verdict": bn_parsed.get("verdict"),
        }

        if "unresolvable" in {sources["fastbins_verdict"], sources["bins_verdict"]}:
            return build_backend_response(
                "classify_heap_corruption",
                parsed={
                    "schema": HEAP_CORRUPTION_SCHEMA,
                    "supported": True,
                    "verdict": "unresolvable",
                    "reason": (
                        "arena bin layout could not be resolved via debug "
                        "symbols; classification skipped"
                    ),
                    "arena_address": arena_addr_out,
                    "is_main_arena": is_main_arena_out,
                    "allocator": allocator_out,
                    "target_profile": profile,
                    "findings": [],
                    "chunks_inspected": 0,
                    "bins_scanned": {"fastbins": 0, "tcache": 0, "regular": 0},
                    "checks_run": [],
                    "sources": sources,
                    "component_status": component_status,
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=warnings or None,
            )

        findings: List[Dict[str, Any]] = []
        chunks_by_address: Dict[int, List[Dict[str, Any]]] = {}
        chunks_inspected = 0

        # Fastbins
        fastbins_scanned = 0
        for bin_ in fb_parsed.get("fastbins", []):
            index = bin_.get("index")
            if bin_.get("read_error"):
                findings.append(
                    self._finding(
                        category="unreadable_bin",
                        severity="unreadable",
                        location={"bin_type": "fastbin", "bin_index": index},
                        reason=str(bin_.get("read_error")),
                    )
                )
                continue
            head = bin_.get("head")
            if head is None or head == 0:
                continue
            fastbins_scanned += 1
            if bin_.get("cycle_detected"):
                findings.append(
                    self._finding(
                        category="cycle",
                        severity="corrupted",
                        location={"bin_type": "fastbin", "bin_index": index},
                        reason=(
                            "fastbin fd chain cycles back to a previously-seen " "chunk"
                        ),
                        evidence={"chain_length": bin_.get("length")},
                    )
                )
            for entry in bin_.get("chain", []):
                chunks_inspected += 1
                self._analyze_entry(
                    entry=entry,
                    bin_type="fastbin",
                    bin_index=index,
                    alignment=alignment,
                    findings=findings,
                    chunks_by_address=chunks_by_address,
                )

        # Tcache (thread-local; only sweep when the inspector parsed successfully)
        tcache_scanned = 0
        if tc_parsed.get("verdict") == "parsed":
            for bin_ in tc_parsed.get("bins", []):
                index = bin_.get("index")
                head = bin_.get("head")
                if head is None or head == 0:
                    continue
                tcache_scanned += 1
                if bin_.get("cycle_detected"):
                    findings.append(
                        self._finding(
                            category="cycle",
                            severity="corrupted",
                            location={"bin_type": "tcache", "bin_index": index},
                            reason=(
                                "tcache chain cycles back to a previously-seen " "entry"
                            ),
                            evidence={"chain_length": bin_.get("length")},
                        )
                    )
                for entry in bin_.get("chain", []):
                    chunks_inspected += 1
                    self._analyze_entry(
                        entry=entry,
                        bin_type="tcache",
                        bin_index=index,
                        alignment=alignment,
                        findings=findings,
                        chunks_by_address=chunks_by_address,
                    )

        # Regular bins (unsorted / small / large)
        regular_scanned = 0
        for bin_ in bn_parsed.get("bins", []):
            if bin_.get("empty"):
                continue
            regular_scanned += 1
            index = bin_.get("index")
            category = bin_.get("category", "regular")
            if bin_.get("cycle_detected"):
                findings.append(
                    self._finding(
                        category="cycle",
                        severity="corrupted",
                        location={"bin_type": category, "bin_index": index},
                        reason=f"{category} bin fd chain cycles",
                        evidence={"chain_length": bin_.get("length")},
                    )
                )
            chain = bin_.get("chain", [])
            for entry in chain:
                chunks_inspected += 1
                self._analyze_entry(
                    entry=entry,
                    bin_type=category,
                    bin_index=index,
                    alignment=alignment,
                    findings=findings,
                    chunks_by_address=chunks_by_address,
                )
            self._check_fd_bk_consistency(chain=chain, bin_=bin_, findings=findings)

        # Cross-bin duplicate detection.  Each chain's own walker guarantees no
        # intra-chain repeats via cycle detection, so a repeated address here
        # is genuinely a cross-bin appearance -- a strong double-free signal.
        for chunk_addr, occurrences in chunks_by_address.items():
            if len(occurrences) < 2:
                continue
            findings.append(
                self._finding(
                    category="duplicate_entry",
                    severity="corrupted",
                    location={"chunk_address": chunk_addr},
                    reason=(
                        f"chunk 0x{chunk_addr:x} appears in {len(occurrences)} "
                        "distinct bin chains; possible double-free"
                    ),
                    evidence={"occurrences": occurrences},
                )
            )

        verdict = self._aggregate_verdict(findings)

        # Additive S6.6 technique hints over the already-collected inspector
        # outputs.  Purely additive: never mutates findings, verdicts, or the
        # check list.  Lazy import keeps the module-level import cycle broken
        # (heap_techniques imports this module at its top).
        from .heap_techniques import (
            resolve_top_chunk,
            technique_hints_for,
        )  # noqa: PLC0415

        top_chunk = resolve_top_chunk(
            bins=self.bins,
            arenas=self.arenas,
            mappings=_mappings,
            profile=profile,
            cancellation_check=cancellation_check,
        )
        technique_hints = technique_hints_for(
            fb_parsed=fb_parsed,
            tc_parsed=tc_parsed,
            bn_parsed=bn_parsed,
            classification=classification,
            mappings=_mappings,
            profile=profile,
            top_chunk=top_chunk,
            read=self.bins._read_bytes,
        )

        return build_backend_response(
            "classify_heap_corruption",
            parsed={
                "schema": HEAP_CORRUPTION_SCHEMA,
                "supported": True,
                "verdict": verdict,
                "arena_address": arena_addr_out,
                "is_main_arena": is_main_arena_out,
                "allocator": allocator_out,
                "target_profile": profile,
                "findings": findings,
                "technique_hints": technique_hints,
                "chunks_inspected": chunks_inspected,
                "bins_scanned": {
                    "fastbins": fastbins_scanned,
                    "tcache": tcache_scanned,
                    "regular": regular_scanned,
                },
                "checks_run": list(self._CHECKS_RUN),
                "sources": sources,
                "component_status": component_status,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def _analyze_entry(
        self,
        *,
        entry: Mapping[str, Any],
        bin_type: str,
        bin_index: Optional[int],
        alignment: int,
        findings: List[Dict[str, Any]],
        chunks_by_address: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        chunk_addr = entry.get("chunk_address")
        if chunk_addr is None:
            return
        location = {
            "bin_type": bin_type,
            "bin_index": bin_index,
            "chunk_address": chunk_addr,
        }
        # The address itself is corrupt data even when the read failed, so
        # alignment / mapping are evaluated regardless of readability.
        if chunk_addr % alignment != 0:
            findings.append(
                self._finding(
                    category="alignment",
                    severity="corrupted",
                    location=location,
                    reason=(
                        f"chunk address 0x{chunk_addr:x} is not aligned to "
                        f"{alignment} bytes"
                    ),
                    evidence={"alignment": alignment},
                )
            )
        if entry.get("in_mapping") is False:
            findings.append(
                self._finding(
                    category="pointer_out_of_mapping",
                    severity="corrupted",
                    location=location,
                    reason=(
                        f"chunk address 0x{chunk_addr:x} lies outside every "
                        "known memory mapping"
                    ),
                )
            )
        # Different inspectors use different field names for the read flag:
        #   fastbins:      entry["readable"]
        #   tcache / bins: entry["chunk_readable"]
        readable = entry.get("chunk_readable")
        if readable is None:
            readable = entry.get("readable")
        if readable is False:
            findings.append(
                self._finding(
                    category="unreadable_chunk",
                    severity="unreadable",
                    location=location,
                    reason=str(
                        entry.get("chunk_read_error")
                        or entry.get("read_error")
                        or "chunk header could not be read"
                    ),
                )
            )
        chunks_by_address.setdefault(chunk_addr, []).append(
            {"bin_type": bin_type, "bin_index": bin_index}
        )

    def _check_fd_bk_consistency(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        bin_: Mapping[str, Any],
        findings: List[Dict[str, Any]],
    ) -> None:
        """Verify Y.bk == X for every consecutive (X, Y) pair in a regular bin.

        A truncated or cycle-detected walk is skipped -- the walker already
        surfaced those signals, and a pair check on a cut-off chain would just
        add noise.  Entries whose header read failed are skipped (bk is None).
        """
        if bin_.get("truncated") or bin_.get("cycle_detected"):
            return
        for i in range(len(chain) - 1):
            x = chain[i]
            y = chain[i + 1]
            if x.get("chunk_readable") is False or y.get("chunk_readable") is False:
                continue
            x_addr = x.get("chunk_address")
            y_addr = y.get("chunk_address")
            bk = y.get("bk")
            if x_addr is None or y_addr is None or bk is None:
                continue
            if bk != x_addr:
                findings.append(
                    self._finding(
                        category="fd_bk_inconsistency",
                        severity="corrupted",
                        location={
                            "bin_type": bin_.get("category"),
                            "bin_index": bin_.get("index"),
                            "chunk_address": y_addr,
                        },
                        reason=(
                            f"chunk 0x{y_addr:x} bk pointer (0x{bk:x}) does "
                            f"not point back to previous chain entry "
                            f"0x{x_addr:x}"
                        ),
                        evidence={"expected_bk": x_addr, "observed_bk": bk},
                    )
                )

    # ------------------------------------------------------------------
    # validate_free
    # ------------------------------------------------------------------
    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Return a safe-to-free verdict for a glibc chunk.

        Runs the S6.2 chunk parse + containment checks and cross-references the
        chunk against the current thread's tcache and main_arena's fastbins to
        catch a double-free before ``free()`` fires.  A chunk already present
        in tcache or fastbins is a double-free candidate; a chunk that fails
        alignment / mapping / min-size checks is broken by construction.

        Verdict vocabulary:

        * ``unsupported`` -- target is not glibc
        * ``unreadable``  -- chunk header could not be read
        * ``double_free`` -- chunk already present in tcache or fastbins
        * ``corrupted``   -- alignment / mapping / min-size check failed
        * ``suspicious`` -- non-fatal check failed (e.g. payload overflow)
        * ``safe``        -- every check passed
        """
        inspection = self.glibc.inspect_chunk(
            address=address,
            address_kind=address_kind,
            include_next=True,
            cancellation_check=cancellation_check,
        )
        parsed_inspect = inspection.get("parsed", {}) or {}
        allocator_out = parsed_inspect.get("allocator")
        target_profile = parsed_inspect.get("target_profile")

        if not parsed_inspect.get("supported"):
            return build_backend_response(
                "validate_free",
                parsed={
                    "schema": FREE_VALIDATION_SCHEMA,
                    "address": parsed_inspect.get("address"),
                    "address_kind": parsed_inspect.get("address_kind"),
                    "verdict": "unsupported",
                    "failures": [],
                    "checks": {},
                    "reason": parsed_inspect.get("reason"),
                    "chunk_summary": None,
                    "allocator": allocator_out,
                    "target_profile": target_profile,
                    "membership": {"tcache": None, "fastbins": None},
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=inspection.get("warnings"),
            )

        chunk = parsed_inspect.get("chunk") or {}
        containment = parsed_inspect.get("mapping_containment") or {}
        warnings: List[str] = list(inspection.get("warnings") or [])
        failures: List[str] = []
        checks: Dict[str, Dict[str, Any]] = {}
        summary: Optional[Dict[str, Any]] = None
        verdict: str

        if not chunk.get("readable"):
            checks["readable"] = {
                "ok": False,
                "reason": chunk.get("reason") or "chunk_header_unreadable",
            }
            failures.append("readable")
            verdict = "unreadable"
        else:
            checks["readable"] = {"ok": True, "reason": None}
            structural = chunk.get("structural_checks", {}) or {}
            alignment_ok = bool(structural.get("alignment_ok"))
            min_size_ok = bool(structural.get("min_size_ok"))
            chunk_in_mapping = bool(containment.get("chunk_in_mapping"))
            header_in_mapping = bool(containment.get("header_in_mapping"))
            payload_in_mapping = containment.get("payload_in_mapping")
            payload_ok = payload_in_mapping is None or bool(payload_in_mapping)

            checks["alignment"] = {
                "ok": alignment_ok,
                "reason": (
                    None if alignment_ok else "chunk_address_or_size_not_aligned"
                ),
            }
            checks["min_size"] = {
                "ok": min_size_ok,
                "reason": None if min_size_ok else "size_below_min_chunk_size",
            }
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

            summary = {
                "chunk_address": chunk.get("chunk_address"),
                "user_address": chunk.get("user_address"),
                "size": chunk.get("fields", {}).get("size"),
                "flags": chunk.get("fields", {}).get("flags"),
                "allocation_status": (parsed_inspect.get("allocation") or {}).get(
                    "status"
                ),
            }

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
                verdict = "safe"

        tcache_membership: Optional[Dict[str, Any]] = None
        fastbins_membership: Optional[Dict[str, Any]] = None
        if chunk.get("readable"):
            tcache_membership = self._locate_in_tcache(
                user_address=chunk.get("user_address"),
                cancellation_check=cancellation_check,
            )
            fastbins_membership = self._locate_in_fastbins(
                chunk_address=chunk.get("chunk_address"),
                cancellation_check=cancellation_check,
            )
            in_tcache = bool((tcache_membership or {}).get("found"))
            in_fastbins = bool((fastbins_membership or {}).get("found"))
            checks["not_in_tcache"] = {
                "ok": not in_tcache,
                "reason": (
                    None if not in_tcache else "chunk_already_present_in_tcache"
                ),
            }
            checks["not_in_fastbins"] = {
                "ok": not in_fastbins,
                "reason": (
                    None if not in_fastbins else "chunk_already_present_in_fastbins"
                ),
            }
            if in_tcache:
                failures.append("not_in_tcache")
            if in_fastbins:
                failures.append("not_in_fastbins")
            # Double-free takes precedence over any earlier structural verdict.
            if in_tcache or in_fastbins:
                verdict = "double_free"

        return build_backend_response(
            "validate_free",
            parsed={
                "schema": FREE_VALIDATION_SCHEMA,
                "address": parsed_inspect.get("address"),
                "address_kind": parsed_inspect.get("address_kind"),
                "verdict": verdict,
                "failures": failures,
                "checks": checks,
                "chunk_summary": summary,
                "allocator": allocator_out,
                "target_profile": target_profile,
                "mapping_containment": containment,
                "allocation": parsed_inspect.get("allocation"),
                "membership": {
                    "tcache": tcache_membership,
                    "fastbins": fastbins_membership,
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def _locate_in_tcache(
        self,
        *,
        user_address: Optional[int],
        cancellation_check: Optional[Callable[[], None]],
    ) -> Optional[Dict[str, Any]]:
        if user_address is None:
            return None
        response = self.bins.inspect_tcache(cancellation_check=cancellation_check)
        parsed = response.get("parsed", {}) or {}
        inspector_verdict = parsed.get("verdict")
        # ``uninitialized`` (NULL tcache) is a well-defined empty state -- the
        # chunk is trivially not in tcache.  ``unresolvable`` / ``unreadable``
        # cannot answer the membership question at all.
        if inspector_verdict == "uninitialized":
            return {"found": False, "checked": True}
        if inspector_verdict != "parsed":
            return {
                "found": False,
                "checked": False,
                "reason": parsed.get("reason") or inspector_verdict,
            }
        for bin_ in parsed.get("bins", []):
            for entry in bin_.get("chain", []):
                if entry.get("user_address") == user_address:
                    return {
                        "found": True,
                        "checked": True,
                        "bin_index": bin_.get("index"),
                        "chunk_address": entry.get("chunk_address"),
                    }
        return {"found": False, "checked": True}

    def _locate_in_fastbins(
        self,
        *,
        chunk_address: Optional[int],
        cancellation_check: Optional[Callable[[], None]],
    ) -> Optional[Dict[str, Any]]:
        if chunk_address is None:
            return None
        response = self.bins.inspect_fastbins(cancellation_check=cancellation_check)
        parsed = response.get("parsed", {}) or {}
        if parsed.get("verdict") != "parsed":
            return {
                "found": False,
                "checked": False,
                "reason": parsed.get("reason") or parsed.get("verdict"),
            }
        for bin_ in parsed.get("fastbins", []):
            for entry in bin_.get("chain", []):
                if entry.get("chunk_address") == chunk_address:
                    return {
                        "found": True,
                        "checked": True,
                        "bin_index": bin_.get("index"),
                    }
        return {"found": False, "checked": True}


# ----------------------------------------------------------------------
# Stage 6.5 -- heap snapshot history and diff
# ----------------------------------------------------------------------
HEAP_SNAPSHOT_SCHEMA = "pao.heap-snapshot.v1"
HEAP_SNAPSHOT_LIST_SCHEMA = "pao.heap-snapshot-list.v1"
HEAP_SNAPSHOT_DIFF_SCHEMA = "pao.heap-snapshot-diff.v1"
HEAP_SNAPSHOT_LINK_SCHEMA = "pao.heap-snapshot-link.v1"

DEFAULT_MAX_HEAP_SNAPSHOTS = 16
DEFAULT_MAX_HEAP_SNAPSHOT_BYTES = 64 * 1024 * 1024


class HeapSnapshotError(RuntimeError):
    """Raised when a heap snapshot cannot be resolved or is over budget."""


@dataclass
class _HeapSnapshotEntry:
    snapshot_id: str
    name: Optional[str]
    captured_at: str
    payload_json: bytes
    linked_crash_ids: List[str] = field(default_factory=list)

    @property
    def payload_bytes(self) -> int:
        return len(self.payload_json)


class HeapHistory:
    """Bounded, named-snapshot registry for the current session's heap state.

    Captures a typed heap snapshot by driving the sibling S6.1 / S6.3
    inspectors (allocator discovery, arena / bin / tcache walkers) into a
    single reproducible payload.  Snapshots are stored session-locally
    under the same bounded-history budget the crash triage uses (16
    entries and 64 MiB total by default; oldest-first eviction).

    A pure ``diff_ledgers`` classifier computes ``new`` / ``removed`` /
    ``moved`` chunk sets between two snapshots and lives as a
    ``@staticmethod`` so it stays unit-testable without a debugger.  A
    sibling ``diff_arena_summaries`` reports per-arena top-address deltas.
    Snapshots may be tagged with one or more crash IDs via
    :meth:`link_snapshot_to_crash`; the association is annotation-only and
    does not verify the crash exists on the crash-history side.

    Live per-allocation tracing (breakpoint-based malloc / free hooks) is
    intentionally deferred here: it needs breakpoint orchestration and
    symbol availability that must be an opt-in feature, and the
    snapshot-diff already covers the practical investigation surface.
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
        bins: "GlibcBins",
        max_snapshots: int = DEFAULT_MAX_HEAP_SNAPSHOTS,
        max_snapshot_bytes: int = DEFAULT_MAX_HEAP_SNAPSHOT_BYTES,
    ) -> None:
        if max_snapshots <= 0:
            raise ValueError("max_snapshots must be positive")
        if max_snapshot_bytes <= 0:
            raise ValueError("max_snapshot_bytes must be positive")
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self.arenas = arenas
        self.bins = bins
        self._history: "OrderedDict[str, _HeapSnapshotEntry]" = OrderedDict()
        self._name_index: Dict[str, str] = {}
        self._history_bytes = 0
        self._history_lock = threading.RLock()
        self._max_snapshots = max_snapshots
        self._max_snapshot_bytes = max_snapshot_bytes

    # ------------------------------------------------------------------
    # Live-target helpers
    # ------------------------------------------------------------------
    def _require_live(self) -> Dict[str, Any]:
        record = self.gdb.get_last_stop_record()
        return LiveInferiorGuard.require_live(
            record,
            self.gdb.get_inferior_pid(),
            detail=(
                "Heap snapshot capture requires a live, stopped inferior; "
                "run the target to a stop before capturing the heap"
            ),
        )

    def _prepare_context(
        self, cancellation_check: Optional[Callable[[], None]]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        self._require_live()
        mappings, libcinfo_parsed, component_status = self.discovery.collect_evidence(
            cancellation_check
        )
        classification = AllocatorDiscovery.classify(mappings, libcinfo_parsed)
        profile = self.discovery.target_profile(mappings)
        return classification, mappings, profile, component_status

    # ------------------------------------------------------------------
    # Pure helpers (unit-testable without a live inferior)
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_ledger_from_arena_bins(
        arena_address: Optional[int],
        fastbins_parsed: Optional[Mapping[str, Any]],
        bins_parsed: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Flatten one arena's fastbins + regular bins into ledger entries."""
        ledger: List[Dict[str, Any]] = []
        if fastbins_parsed and fastbins_parsed.get("verdict") == "parsed":
            for bin_ in fastbins_parsed.get("fastbins", []) or []:
                bin_index = bin_.get("index")
                for entry in bin_.get("chain", []) or []:
                    addr = entry.get("chunk_address")
                    if addr is None:
                        continue
                    ledger.append(
                        {
                            "chunk_address": int(addr),
                            "size": entry.get("size"),
                            "arena_address": arena_address,
                            "location": {"kind": "fastbin", "index": bin_index},
                        }
                    )
        if bins_parsed and bins_parsed.get("verdict") == "parsed":
            for bin_ in bins_parsed.get("bins", []) or []:
                bin_index = bin_.get("index")
                bin_category = bin_.get("category")
                for entry in bin_.get("chain", []) or []:
                    addr = entry.get("chunk_address")
                    if addr is None:
                        continue
                    ledger.append(
                        {
                            "chunk_address": int(addr),
                            "size": entry.get("size"),
                            "arena_address": arena_address,
                            "location": {
                                "kind": bin_category or "regular",
                                "index": bin_index,
                            },
                        }
                    )
        return ledger

    @staticmethod
    def _chunk_ledger_from_tcache(
        tcache_parsed: Optional[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        ledger: List[Dict[str, Any]] = []
        if not tcache_parsed or tcache_parsed.get("verdict") != "parsed":
            return ledger
        for bin_ in tcache_parsed.get("bins", []) or []:
            bin_index = bin_.get("index")
            for entry in bin_.get("chain", []) or []:
                addr = entry.get("chunk_address")
                if addr is None:
                    continue
                ledger.append(
                    {
                        "chunk_address": int(addr),
                        "size": entry.get("size"),
                        "arena_address": None,
                        "location": {"kind": "tcache", "index": bin_index},
                    }
                )
        return ledger

    @staticmethod
    def diff_ledgers(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Compute new / removed / moved chunks between two ledgers.

        Address is the identity key.  A chunk with the same address in
        both ledgers whose location set differs is reported as ``moved``.
        A single address may legitimately appear multiple times per
        snapshot (cross-bin double-free), and each occurrence is
        preserved in the ``locations`` list.
        """

        def _index(
            items: Sequence[Mapping[str, Any]],
        ) -> Dict[int, Dict[str, Any]]:
            out: Dict[int, Dict[str, Any]] = {}
            for entry in items:
                addr = entry.get("chunk_address")
                if addr is None:
                    continue
                bucket = out.setdefault(
                    int(addr),
                    {
                        "chunk_address": int(addr),
                        "size": entry.get("size"),
                        "locations": [],
                    },
                )
                bucket["locations"].append(entry.get("location"))
            return out

        before_index = _index(before)
        after_index = _index(after)

        new_chunks: List[Dict[str, Any]] = []
        removed_chunks: List[Dict[str, Any]] = []
        moved_chunks: List[Dict[str, Any]] = []

        for addr in sorted(after_index.keys() - before_index.keys()):
            entry = after_index[addr]
            new_chunks.append(
                {
                    "chunk_address": addr,
                    "size": entry["size"],
                    "locations_after": entry["locations"],
                }
            )
        for addr in sorted(before_index.keys() - after_index.keys()):
            entry = before_index[addr]
            removed_chunks.append(
                {
                    "chunk_address": addr,
                    "size": entry["size"],
                    "locations_before": entry["locations"],
                }
            )
        for addr in sorted(after_index.keys() & before_index.keys()):
            before_entry = before_index[addr]
            after_entry = after_index[addr]
            if before_entry["locations"] != after_entry["locations"]:
                moved_chunks.append(
                    {
                        "chunk_address": addr,
                        "size": after_entry["size"],
                        "locations_before": before_entry["locations"],
                        "locations_after": after_entry["locations"],
                    }
                )

        return {
            "new_chunks": new_chunks,
            "removed_chunks": removed_chunks,
            "moved_chunks": moved_chunks,
            "summary": {
                "new_count": len(new_chunks),
                "removed_count": len(removed_chunks),
                "moved_count": len(moved_chunks),
                "before_chunk_count": sum(
                    len(v["locations"]) for v in before_index.values()
                ),
                "after_chunk_count": sum(
                    len(v["locations"]) for v in after_index.values()
                ),
            },
        }

    @staticmethod
    def diff_arena_summaries(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Per-arena presence + top / last_remainder delta between snapshots."""

        def _by_addr(
            items: Sequence[Mapping[str, Any]],
        ) -> Dict[int, Mapping[str, Any]]:
            out: Dict[int, Mapping[str, Any]] = {}
            for arena in items:
                addr = arena.get("arena_address")
                if addr is None:
                    continue
                out[int(addr)] = arena
            return out

        before_map = _by_addr(before)
        after_map = _by_addr(after)
        addresses = sorted(set(before_map.keys()) | set(after_map.keys()))
        deltas: List[Dict[str, Any]] = []
        for addr in addresses:
            b = before_map.get(addr)
            a = after_map.get(addr)
            entry: Dict[str, Any] = {
                "arena_address": addr,
                "present_before": b is not None,
                "present_after": a is not None,
                "top_before": b.get("top") if b else None,
                "top_after": a.get("top") if a else None,
                "last_remainder_before": b.get("last_remainder") if b else None,
                "last_remainder_after": a.get("last_remainder") if a else None,
            }
            top_b = entry["top_before"]
            top_a = entry["top_after"]
            if isinstance(top_b, int) and isinstance(top_a, int):
                entry["top_delta"] = top_a - top_b
            else:
                entry["top_delta"] = None
            deltas.append(entry)
        return deltas

    # ------------------------------------------------------------------
    # Snapshot store (bounded, thread-safe)
    # ------------------------------------------------------------------
    def _remember(
        self,
        *,
        name: Optional[str],
        payload: Dict[str, Any],
    ) -> _HeapSnapshotEntry:
        snapshot_id = payload["snapshot_id"]
        payload_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        if len(payload_json) > self._max_snapshot_bytes:
            raise HeapSnapshotError(
                "Heap snapshot exceeds the bounded history byte budget"
            )
        entry = _HeapSnapshotEntry(
            snapshot_id=snapshot_id,
            name=name,
            captured_at=payload["captured_at"],
            payload_json=payload_json,
        )
        with self._history_lock:
            if name is not None:
                existing_id = self._name_index.pop(name, None)
                if existing_id is not None:
                    previous = self._history.pop(existing_id, None)
                    if previous is not None:
                        self._history_bytes -= previous.payload_bytes
            while self._history and (
                len(self._history) >= self._max_snapshots
                or self._history_bytes + entry.payload_bytes > self._max_snapshot_bytes
            ):
                _, removed = self._history.popitem(last=False)
                self._history_bytes -= removed.payload_bytes
                if removed.name is not None:
                    self._name_index.pop(removed.name, None)
            self._history[snapshot_id] = entry
            self._history_bytes += entry.payload_bytes
            if name is not None:
                self._name_index[name] = snapshot_id
        return entry

    def _resolve_snapshot(self, identifier: str) -> _HeapSnapshotEntry:
        if not identifier or not isinstance(identifier, str):
            raise HeapSnapshotError("snapshot identifier must be a non-empty string")
        with self._history_lock:
            entry = self._history.get(identifier)
            if entry is None:
                mapped_id = self._name_index.get(identifier)
                if mapped_id is not None:
                    entry = self._history.get(mapped_id)
            if entry is None:
                raise HeapSnapshotError(
                    f"heap snapshot not found for identifier '{identifier}'"
                )
            return entry

    def _clone(self, entry: _HeapSnapshotEntry) -> Dict[str, Any]:
        payload = json.loads(entry.payload_json)
        payload["linked_crash_ids"] = list(entry.linked_crash_ids)
        return payload

    def _descriptor(self, entry: _HeapSnapshotEntry) -> Dict[str, Any]:
        payload = json.loads(entry.payload_json)
        summary = payload.get("summary") or {}
        allocator = payload.get("allocator") or {}
        return {
            "snapshot_id": entry.snapshot_id,
            "name": entry.name,
            "captured_at": entry.captured_at,
            "payload_bytes": entry.payload_bytes,
            "allocator_family": allocator.get("family"),
            "verdict": payload.get("verdict"),
            "chunk_count": summary.get("chunk_count"),
            "arena_count": summary.get("arena_count"),
            "linked_crash_ids": list(entry.linked_crash_ids),
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture and store a typed heap snapshot for this session."""
        if name is not None:
            if not isinstance(name, str):
                raise ValueError(
                    f"snapshot name must be a string, got {type(name).__name__}"
                )
            if not name.strip():
                raise ValueError("snapshot name may not be empty or whitespace-only")
        if max_chunks_per_bin <= 0:
            raise ValueError(
                f"max_chunks_per_bin must be positive, got {max_chunks_per_bin!r}"
            )

        classification, _mappings, profile, component_status = self._prepare_context(
            cancellation_check
        )
        pid = self.gdb.get_inferior_pid()
        epoch = self.gdb.get_inferior_epoch()
        allocator_summary = {
            "family": classification.get("family"),
            "variant": classification.get("variant"),
            "version": classification.get("version"),
            "confidence": classification.get("confidence"),
            "safe_linking": classification.get("safe_linking"),
        }

        snapshot_id = uuid.uuid4().hex[:16]
        captured_at = datetime.now(timezone.utc).isoformat()

        if classification["family"] != "glibc":
            payload: Dict[str, Any] = {
                "schema": HEAP_SNAPSHOT_SCHEMA,
                "snapshot_id": snapshot_id,
                "name": name,
                "captured_at": captured_at,
                "session_id": self.session_id,
                "inferior": {"pid": pid, "epoch": epoch},
                "allocator": allocator_summary,
                "target_profile": profile,
                "supported": False,
                "verdict": "unsupported",
                "reason": (
                    "heap snapshot supports glibc only; detected "
                    f"family={classification.get('family')}"
                ),
                "arenas": [],
                "chunk_ledger": [],
                "tcache_summary": None,
                "component_status": component_status,
                "summary": {"chunk_count": 0, "arena_count": 0},
            }
            entry = self._remember(name=name, payload=payload)
            return build_backend_response(
                "capture_heap_snapshot",
                parsed=self._clone(entry),
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        warnings: List[str] = []
        arena_list_response = self.arenas.list_arenas(
            cancellation_check=cancellation_check
        )
        arena_list_parsed = arena_list_response.get("parsed", {}) or {}
        for w in arena_list_response.get("warnings") or []:
            warnings.append(f"list_arenas: {w}")

        raw_arenas = arena_list_parsed.get("arenas", []) or []
        arena_summaries: List[Dict[str, Any]] = []
        chunk_ledger: List[Dict[str, Any]] = []

        for arena_entry in raw_arenas:
            addr = arena_entry.get("address")
            if addr is None:
                continue
            addr_int = int(addr)
            addr_str = str(addr_int)
            arena_inspect = self.arenas.inspect_arena(
                arena_address=addr_str, cancellation_check=cancellation_check
            )
            arena_parsed = arena_inspect.get("parsed", {}) or {}
            fields = arena_parsed.get("fields") or {}
            arena_summaries.append(
                {
                    "arena_address": addr_int,
                    "is_main_arena": arena_parsed.get("is_main"),
                    "top": fields.get("top"),
                    "last_remainder": fields.get("last_remainder"),
                    "system_mem": fields.get("system_mem"),
                    "next": fields.get("next"),
                    "verdict": arena_parsed.get("verdict"),
                }
            )
            for w in arena_inspect.get("warnings") or []:
                warnings.append(f"inspect_arena[{addr_int:#x}]: {w}")

            fastbins_response = self.bins.inspect_fastbins(
                arena_address=addr_str,
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=cancellation_check,
            )
            for w in fastbins_response.get("warnings") or []:
                warnings.append(f"inspect_fastbins[{addr_int:#x}]: {w}")
            bins_response = self.bins.inspect_bins(
                arena_address=addr_str,
                max_chunks_per_bin=max_chunks_per_bin,
                cancellation_check=cancellation_check,
            )
            for w in bins_response.get("warnings") or []:
                warnings.append(f"inspect_bins[{addr_int:#x}]: {w}")

            chunk_ledger.extend(
                self._chunk_ledger_from_arena_bins(
                    arena_address=addr_int,
                    fastbins_parsed=fastbins_response.get("parsed"),
                    bins_parsed=bins_response.get("parsed"),
                )
            )

        tcache_response = self.bins.inspect_tcache(
            cancellation_check=cancellation_check
        )
        tcache_parsed = tcache_response.get("parsed", {}) or {}
        for w in tcache_response.get("warnings") or []:
            warnings.append(f"inspect_tcache: {w}")
        tcache_summary = {
            "verdict": tcache_parsed.get("verdict"),
            "bin_count": len(tcache_parsed.get("bins", []) or []),
        }
        chunk_ledger.extend(self._chunk_ledger_from_tcache(tcache_parsed))

        payload = {
            "schema": HEAP_SNAPSHOT_SCHEMA,
            "snapshot_id": snapshot_id,
            "name": name,
            "captured_at": captured_at,
            "session_id": self.session_id,
            "inferior": {"pid": pid, "epoch": epoch},
            "allocator": allocator_summary,
            "target_profile": profile,
            "supported": True,
            "verdict": "captured",
            "arenas": arena_summaries,
            "chunk_ledger": chunk_ledger,
            "tcache_summary": tcache_summary,
            "component_status": component_status,
            "summary": {
                "chunk_count": len(chunk_ledger),
                "arena_count": len(arena_summaries),
            },
        }
        entry = self._remember(name=name, payload=payload)
        return build_backend_response(
            "capture_heap_snapshot",
            parsed=self._clone(entry),
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=warnings or None,
        )

    def list_heap_snapshots(self) -> Dict[str, Any]:
        with self._history_lock:
            descriptors = [self._descriptor(entry) for entry in self._history.values()]
            total_bytes = self._history_bytes
        return build_backend_response(
            "list_heap_snapshots",
            parsed={
                "schema": HEAP_SNAPSHOT_LIST_SCHEMA,
                "snapshots": descriptors,
                "budget": {
                    "count": len(descriptors),
                    "max_count": self._max_snapshots,
                    "bytes": total_bytes,
                    "max_bytes": self._max_snapshot_bytes,
                },
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def get_heap_snapshot(self, *, identifier: str) -> Dict[str, Any]:
        entry = self._resolve_snapshot(identifier)
        return build_backend_response(
            "get_heap_snapshot",
            parsed=self._clone(entry),
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def diff_heap_snapshots(self, *, before: str, after: str) -> Dict[str, Any]:
        before_entry = self._resolve_snapshot(before)
        after_entry = self._resolve_snapshot(after)
        if before_entry.snapshot_id == after_entry.snapshot_id:
            raise HeapSnapshotError(
                "cannot diff a snapshot against itself "
                f"('{before_entry.snapshot_id}')"
            )
        before_payload = json.loads(before_entry.payload_json)
        after_payload = json.loads(after_entry.payload_json)
        header_before = {
            "snapshot_id": before_entry.snapshot_id,
            "name": before_entry.name,
            "captured_at": before_entry.captured_at,
            "allocator_family": (before_payload.get("allocator") or {}).get("family"),
        }
        header_after = {
            "snapshot_id": after_entry.snapshot_id,
            "name": after_entry.name,
            "captured_at": after_entry.captured_at,
            "allocator_family": (after_payload.get("allocator") or {}).get("family"),
        }

        if not (
            before_payload.get("supported") is True
            and after_payload.get("supported") is True
        ):
            return build_backend_response(
                "diff_heap_snapshots",
                parsed={
                    "schema": HEAP_SNAPSHOT_DIFF_SCHEMA,
                    "before": header_before,
                    "after": header_after,
                    "supported": False,
                    "verdict": "unsupported",
                    "reason": (
                        "diff supports snapshots captured on glibc only; "
                        "one or both snapshots were unsupported"
                    ),
                    "chunk_diff": None,
                    "arena_deltas": [],
                },
                raw_responses=[],
                state=self.gdb.get_state(),
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        chunk_diff = self.diff_ledgers(
            before_payload.get("chunk_ledger") or [],
            after_payload.get("chunk_ledger") or [],
        )
        arena_deltas = self.diff_arena_summaries(
            before_payload.get("arenas") or [],
            after_payload.get("arenas") or [],
        )
        return build_backend_response(
            "diff_heap_snapshots",
            parsed={
                "schema": HEAP_SNAPSHOT_DIFF_SCHEMA,
                "before": header_before,
                "after": header_after,
                "supported": True,
                "verdict": "diffed",
                "chunk_diff": chunk_diff,
                "arena_deltas": arena_deltas,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def link_snapshot_to_crash(
        self, *, identifier: str, crash_id: str
    ) -> Dict[str, Any]:
        if not crash_id or not isinstance(crash_id, str):
            raise HeapSnapshotError("crash_id must be a non-empty string")
        with self._history_lock:
            entry = self._resolve_snapshot(identifier)
            if crash_id not in entry.linked_crash_ids:
                entry.linked_crash_ids.append(crash_id)
            linked = list(entry.linked_crash_ids)
        return build_backend_response(
            "link_snapshot_to_crash",
            parsed={
                "schema": HEAP_SNAPSHOT_LINK_SCHEMA,
                "snapshot_id": entry.snapshot_id,
                "name": entry.name,
                "linked_crash_ids": linked,
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
            warnings=None,
        )


class HeapTools:
    """Facade that groups every heap-inspection component under one object.

    Later stages (S6.3 arenas/bins, S6.6 musl/jemalloc parsers) attach here so
    the tool layer keeps a single ``session.tools.heap`` entry point regardless
    of the detected allocator family.  Delegation is intentionally boring:
    each public method routes to the appropriate sub-component untouched.

    Allocator-plugin contract
    -------------------------
    Every allocator family must be identified by :class:`AllocatorDiscovery`
    (extend ``_FAMILY_FILENAME_PATTERNS`` and the runtime-command map) and
    -- when supported -- have a sibling introspection class hanging off
    this facade (mirror :class:`GlibcHeap`, :class:`GlibcArenas`,
    :class:`GlibcBins`).

    The twelve allocator-layout-dependent methods (chunks, arenas, bins,
    corruption, free validation, techniques, UAF, snapshot capture) are
    dispatched per call through an :class:`AllocatorAnalyzer` resolved from
    the detected family (``heap_analyzers.analyzer_for_family``).  ``glibc``
    is the only family with a typed parser in Phase 0; every other family
    resolves to ``UnsupportedAnalyzer``, which forwards to the glibc
    components that already return an honest typed ``unsupported`` verdict
    for non-glibc allocators.  A new family's parser plugs in by implementing
    the analyzer surface and adding its name to the dispatch; nothing else
    changes.
    """

    def __init__(
        self,
        gdb: GdbController,
        inspection: PwndbgInspection,
        debugger: DebuggerOperations,
        session_id: str,
        *,
        binary_path_provider: Optional[Callable[[], Optional[str]]] = None,
        backend: Optional[BackendSpec] = None,
    ) -> None:
        self.backend = backend
        self.session_id = session_id
        # TTL cache of the last detected allocator family (monotonic clock
        # seconds + family).  Consulted only by ``_analyzer()``; the glibc
        # components re-classify per call regardless, so a stale hit never
        # changes behavior -- it only picks which forwarder runs.
        self._family_cache: Optional[Tuple[float, Optional[str]]] = None
        self.gdb = gdb
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
        self.bins = GlibcBins(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
        )
        self.corruption = GlibcHeapCorruption(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
            glibc=self.glibc,
        )
        self.history = HeapHistory(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
        )
        # Lazy imports to break module-level cycles: heap_techniques and
        # heap_analyzers both import this module at their top, so heap.py must
        # never import them at import time -- only here, when the facade is
        # actually built.
        from .attack_helpers import GlibcAttackHelper  # noqa: PLC0415
        from .heap_analyzers import analyzer_for_family  # noqa: PLC0415
        from .heap_techniques import GlibcTechniqueAnalyzer  # noqa: PLC0415
        from .slub import KernelSlab  # noqa: PLC0415

        self._analyzer_for_family = analyzer_for_family

        self.techniques = GlibcTechniqueAnalyzer(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            arenas=self.arenas,
            bins=self.bins,
            glibc=self.glibc,
            corruption=self.corruption,
        )

        # Attack helpers are the write-capable extension of the heap surface:
        # they execute primitives against the live target via the guarded
        # writer, gated behind an explicit ``dry_run=false`` opt-in.
        self._attack_for_family = GlibcAttackHelper
        self.attack = GlibcAttackHelper(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
            glibc=self.glibc,
            bins=self.bins,
            history=self.history,
        )

        # Kernel SLUB is a dedicated self-gated surface: kernel targets are
        # not classified by the userspace allocator protocol, so this
        # component never routes through ``classify`` -- the caller names a
        # cache address and BTF resolves the layout.
        self.slub = KernelSlab(
            gdb,
            inspection,
            debugger,
            session_id,
            discovery=self.discovery,
        )

    # How long a detected family stays cached on this facade.  Short on
    # purpose: the components re-classify per call anyway, so this only avoids
    # a second full classification when consecutive dispatched calls land
    # within the window.
    _FAMILY_CACHE_TTL_SEC = 1.0

    def _detected_family(self) -> Optional[str]:
        """Classify the current inferior's allocator family (TTL-cached).

        Returns ``None`` when the detection cannot resolve a family (no live
        inferior, classification failure); the dispatch treats that exactly
        like ``unknown``.  Failures never propagate here -- the analyzer
        forwarders surface the same typed errors the components already raise.
        """
        now = time.monotonic()
        cached = getattr(self, "_family_cache", None)
        if cached is not None:
            ts, family = cached
            if now - ts < self._FAMILY_CACHE_TTL_SEC:
                return family
        try:
            envelope = self.discovery.detect_allocator()
        except Exception:  # noqa: BLE001 -- detection is best-effort here
            self._family_cache = (now, None)
            return None
        parsed = envelope.get("parsed") if isinstance(envelope, dict) else None
        family = (parsed or {}).get("family")
        self._family_cache = (now, family)
        return family

    def _analyzer(self) -> Any:
        """Resolve the per-family analyzer for this call's detected family.

        ``_analyzer_for_family`` is normally set in ``__init__``; fall back to
        a lazy import when the facade was assembled without ``__init__`` (the
        allocator-plugin test pattern builds ``HeapTools`` shells via
        ``__new__`` and attaches components directly).  The import is deferred
        to call time to keep the ``heap_analyzers`` <-> ``heap`` module cycle
        broken.
        """
        resolver = getattr(self, "_analyzer_for_family", None)
        if resolver is None:
            from .heap_analyzers import analyzer_for_family  # noqa: PLC0415

            resolver = analyzer_for_family
        return resolver(self._detected_family(), self)

    @property
    def _is_lldb(self) -> bool:
        backend = getattr(self, "backend", None)
        return backend is not None and getattr(backend, "protocol", None) == "lldb-cli"

    def _gdb_gate(self, capability: str) -> Optional[Dict[str, Any]]:
        """Return a typed verdict when a heap parse requires the gdb backend."""
        if not self._is_lldb:
            return None
        backend_name = getattr(self.backend, "name", "lldb")
        return {
            "success": False,
            "command": capability,
            "state": self.gdb.get_state(),
            "type": "GdbOnlyBackendError",
            "error": (
                f"'{capability}' requires the gdb/pwndbg backend; "
                f"session uses {backend_name}"
            ),
        }

    # ---- S6.1 allocator discovery ------------------------------------
    def detect_allocator(
        self,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("detect_allocator")
        if gate is not None:
            return gate
        return self.discovery.detect_allocator(cancellation_check=cancellation_check)

    def get_allocator_capabilities(
        self,
        *,
        debugger_command_names: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("get_allocator_capabilities")
        if gate is not None:
            return gate
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
        gate = self._gdb_gate("inspect_chunk")
        if gate is not None:
            return gate
        return self._analyzer().inspect_chunk(
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
        gate = self._gdb_gate("validate_chunk")
        if gate is not None:
            return gate
        return self._analyzer().validate_chunk(
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
        gate = self._gdb_gate("list_arenas")
        if gate is not None:
            return gate
        return self._analyzer().list_arenas(cancellation_check=cancellation_check)

    def inspect_arena(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_arena")
        if gate is not None:
            return gate
        return self._analyzer().inspect_arena(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

    # ---- S6.3b glibc fastbin traversal --------------------------------
    def inspect_fastbins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_FASTBIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_fastbins")
        if gate is not None:
            return gate
        return self._analyzer().inspect_fastbins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- S6.3c glibc tcache traversal ---------------------------------
    def inspect_tcache(
        self,
        *,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_tcache")
        if gate is not None:
            return gate
        return self._analyzer().inspect_tcache(
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- S6.3d glibc regular bins (unsorted/small/large) --------------
    def inspect_bins(
        self,
        *,
        arena_address: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("inspect_bins")
        if gate is not None:
            return gate
        return self._analyzer().inspect_bins(
            arena_address=arena_address,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- S6.9 attack helpers ------------------------------------------
    def poison_tcache_link(
        self,
        *,
        victim_address: str,
        target_address: str,
        address_kind: str = "chunk",
        dry_run: bool = True,
        snapshot_before: bool = True,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Re-point a freed tcache chunk's ``fd`` (dry-run by default)."""
        gate = self._gdb_gate("poison_tcache_link")
        if gate is not None:
            return gate
        attack = getattr(self, "attack", None)
        if attack is None:
            # Facade assembled without ``__init__`` (the allocator-plugin
            # test pattern); build a throwaway helper on the live components.
            from .attack_helpers import GlibcAttackHelper  # noqa: PLC0415

            attack = GlibcAttackHelper(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
                glibc=self.glibc,
                bins=self.bins,
                history=self.history,
            )
        return attack.poison_tcache_link(
            victim_address=victim_address,
            target_address=target_address,
            address_kind=address_kind,
            dry_run=dry_run,
            snapshot_before=snapshot_before,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    # ---- S6.10 kernel SLUB (dedicated self-gated surface) --------------
    def inspect_slub_cache(
        self,
        *,
        cache_address: str,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode a kernel SLUB cache's geometry + hardening state from BTF."""
        gate = self._gdb_gate("inspect_slub_cache")
        if gate is not None:
            return gate
        slub = getattr(self, "slub", None)
        if slub is None:
            # Facade assembled without ``__init__`` (the allocator-plugin
            # test pattern); build a throwaway component on the discovery.
            from .slub import KernelSlab  # noqa: PLC0415

            slub = KernelSlab(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
            )
        return slub.inspect_slub_cache(
            cache_address=cache_address,
            btf_path=btf_path,
            page_size=page_size,
            cancellation_check=cancellation_check,
        )

    def decode_slub_freelist(
        self,
        *,
        cache_address: str,
        slab_address: str,
        slab_mem_address: Optional[str] = None,
        btf_path: Optional[str] = None,
        page_size: int = DEFAULT_SLUB_PAGE_SIZE,
        max_objects: Optional[int] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Decode one kernel SLUB slab's obfuscated freelist chain."""
        gate = self._gdb_gate("decode_slub_freelist")
        if gate is not None:
            return gate
        slub = getattr(self, "slub", None)
        if slub is None:
            from .slub import KernelSlab  # noqa: PLC0415

            slub = KernelSlab(
                self.discovery.gdb,
                self.discovery.inspection,
                self.discovery.debugger,
                self.session_id,
                discovery=self.discovery,
            )
        return slub.decode_slub_freelist(
            cache_address=cache_address,
            slab_address=slab_address,
            slab_mem_address=slab_mem_address,
            btf_path=btf_path,
            page_size=page_size,
            max_objects=max_objects,
            cancellation_check=cancellation_check,
        )

    # ---- S6.4 heap corruption classifier ------------------------------
    def classify_heap_corruption(
        self,
        *,
        arena_address: Optional[str] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("classify_heap_corruption")
        if gate is not None:
            return gate
        return self._analyzer().classify_heap_corruption(
            arena_address=arena_address,
            cancellation_check=cancellation_check,
        )

    def validate_free(
        self,
        *,
        address: str,
        address_kind: str = "user",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("validate_free")
        if gate is not None:
            return gate
        return self._analyzer().validate_free(
            address=address,
            address_kind=address_kind,
            cancellation_check=cancellation_check,
        )

    # ---- S6.6 heap technique analysis --------------------------------
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
        """Run House-of-* / exploit-primitive detectors over the live heap.

        Delegates to :class:`GlibcTechniqueAnalyzer`; every finding is a
        heuristic tagged with raw evidence, confidence, confirmation tools and
        a playbook.  This method requires the gdb/pwndbg backend (rich heap
        parsing).
        """
        gate = self._gdb_gate("analyze_heap_technique")
        if gate is not None:
            return gate
        return self._analyzer().analyze_heap_technique(
            arena_address=arena_address,
            techniques=techniques,
            scan_mmap_mappings=scan_mmap_mappings,
            mmap_max_mappings=mmap_max_mappings,
            mmap_max_bytes=mmap_max_bytes,
            max_overlap_chunks=max_overlap_chunks,
            cancellation_check=cancellation_check,
        )

    def detect_uaf(
        self,
        *,
        max_bytes: int = 1 << 22,
        max_mappings: int = 8,
        max_chunks: int = 512,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan writable mappings for pointers into currently-freed chunks.

        The dangling-reference signature of a use-after-free.  Bounded
        (``max_bytes`` / ``max_mappings`` / ``max_chunks``) and excludes the
        allocator's own bookkeeping (bin link slots, arena struct, tcache
        struct).  Requires the gdb/pwndbg backend.
        """
        gate = self._gdb_gate("detect_uaf")
        if gate is not None:
            return gate
        return self._analyzer().detect_uaf(
            max_bytes=max_bytes,
            max_mappings=max_mappings,
            max_chunks=max_chunks,
            cancellation_check=cancellation_check,
        )

    def heap_technique_catalog(
        self,
        *,
        technique: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the static House-of-* technique catalog (no debugger needed)."""
        return self.techniques.heap_technique_catalog(technique=technique)

    # ---- S6.5 heap snapshot history ----------------------------------
    def capture_heap_snapshot(
        self,
        *,
        name: Optional[str] = None,
        max_chunks_per_bin: int = DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        gate = self._gdb_gate("capture_heap_snapshot")
        if gate is not None:
            return gate
        return self._analyzer().capture_heap_snapshot(
            name=name,
            max_chunks_per_bin=max_chunks_per_bin,
            cancellation_check=cancellation_check,
        )

    def list_heap_snapshots(self) -> Dict[str, Any]:
        return self.history.list_heap_snapshots()

    def get_heap_snapshot(self, *, identifier: str) -> Dict[str, Any]:
        return self.history.get_heap_snapshot(identifier=identifier)

    def diff_heap_snapshots(self, *, before: str, after: str) -> Dict[str, Any]:
        return self.history.diff_heap_snapshots(before=before, after=after)

    def link_snapshot_to_crash(
        self, *, identifier: str, crash_id: str
    ) -> Dict[str, Any]:
        return self.history.link_snapshot_to_crash(
            identifier=identifier, crash_id=crash_id
        )


__all__ = [
    "AllocatorDiscovery",
    "GlibcHeap",
    "GlibcArenas",
    "GlibcBins",
    "GlibcHeapCorruption",
    "HeapHistory",
    "HeapSnapshotError",
    "ArenaLayoutResolver",
    "TcacheLayoutResolver",
    "HeapTools",
    "ALLOCATOR_DETECTION_SCHEMA",
    "ALLOCATOR_CAPABILITIES_SCHEMA",
    "CHUNK_INSPECTION_SCHEMA",
    "CHUNK_VALIDATION_SCHEMA",
    "ARENA_LIST_SCHEMA",
    "ARENA_INSPECTION_SCHEMA",
    "FASTBIN_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_FASTBIN",
    "TCACHE_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_TCACHE_BIN",
    "TCACHE_MAX_BINS",
    "BIN_INSPECTION_SCHEMA",
    "DEFAULT_MAX_CHUNKS_PER_REGULAR_BIN",
    "DEFAULT_SLUB_PAGE_SIZE",
    "HEAP_CORRUPTION_SCHEMA",
    "HEAP_TECHNIQUE_SCHEMA",
    "UAF_SCAN_SCHEMA",
    "HEAP_TECHNIQUE_CATALOG_SCHEMA",
    "FREE_VALIDATION_SCHEMA",
    "HEAP_SNAPSHOT_SCHEMA",
    "HEAP_SNAPSHOT_LIST_SCHEMA",
    "HEAP_SNAPSHOT_DIFF_SCHEMA",
    "HEAP_SNAPSHOT_LINK_SCHEMA",
    "DEFAULT_MAX_HEAP_SNAPSHOTS",
    "DEFAULT_MAX_HEAP_SNAPSHOT_BYTES",
    "SAFE_LINKING_MIN_VERSION",
    "encode_safe_linked",
    "MUSL_MALLOCNG_MIN_VERSION",
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


# PEP 562 lazy re-export.  The S6.6 schema constants live in heap_techniques,
# which imports this module at its top; a module-level ``from
# .heap_techniques import ...`` here would create an import cycle.  Keeping the
# names in __all__ and resolving them on attribute access preserves the
# ``from paomcp.tools.backends.heap import HEAP_TECHNIQUE_SCHEMA`` spelling.
def __getattr__(name: str) -> Any:
    if name in (
        "HEAP_TECHNIQUE_SCHEMA",
        "UAF_SCAN_SCHEMA",
        "HEAP_TECHNIQUE_CATALOG_SCHEMA",
    ):
        from .heap_techniques import (  # noqa: PLC0415
            HEAP_TECHNIQUE_CATALOG_SCHEMA,
            HEAP_TECHNIQUE_SCHEMA,
            UAF_SCAN_SCHEMA,
        )

        return {
            "HEAP_TECHNIQUE_SCHEMA": HEAP_TECHNIQUE_SCHEMA,
            "UAF_SCAN_SCHEMA": UAF_SCAN_SCHEMA,
            "HEAP_TECHNIQUE_CATALOG_SCHEMA": HEAP_TECHNIQUE_CATALOG_SCHEMA,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
