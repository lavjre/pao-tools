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

    def _target_profile(self, mappings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
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

    def _collect_evidence(
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
        mappings, libcinfo_parsed, component_status = self._collect_evidence(
            cancellation_check
        )
        classification = self.classify(mappings, libcinfo_parsed)
        profile = self._target_profile(mappings)

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
        mappings, libcinfo_parsed, component_status = self._collect_evidence(
            cancellation_check
        )
        classification = self.classify(mappings, libcinfo_parsed)
        profile = self._target_profile(mappings)

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


__all__ = [
    "AllocatorDiscovery",
    "ALLOCATOR_DETECTION_SCHEMA",
    "ALLOCATOR_CAPABILITIES_SCHEMA",
    "SAFE_LINKING_MIN_VERSION",
    "ALLOCATOR_RUNTIME_COMMANDS",
]
