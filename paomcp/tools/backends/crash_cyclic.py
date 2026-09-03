# mypy: disable-error-code="attr-defined"
"""De Bruijn cyclic-pattern engine (split from crash.py)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from elftools.elf.sections import NoteSection
from pwnlib.util.cyclic import de_bruijn  # type: ignore[import-untyped]

from paomcp.artifacts import (
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    WriterNotQuiescentError,
)
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorError,
    LiveInferiorGuard,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

import logging

logger = logging.getLogger(__name__)

from paomcp.tools.backends.crash_schemas import (
    PARSER_VERSION,
    CRASH_SNAPSHOT_VERSION,
    DEFAULT_FATAL_SIGNALS,
    SAFE_SIGNAL_PATTERN,
    MAX_RUN_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    DEFAULT_CORE_TIMEOUT_SECONDS,
    DEFAULT_MAX_CORE_DUMP_BYTES,
    MAX_CONFIGURED_CORE_DUMP_BYTES,
    MAX_STACK_BYTES,
    MAX_CYCLIC_LENGTH,
    MAX_CYCLIC_STACK_SCAN_BYTES,
    MAX_CYCLIC_SCAN_MATCHES,
    MAX_CYCLIC_VALUE_CHARS,
    MAX_CYCLIC_ALPHABET_CHARS,
    MAX_ELF_CORE_NOTE_ENTRIES,
    MAX_CRASH_HISTORY_ENTRIES,
    MAX_CRASH_HISTORY_BYTES,
    CRASH_CLASSIFICATION_SCHEMA,
    CRASH_FINGERPRINT_SCHEMA,
    CRASH_COMPARISON_SCHEMA,
)


def _record_error(message: str) -> Exception:
    """Build CrashRecordError with a deferred import (crash.py owns it).

    A top-level import would cycle: crash.py imports these mixins for the
    CrashTriage bases. Resolving at call time is always safe because the
    class (and its module) is fully loaded before any method runs.
    """

    from paomcp.tools.backends.crash import CrashRecordError  # noqa: PLC0415

    return CrashRecordError(message)


class CrashCyclicMixin:
    """Mixin supplying CrashTriage.cyclic behavior; never instantiated alone."""

    @staticmethod
    def _cyclic_parameters(
        *,
        n: int,
        endian: str,
        max_length: int,
        alphabet: str,
    ) -> tuple[bytes, int, int]:
        if not 2 <= n <= 8:
            raise ValueError("n must be between 2 and 8")
        if endian not in {"little", "big"}:
            raise ValueError("endian must be 'little' or 'big'")
        if not 1 <= max_length <= MAX_CYCLIC_LENGTH:
            raise ValueError(f"max_length must be between 1 and {MAX_CYCLIC_LENGTH}")
        if not isinstance(alphabet, str):
            raise ValueError("alphabet must be a string")
        if not 2 <= len(alphabet) <= MAX_CYCLIC_ALPHABET_CHARS:
            raise ValueError("alphabet must contain 2-256 unique ASCII characters")
        alphabet_bytes = alphabet.encode("ascii", errors="strict")
        if not 2 <= len(alphabet_bytes) <= 256 or len(set(alphabet_bytes)) != len(
            alphabet_bytes
        ):
            raise ValueError("alphabet must contain 2-256 unique ASCII characters")
        pattern_space = len(alphabet_bytes) ** n
        searched_length = min(max_length, pattern_space)
        return alphabet_bytes, pattern_space, searched_length

    @staticmethod
    def _build_cyclic_pattern(
        alphabet_bytes: bytes,
        n: int,
        searched_length: int,
        cancellation_check: Optional[Callable[[], None]],
    ) -> bytes:
        """Generate a bounded pattern with cooperative cancellation points."""
        if cancellation_check is not None:
            cancellation_check()
        pattern = bytearray()
        for index, item in enumerate(de_bruijn(alphabet_bytes, n)):
            if index >= searched_length:
                break
            pattern.append(int(item))
            if cancellation_check is not None and not index % (64 * 1024):
                cancellation_check()
        if cancellation_check is not None:
            cancellation_check()
        return bytes(pattern)

    @staticmethod
    def _decode_cyclic_value(
        value: str,
        *,
        endian: str,
        width: Optional[int],
    ) -> tuple[bytes, str]:
        if not isinstance(value, str):
            raise ValueError("value must be a string")
        if not 1 <= len(value) <= MAX_CYCLIC_VALUE_CHARS:
            raise ValueError(
                f"value must contain between 1 and {MAX_CYCLIC_VALUE_CHARS} characters"
            )
        raw_value = value.strip()
        if not raw_value:
            raise ValueError("value cannot be empty")
        input_kind = "text"
        if raw_value.lower().startswith("hex:"):
            hex_value = raw_value[4:]
            if len(hex_value) % 2:
                hex_value = "0" + hex_value
            query = bytes.fromhex(hex_value)
            input_kind = "hex-bytes"
        elif raw_value.lower().startswith("text:"):
            query = raw_value[5:].encode("utf-8")
        elif raw_value.lower().startswith("0x"):
            integer = int(raw_value, 16)
            inferred_width = max(1, (integer.bit_length() + 7) // 8)
            resolved_width = inferred_width if width is None else width
            if not 1 <= resolved_width <= 64:
                raise ValueError("width must be between 1 and 64")
            if integer >= 1 << (resolved_width * 8):
                raise ValueError(
                    f"integer value does not fit in the requested {resolved_width} bytes"
                )
            integer_byteorder: Literal["little", "big"] = (
                "little" if endian == "little" else "big"
            )
            query = integer.to_bytes(
                resolved_width, byteorder=integer_byteorder, signed=False
            )
            input_kind = "integer"
        else:
            query = raw_value.encode("utf-8")
        if not query or len(query) > 64:
            raise ValueError("decoded value must contain between 1 and 64 bytes")
        return query, input_kind

    @staticmethod
    def _search_cyclic_query(
        pattern: bytes,
        query: bytes,
        *,
        input_kind: str,
        n: int,
        endian: str,
        cancellation_check: Optional[Callable[[], None]],
    ) -> List[Dict[str, Any]]:
        matches: List[Dict[str, Any]] = []
        if cancellation_check is not None:
            cancellation_check()
        if len(query) < n:
            # A de Bruijn sequence only guarantees uniqueness for n-byte windows.
            # Enumerate bounded candidates for shorter evidence and never present
            # the first hit as a unique/exact offset.
            start = 0
            while len(matches) <= MAX_CYCLIC_SCAN_MATCHES:
                offset = pattern.find(query, start)
                if offset < 0:
                    break
                matches.append(
                    {
                        "kind": "partial_candidate",
                        "offset": offset,
                        "query_hex": query.hex(),
                        "byte_count": len(query),
                    }
                )
                start = offset + 1
                if cancellation_check is not None and not len(matches) % 32:
                    cancellation_check()
            return matches

        exact_offset = pattern.find(query)
        if exact_offset >= 0:
            matches.append(
                {
                    "kind": "exact",
                    "offset": exact_offset,
                    "query_hex": query.hex(),
                    "byte_count": len(query),
                }
            )
        if len(query) != n:
            windows: List[tuple[str, bytes]] = []
            if len(query) > n:
                # Label by byte position (unambiguous regardless of endian).
                leading = query[:n]
                trailing = query[-n:]
                windows.extend([("leading_n", leading), ("trailing_n", trailing)])
                # For integer input the caller usually wants the low/high-order
                # window.  Which byte position that maps to depends on endian:
                # little-endian puts the least-significant byte first, big-endian
                # puts it last.  Only add these labels for integer input so text
                # or raw-byte queries are not mislabeled as having an "order".
                if input_kind == "integer":
                    if endian == "little":
                        windows.extend(
                            [("low_order_n", leading), ("high_order_n", trailing)]
                        )
                    else:
                        windows.extend(
                            [("low_order_n", trailing), ("high_order_n", leading)]
                        )
            seen = set()
            for kind, candidate in windows:
                if cancellation_check is not None:
                    cancellation_check()
                offset = pattern.find(candidate)
                identity = (kind, offset, candidate)
                if offset >= 0 and identity not in seen:
                    seen.add(identity)
                    matches.append(
                        {
                            "kind": kind,
                            "offset": offset,
                            "query_hex": candidate.hex(),
                            "byte_count": len(candidate),
                        }
                    )
        return matches

    @staticmethod
    def find_cyclic_offset(
        value: str,
        *,
        n: int = 4,
        endian: str = "little",
        width: Optional[int] = None,
        max_length: int = 1024 * 1024,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Find exact and n-byte cyclic-pattern offsets without silent truncation."""
        alphabet_bytes, pattern_space, searched_length = (
            CrashCyclicMixin._cyclic_parameters(
                n=n,
                endian=endian,
                max_length=max_length,
                alphabet=alphabet,
            )
        )
        query, input_kind = CrashCyclicMixin._decode_cyclic_value(
            value, endian=endian, width=width
        )
        pattern = CrashCyclicMixin._build_cyclic_pattern(
            alphabet_bytes, n, searched_length, cancellation_check
        )
        matches = CrashCyclicMixin._search_cyclic_query(
            pattern,
            query,
            input_kind=input_kind,
            n=n,
            endian=endian,
            cancellation_check=cancellation_check,
        )
        candidates_truncated = len(matches) > MAX_CYCLIC_SCAN_MATCHES
        if candidates_truncated:
            matches = matches[:MAX_CYCLIC_SCAN_MATCHES]
        ambiguous = len(query) < n and bool(matches)
        primary = matches[0] if matches and not ambiguous else None
        return {
            "success": True,
            "command": "find_cyclic_offset",
            "state": "local",
            "parser_version": PARSER_VERSION,
            "parsed": {
                "found": bool(matches),
                "offset": primary.get("offset") if primary else None,
                "match_kind": (
                    "ambiguous"
                    if ambiguous
                    else (primary.get("kind") if primary else None)
                ),
                "resolution": (
                    "ambiguous"
                    if ambiguous
                    else ("unique_in_searched_pattern" if primary else "not_found")
                ),
                "ambiguous": ambiguous,
                "candidate_count": len(matches),
                "candidates_truncated": candidates_truncated,
                "matches": matches,
                "input": {
                    "value": value,
                    "kind": input_kind,
                    "query_hex": query.hex(),
                    "endian": endian,
                    "width": len(query),
                },
                "pattern": {
                    "n": n,
                    "alphabet": alphabet,
                    "space": pattern_space,
                    "searched_length": searched_length,
                    "search_complete": searched_length == pattern_space,
                },
                "source": "local",
            },
            "raw_responses": [],
        }

    def find_cyclic_offsets_for_crash(
        self,
        expected_crash_id: str,
        *,
        value: Optional[str] = None,
        n: int = 4,
        endian: str = "little",
        width: Optional[int] = None,
        max_length: int = 1024 * 1024,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz",
        scan_registers: bool = True,
        scan_stack: bool = True,
        max_stack_scan_bytes: int = 4096,
        max_matches: int = 64,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Scan bounded retained register/stack evidence for cyclic values."""
        if not scan_registers and not scan_stack and value is None:
            raise ValueError(
                "Provide value or enable at least one crash evidence scan source"
            )
        if not 0 <= max_stack_scan_bytes <= MAX_CYCLIC_STACK_SCAN_BYTES:
            raise ValueError(
                "max_stack_scan_bytes must be between 0 and "
                f"{MAX_CYCLIC_STACK_SCAN_BYTES}"
            )
        if not 1 <= max_matches <= MAX_CYCLIC_SCAN_MATCHES:
            raise ValueError(
                f"max_matches must be between 1 and {MAX_CYCLIC_SCAN_MATCHES}"
            )
        snapshot = self._history_snapshot(expected_crash_id)
        if snapshot is None:
            # The exact current live crash can still be captured lazily. Historical
            # crashes must already have an immutable retained snapshot.
            record = self.gdb.get_last_stop_record()
            if (
                record is None
                or self._crash_id(record) != expected_crash_id
                or not LiveInferiorGuard.is_live(record, self.gdb.get_inferior_pid())
            ):
                raise _record_error(
                    f"Crash snapshot is no longer retained: {expected_crash_id}"
                )
            captured = self.capture_crash_snapshot(
                expected_crash_id,
                cancellation_check=cancellation_check,
            )
            snapshot = captured.get("parsed", {}).get("crash_snapshot")
        if not isinstance(snapshot, dict):
            raise _record_error(f"Crash snapshot is unavailable: {expected_crash_id}")
        alphabet_bytes, pattern_space, searched_length = self._cyclic_parameters(
            n=n,
            endian=endian,
            max_length=max_length,
            alphabet=alphabet,
        )
        pattern = self._build_cyclic_pattern(
            alphabet_bytes, n, searched_length, cancellation_check
        )
        matches: List[Dict[str, Any]] = []
        explicit_result = None
        if value is not None:
            query, input_kind = self._decode_cyclic_value(
                value, endian=endian, width=width
            )
            explicit_matches = self._search_cyclic_query(
                pattern,
                query,
                input_kind=input_kind,
                n=n,
                endian=endian,
                cancellation_check=cancellation_check,
            )
            explicit_candidates_truncated = (
                len(explicit_matches) > MAX_CYCLIC_SCAN_MATCHES
            )
            if explicit_candidates_truncated:
                explicit_matches = explicit_matches[:MAX_CYCLIC_SCAN_MATCHES]
            explicit_ambiguous = len(query) < n and bool(explicit_matches)
            explicit_result = {
                "value": value,
                "kind": input_kind,
                "query_hex": query.hex(),
                "resolution": (
                    "ambiguous"
                    if explicit_ambiguous
                    else (
                        "unique_in_searched_pattern"
                        if explicit_matches
                        else "not_found"
                    )
                ),
                "ambiguous": explicit_ambiguous,
                "candidate_count": len(explicit_matches),
                "candidates_truncated": explicit_candidates_truncated,
                "matches": explicit_matches,
            }
            matches.extend(
                {**match, "evidence": {"kind": "explicit_value"}}
                for match in explicit_matches[:max_matches]
            )

        debugger_snapshot = snapshot.get("debugger", {})
        if not isinstance(debugger_snapshot, dict):
            debugger_snapshot = {}
        if scan_registers and len(matches) < max_matches:
            registers = debugger_snapshot.get("registers", [])
            if isinstance(registers, list):
                for register in registers[:512]:
                    if cancellation_check is not None:
                        cancellation_check()
                    if not isinstance(register, dict) or register.get("value") is None:
                        continue
                    integer = self._integer(register.get("value"))
                    if integer is None or integer < 0:
                        continue
                    register_value = f"0x{integer:x}"
                    # ``width`` describes the caller-supplied value. Applying it
                    # to every architectural register would make one wide register
                    # abort the whole scan when the requested width is smaller.
                    register_width = max(1, (integer.bit_length() + 7) // 8)
                    query, input_kind = self._decode_cyclic_value(
                        register_value,
                        endian=endian,
                        width=register_width,
                    )
                    register_matches = self._search_cyclic_query(
                        pattern,
                        query,
                        input_kind=input_kind,
                        n=n,
                        endian=endian,
                        cancellation_check=cancellation_check,
                    )
                    for match in register_matches:
                        matches.append(
                            {
                                **match,
                                "evidence": {
                                    "kind": "register",
                                    "name": register.get("name"),
                                    "value": register.get("value"),
                                    "source": "gdb-mi",
                                },
                            }
                        )
                        if len(matches) >= max_matches:
                            break
                    if len(matches) >= max_matches:
                        break

        stack = snapshot.get("stack", {})
        stack_scanned_bytes = 0
        if (
            scan_stack
            and len(matches) < max_matches
            and isinstance(stack, dict)
            and isinstance(stack.get("data_hex"), str)
            and max_stack_scan_bytes
        ):
            try:
                stack_bytes = bytes.fromhex(stack["data_hex"])[:max_stack_scan_bytes]
            except ValueError:
                stack_bytes = b""
            stack_scanned_bytes = len(stack_bytes)
            candidate_offsets: Dict[bytes, List[int]] = {}
            alphabet_set = set(alphabet_bytes)
            for stack_offset in range(max(0, len(stack_bytes) - n + 1)):
                if cancellation_check is not None and not stack_offset % (64 * 1024):
                    cancellation_check()
                candidate = stack_bytes[stack_offset : stack_offset + n]
                if all(byte in alphabet_set for byte in candidate):
                    candidate_offsets.setdefault(candidate, []).append(stack_offset)
            found_pattern_offsets: Dict[bytes, int] = {}
            if candidate_offsets:
                for pattern_offset in range(max(0, len(pattern) - n + 1)):
                    if cancellation_check is not None and not pattern_offset % (
                        64 * 1024
                    ):
                        cancellation_check()
                    candidate = pattern[pattern_offset : pattern_offset + n]
                    if (
                        candidate in candidate_offsets
                        and candidate not in found_pattern_offsets
                    ):
                        found_pattern_offsets[candidate] = pattern_offset
                        if len(found_pattern_offsets) == len(candidate_offsets):
                            break
            stack_base = self._integer(stack.get("address"))
            for candidate, offsets in candidate_offsets.items():
                found_offset = found_pattern_offsets.get(candidate)
                if found_offset is None:
                    continue
                for stack_offset in offsets:
                    matches.append(
                        {
                            "kind": "stack_window",
                            "offset": found_offset,
                            "query_hex": candidate.hex(),
                            "byte_count": len(candidate),
                            "evidence": {
                                "kind": "stack",
                                "stack_offset": stack_offset,
                                "address": (
                                    f"0x{stack_base + stack_offset:x}"
                                    if stack_base is not None
                                    else None
                                ),
                                "source": "gdb-mi",
                            },
                        }
                    )
                    if len(matches) >= max_matches:
                        break
                if len(matches) >= max_matches:
                    break
        if cancellation_check is not None:
            cancellation_check()
        unambiguous_matches = [
            match for match in matches if match.get("kind") != "partial_candidate"
        ]
        primary = unambiguous_matches[0] if unambiguous_matches else None
        ambiguous = bool(matches) and primary is None
        return {
            "success": True,
            "command": "find_cyclic_offset",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "found": bool(matches),
                "offset": primary.get("offset") if primary else None,
                "match_kind": (
                    primary.get("kind")
                    if primary
                    else ("ambiguous" if ambiguous else None)
                ),
                "resolution": (
                    "unique_in_searched_pattern"
                    if primary
                    else ("ambiguous" if ambiguous else "not_found")
                ),
                "ambiguous": ambiguous,
                "matches": matches,
                "input": explicit_result,
                "crash_scan": {
                    "crash_id": expected_crash_id,
                    "registers_scanned": scan_registers,
                    "stack_scanned": scan_stack,
                    "stack_scanned_bytes": stack_scanned_bytes,
                    "max_matches": max_matches,
                    "truncated": len(matches) >= max_matches,
                },
                "pattern": {
                    "n": n,
                    "alphabet": alphabet,
                    "space": pattern_space,
                    "searched_length": searched_length,
                    "search_complete": searched_length == pattern_space,
                },
                "source": "local-over-typed-crash-snapshot",
            },
            "raw_responses": [],
        }
