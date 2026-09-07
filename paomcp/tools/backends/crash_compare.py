# mypy: disable-error-code="attr-defined"
"""Crash comparison (split from crash.py)."""

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


class CrashCompareMixin:
    """Mixin supplying CrashTriage.compare behavior; never instantiated alone."""

    @staticmethod
    def _comparable_value(value: Any) -> bool:
        if value is None or value == "" or value == []:
            return False
        if isinstance(value, dict) and value.get("kind") in {"unknown", "unmapped"}:
            return False
        return True

    @classmethod
    def _comparison_fields(
        cls, left: Dict[str, Any], right: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        left_canonical = left.get("signature", {}).get("canonical", {})
        right_canonical = right.get("signature", {}).get("canonical", {})
        left_instruction = left_canonical.get("instruction", {})
        right_instruction = right_canonical.get("instruction", {})
        left_frames = left_canonical.get("frames", [])
        right_frames = right_canonical.get("frames", [])
        fields = (
            (
                "target_identity",
                left_canonical.get("target_identity"),
                right_canonical.get("target_identity"),
            ),
            ("signal", left_canonical.get("signal"), right_canonical.get("signal")),
            (
                "category",
                left_canonical.get("category"),
                right_canonical.get("category"),
            ),
            (
                "primitive",
                left_canonical.get("primitive"),
                right_canonical.get("primitive"),
            ),
            (
                "program_counter",
                left_canonical.get("program_counter"),
                right_canonical.get("program_counter"),
            ),
            (
                "fault_address",
                left_canonical.get("fault_address"),
                right_canonical.get("fault_address"),
            ),
            (
                "instruction.mnemonic",
                (
                    left_instruction.get("mnemonic")
                    if isinstance(left_instruction, dict)
                    else None
                ),
                (
                    right_instruction.get("mnemonic")
                    if isinstance(right_instruction, dict)
                    else None
                ),
            ),
            (
                "instruction.function",
                (
                    left_instruction.get("function")
                    if isinstance(left_instruction, dict)
                    else None
                ),
                (
                    right_instruction.get("function")
                    if isinstance(right_instruction, dict)
                    else None
                ),
            ),
            (
                # Full normalized stack (capped at 5 upstream): two crashes
                # sharing only a libc top frame (raise/abort) but diverging
                # below it are different families, not the same bug.
                "stack_top5",
                (
                    left_frames[:5]
                    if isinstance(left_frames, list) and left_frames
                    else None
                ),
                (
                    right_frames[:5]
                    if isinstance(right_frames, list) and right_frames
                    else None
                ),
            ),
        )
        evidence: List[Dict[str, Any]] = []
        for field, left_value, right_value in fields:
            comparable = cls._comparable_value(left_value) and cls._comparable_value(
                right_value
            )
            evidence.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                    "status": (
                        "match"
                        if comparable and left_value == right_value
                        else ("different" if comparable else "unknown")
                    ),
                    "comparable": comparable,
                }
            )
        return evidence

    def compare_crashes(
        self,
        left_crash_id: str,
        right_crash_id: str,
        *,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Compare two retained crash descriptors using ASLR-stable fields."""
        if not left_crash_id or not right_crash_id:
            raise ValueError("left_crash_id and right_crash_id cannot be empty")
        if cancellation_check is not None:
            cancellation_check()
        left = self._history_descriptor(left_crash_id)
        right = self._history_descriptor(right_crash_id)
        missing = [
            crash_id
            for crash_id, descriptor in (
                (left_crash_id, left),
                (right_crash_id, right),
            )
            if descriptor is None
        ]
        if missing:
            raise _record_error(
                "Crash descriptor is no longer retained: " + ", ".join(missing)
            )
        assert left is not None and right is not None
        evidence = self._comparison_fields(left, right)
        if cancellation_check is not None:
            cancellation_check()
        left_signature = left.get("signature", {})
        right_signature = right.get("signature", {})
        digest_equal = left_signature.get("digest") and left_signature.get(
            "digest"
        ) == right_signature.get("digest")
        statuses = {item["field"]: item["status"] for item in evidence}
        pc_values = {
            item["field"]: (item.get("left"), item.get("right")) for item in evidence
        }.get("program_counter", (None, None))
        stable_pc_difference = statuses.get("program_counter") == "different" and all(
            isinstance(value, dict) and value.get("kind") == "module_offset"
            for value in pc_values
        )
        target_identity_difference = statuses.get("target_identity") == "different"
        match_count = sum(item["status"] == "match" for item in evidence)
        differing = [
            item["field"] for item in evidence if item["status"] == "different"
        ]
        if (
            target_identity_difference
            or statuses.get("signal") == "different"
            or stable_pc_difference
        ):
            relationship = "different"
            confidence = "high"
            reasons = [
                "A stable crash-family field differs: "
                + (
                    "target identity"
                    if target_identity_difference
                    else (
                        "signal"
                        if statuses.get("signal") == "different"
                        else "module-relative program counter"
                    )
                )
            ]
        elif digest_equal and left_signature.get("confidence") in {"high", "medium"}:
            relationship = "same"
            confidence = str(left_signature.get("confidence"))
            reasons = ["ASLR-normalized crash signatures match"]
        elif (
            statuses.get("signal") == "match"
            and statuses.get("program_counter") == "match"
            and match_count >= 3
        ):
            relationship = "same"
            confidence = "medium"
            reasons = ["Signal and module-relative program counter match"]
        elif len(differing) >= 2 and statuses.get("signal") != "unknown":
            relationship = "different"
            confidence = "medium"
            reasons = ["Multiple comparable crash-family fields differ"]
        else:
            relationship = "unknown"
            confidence = "low"
            reasons = [
                "Retained evidence lacks enough stable matching or conflicting fields"
            ]
        return {
            "success": True,
            "complete": relationship != "unknown",
            "command": "compare_crashes",
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "schema": CRASH_COMPARISON_SCHEMA,
                "left_crash_id": left_crash_id,
                "right_crash_id": right_crash_id,
                "relationship": relationship,
                "same_crash_family": (
                    True
                    if relationship == "same"
                    else (False if relationship == "different" else None)
                ),
                "confidence": confidence,
                "reasons": reasons,
                "field_evidence": evidence,
                "left_signature": left_signature,
                "right_signature": right_signature,
                "source": "local",
            },
            "raw_responses": [],
        }
