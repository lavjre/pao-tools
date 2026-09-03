"""Reusable primitives shared across typed backend research stages.

This module holds the safety and shape invariants that Stage 5 (crash triage)
established and that later stages (S6 heap analysis, S7 ROP/libc, S8 target
providers) must reuse rather than re-implement.  Keeping a single copy of each
invariant prevents the codebase from forking into several slightly different
definitions of "a valid response", "a live inferior", or "parse this integer".

Nothing here talks to GDB directly; every helper operates on already-collected
values so it stays trivially unit-testable without a debugger.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from paomcp.artifacts import WriterNotQuiescentError

DEFAULT_PARSER_VERSION = "1"
_HEX_VALUE_PATTERN = re.compile(r"0x[0-9a-fA-F]+")
# A bare hex integer with no 0x prefix ("40113a").  Only consulted when
# base-auto-detection fails, so decimal-looking values never reach it.
_BARE_HEX_PATTERN = re.compile(r"^-?[0-9a-fA-F]+$")


def build_backend_response(
    command: str,
    *,
    parsed: Any,
    raw_responses: Any,
    state: str,
    success: bool = True,
    parser_version: str = DEFAULT_PARSER_VERSION,
    warnings: Optional[Sequence[str]] = None,
    error: Optional[str] = None,
    error_type: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical inner response envelope emitted by every backend.

    The public MCP envelope (``ToolResponse`` in :mod:`paomcp.api.models`) wraps
    this shape; this is the *inner* dict every typed backend method returns:
    ``{success, command, state, parser_version, parsed, raw_responses}`` plus an
    optional ``warnings`` list and, on failure, ``error``/``type``.

    Payload is never truncated.  ``extra`` merges additional top-level keys
    (for example ``usable``/``complete`` on crash snapshots) without letting a
    caller silently override the core contract keys.
    """
    response: Dict[str, Any] = {
        "success": success,
        "command": command,
        "state": state,
        "parser_version": parser_version,
        "parsed": parsed,
        "raw_responses": raw_responses,
    }
    if extra:
        for key, value in extra.items():
            if key in response:
                raise ValueError(f"extra cannot override reserved envelope key: {key}")
            response[key] = value
    normalized_warnings = [str(item) for item in warnings] if warnings else []
    if normalized_warnings:
        response["warnings"] = normalized_warnings
    if not success:
        response["error"] = error or "Backend operation failed"
        response["type"] = error_type or "BackendError"
    return response


def parse_int(value: Any) -> Optional[int]:
    """Parse an integer from GDB/MI text, tolerating ``0x`` and decorated values.

    Order matters: an embedded ``0x...`` token wins (GDB frequently renders
    values as ``(void *) 0x40113a``), otherwise the whole value is parsed with
    base auto-detection.  Returns ``None`` when nothing usable is present rather
    than raising, because missing register/pointer values are an expected,
    non-fatal condition for conservative analysis.
    """
    if value is None:
        return None
    match = _HEX_VALUE_PATTERN.search(str(value))
    if match is not None:
        return int(match.group(0), 16)
    text = str(value).strip()
    try:
        return int(text, 0)
    except (TypeError, ValueError):
        # GDB / crash tooling sometimes renders pointers as plain bare hex
        # with no 0x prefix.  Accept the WHOLE value as hex only when base
        # auto-detection failed -- an unambiguous decimal ("12345") parses
        # above, so this fallback can only ever see strings containing a-f.
        if _BARE_HEX_PATTERN.match(text) is not None:
            return int(text, 16)
        return None


def mapping_for(
    value: Optional[int], mappings: Sequence[Any]
) -> Optional[Dict[str, Any]]:
    """Return the first memory mapping whose ``[start, end)`` contains ``value``.

    Shared by crash-fault attribution today and by heap/ROP address attribution
    in later stages; the containment rule must stay identical everywhere.
    """
    if value is None:
        return None
    for item in mappings:
        if not isinstance(item, dict):
            continue
        try:
            start = int(str(item.get("start")), 16)
            end = int(str(item.get("end")), 16)
        except (TypeError, ValueError):
            continue
        if start <= value < end:
            return dict(item)
    return None


class LiveInferiorError(RuntimeError):
    """Raised when a stop record does not match the current live inferior."""


class LiveInferiorGuard:
    """Enforce that an operation targets the exact current stopped live inferior.

    Stage 5 learned that a crash id must never be honored once the process has
    resumed, exited, or been replaced.  Heap and ROP snapshots need the same
    rule, so the check lives here.  The guard is intentionally tiny and pure:
    it takes a stop ``record`` (already copied out of the controller) and the
    controller's current inferior pid, and decides liveness from
    ``record.is_live`` plus a present pid.
    """

    @staticmethod
    def is_live(record: Optional[Dict[str, Any]], inferior_pid: Optional[int]) -> bool:
        """Return whether ``record`` still describes the attached live inferior."""
        if not isinstance(record, dict):
            return False
        return bool(record.get("is_live")) and inferior_pid is not None

    @classmethod
    def require_live(
        cls,
        record: Optional[Dict[str, Any]],
        inferior_pid: Optional[int],
        *,
        detail: str = "The stop record is no longer attached to a live inferior",
    ) -> Dict[str, Any]:
        """Return ``record`` when live, else raise :class:`LiveInferiorError`."""
        if not cls.is_live(record, inferior_pid):
            raise LiveInferiorError(detail)
        assert record is not None
        return record


def merge_warnings(
    response: Dict[str, Any], warnings: Optional[Sequence[str]]
) -> Dict[str, Any]:
    """Append ``warnings`` to a response's warning list without duplicating keys."""
    if not warnings:
        return response
    existing: List[str] = list(response.get("warnings", []))
    existing.extend(str(item) for item in warnings)
    response["warnings"] = existing
    return response


def validate_attach_pid(pid: Any) -> int:
    """Validate a PID before any debugger/gdbserver attach.

    Rejects PID 0/1 (kernel/init — attaching destabilizes the host),
    our own supervisor process, and our parent process. Non-integer
    inputs are rejected; bools are rejected explicitly (``True == 1``).
    Returns the normalized int on success, raises ValueError otherwise.
    Every attach path (GDB, LLDB, attach_pid, gdbserver, /attach) must
    funnel through here instead of reimplementing the blocklist.
    """
    if isinstance(pid, bool) or not isinstance(pid, int):
        raise ValueError(f"attach pid must be an integer, got {pid!r}")
    if pid <= 1:
        raise ValueError(
            f"Refusing to attach to PID {pid}: PID 0/1 would destabilize the host"
        )
    if pid == os.getpid():
        raise ValueError("Refusing to attach to our own debugger supervisor process")
    try:
        parent = os.getppid()
    except Exception:
        parent = None
    if parent is not None and pid == parent:
        raise ValueError("Refusing to attach to our parent process")
    return pid


__all__ = [
    "DEFAULT_PARSER_VERSION",
    "build_backend_response",
    "parse_int",
    "mapping_for",
    "LiveInferiorError",
    "WriterNotQuiescentError",
    "LiveInferiorGuard",
    "merge_warnings",
    "validate_attach_pid",
]
