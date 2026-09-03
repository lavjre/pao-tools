"""Typed Linux kernel oops/panic triage from log text (Phase 3).

Mirrors the userland crash pipeline's contract (classify -> fingerprint ->
dedup) but over kernel log text instead of a live inferior: no debugger is
required, so this runs on pasted dmesg/serial output. Every conclusion
carries the exact log lines it was derived from, and anything unrecognized
is reported as ``unknown`` rather than guessed.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from paomcp.tools.backends.foundation import build_backend_response

logger = logging.getLogger(__name__)

KERNEL_OOPS_SCHEMA = "pao.kernel-oops.v1"
PARSER_VERSION = "1"
MAX_LOG_BYTES = 1024 * 1024
MAX_TRACE_FRAMES = 128

_FAULT_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        "null_deref",
        re.compile(
            r"unable to handle kernel (NULL pointer dereference|paging request)",
            re.IGNORECASE,
        ),
    ),
    ("null_deref", re.compile(r"NULL pointer dereference", re.IGNORECASE)),
    ("kasan_uaf", re.compile(r"KASAN:\s*use-after-free", re.IGNORECASE)),
    (
        "kasan_oob",
        re.compile(
            r"KASAN:\s*(?:slab-out-of-bounds|out-of-bounds|global-out-of-bounds|stack-out-of-bounds)",
            re.IGNORECASE,
        ),
    ),
    ("kasan", re.compile(r"KASAN:", re.IGNORECASE)),
    ("general_protection", re.compile(r"general protection fault", re.IGNORECASE)),
    ("divide_error", re.compile(r"divide error", re.IGNORECASE)),
    ("stack_segment", re.compile(r"stack-segment fault", re.IGNORECASE)),
    ("page_fault", re.compile(r"#PF:|page fault", re.IGNORECASE)),
    ("oops", re.compile(r"^Oops:|\bOops\b", re.IGNORECASE | re.MULTILINE)),
    ("bug", re.compile(r"^BUG:", re.IGNORECASE | re.MULTILINE)),
    ("warning", re.compile(r"^WARNING:", re.IGNORECASE | re.MULTILINE)),
)

_FAULT_ADDRESS_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"at\s+((?:0x)?[0-9a-fA-F]{4,}|\(null\)|0000+#\d+)", re.IGNORECASE),
    re.compile(r"address\s+((?:0x)?[0-9a-fA-F]{4,})", re.IGNORECASE),
    re.compile(r"Write of size \d+ at addr\s+((?:0x)?[0-9a-fA-F]{4,})", re.IGNORECASE),
    re.compile(r"Read of size \d+ at addr\s+((?:0x)?[0-9a-fA-F]{4,})", re.IGNORECASE),
)

_RIP_PATTERN = re.compile(
    r"RIP:\s*(?:[0-9a-fA-F]{4}:)?([A-Za-z_][\w.]*)\+0x([0-9a-fA-F]+)/0x([0-9a-fA-F]+)"
)
_RIP_BARE_PATTERN = re.compile(r"RIP:\s*[0-9a-fA-F]{4}:\[<([0-9a-fA-F]+)\>\]")
_CALL_TRACE_PATTERN = re.compile(
    r"^\s*(?:\[\s*\d+\.\d+\]\s*)?([A-Za-z_][\w.]*)\+0x[0-9a-fA-F]+/0x[0-9a-fA-F]+",
    re.MULTILINE,
)
_CPU_PATTERN = re.compile(r"CPU:\s*(\d+)\s+PID:\s*(\d+)\s+Comm:\s*(\S+)")
_TAINT_PATTERN = re.compile(r"Tainted:\s*([A-Z ]+?)(?:\s+[A-Z][a-z]+|$)", re.MULTILINE)
_OOPS_CODE_PATTERN = re.compile(r"Oops:\s*([0-9a-fA-F]+)\s*\[#(\d+)\]")
_SLAB_PATTERN = re.compile(
    r"(?:cache|slab)\s+([A-Za-z0-9_\-]+)(?:\s+.*?(?:object|size))?", re.IGNORECASE
)
_KASAN_ACCESS_PATTERN = re.compile(r"(Read|Write) of size (\d+)", re.IGNORECASE)


def _first_group_match(
    patterns: Sequence[re.Pattern], text: str, group: int = 1
) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            try:
                return match.group(group)
            except IndexError:
                continue
    return None


def _fault_kind(text: str) -> Tuple[str, List[str]]:
    evidence: List[str] = []
    for kind, pattern in _FAULT_PATTERNS:
        match = pattern.search(text)
        if match:
            evidence.append(match.group(0).strip()[:120])
            return kind, evidence
    return "unknown", evidence


def _call_trace(text: str, limit: int = MAX_TRACE_FRAMES) -> List[str]:
    frames = _CALL_TRACE_PATTERN.findall(text)
    return frames[:limit]


def _fingerprint(
    kind: str,
    rip_symbol: Optional[str],
    rip_offset: Optional[str],
    trace: Sequence[str],
) -> str:
    """Stable digest over normalized (ASLR-free) crash identity."""
    normalized = "|".join(
        [
            kind,
            rip_symbol or "?",
            rip_offset or "?",
            ",".join(trace[:8]),
        ]
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def analyze_kernel_oops(log_text: str) -> Dict[str, Any]:
    """Triage one kernel oops/panic log into a typed record."""
    if not isinstance(log_text, str) or not log_text.strip():
        return build_backend_response(
            "analyze_kernel_oops",
            parsed=None,
            raw_responses=[],
            state="STATIC",
            success=False,
            error="log_text cannot be empty",
            error_type="ValueError",
        )
    if len(log_text.encode("utf-8")) > MAX_LOG_BYTES:
        return build_backend_response(
            "analyze_kernel_oops",
            parsed=None,
            raw_responses=[],
            state="STATIC",
            success=False,
            error=f"log_text exceeds the {MAX_LOG_BYTES}-byte cap; trim to the oops",
            error_type="ValueError",
        )
    kind, kind_evidence = _fault_kind(log_text)
    fault_address = _first_group_match(_FAULT_ADDRESS_PATTERNS, log_text)
    rip_symbol: Optional[str] = None
    rip_offset: Optional[str] = None
    rip_match = _RIP_PATTERN.search(log_text)
    if rip_match:
        rip_symbol, rip_offset = rip_match.group(1), rip_match.group(2).lower()
    else:
        bare = _RIP_BARE_PATTERN.search(log_text)
        rip_offset = bare.group(1).lower() if bare else None
    trace = _call_trace(log_text)
    cpu_match = _CPU_PATTERN.search(log_text)
    taint_match = _TAINT_PATTERN.search(log_text)
    oops_match = _OOPS_CODE_PATTERN.search(log_text)
    slab_match = _SLAB_PATTERN.search(log_text)
    kasan_match = _KASAN_ACCESS_PATTERN.search(log_text)
    parsed = {
        "schema": KERNEL_OOPS_SCHEMA,
        "verdict": kind if kind != "unknown" else "unknown",
        "fault_kind": kind,
        "kind_evidence": kind_evidence,
        "fault_address": fault_address,
        "rip": {"symbol": rip_symbol, "offset": rip_offset},
        "call_trace": trace,
        "frame_count": len(trace),
        "cpu": (
            {
                "cpu": cpu_match.group(1),
                "pid": cpu_match.group(2),
                "comm": cpu_match.group(3),
            }
            if cpu_match
            else None
        ),
        "tainted": taint_match.group(1).strip() if taint_match else None,
        "oops_code": oops_match.group(1) if oops_match else None,
        "oops_number": oops_match.group(2) if oops_match else None,
        "slab_cache": slab_match.group(1) if slab_match else None,
        "kasan_access": (
            {"kind": kasan_match.group(1).lower(), "size": int(kasan_match.group(2))}
            if kasan_match
            else None
        ),
        "fingerprint": _fingerprint(kind, rip_symbol, rip_offset, trace),
    }
    warnings = []
    if kind == "unknown":
        warnings.append(
            "No recognized fault signature; fingerprint is still stable for dedup"
        )
    return build_backend_response(
        "analyze_kernel_oops",
        parsed=parsed,
        raw_responses=[],
        state="STATIC",
        parser_version=PARSER_VERSION,
        warnings=warnings or None,
    )


__all__ = [
    "analyze_kernel_oops",
    "KERNEL_OOPS_SCHEMA",
    "MAX_LOG_BYTES",
]
