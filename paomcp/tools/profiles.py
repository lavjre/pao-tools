"""Task profiles: curated module subsets behind ``PAO_PROFILE`` (Phase 4).

Listing 199 tools costs ~26k tokens before any work happens. Profiles let
an operator pay only for the mission at hand: each profile names the tool
*modules* it keeps (see :func:`register_all_tools`), everything else is
never registered. ``full`` (the default) keeps the entire surface.
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Optional

ALL_MODULES: FrozenSet[str] = frozenset(
    {
        "artifacts",
        "attack",
        "compatibility",
        "crash",
        "decompiler",
        "debug",
        "debugger_memory",
        "elf_runtime",
        "fuzz",
        "heap",
        "ida",
        "inspect",
        "kernel_slub",
        "macho",
        "operations",
        "platform",
        "processes",
        "paocli",
        "python_env",
        "repos",
        "regions",
        "research",
        "rop",
        "static_analysis",
        "targets",
    }
)

PWN_MODULES: FrozenSet[str] = ALL_MODULES - {"ida", "macho", "kernel_slub"}
"""Exploit dev on Linux userland: Ghidra covers static, IDA/Mach-O/kernel stay out."""

REV_MODULES: FrozenSet[str] = frozenset(
    {
        "artifacts",
        "compatibility",
        "debug",
        "debugger_memory",
        "decompiler",
        "elf_runtime",
        "ida",
        "inspect",
        "macho",
        "operations",
        "platform",
        "processes",
        "python_env",
        "repos",
        "research",
        "static_analysis",
    }
)
"""Reverse engineering: static + dynamic reading; heap/rop/crash/attack stay out."""

BOUNTY_MODULES: FrozenSet[str] = ALL_MODULES - {"attack", "paocli"}
"""Vuln research + triage + reporting: everything except write primitives
and interactive exploit drivers."""

PROFILES: Dict[str, Optional[FrozenSet[str]]] = {
    "full": None,
    "pwn": PWN_MODULES,
    "rev": REV_MODULES,
    "bounty": BOUNTY_MODULES,
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "full": "Every tool module (default).",
    "pwn": "Debugger, crash triage, heap, ROP, targets, and exploit running.",
    "rev": "Static analysis, decompilers (Ghidra + IDA), and dynamic reading.",
    "bounty": "Research, triage, fuzzing, and reporting across target kinds.",
}

DEFAULT_PROFILE = "full"


def resolve_profile(name: Optional[str]) -> Optional[FrozenSet[str]]:
    """Validate a profile name; returns None for ``full`` (keep everything)."""
    profile = (name or DEFAULT_PROFILE).strip().lower() or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise ValueError(
            f"unknown tool profile {name!r}; expected one of {sorted(PROFILES)}"
        )
    return PROFILES[profile]


def profile_from_env() -> Optional[FrozenSet[str]]:
    """Read the active profile from ``PAO_PROFILE`` (default ``full``)."""
    return resolve_profile(os.environ.get("PAO_PROFILE", DEFAULT_PROFILE))


__all__ = [
    "ALL_MODULES",
    "PWN_MODULES",
    "REV_MODULES",
    "BOUNTY_MODULES",
    "PROFILES",
    "PROFILE_DESCRIPTIONS",
    "DEFAULT_PROFILE",
    "resolve_profile",
    "profile_from_env",
]
