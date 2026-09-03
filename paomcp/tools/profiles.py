"""Task profiles: curated module subsets behind ``PAO_PROFILE``.

Listing 199 tools costs ~28k tokens before any work happens. Profiles let
an operator pay only for the mission at hand: each profile names the tool
*modules* it keeps (see :func:`register_all_tools`), everything else is
never registered. ``full`` (the default) keeps the entire surface.

Profiles are token budgets, never authority boundaries: the backend bundle
is still fully constructed per session, so an excluded module only removes
its ``list_tools`` entries. A profile that omits e.g. write paths still
runs in the same single trust domain as ``full`` -- do not rely on profiles
for isolation.
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
"""Vuln research + triage + reporting: omits the dedicated write-primitive
and interactive-driver modules (memory writes stay reachable through the
debugger and heap modules, so this is scoping, not a security boundary)."""

CHROME_MODULES: FrozenSet[str] = frozenset(
    {
        "artifacts",
        "compatibility",
        "crash",
        "debug",
        "debugger_memory",
        "decompiler",
        "elf_runtime",
        "fuzz",
        "ida",
        "inspect",
        "operations",
        "platform",
        "processes",
        "python_env",
        "regions",
        "repos",
        "research",
        "static_analysis",
        "targets",
    }
)
"""Browser/JIT exploitation: debugger, crash, JIT regions, decompilers. No
glibc heap, ROP chains, write primitives, or kernel/Mach-O surface."""

KERNEL_MODULES: FrozenSet[str] = frozenset(
    {
        "artifacts",
        "compatibility",
        "crash",
        "debug",
        "debugger_memory",
        "decompiler",
        "elf_runtime",
        "ida",
        "inspect",
        "kernel_slub",
        "operations",
        "platform",
        "processes",
        "python_env",
        "regions",
        "repos",
        "research",
        "rop",
        "static_analysis",
        "targets",
    }
)
"""Kernel exploitation: oops triage, SLUB, QEMU/gdbserver targets. No
userland heap, write primitives, drivers, Mach-O, or (yet) syzkaller fuzz."""

FIRMWARE_MODULES: FrozenSet[str] = frozenset(
    {
        "artifacts",
        "compatibility",
        "decompiler",
        "elf_runtime",
        "ida",
        "inspect",
        "operations",
        "platform",
        "processes",
        "python_env",
        "repos",
        "research",
        "static_analysis",
    }
)
"""Firmware reverse engineering: static analysis only, no live inferior.
No debugger, heap, ROP, crash, or target lifecycle. (Filesystem
extraction like binwalk/unblob is not yet a tool; track there.)"""

CTF_MODULES: FrozenSet[str] = PWN_MODULES
"""CTF is pwn under time pressure: same surface, plus the ctf skill
workflow (recon, triage, exploit, flag). Alias kept for discoverability."""

PROFILES: Dict[str, Optional[FrozenSet[str]]] = {
    "full": None,
    "pwn": PWN_MODULES,
    "rev": REV_MODULES,
    "bounty": BOUNTY_MODULES,
    "chrome": CHROME_MODULES,
    "kernel": KERNEL_MODULES,
    "firmware": FIRMWARE_MODULES,
    "ctf": CTF_MODULES,
}

PROFILE_DESCRIPTIONS: Dict[str, str] = {
    "full": "Every tool module (default).",
    "pwn": "Debugger, crash triage, heap, ROP, targets, and exploit running.",
    "rev": "Static analysis, decompilers (Ghidra + IDA), and dynamic reading.",
    "bounty": "Research, triage, fuzzing, and reporting across target kinds.",
    "chrome": "Browser/JIT exploitation without the userland heap/ROP surface.",
    "kernel": "Kernel oops triage, SLUB, and VM targets without userland heap.",
    "firmware": "Static-only firmware reversing without any live target.",
    "ctf": "Alias of pwn, paired with the ctf time-boxed workflow skill.",
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
    "CHROME_MODULES",
    "KERNEL_MODULES",
    "FIRMWARE_MODULES",
    "CTF_MODULES",
    "PROFILES",
    "PROFILE_DESCRIPTIONS",
    "DEFAULT_PROFILE",
    "resolve_profile",
    "profile_from_env",
]
