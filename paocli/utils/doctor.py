"""Native dependency doctor for pao-tools (no Docker).

Pure check logic lives here so both the ``paocli doctor`` command and the
test-suite drive it without spawning processes: callers inject ``which_fn``
and ``env``. A check is ``required`` only when the server cannot boot or run
its core loop without it; everything else is optional with an install hint
per OS.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Mapping, Optional

# Each entry: name shown to the user, how to probe it, whether the core
# server loop needs it on each OS, and the install hint per OS.
_CHECKS: List[Dict[str, Any]] = [
    {
        "name": "python3 (>=3.12)",
        "kind": "python",
        "required_linux": True,
        "required_macos": True,
        "hint_linux": "apt install python3.12 / dnf install python3.12 (or uv python install 3.12)",
        "hint_macos": "brew install python@3.12 (or uv python install 3.12)",
    },
    {
        "name": "uv",
        "kind": "binary",
        "target": "uv",
        "required_linux": True,
        "required_macos": True,
        "hint_linux": "curl -LsSf astral.sh/uv/install.sh | sh",
        "hint_macos": "brew install uv (or astral.sh/uv/install.sh | sh)",
    },
    {
        "name": "pwndbg",
        "kind": "binary",
        "target": "pwndbg",
        "required_linux": True,
        "required_macos": False,
        "hint_linux": "pipx install pwndbg (server execs `pwndbg --interpreter=mi3`)",
        "hint_macos": "optional: pipx install pwndbg (ldd backend covers Mach-O; Linux ELF still needs Docker/VM)",
    },
    {
        "name": "gdb",
        "kind": "binary",
        "target": "gdb",
        "required_linux": True,
        "required_macos": False,
        "hint_linux": "apt install gdb gdbserver / dnf install gdb-gdbserver",
        "hint_macos": "optional (unsigned gdb cannot debug on macOS; use the lldb backend)",
    },
    {
        "name": "lldb",
        "kind": "binary",
        "target": "lldb",
        "required_linux": False,
        "required_macos": True,
        "hint_linux": "optional (apt install lldb)",
        "hint_macos": "xcode-select --install (ships lldb; PAO_DEBUGGER_BACKEND=lldb)",
    },
    {
        "name": "git",
        "kind": "binary",
        "target": "git",
        "required_linux": True,
        "required_macos": True,
        "hint_linux": "apt install git",
        "hint_macos": "xcode-select --install (ships git)",
    },
    {
        "name": "java (>=17, Ghidra)",
        "kind": "binary",
        "target": "java",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "apt install openjdk-21-jdk-headless (only for local Ghidra decompile)",
        "hint_macos": "brew install openjdk@21 (only for local Ghidra decompile)",
    },
    {
        "name": "Ghidra headless",
        "kind": "env_or_binary",
        "target": "analyzeHeadless",
        "env_key": "PAO_GHIDRA_HEADLESS",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "./scripts/setup-native.sh --with-ghidra (pinned 12.1.2, sha-verified)",
        "hint_macos": "./scripts/setup-native.sh --with-ghidra (pinned 12.1.2, sha-verified)",
    },
    {
        "name": "qemu-system-x86_64",
        "kind": "binary",
        "target": "qemu-system-x86_64",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "apt install qemu-system-x86 qemu-user (only for QEMU targets)",
        "hint_macos": "brew install qemu (only for QEMU targets; slow under emulation)",
    },
    {
        "name": "afl-fuzz",
        "kind": "binary",
        "target": "afl-fuzz",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "apt install afl++ (only for run_afl_batch coverage campaigns)",
        "hint_macos": "brew install afl-fuzz (only for run_afl_batch coverage campaigns)",
    },
    {
        "name": "unblob",
        "kind": "binary",
        "target": "unblob",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "pipx install unblob (else extract_firmware falls back to binwalk/pure carve)",
        "hint_macos": "pipx install unblob (else extract_firmware falls back to binwalk/pure carve)",
    },
    {
        "name": "pwninit",
        "kind": "binary",
        "target": "pwninit",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "./scripts/setup-native.sh --with-pwninit (opt-in helper)",
        "hint_macos": "not supported on macOS (Linux ELF helper)",
    },
    {
        "name": "one_gadget",
        "kind": "binary",
        "target": "one_gadget",
        "required_linux": False,
        "required_macos": False,
        "hint_linux": "gem install one_gadget (opt-in libc helper)",
        "hint_macos": "gem install one_gadget (opt-in libc helper)",
    },
]


def detect_os(platform: Optional[str] = None) -> str:
    """Return ``linux`` or ``macos`` for a ``sys.platform`` value."""
    value = (platform if platform is not None else sys.platform).lower()
    if value.startswith("darwin"):
        return "macos"
    return "linux"


def check_one(
    spec: Mapping[str, Any],
    *,
    os_name: str,
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    env: Mapping[str, str] = os.environ,
) -> Dict[str, Any]:
    """Probe one dependency spec; never raises, never spawns."""
    kind = spec.get("kind")
    found: Optional[str] = None
    if kind == "python":
        found = sys.executable or None
        ok = sys.version_info >= (3, 12)
        detail = f"{sys.version.split()[0]} at {found}" if found else "no interpreter"
        if found and not ok:
            detail += " (needs >=3.12)"
    elif kind == "binary":
        found = which_fn(str(spec.get("target")))
        ok = found is not None
        detail = found or "not on PATH"
    elif kind == "env_or_binary":
        env_key = str(spec.get("env_key") or "")
        configured = (env.get(env_key) or "").strip()
        if configured and os.path.isfile(configured):
            return {
                "name": spec.get("name"),
                "ok": True,
                "required": bool(spec.get(f"required_{os_name}")),
                "detail": f"{env_key}={configured}",
            }
        found = which_fn(str(spec.get("target")))
        ok = found is not None
        detail = found or f"neither {env_key} nor PATH provides it"
    else:  # pragma: no cover - defensive, spec table is closed
        ok, detail = False, "unknown check kind"
    required = bool(spec.get(f"required_{os_name}"))
    hint = str(spec.get(f"hint_{os_name}") or spec.get("hint_linux") or "")
    return {
        "name": spec.get("name"),
        "ok": bool(ok),
        "required": required,
        "detail": str(detail),
        "hint": hint,
    }


def run_doctor(
    *,
    platform: Optional[str] = None,
    which_fn: Callable[[str], Optional[str]] = shutil.which,
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Run every check; returns os, per-check rows, and missing required."""
    os_name = detect_os(platform)
    env_map = dict(os.environ) if env is None else dict(env)
    checks = [
        check_one(spec, os_name=os_name, which_fn=which_fn, env=env_map)
        for spec in _CHECKS
    ]
    missing = [c for c in checks if c["required"] and not c["ok"]]
    return {
        "os": os_name,
        "checks": checks,
        "ok_required": not missing,
        "missing_required": [c["name"] for c in missing],
        "required_total": sum(1 for c in checks if c["required"]),
    }


__all__ = ["detect_os", "check_one", "run_doctor"]
