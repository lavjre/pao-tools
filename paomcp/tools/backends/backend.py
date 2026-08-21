"""Debugger backend decision layer: ``gdb`` (pwndbg/MI) vs ``lldb`` (pwndbg-lldb CLI).

The server can drive two debugger runtimes:

* ``gdb`` -- ``pwndbg`` speaking GDB/MI (``--interpreter=mi3``). This is the
  full-surface backend: every typed tool (heap parsers, elf runtime, crash
  triage) is implemented on top of it.
* ``lldb`` -- ``pwndbg-lldb`` speaking the lldb command language over stdin.
  A minimal ``LldbController`` exposes the lifecycle primitives (set file,
  run, attach, stepping, memory/register reads); the rich pwndbg-only tools
  are gated at the tool layer and report a typed ``GdbOnlyBackendError``
  verdict instead of surfacing raw lldb text.

:func:`resolve_backend` decides which backend a session uses.  Selection
order (see the function docstring): an explicit per-session ``backend`` wins
over the ``PAO_DEBUGGER_BACKEND`` environment variable; ``auto`` prefers gdb
on non-darwin hosts, prefers lldb on darwin/Mach-O targets, and falls back
across the pair based on availability.  Availability is a ``shutil.which``
check for gdb plus a launch probe for lldb, because the bundled
``_vendor/bin/lldb`` shipped with pwndbg-lldb can fail to load glibc symbols
(``libm.so.6: cannot open shared object file``) while still present on PATH.

``build_controller`` resolves the controller classes at call time from this
module's globals so standalone callers can construct a controller from a spec.
The session registry deliberately keeps its *own* tiny factory (referencing
its module-global ``GdbController``/``LldbController`` names) so existing
tests that monkeypatch ``registry.GdbController`` keep working.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

BACKEND_CHOICES = ("gdb", "lldb", "auto")
BACKEND_NAMES = ("gdb", "lldb")

# Failure signatures observed when pwndbg-lldb's bundled lldb cannot load the
# host libc/libm.  Any of these in the launch output means the binary is
# present but unusable on this machine.
LLDB_FAILURE_MARKERS = (
    b"Could not find the LLDB Python Path",
    b"cannot open shared object file",
    b"error while loading shared libraries",
)


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested backend cannot be used."""


class UnknownBackendError(ValueError):
    """Raised when a backend name is not one of ``gdb``/``lldb``/``auto``."""


@dataclass(frozen=True)
class BackendSpec:
    """An immutable description of a resolved debugger backend."""

    name: str  # "gdb" | "lldb"
    protocol: str  # "gdb-mi3" | "lldb-cli"
    executable: str  # "pwndbg" | "pwndbg-lldb" (or resolved path)
    available: bool
    reason: str  # "default" | "explicit" | "auto-availability" | "target-os" | ...
    target_hint: Optional[str] = None  # e.g. "darwin" when chosen by target OS

    def to_dict(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "name": self.name,
            "protocol": self.protocol,
            "executable": self.executable,
            "available": self.available,
            "reason": self.reason,
        }
        if self.target_hint is not None:
            base["target_hint"] = self.target_hint
        return base


def validate_backend_name(value: Optional[str]) -> Optional[str]:
    """Validate a user-supplied backend name; ``None`` passes through."""
    if value is None:
        return None
    if value not in BACKEND_CHOICES:
        raise UnknownBackendError(
            f"Unknown debugger backend {value!r}; expected one of {BACKEND_CHOICES}"
        )
    return value


def gdb_available(which: Callable[[str], Optional[str]] = shutil.which) -> bool:
    """Return whether ``pwndbg`` is on PATH (the gdb backend prerequisite)."""
    return which("pwndbg") is not None


@functools.lru_cache(maxsize=1)
def default_probe_lldb(
    executable: str = "pwndbg-lldb", timeout_sec: float = 5.0
) -> bool:
    """Return whether ``pwndbg-lldb`` is on PATH *and* actually launches.

    A plain ``which`` is not enough: the pwndbg-lldb distribution bundles its
    own ``_vendor/bin/lldb`` which can fail to load host glibc symbols while
    the launcher still appears on PATH.  The probe spawns the launcher with no
    arguments -- the exact path a session uses -- because flag paths like
    ``-h`` are handled by argparse *before* the bundled lldb is ever loaded,
    so a broken distribution still exits cleanly.  A launcher that reaches its
    REPL either keeps running (timeout -> available) or exits without a vendor
    failure signature (available); one that prints a failure marker while
    trying to load the bundled lldb is reported unavailable.  The result is
    cached so a session registry never pays the probe more than once.
    """
    path = shutil.which(executable)
    if path is None:
        return False
    try:
        proc = subprocess.Popen(
            [path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        logger.debug("pwndbg-lldb probe could not spawn %s", path, exc_info=True)
        return False
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            # Still alive after the budget: the lldb REPL loaded and is
            # waiting on input -- the vendor-lib failure would have died
            # within milliseconds.
            proc.kill()
            proc.communicate()
            return True
    except OSError:
        return False
    combined = (stdout or b"") + (stderr or b"")
    if any(marker in combined for marker in LLDB_FAILURE_MARKERS):
        logger.warning(
            "pwndbg-lldb is present but its bundled lldb cannot load host "
            "libraries; lldb backend is unavailable"
        )
        return False
    return True


def available_backends(
    which: Callable[[str], Optional[str]] = shutil.which,
    probe_lldb: Optional[Callable[[], bool]] = None,
) -> List[Dict[str, Any]]:
    """Return the availability report for every known backend."""
    lldb_probe = probe_lldb or default_probe_lldb
    gdb_path = which("pwndbg")
    lldb_path = which("pwndbg-lldb")
    return [
        {
            "name": "gdb",
            "protocol": "gdb-mi3",
            "executable": "pwndbg",
            "path": gdb_path,
            "available": gdb_path is not None,
        },
        {
            "name": "lldb",
            "protocol": "lldb-cli",
            "executable": "pwndbg-lldb",
            "path": lldb_path,
            "available": lldb_path is not None and bool(lldb_probe()),
        },
    ]


def _spec(
    name: str,
    protocol: str,
    executable: str,
    available: bool,
    reason: str,
    target_hint: Optional[str] = None,
) -> BackendSpec:
    return BackendSpec(
        name=name,
        protocol=protocol,
        executable=executable,
        available=available,
        reason=reason,
        target_hint=target_hint,
    )


def resolve_backend(
    *,
    env_backend: Optional[str] = None,
    session_backend: Optional[str] = None,
    target_kind: Optional[str] = None,
    target_os: Optional[str] = None,
    target_platform: Optional[str] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    probe_lldb: Optional[Callable[[], bool]] = None,
) -> BackendSpec:
    """Resolve which debugger backend a session should drive.

    Rule ordering:

    1. ``requested = session_backend or env_backend``; any value must be one
       of ``gdb``/``lldb``/``auto`` (else :class:`UnknownBackendError`).
    2. Availability for both backends is computed once.
    3. Explicit ``gdb`` / ``lldb`` is honored when the backend is available,
       else :class:`BackendUnavailableError`.
    4. ``auto``: a darwin/Mach-O target hint prefers lldb (then gdb); any
       other target prefers gdb (full tool surface), falling back to lldb.
       Neither available raises :class:`BackendUnavailableError`.
    5. No request at all returns a ``gdb`` spec with ``reason="default"`` --
       preserving the pre-decision-layer behavior exactly (including for
       environments where ``pwndbg`` is absent, so construction-time tests
       that patch the controller keep working).

    ``target_kind`` is accepted for API symmetry and remote-target host-OS
    hints; today it does not change the ordering beyond the OS/platform hint.
    """
    requested = session_backend or env_backend
    validate_backend_name(requested)
    lldb_probe = probe_lldb or default_probe_lldb
    g_avail = gdb_available(which)
    l_avail = lldb_probe()

    if requested is None:
        return _spec("gdb", "gdb-mi3", "pwndbg", g_avail, "default")

    if requested == "gdb":
        if not g_avail:
            raise BackendUnavailableError(
                "gdb backend requested but 'pwndbg' is not on PATH"
            )
        return _spec("gdb", "gdb-mi3", "pwndbg", True, "explicit")

    if requested == "lldb":
        if not l_avail:
            raise BackendUnavailableError(
                "lldb backend requested but pwndbg-lldb is unavailable"
            )
        return _spec("lldb", "lldb-cli", "pwndbg-lldb", True, "explicit")

    # auto
    is_darwin = (
        "darwin" in (target_os or "").lower()
        or "mach-o" in (target_platform or "").lower()
    )
    if is_darwin:
        if l_avail:
            return _spec(
                "lldb",
                "lldb-cli",
                "pwndbg-lldb",
                True,
                "target-os",
                target_hint="darwin",
            )
        if g_avail:
            return _spec(
                "gdb", "gdb-mi3", "pwndbg", True, "auto-fallback", target_hint="darwin"
            )
        raise BackendUnavailableError(
            "no debugger backend available (auto requested for a darwin target)"
        )

    if g_avail:
        return _spec("gdb", "gdb-mi3", "pwndbg", True, "auto-availability")
    if l_avail:
        return _spec("lldb", "lldb-cli", "pwndbg-lldb", True, "auto-availability")
    raise BackendUnavailableError(
        "no debugger backend available (pwndbg or pwndbg-lldb is required)"
    )


def build_controller(spec: BackendSpec) -> Any:
    """Instantiate the controller class for a resolved backend spec.

    Controller classes are resolved lazily at call time so importing this
    module never pulls in pygdbmi/lldb machinery.  Note that the session
    registry uses its *own* factory referencing its module-global controller
    names (so its tests' monkeypatches keep working); this helper is for
    standalone construction.
    """
    from .gdb import GdbController  # noqa: PLC0415
    from .lldb import LldbController  # noqa: PLC0415

    if spec.protocol == "lldb-cli":
        if spec.executable and spec.executable != "pwndbg-lldb":
            return LldbController(executable=spec.executable)
        return LldbController()
    if spec.executable and spec.executable != "pwndbg":
        return GdbController(pwndbg_executable=spec.executable)
    return GdbController()


__all__ = [
    "BACKEND_CHOICES",
    "BACKEND_NAMES",
    "BackendSpec",
    "BackendUnavailableError",
    "UnknownBackendError",
    "validate_backend_name",
    "gdb_available",
    "default_probe_lldb",
    "available_backends",
    "resolve_backend",
    "build_controller",
]
