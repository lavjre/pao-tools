"""Shared timeout / signal clamps for backend subprocess and tool timeouts.

Extracted from SubprocessTools so PythonTools, IDA bridge, and future
backends share one definition of "never wait forever". Subproc re-exports
these names for backward compatibility.
"""

from __future__ import annotations

import os
import signal as signal_module
from typing import Optional

DEFAULT_COMMAND_TIMEOUT = 30.0
MAX_COMMAND_TIMEOUT = float(os.environ.get("PAO_MAX_COMMAND_TIMEOUT", "600"))
ALLOWED_KILL_SIGNALS = {
    int(getattr(signal_module, "SIGTERM", 15)),
    int(getattr(signal_module, "SIGKILL", 9)),
    int(getattr(signal_module, "SIGINT", 2)),
}


def clamp_timeout(
    value: Optional[float], *, default: float = DEFAULT_COMMAND_TIMEOUT
) -> float:
    """Clamp caller timeouts so None can never wait forever holding a slot."""
    if value is None:
        return default
    try:
        clamped = float(value)
    except (TypeError, ValueError):
        return default
    if clamped <= 0:
        return default
    return min(clamped, MAX_COMMAND_TIMEOUT)


__all__ = [
    "DEFAULT_COMMAND_TIMEOUT",
    "MAX_COMMAND_TIMEOUT",
    "ALLOWED_KILL_SIGNALS",
    "clamp_timeout",
]
