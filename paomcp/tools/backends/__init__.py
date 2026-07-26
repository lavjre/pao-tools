"""Internal backend adapters used by MCP tool registrations."""

from .decompiler import GhidraDecompiler
from .crash import CrashTriage
from .git import GitTools
from .gdb import GdbController
from .heap import (
    AllocatorDiscovery,
    ArenaLayoutResolver,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    HeapTools,
    TcacheLayoutResolver,
)
from .pwndbg import PwndbgTools
from .python import PythonTools
from .rop import (
    GadgetIndex,
    GadgetIndexError,
    LibcIdentifier,
    LibcIdentifierError,
    RopTools,
)
from .subproc import SubprocessTools

__all__ = [
    "GdbController",
    "CrashTriage",
    "AllocatorDiscovery",
    "ArenaLayoutResolver",
    "GlibcArenas",
    "GlibcBins",
    "GlibcHeap",
    "HeapTools",
    "TcacheLayoutResolver",
    "PwndbgTools",
    "SubprocessTools",
    "GitTools",
    "PythonTools",
    "GhidraDecompiler",
    "GadgetIndex",
    "GadgetIndexError",
    "LibcIdentifier",
    "LibcIdentifierError",
    "RopTools",
]
