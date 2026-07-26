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
)
from .pwndbg import PwndbgTools
from .python import PythonTools
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
    "PwndbgTools",
    "SubprocessTools",
    "GitTools",
    "PythonTools",
    "GhidraDecompiler",
]
