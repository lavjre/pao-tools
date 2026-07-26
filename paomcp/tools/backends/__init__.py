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
    ChainBuilder,
    ChainBuilderError,
    ChainValidator,
    ChainValidatorError,
    GadgetIndex,
    GadgetIndexError,
    LibcIdentifier,
    LibcIdentifierError,
    RopTools,
)
from .subproc import SubprocessTools
from .targets import (
    AttachPidProvider,
    GdbserverProvider,
    LocalProcessProvider,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
    TcpRemoteProvider,
)

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
    "ChainBuilder",
    "ChainBuilderError",
    "ChainValidator",
    "ChainValidatorError",
    "RopTools",
    "TargetKind",
    "TargetState",
    "TargetError",
    "TargetSpec",
    "TargetHandle",
    "TargetRegistry",
    "LocalProcessProvider",
    "AttachPidProvider",
    "GdbserverProvider",
    "TcpRemoteProvider",
]
