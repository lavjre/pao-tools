"""Internal backend adapters used by MCP tool registrations.

Re-export policy: this package exposes the stable facade surface (session
components, analyzers, registries). Split submodules (``heap_glibc_*``,
``rop_*``, ``crash_*``, ``targets_*``) are importable directly, but only
names listed here are covered by the cross-module compatibility tests --
import anything else from its defining submodule.
"""

from .backend import (
    BackendSpec,
    BackendUnavailableError,
    UnknownBackendError,
    available_backends,
    build_controller,
    resolve_backend,
)
from .decompiler import GhidraDecompiler
from .ida import IdaProMCP
from .crash import CrashTriage
from .git import GitTools
from .attack_helpers import ATTACK_TCACHE_POISON_SCHEMA, GlibcAttackHelper
from .gdb import GdbController
from .lldb import LldbController
from .heap import (
    AllocatorDiscovery,
    ArenaLayoutResolver,
    GlibcArenas,
    GlibcBins,
    GlibcHeap,
    HeapTools,
    TcacheLayoutResolver,
)
from .heap_techniques import GlibcTechniqueAnalyzer
from .libmalloc import (
    LIBMALLOC_POINTER_SCHEMA,
    LIBMALLOC_ZONE_SCHEMA,
    LIBMALLOC_ZONES_SCHEMA,
    LibmallocAnalyzer,
)
from .macho import MachOFile, load_macho, parse_macho
from .pwndbg import PwndbgTools
from .regions import JIT_REGIONS_SCHEMA, MEMORY_REGION_SCHEMA, JitRegionTracker
from .slub import SLUB_CACHE_SCHEMA, SLUB_FREELIST_SCHEMA, KernelSlab
from .python import PythonTools
from .research import (
    ResearchError,
    ResearchEvidence,
    ResearchHypothesis,
    ResearchNote,
    ResearchRegistry,
    ResearchSession,
    ResearchSnapshot,
    TimelineEvent,
)
from .fuzz import FUZZ_BATCH_SCHEMA, FUZZ_INPUTS_SCHEMA, FuzzBatch, generate_inputs
from .research_bundle import BundleError, ResearchBundle
from .research_report import ReportError, ResearchReport
from .research_store import CaseStore
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
    DockerProvider,
    GdbserverProvider,
    LocalProcessProvider,
    QemuSystemProvider,
    QemuUserProvider,
    SshProvider,
    TargetError,
    TargetHandle,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
    TcpRemoteProvider,
)

__all__ = [
    "BackendSpec",
    "BackendUnavailableError",
    "UnknownBackendError",
    "available_backends",
    "build_controller",
    "resolve_backend",
    "GdbController",
    "LldbController",
    "CrashTriage",
    "AllocatorDiscovery",
    "ArenaLayoutResolver",
    "GlibcArenas",
    "GlibcBins",
    "GlibcHeap",
    "HeapTools",
    "TcacheLayoutResolver",
    "GlibcTechniqueAnalyzer",
    "GlibcAttackHelper",
    "ATTACK_TCACHE_POISON_SCHEMA",
    "KernelSlab",
    "SLUB_CACHE_SCHEMA",
    "SLUB_FREELIST_SCHEMA",
    "LibmallocAnalyzer",
    "LIBMALLOC_ZONES_SCHEMA",
    "LIBMALLOC_ZONE_SCHEMA",
    "LIBMALLOC_POINTER_SCHEMA",
    "JitRegionTracker",
    "JIT_REGIONS_SCHEMA",
    "MEMORY_REGION_SCHEMA",
    "MachOFile",
    "load_macho",
    "parse_macho",
    "PwndbgTools",
    "SubprocessTools",
    "GitTools",
    "PythonTools",
    "GhidraDecompiler",
    "IdaProMCP",
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
    "SshProvider",
    "DockerProvider",
    "QemuUserProvider",
    "QemuSystemProvider",
    "ResearchError",
    "ResearchEvidence",
    "ResearchHypothesis",
    "ResearchNote",
    "ResearchRegistry",
    "ResearchSession",
    "ResearchSnapshot",
    "TimelineEvent",
    "BundleError",
    "ResearchBundle",
    "ReportError",
    "ResearchReport",
    "CaseStore",
    "FuzzBatch",
    "generate_inputs",
    "FUZZ_BATCH_SCHEMA",
    "FUZZ_INPUTS_SCHEMA",
]
