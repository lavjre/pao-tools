"""Typed ROP gadget indexing (Stage 7.1).

Stage 7 turns raw gadget dumps into a typed exploit-research surface.  S7.1
is the foundation: index every gadget in a target binary, normalize the
per-gadget shape (address, bytes, instructions, side effects, stack
delta), deduplicate across sightings, filter by bad bytes, and rebase
under a caller-supplied module base.  No live inferior is required --
gadget indexing is pure static analysis of a binary file -- so the whole
class stays cheaply unit-testable by injecting a fake engine loader.

The real engine is :mod:`ropper` (available in this environment); the
backend consumes an engine loader callable ``(binary_path, arch,
image_base) -> Sequence[GadgetRecord]`` so tests can drive it with a
fixed set of records instead of a real binary.  A ``GadgetRecord`` is
intentionally a plain dict::

    {
        "address": int,
        "bytes": bytes,
        "instructions": List[str],   # normalized "mnemonic op1, op2" lines
        "arch": Optional[str],
    }

Side-effect analysis for common gadget shapes (pop/mov/xor/push/ret/jmp
reg/call reg/syscall/add rsp, N) is derived directly from the mnemonic
+ operand parse, and unrecognized mnemonics are surfaced as ``unknown``
rather than silently guessed.  This matches the Stage 5 / 6 rule: a
heuristic is never promoted to a fact.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from paomcp.tools.backends.foundation import build_backend_response, parse_int

logger = logging.getLogger(__name__)

PARSER_VERSION = "1"
GADGET_LIST_SCHEMA = "pao.gadget-list.v1"
GADGET_SEQUENCE_SCHEMA = "pao.gadget-sequence.v1"
GADGET_INSPECTION_SCHEMA = "pao.gadget-inspection.v1"
LIBC_IDENTIFICATION_SCHEMA = "pao.libc-identification.v1"
ONE_GADGET_SCHEMA = "pao.one-gadgets.v1"
ROP_CHAIN_SCHEMA = "pao.rop-chain.v1"
SIGRETURN_FRAME_SCHEMA = "pao.sigreturn-frame.v1"
CHAIN_CONSTRAINTS_SCHEMA = "pao.chain-constraints.v1"
CHAIN_VALIDATION_SCHEMA = "pao.chain-validation.v1"
CHAIN_EXPORT_SCHEMA = "pao.chain-export.v1"

CHAIN_EXPORT_FORMATS: Tuple[str, ...] = ("raw", "python", "pwntools")

DEFAULT_ONE_GADGET_TIMEOUT_SEC = 30.0

# SysV AMD64 calling convention: first six integer / pointer args land in
# these registers in order.  A ret2libc chain uses this ordering to load
# each argument via a matching pop gadget.
_SYSV_AMD64_ARG_REGS: Tuple[str, ...] = ("rdi", "rsi", "rdx", "rcx", "r8", "r9")
# Linux syscall calling convention on x86_64 replaces rcx with r10 because
# the SYSCALL instruction itself clobbers rcx (return-address slot).
_LINUX_AMD64_SYSCALL_ARG_REGS: Tuple[str, ...] = (
    "rdi",
    "rsi",
    "rdx",
    "r10",
    "r8",
    "r9",
)

DEFAULT_MAX_GADGETS = 500
MAX_MAX_GADGETS = 20000
DEFAULT_MAX_INSTRUCTIONS = 8  # per-gadget instruction budget
MAX_SEQUENCE_STEPS = 32

# First-analysis pass parallelism.  Analyzing side effects is pure per-record
# work, so large binaries (libc dumps are 50k+ gadgets) can be chunked across
# worker processes the first time they are cached.  The threshold is high
# enough that small binaries never pay pool-spawn overhead; both knobs are
# overridable for sandboxes that forbid multiprocessing.
PARALLEL_ANALYSIS_THRESHOLD = int(
    os.environ.get("PAO_PARALLEL_GADGET_THRESHOLD", "20000")
)
PARALLEL_ANALYSIS_CHUNK = 4096
MAX_ANALYSIS_WORKERS = 8
PARALLEL_ANALYSIS_TIMEOUT_SEC = 300
_PARALLEL_ANALYSIS_DISABLED = os.environ.get(
    "PAO_DISABLE_PARALLEL_GADGETS", ""
).strip().lower() in ("1", "true", "yes")

# Canonical stack pointer per architecture; used for stack-delta accounting.
_STACK_POINTERS: Dict[str, str] = {
    "x86_64": "rsp",
    "i386": "esp",
    "aarch64": "sp",
    "arm": "sp",
    "mips": "sp",
    "mipsel": "sp",
    "riscv64": "sp",
}

# Pointer width in bytes per architecture.  Falls back to 8 when a caller
# passes an unknown arch string; the fallback is reported as ``unknown``
# in the response so nothing silently overrides evidence.
_POINTER_BYTES: Dict[str, int] = {
    "x86_64": 8,
    "i386": 4,
    "aarch64": 8,
    "arm": 4,
    "mips": 4,
    "mipsel": 4,
    "riscv64": 8,
}

_X86_REGISTERS: Tuple[str, ...] = (
    "rax",
    "rbx",
    "rcx",
    "rdx",
    "rsi",
    "rdi",
    "rbp",
    "rsp",
    "r8",
    "r9",
    "r10",
    "r11",
    "r12",
    "r13",
    "r14",
    "r15",
    "eax",
    "ebx",
    "ecx",
    "edx",
    "esi",
    "edi",
    "ebp",
    "esp",
    "ax",
    "bx",
    "cx",
    "dx",
    "si",
    "di",
    "bp",
    "sp",
    "al",
    "bl",
    "cl",
    "dl",
    "sil",
    "dil",
    "bpl",
    "spl",
    "ah",
    "bh",
    "ch",
    "dh",
    "r8d",
    "r9d",
    "r10d",
    "r11d",
    "r12d",
    "r13d",
    "r14d",
    "r15d",
    "r8w",
    "r9w",
    "r10w",
    "r11w",
    "r12w",
    "r13w",
    "r14w",
    "r15w",
    "r8b",
    "r9b",
    "r10b",
    "r11b",
    "r12b",
    "r13b",
    "r14b",
    "r15b",
)

_ARM_REGISTERS: Tuple[str, ...] = (
    tuple(f"r{i}" for i in range(16))
    + (
        "sp",
        "lr",
        "pc",
        "fp",
    )
    + tuple(f"x{i}" for i in range(31))
    + ("xzr", "sp")
)

_KNOWN_REGISTERS: Dict[str, frozenset] = {
    "x86_64": frozenset(_X86_REGISTERS),
    "i386": frozenset(_X86_REGISTERS),
    "aarch64": frozenset(_ARM_REGISTERS),
    "arm": frozenset(_ARM_REGISTERS),
}

# Mnemonics whose x86 semantics we can reason about at the level a gadget
# needs.  Unlisted mnemonics fall through to ``unknown`` semantics so a
# caller can decide how much to trust a side-effect summary.
_X86_WRITE_MNEMONICS: frozenset = frozenset(
    {
        "mov",
        "movabs",
        "movsx",
        "movsxd",
        "movzx",
        "lea",
        "pop",
        "xor",
        "add",
        "sub",
        "and",
        "or",
        "shl",
        "shr",
        "sar",
        "not",
        "neg",
        "inc",
        "dec",
        "mul",
        "imul",
        "div",
        "idiv",
        "bswap",
        "cmov",
    }
)
_X86_READ_ONLY_MNEMONICS: frozenset = frozenset({"cmp", "test", "push"})
_X86_TRANSFER_MNEMONICS: frozenset = frozenset(
    {"jmp", "call", "ret", "retn", "iret", "int", "syscall", "sysenter"}
)

# A single-instruction line broken into mnemonic and operand tokens.
_INSTRUCTION_LINE = re.compile(
    r"^\s*(?P<mnemonic>[A-Za-z][A-Za-z0-9\.]*)" r"(?:\s+(?P<operands>.+?))?\s*$"
)

# Ropper commonly prints "0x400123: pop rdi ; ret" style; the split
# leaves the instruction stream after the colon.
_ROPPER_LINE = re.compile(r"^(?:0x[0-9a-fA-F]+:\s*)?(?P<body>.+)$")


class GadgetIndexError(RuntimeError):
    """Raised when an engine load or a caller argument is invalid."""


GadgetLoader = Callable[
    [str, Optional[str], Optional[int]],
    Sequence[Mapping[str, Any]],
]


# ---------------------------------------------------------------------------
# Ropper adapter (real engine).  Isolated so the backend proper does not
# import ropper unless it is actually used, keeping the test import cheap.
# ---------------------------------------------------------------------------
def ropper_gadget_loader(
    binary_path: str,
    arch: Optional[str],
    image_base: Optional[int],
) -> Sequence[Mapping[str, Any]]:
    """Load gadgets for ``binary_path`` via :mod:`ropper`'s Python API."""
    import ropper  # type: ignore[import-untyped]  # local import so importing this module stays cheap

    service = ropper.RopperService()
    service.addFile(binary_path)
    if arch:
        service.setArchitectureFor(name=binary_path, arch=arch)
    if image_base is not None:
        service.setImageBaseFor(name=binary_path, imagebase=image_base)
    service.loadGadgetsFor(name=binary_path)
    file_ = service.getFileFor(name=binary_path)
    if file_ is None:
        raise GadgetIndexError(
            f"ropper did not surface a loaded file for '{binary_path}'"
        )
    records: List[Dict[str, Any]] = []
    for gadget in file_.gadgets or []:
        instructions = [_extract_instruction(line) for line in gadget.lines or []]
        instructions = [line for line in instructions if line]
        records.append(
            {
                "address": int(gadget.address),
                "bytes": bytes(gadget.bytes or b""),
                "instructions": instructions,
                "arch": str(gadget.arch) if gadget.arch is not None else None,
            }
        )
    service.removeFile(name=binary_path)
    return records


def _extract_instruction(line: Any) -> Optional[str]:
    """Pull the mnemonic/operand text out of a ropper line.

    Ropper's newer releases emit each instruction as a tuple of
    ``(address, full_text, mnemonic, operands)``; older releases emit
    a plain ``"0x400123: mnemonic operands"`` string.  Both shapes are
    normalized here to a single ``"mnemonic operands"`` line.
    """
    if isinstance(line, (list, tuple)):
        # (address, full_text, mnemonic, operands) -- prefer full_text
        if len(line) >= 2:
            text = line[1]
        elif line:
            text = line[-1]
        else:
            text = ""
    else:
        text = line
    text = str(text).strip()
    if not text:
        return None
    match = _ROPPER_LINE.match(text)
    body = match.group("body") if match else text
    return body.strip()


def _analyze_gadget_batch(
    payload: Tuple[List[Dict[str, Any]], Optional[str]],
) -> List[Dict[str, Any]]:
    """Analyze a chunk of raw gadget records (module-level for pickling).

    Runs inside ``multiprocessing`` workers during the first-analysis pass.
    ``analyze_gadget`` is a pure per-record transform with no shared state,
    so chunked execution is order-preserving and byte-identical to a
    sequential pass -- which is exactly what the fallback path relies on.
    ``arch_override`` is threaded through so the cached analysis matches
    the original caller-supplied arch exactly.
    """
    records, arch_override = payload
    return [
        GadgetIndex.analyze_gadget(record, arch_override=arch_override)
        for record in records
    ]


# ---------------------------------------------------------------------------
# GadgetIndex backend
# ---------------------------------------------------------------------------
_CacheKey = Tuple[str, float, Optional[str], Optional[int]]


class GadgetIndex:
    """Typed gadget indexing over an injectable engine loader.

    The backend keeps a per-binary cache keyed by ``(abspath, mtime, arch,
    image_base)`` so an interactive workflow that dumps then searches the
    same binary many times does not re-invoke the loader.  Cache eviction
    is intentionally file-driven -- an mtime bump invalidates -- so the
    caller never has to remember to flush after rebuilding.

    What is cached is the *analyzed* gadget stream (normalized records run
    through :meth:`analyze_gadget` once, chunked across worker processes
    when the binary is large) plus two lookup indexes over it: a
    first-mnemonic inverted index used to short-circuit anchored regex
    steps in :meth:`find_gadget_sequence`, and an address -> gadget index
    used by :meth:`inspect_gadget`.  Query-time concerns that vary per call
    -- bad bytes, regex filters, dedupe/sort order -- are deliberately
    applied *after* the cached pass so one binary's analysis is paid once
    and every repeated search is a pure in-memory filter.
    """

    def __init__(
        self,
        loader: GadgetLoader = ropper_gadget_loader,
        *,
        max_cached_binaries: int = 8,
    ) -> None:
        if max_cached_binaries <= 0:
            raise ValueError("max_cached_binaries must be positive")
        self._loader = loader
        self._analyzed: Dict[_CacheKey, List[Dict[str, Any]]] = {}
        self._first_mnemonic: Dict[_CacheKey, Dict[str, List[int]]] = {}
        self._by_address: Dict[_CacheKey, Dict[int, int]] = {}
        self._cache_order: List[_CacheKey] = []
        self._max_cached_binaries = max_cached_binaries

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _load_analyzed(
        self,
        binary_path: str,
        arch: Optional[str],
        image_base: Optional[int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], Dict[int, int]]:
        """Return the analyzed gadget stream plus its lookup indexes.

        The first call for a given key runs the engine loader, normalizes
        every record, and analyzes side effects (in parallel for large
        dumps); every later call is a dict hit.  A mutated binary changes
        its mtime and therefore its key, so rebuilds are implicit.
        """
        abspath = os.path.abspath(binary_path)
        try:
            mtime = os.path.getmtime(abspath)
        except OSError as exc:
            raise GadgetIndexError(
                f"binary '{binary_path}' is not readable: {exc}"
            ) from exc
        key = (abspath, mtime, arch, image_base)
        cached = self._analyzed.get(key)
        if cached is not None:
            return cached, self._first_mnemonic[key], self._by_address[key]
        records = list(self._loader(abspath, arch, image_base) or [])
        normalized: List[Dict[str, Any]] = []
        for raw in records:
            parsed = self._normalize_record(raw)
            if parsed is not None:
                normalized.append(parsed)
        analyzed = self._analyze_many(normalized, arch)
        self._analyzed[key] = analyzed
        self._first_mnemonic[key] = self._build_first_mnemonic_index(analyzed)
        self._by_address[key] = {
            int(gadget["address"]): index for index, gadget in enumerate(analyzed)
        }
        self._cache_order.append(key)
        while len(self._cache_order) > self._max_cached_binaries:
            evicted = self._cache_order.pop(0)
            for store in (self._analyzed, self._first_mnemonic, self._by_address):
                store.pop(evicted, None)
        return analyzed, self._first_mnemonic[key], self._by_address[key]

    def _analyze_many(
        self,
        records: List[Dict[str, Any]],
        arch_override: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Run :meth:`analyze_gadget` over every record, optionally in parallel."""
        if (
            not _PARALLEL_ANALYSIS_DISABLED
            and len(records) >= PARALLEL_ANALYSIS_THRESHOLD
            and os.name != "nt"
        ):
            try:
                chunks = [
                    (records[i : i + PARALLEL_ANALYSIS_CHUNK], arch_override)
                    for i in range(0, len(records), PARALLEL_ANALYSIS_CHUNK)
                ]
                context = multiprocessing.get_context("fork")
                with context.Pool(
                    processes=min(len(chunks), MAX_ANALYSIS_WORKERS),
                    maxtasksperchild=2,
                ) as pool:
                    async_result = pool.map_async(_analyze_gadget_batch, chunks)
                    result_chunks = async_result.get(
                        timeout=PARALLEL_ANALYSIS_TIMEOUT_SEC
                    )
                return [gadget for chunk in result_chunks for gadget in chunk]
            except Exception as exc:  # pragma: no cover - environment dependent
                # Any pool failure degrades to the sequential pass; the
                # analysis itself is pure, so results stay identical.
                logger.warning(
                    "Parallel gadget analysis failed (%s); falling back to "
                    "sequential analysis",
                    exc,
                )
        return _analyze_gadget_batch((records, arch_override))

    @staticmethod
    def _build_first_mnemonic_index(
        gadgets: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[int]]:
        """Invert the analyzed stream by each gadget's first instruction."""
        index: Dict[str, List[int]] = {}
        for gadget_index, gadget in enumerate(gadgets):
            analyzed_instructions = gadget.get("instructions_analyzed") or []
            if not analyzed_instructions:
                continue
            mnemonic = analyzed_instructions[0].get("mnemonic")
            if mnemonic:
                index.setdefault(mnemonic, []).append(gadget_index)
        return index

    @classmethod
    def _leading_mnemonic_for_anchor(cls, raw_pattern: str) -> Optional[str]:
        """Return the literal first mnemonic an anchored pattern pins, if any.

        A pattern like ``"^pop rdi ; ret"`` can only match a gadget whose
        instruction text *starts* with ``pop`` -- i.e. whose first
        instruction's mnemonic is ``pop``.  Extracting that literal token
        lets :meth:`find_gadget_sequence` narrow to the inverted index.
        Returns ``None`` when the pattern is not anchored, or when the
        leading token is a regex (metacharacters) rather than a literal
        mnemonic; both cases fall back to a full scan.
        """
        stripped = raw_pattern.lstrip()
        if not stripped.startswith("^"):
            return None
        rest = stripped[1:].lstrip()
        match = re.match(r"[A-Za-z][A-Za-z0-9_.]*", rest)
        if match is None:
            return None
        token = match.group(0)
        # A token carrying a regex wildcard (e.g. "p.op") is not a literal
        # mnemonic; the caller checks membership against the index, which
        # only ever contains real disassembled mnemonics, so a metachar
        # token fails the membership test and degrades to a full scan.
        return token if token.isalnum() else None

    @classmethod
    def _filter_and_sort(
        cls,
        gadgets: Sequence[Dict[str, Any]],
        normalized_bad_bytes: Sequence[int],
        prefer_shortest: bool,
    ) -> List[Dict[str, Any]]:
        """Bad-byte filter, dedupe, then order for a sequence search."""
        clean = [
            gadget
            for gadget in gadgets
            if not cls._address_contains_bad_byte(
                gadget["address"], gadget["pointer_bytes"], normalized_bad_bytes
            )
        ]
        deduped = cls._dedupe(clean)
        if prefer_shortest:
            deduped.sort(key=lambda g: (len(g["instructions"]), g["address"]))
        else:
            deduped.sort(key=lambda g: g["address"])
        return deduped

    @staticmethod
    def _normalize_record(record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        addr = record.get("address")
        if not isinstance(addr, int):
            addr_parsed = parse_int(addr)
            if addr_parsed is None:
                return None
            addr = addr_parsed
        raw_bytes = record.get("bytes") or b""
        if not isinstance(raw_bytes, (bytes, bytearray)):
            try:
                raw_bytes = bytes(raw_bytes)
            except Exception:
                raw_bytes = b""
        instructions_field = record.get("instructions") or []
        instructions: List[str] = []
        for item in instructions_field:
            text = str(item).strip()
            if text:
                instructions.append(text)
        arch = record.get("arch")
        arch_str = str(arch) if arch else None
        return {
            "address": int(addr),
            "bytes": bytes(raw_bytes),
            "instructions": instructions,
            "arch": arch_str,
        }

    # ------------------------------------------------------------------
    # Pure helpers (unit-testable without an engine)
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_arch(arch: Optional[str]) -> str:
        if not arch:
            return "unknown"
        lowered = arch.lower()
        alias = {
            "amd64": "x86_64",
            "x64": "x86_64",
            "x86-64": "x86_64",
            "x86": "i386",
            "i686": "i386",
            "arm64": "aarch64",
            "armv7": "arm",
        }
        return alias.get(lowered, lowered)

    @classmethod
    def _pointer_bytes_for_arch(cls, arch: Optional[str]) -> Optional[int]:
        return _POINTER_BYTES.get(cls._canonical_arch(arch))

    @classmethod
    def _stack_pointer_for_arch(cls, arch: Optional[str]) -> Optional[str]:
        return _STACK_POINTERS.get(cls._canonical_arch(arch))

    @classmethod
    def _split_operands(cls, operands: str) -> List[str]:
        if not operands:
            return []
        tokens: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in operands:
            if ch in "([{":
                depth += 1
                current.append(ch)
            elif ch in ")]}":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                tokens.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            tokens.append(tail)
        return tokens

    @classmethod
    def _extract_registers(cls, operand: str, arch: Optional[str]) -> List[str]:
        """Return every register token embedded in a single operand."""
        registers = _KNOWN_REGISTERS.get(cls._canonical_arch(arch), frozenset())
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", operand.lower())
        return [t for t in tokens if t in registers]

    @classmethod
    def _parse_instruction(cls, text: str) -> Optional[Tuple[str, List[str]]]:
        match = _INSTRUCTION_LINE.match(text)
        if match is None:
            return None
        mnemonic = match.group("mnemonic").lower()
        operands_text = match.group("operands") or ""
        return mnemonic, cls._split_operands(operands_text)

    @classmethod
    def _instruction_side_effects(
        cls, mnemonic: str, operands: Sequence[str], arch: Optional[str]
    ) -> Dict[str, Any]:
        """Derive reads / writes / clobbers / notes for a single instruction."""
        reads: List[str] = []
        writes: List[str] = []
        clobbers: List[str] = []
        notes: List[str] = []

        stack_ptr = cls._stack_pointer_for_arch(arch) or "sp"

        canonical_arch = cls._canonical_arch(arch)
        if canonical_arch in ("x86_64", "i386"):
            if mnemonic in _X86_WRITE_MNEMONICS and operands:
                destination = operands[0]
                dest_is_memory = "[" in destination
                dest_regs = cls._extract_registers(destination, arch)
                if dest_is_memory:
                    # Memory-destination writes touch memory, not the
                    # base/index registers themselves; those registers
                    # feed the effective-address computation and are
                    # therefore reads.
                    reads.extend(dest_regs)
                else:
                    writes.extend(dest_regs)
                for extra in operands[1:]:
                    reads.extend(cls._extract_registers(extra, arch))
                if mnemonic == "pop":
                    reads.append(stack_ptr)
                    writes.append(stack_ptr)
                if (
                    not dest_is_memory
                    and mnemonic == "xor"
                    and len(dest_regs) == 1
                    and dest_regs
                    and operands[1:]
                ):
                    source_regs = cls._extract_registers(operands[1], arch)
                    if source_regs == dest_regs:
                        # xor reg, reg is a zeroing idiom -- not a read of reg
                        reads = [r for r in reads if r not in dest_regs]
            elif mnemonic in _X86_READ_ONLY_MNEMONICS:
                for operand in operands:
                    reads.extend(cls._extract_registers(operand, arch))
                if mnemonic == "push":
                    reads.append(stack_ptr)
                    writes.append(stack_ptr)
            elif mnemonic in _X86_TRANSFER_MNEMONICS:
                if mnemonic in ("ret", "retn", "iret"):
                    reads.append(stack_ptr)
                    writes.append(stack_ptr)
                elif mnemonic in ("call", "jmp"):
                    for operand in operands:
                        reads.extend(cls._extract_registers(operand, arch))
                    if mnemonic == "call":
                        # Calls clobber the SysV caller-saved regs at the
                        # boundary; we conservatively surface that so a
                        # register-clobber summary is not overly optimistic.
                        clobbers.extend(
                            [
                                "rax",
                                "rcx",
                                "rdx",
                                "rsi",
                                "rdi",
                                "r8",
                                "r9",
                                "r10",
                                "r11",
                            ]
                            if canonical_arch == "x86_64"
                            else ["eax", "ecx", "edx"]
                        )
                elif mnemonic in ("syscall", "sysenter", "int"):
                    # Syscalls read the syscall-argument regs and clobber
                    # rax/rcx/r11 on x86_64 (rcx/r11 are call-clobbered
                    # by the kernel).
                    if canonical_arch == "x86_64":
                        reads.extend(["rax", "rdi", "rsi", "rdx", "r10", "r8", "r9"])
                        clobbers.extend(["rax", "rcx", "r11"])
                    else:
                        reads.extend(["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp"])
                        clobbers.extend(["eax"])
            else:
                notes.append(f"unknown mnemonic '{mnemonic}'")
        else:
            notes.append(
                f"side-effect analysis for arch '{canonical_arch}' is not modeled"
            )

        # Deduplicate while preserving order for stable comparisons.
        def _dedup(seq: Iterable[str]) -> List[str]:
            seen: List[str] = []
            for item in seq:
                if item not in seen:
                    seen.append(item)
            return seen

        return {
            "reads": _dedup(reads),
            "writes": _dedup(writes),
            "clobbers": _dedup(clobbers),
            "notes": notes,
        }

    @classmethod
    def _instruction_stack_delta(
        cls, mnemonic: str, operands: Sequence[str], arch: Optional[str]
    ) -> Optional[int]:
        """Return the stack-pointer delta this instruction produces in bytes.

        ``None`` means "unknown for this mnemonic"; the caller aggregates
        by summing when every step is known and reporting ``None`` when
        anything is unknown.
        """
        ptr_bytes = cls._pointer_bytes_for_arch(arch)
        if ptr_bytes is None:
            return None
        canonical_arch = cls._canonical_arch(arch)
        if canonical_arch not in ("x86_64", "i386"):
            return None
        if mnemonic == "pop":
            return ptr_bytes
        if mnemonic == "push":
            return -ptr_bytes
        if mnemonic in ("ret", "retn"):
            if operands:
                imm = parse_int(operands[0])
                return ptr_bytes + imm if imm is not None else None
            return ptr_bytes
        if mnemonic == "add" and len(operands) == 2:
            dst_regs = cls._extract_registers(operands[0], arch)
            if dst_regs == [cls._stack_pointer_for_arch(arch)]:
                imm = parse_int(operands[1])
                return imm if imm is not None else None
        if mnemonic == "sub" and len(operands) == 2:
            dst_regs = cls._extract_registers(operands[0], arch)
            if dst_regs == [cls._stack_pointer_for_arch(arch)]:
                imm = parse_int(operands[1])
                return -imm if imm is not None else None
        if mnemonic in ("call",):
            return -ptr_bytes  # pushes return address
        if mnemonic in ("nop", "jmp"):
            return 0
        return None

    @classmethod
    def analyze_gadget(
        cls, record: Mapping[str, Any], arch_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Turn a raw gadget record into a typed dict with side effects."""
        instructions = list(record.get("instructions") or [])
        arch = arch_override or record.get("arch")
        analyzed_instructions: List[Dict[str, Any]] = []
        aggregate_reads: List[str] = []
        aggregate_writes: List[str] = []
        aggregate_clobbers: List[str] = []
        aggregate_notes: List[str] = []
        stack_delta: Optional[int] = 0
        for text in instructions:
            parsed = cls._parse_instruction(text)
            if parsed is None:
                analyzed_instructions.append(
                    {
                        "text": text,
                        "mnemonic": None,
                        "operands": [],
                        "reads": [],
                        "writes": [],
                        "clobbers": [],
                        "stack_delta": None,
                        "notes": ["could not parse"],
                    }
                )
                stack_delta = None
                aggregate_notes.append(f"could not parse: '{text}'")
                continue
            mnemonic, operands = parsed
            effects = cls._instruction_side_effects(mnemonic, operands, arch)
            delta = cls._instruction_stack_delta(mnemonic, operands, arch)
            analyzed_instructions.append(
                {
                    "text": text,
                    "mnemonic": mnemonic,
                    "operands": operands,
                    "reads": effects["reads"],
                    "writes": effects["writes"],
                    "clobbers": effects["clobbers"],
                    "stack_delta": delta,
                    "notes": effects["notes"],
                }
            )
            aggregate_reads.extend(effects["reads"])
            aggregate_writes.extend(effects["writes"])
            aggregate_clobbers.extend(effects["clobbers"])
            aggregate_notes.extend(effects["notes"])
            if stack_delta is not None and delta is not None:
                stack_delta += delta
            elif delta is None:
                stack_delta = None

        def _dedup_ordered(seq: Iterable[str]) -> List[str]:
            out: List[str] = []
            for item in seq:
                if item not in out:
                    out.append(item)
            return out

        instruction_text = " ; ".join(instructions)
        canonical_arch = cls._canonical_arch(arch)
        pointer_bytes = cls._pointer_bytes_for_arch(arch)
        return {
            "address": int(record["address"]),
            "bytes": bytes(record.get("bytes") or b"").hex(),
            "instructions": instructions,
            "instruction_text": instruction_text,
            "instructions_analyzed": analyzed_instructions,
            "reads": _dedup_ordered(aggregate_reads),
            "writes": _dedup_ordered(aggregate_writes),
            "clobbers": _dedup_ordered(aggregate_clobbers),
            "stack_delta": stack_delta,
            "arch": arch,
            "arch_canonical": canonical_arch,
            "pointer_bytes": pointer_bytes,
            "notes": _dedup_ordered(aggregate_notes),
        }

    @classmethod
    def _dedupe(cls, gadgets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate identical (bytes, instruction_text) sightings.

        When identical gadget bytes+text appear at several addresses, the
        lowest-address sighting wins: address-space planners prefer the
        earliest placement, and that choice must not depend on loader scan
        order.
        """
        seen: Dict[Tuple[bytes, str], Dict[str, Any]] = {}
        for gadget in gadgets:
            key = (
                (
                    bytes.fromhex(gadget.get("bytes", "") or "")
                    if isinstance(gadget.get("bytes"), str)
                    else bytes(gadget.get("bytes") or b"")
                ),
                gadget.get("instruction_text")
                or " ; ".join(gadget.get("instructions", []) or []),
            )
            existing = seen.get(key)
            if existing is None:
                seen[key] = dict(gadget)
            elif int(gadget.get("address") or 0) < int(existing.get("address") or 0):
                seen[key] = dict(gadget)
        return list(seen.values())

    @classmethod
    def _address_contains_bad_byte(
        cls, address: int, pointer_bytes: Optional[int], bad_bytes: Sequence[int]
    ) -> bool:
        if not bad_bytes:
            return False
        if pointer_bytes is None:
            # Without a pointer width we cannot honestly check; be
            # conservative and report False, and the caller receives a
            # note explaining the arch was unresolved.
            return False
        try:
            encoded = int(address).to_bytes(pointer_bytes, "little", signed=False)
        except OverflowError:
            return True  # address doesn't fit the pointer width -> bad
        return any(b in encoded for b in bad_bytes)

    @staticmethod
    def _normalize_bad_bytes(bad_bytes: Optional[Sequence[Any]]) -> List[int]:
        if not bad_bytes:
            return []
        normalized: List[int] = []
        for item in bad_bytes:
            if isinstance(item, int):
                if not 0 <= item <= 0xFF:
                    raise GadgetIndexError(
                        f"bad_bytes entry {item:#x} is not a byte (0..0xff)"
                    )
                normalized.append(item)
                continue
            if isinstance(item, str):
                stripped = item.strip()
                if not stripped:
                    continue
                try:
                    value = (
                        int(stripped, 16)
                        if stripped.lower().startswith("0x")
                        else int(stripped, 0)
                    )
                except ValueError:
                    raise GadgetIndexError(
                        f"bad_bytes entry '{item}' is not a byte value"
                    )
                if not 0 <= value <= 0xFF:
                    raise GadgetIndexError(
                        f"bad_bytes entry {value:#x} is not a byte (0..0xff)"
                    )
                normalized.append(value)
                continue
            raise GadgetIndexError(
                f"bad_bytes entry {item!r} is not an int or hex string"
            )
        # dedupe while preserving order
        seen: List[int] = []
        for value in normalized:
            if value not in seen:
                seen.append(value)
        return seen

    @staticmethod
    def _compile_pattern(
        pattern: Optional[str],
    ) -> Optional[re.Pattern[str]]:
        if not pattern:
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise GadgetIndexError(
                f"filter pattern '{pattern}' is not a valid regex: {exc}"
            )

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def find_gadgets(
        self,
        *,
        binary_path: str,
        filter_pattern: Optional[str] = None,
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        max_results: int = DEFAULT_MAX_GADGETS,
        max_instructions: int = DEFAULT_MAX_INSTRUCTIONS,
        include_analysis: bool = True,
    ) -> Dict[str, Any]:
        """Enumerate gadgets from a binary with typed side-effect analysis."""
        if max_results <= 0 or max_results > MAX_MAX_GADGETS:
            raise GadgetIndexError(
                f"max_results must be between 1 and {MAX_MAX_GADGETS}"
            )
        if max_instructions <= 0:
            raise GadgetIndexError("max_instructions must be positive")
        compiled_filter = self._compile_pattern(filter_pattern)
        normalized_bad_bytes = self._normalize_bad_bytes(bad_bytes)
        # The analyzed stream is cached (and built in parallel on first
        # sight), so this method is a pure in-memory filter on repeat calls.
        analyzed, _, _ = self._load_analyzed(binary_path, arch, image_base)
        candidates: List[Dict[str, Any]] = []
        for gadget in analyzed:
            if len(gadget["instructions"]) > max_instructions:
                continue
            if compiled_filter is not None and not compiled_filter.search(
                gadget["instruction_text"]
            ):
                continue
            if self._address_contains_bad_byte(
                gadget["address"], gadget["pointer_bytes"], normalized_bad_bytes
            ):
                continue
            candidates.append(gadget)
        deduped = self._dedupe(candidates)
        deduped.sort(key=lambda g: g["address"])
        truncated = len(deduped) > max_results
        page = deduped[:max_results]
        if not include_analysis:
            page = [
                {k: v for k, v in gadget.items() if k != "instructions_analyzed"}
                for gadget in page
            ]
        return build_backend_response(
            "find_gadgets",
            parsed={
                "schema": GADGET_LIST_SCHEMA,
                "binary_path": os.path.abspath(binary_path),
                "arch": arch,
                "image_base": image_base,
                "filter_pattern": filter_pattern,
                "bad_bytes": normalized_bad_bytes,
                "max_results": max_results,
                "max_instructions": max_instructions,
                "total_scanned": len(analyzed),
                "matched": len(deduped),
                "returned": len(page),
                "truncated": truncated,
                "gadgets": page,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def find_gadget_sequence(
        self,
        *,
        binary_path: str,
        steps: Sequence[str],
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        prefer_shortest: bool = True,
    ) -> Dict[str, Any]:
        """Locate one gadget per pattern in ``steps``; return the chain layout.

        Each step is an independent regex applied against the gadget's
        instruction text (``mnemonic op1, op2 ; ...``).  A step with no
        matching gadget yields ``matched=False`` for that slot; the
        overall verdict is ``complete`` only when every step matched.
        """
        if not steps:
            raise GadgetIndexError("steps must be a non-empty sequence of patterns")
        if len(steps) > MAX_SEQUENCE_STEPS:
            raise GadgetIndexError(
                f"steps count exceeds cap ({len(steps)} > {MAX_SEQUENCE_STEPS})"
            )
        compiled_steps = [self._compile_pattern(step) for step in steps]
        normalized_bad_bytes = self._normalize_bad_bytes(bad_bytes)
        # Reuse the cached analyzed stream (built in parallel on first
        # sight).  Anchored steps narrow to the first-mnemonic index;
        # anything else lazily builds the filtered/deduped/sorted full
        # list once and reuses it across steps.
        analyzed, mnemonic_index, _ = self._load_analyzed(binary_path, arch, image_base)
        full_candidates: Optional[List[Dict[str, Any]]] = None
        results: List[Dict[str, Any]] = []
        for step_index, (raw_step, pattern) in enumerate(zip(steps, compiled_steps)):
            match: Optional[Dict[str, Any]] = None
            if pattern is None:
                if full_candidates is None:
                    full_candidates = self._filter_and_sort(
                        analyzed, normalized_bad_bytes, prefer_shortest
                    )
                if full_candidates:
                    match = full_candidates[0]
            else:
                token = self._leading_mnemonic_for_anchor(raw_step)
                if token is not None and token in mnemonic_index:
                    candidates = self._filter_and_sort(
                        [analyzed[index] for index in mnemonic_index[token]],
                        normalized_bad_bytes,
                        prefer_shortest,
                    )
                else:
                    if full_candidates is None:
                        full_candidates = self._filter_and_sort(
                            analyzed, normalized_bad_bytes, prefer_shortest
                        )
                    candidates = full_candidates
                for gadget in candidates:
                    if pattern.search(gadget["instruction_text"]):
                        match = gadget
                        break
            results.append(
                {
                    "step_index": step_index,
                    "pattern": steps[step_index],
                    "matched": match is not None,
                    "gadget": match,
                }
            )
        verdict = "complete" if all(r["matched"] for r in results) else "partial"
        return build_backend_response(
            "find_gadget_sequence",
            parsed={
                "schema": GADGET_SEQUENCE_SCHEMA,
                "binary_path": os.path.abspath(binary_path),
                "arch": arch,
                "image_base": image_base,
                "bad_bytes": normalized_bad_bytes,
                "prefer_shortest": prefer_shortest,
                "verdict": verdict,
                "steps": results,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def inspect_gadget(
        self,
        *,
        binary_path: str,
        address: int,
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Look up a single gadget by absolute address."""
        target = parse_int(address)
        if target is None:
            raise GadgetIndexError(f"address must be an integer, got {address!r}")
        analyzed, _, by_address = self._load_analyzed(binary_path, arch, image_base)
        index = by_address.get(target)
        if index is not None:
            gadget = analyzed[index]
            return build_backend_response(
                "inspect_gadget",
                parsed={
                    "schema": GADGET_INSPECTION_SCHEMA,
                    "binary_path": os.path.abspath(binary_path),
                    "arch": arch,
                    "image_base": image_base,
                    "verdict": "found",
                    "gadget": gadget,
                },
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        return build_backend_response(
            "inspect_gadget",
            parsed={
                "schema": GADGET_INSPECTION_SCHEMA,
                "binary_path": os.path.abspath(binary_path),
                "arch": arch,
                "image_base": image_base,
                "verdict": "not_found",
                "address": target,
                "gadget": None,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


# ---------------------------------------------------------------------------
# Stage 7.2 -- libc identification and one-gadget lookup
# ---------------------------------------------------------------------------
SymbolExtractor = Callable[[str], Dict[str, int]]
OneGadgetRunner = Callable[[str, float], Tuple[int, str, str]]


class LibcIdentifierError(RuntimeError):
    """Raised when a libc identification input is malformed."""


def _pyelftools_symbol_extractor(binary_path: str) -> Dict[str, int]:
    """Extract defined symbol offsets from an ELF file via pyelftools."""
    from elftools.elf.elffile import ELFFile

    symbols: Dict[str, int] = {}
    with open(binary_path, "rb") as handle:
        elf = ELFFile(handle)
        for section in elf.iter_sections():
            sh_type = getattr(section.header, "sh_type", None)
            if sh_type not in ("SHT_SYMTAB", "SHT_DYNSYM"):
                continue
            for symbol in section.iter_symbols():
                name = symbol.name
                if not name:
                    continue
                value = getattr(symbol.entry, "st_value", 0)
                if value == 0:
                    continue
                shndx = getattr(symbol.entry, "st_shndx", None)
                if shndx == "SHN_UNDEF":
                    continue
                # Prefer the earliest definition; a later duplicate is
                # almost always the versioned alias (same offset) or a
                # weak override we do not want to silently overwrite.
                symbols.setdefault(name, int(value))
    return symbols


def _one_gadget_subprocess(
    binary_path: str, timeout_sec: float
) -> Tuple[int, str, str]:
    """Run ``one_gadget`` on ``binary_path``; returns (rc, stdout, stderr)."""
    import shutil
    import subprocess

    executable = shutil.which("one_gadget")
    if executable is None:
        return 127, "", "one_gadget executable not found on PATH"
    try:
        completed = subprocess.run(
            [executable, "--level", "1", binary_path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"one_gadget timed out after {timeout_sec}s: {exc}"
    return completed.returncode, completed.stdout, completed.stderr


class LibcIdentifier:
    """Score ``{symbol: address}`` leaks against candidate libc.so files.

    Every candidate is checked with the invariant that every leaked
    ``address - libc_symbol_offset`` must yield the same base.  A
    candidate is ``consistent`` when at least two leaks agree on a
    base; ``insufficient`` when only one leak matched; ``inconsistent``
    when the bases disagree.  The overall verdict aggregates:

    * ``confirmed``    -- exactly one consistent candidate, >= 3 leaks
    * ``likely``       -- exactly one consistent candidate, 2 leaks
    * ``ambiguous``    -- more than one candidate is consistent
    * ``insufficient`` -- no candidate matches enough leaks to decide
    * ``rejected``     -- every candidate produced inconsistent bases

    ``find_one_gadgets`` shells out to the Ruby ``one_gadget`` tool
    when it is on ``PATH`` and parses its human-readable output into a
    typed list.  When the tool is missing the response is a typed
    ``unsupported`` -- never a silent empty list.
    """

    def __init__(
        self,
        symbol_extractor: SymbolExtractor = _pyelftools_symbol_extractor,
        *,
        one_gadget_runner: OneGadgetRunner = _one_gadget_subprocess,
        max_cached_libcs: int = 8,
    ) -> None:
        if max_cached_libcs <= 0:
            raise ValueError("max_cached_libcs must be positive")
        self._extractor = symbol_extractor
        self._one_gadget_runner = one_gadget_runner
        self._symbol_cache: "Dict[Tuple[str, float], Dict[str, int]]" = {}
        self._symbol_cache_order: List[Tuple[str, float]] = []
        self._max_cached_libcs = max_cached_libcs

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_leaks(
        leaks: Mapping[str, Any],
    ) -> Dict[str, int]:
        if not isinstance(leaks, Mapping) or not leaks:
            raise LibcIdentifierError(
                "leaks must be a non-empty mapping of {symbol_name: address}"
            )
        normalized: Dict[str, int] = {}
        for name, value in leaks.items():
            if not isinstance(name, str) or not name.strip():
                raise LibcIdentifierError(
                    f"leak symbol name must be a non-empty string, got {name!r}"
                )
            addr = parse_int(value)
            if addr is None or addr < 0:
                raise LibcIdentifierError(
                    f"leak address for '{name}' is not a non-negative integer: {value!r}"
                )
            normalized[name] = int(addr)
        return normalized

    @classmethod
    def score_candidate(
        cls,
        libc_path: str,
        symbols: Mapping[str, int],
        leaks: Mapping[str, int],
    ) -> Dict[str, Any]:
        """Return the per-leak base and overall consistency for one libc."""
        matched: List[Dict[str, Any]] = []
        unmatched: List[str] = []
        bases: Dict[int, List[str]] = {}
        for name, leak_addr in leaks.items():
            offset = symbols.get(name)
            if offset is None:
                unmatched.append(name)
                continue
            base = leak_addr - offset
            matched.append(
                {
                    "symbol": name,
                    "leak_address": leak_addr,
                    "offset": offset,
                    "base": base,
                }
            )
            bases.setdefault(base, []).append(name)
        consistent = len(bases) == 1 and len(matched) >= 2
        insufficient = len(matched) == 1 and not bases or len(matched) < 2 and matched
        # Recompute insufficient cleanly: one match at most and every
        # matched base agrees (trivially -- there's only one).
        insufficient = len(matched) == 1
        inconsistent = len(bases) > 1
        base_value: Optional[int]
        if consistent:
            base_value = next(iter(bases))
        elif insufficient and matched:
            base_value = matched[0]["base"]
        else:
            base_value = None
        if consistent:
            verdict = "consistent"
        elif insufficient:
            verdict = "insufficient"
        elif inconsistent:
            verdict = "inconsistent"
        else:
            verdict = "no_matches"
        return {
            "libc_path": os.path.abspath(libc_path),
            "symbol_count": len(symbols),
            "matched": matched,
            "unmatched": unmatched,
            "base": base_value,
            "distinct_bases": [
                {"base": base, "symbols": names} for base, names in bases.items()
            ],
            "verdict": verdict,
        }

    @classmethod
    def aggregate_verdict(
        cls, candidates: Sequence[Mapping[str, Any]]
    ) -> Tuple[str, Optional[Mapping[str, Any]], List[Mapping[str, Any]]]:
        """Aggregate per-candidate verdicts into an overall decision."""
        consistent: List[Mapping[str, Any]] = [
            c for c in candidates if c.get("verdict") == "consistent"
        ]
        insufficient: List[Mapping[str, Any]] = [
            c for c in candidates if c.get("verdict") == "insufficient"
        ]
        if len(consistent) == 1:
            match_count = len(consistent[0].get("matched") or [])
            confidence = "confirmed" if match_count >= 3 else "likely"
            return confidence, consistent[0], list(consistent)
        if len(consistent) > 1:
            return "ambiguous", None, list(consistent)
        if insufficient:
            return "insufficient", None, list(insufficient)
        return "rejected", None, []

    # ------------------------------------------------------------------
    # Symbol cache
    # ------------------------------------------------------------------
    def _symbols_for(self, libc_path: str) -> Dict[str, int]:
        abspath = os.path.abspath(libc_path)
        try:
            mtime = os.path.getmtime(abspath)
        except OSError as exc:
            raise LibcIdentifierError(
                f"candidate libc '{libc_path}' is not readable: {exc}"
            ) from exc
        key = (abspath, mtime)
        cached = self._symbol_cache.get(key)
        if cached is not None:
            return cached
        try:
            symbols = self._extractor(abspath)
        except Exception as exc:
            raise LibcIdentifierError(
                f"failed to extract symbols from '{libc_path}': {exc}"
            ) from exc
        self._symbol_cache[key] = symbols
        self._symbol_cache_order.append(key)
        while len(self._symbol_cache_order) > self._max_cached_libcs:
            evicted = self._symbol_cache_order.pop(0)
            self._symbol_cache.pop(evicted, None)
        return symbols

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def identify_libc(
        self,
        *,
        leaks: Mapping[str, Any],
        candidate_libcs: Sequence[str],
    ) -> Dict[str, Any]:
        """Score every candidate libc against a set of leaked symbol addresses."""
        normalized_leaks = self._normalize_leaks(leaks)
        if not candidate_libcs:
            raise LibcIdentifierError(
                "candidate_libcs must be a non-empty sequence of paths"
            )
        candidates: List[Dict[str, Any]] = []
        for path in candidate_libcs:
            try:
                symbols = self._symbols_for(path)
            except LibcIdentifierError as exc:
                candidates.append(
                    {
                        "libc_path": os.path.abspath(path),
                        "symbol_count": 0,
                        "matched": [],
                        "unmatched": list(normalized_leaks.keys()),
                        "base": None,
                        "distinct_bases": [],
                        "verdict": "unreadable",
                        "error": str(exc),
                    }
                )
                continue
            candidates.append(self.score_candidate(path, symbols, normalized_leaks))
        verdict, best, tie_break = self.aggregate_verdict(candidates)
        return build_backend_response(
            "identify_libc",
            parsed={
                "schema": LIBC_IDENTIFICATION_SCHEMA,
                "verdict": verdict,
                "leak_count": len(normalized_leaks),
                "leaks": normalized_leaks,
                "candidates": candidates,
                "selected": (
                    {
                        "libc_path": best["libc_path"],
                        "base": best["base"],
                        "matched_count": len(best.get("matched") or []),
                    }
                    if isinstance(best, dict)
                    else None
                ),
                "tie_break": (
                    [
                        {"libc_path": c["libc_path"], "base": c["base"]}
                        for c in tie_break
                    ]
                    if verdict == "ambiguous"
                    else []
                ),
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # one_gadget
    # ------------------------------------------------------------------
    _OFFSET_LINE = re.compile(r"^(0x[0-9a-fA-F]+)\s+(.*)$")

    @classmethod
    def parse_one_gadget_output(cls, text: str) -> List[Dict[str, Any]]:
        """Parse the human-readable output of ``one_gadget`` into typed dicts.

        A gadget block looks like:

            0xdeadbeef execve("/bin/sh", rsi, rdx)
            constraints:
              rsi == 0
              rdx == 0 || (u64)[rdx] == 0

        Blocks are separated by blank lines.  Constraints are captured
        verbatim so a caller can reason about them without re-parsing.
        """
        gadgets: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        collecting_constraints = False
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                if current is not None:
                    gadgets.append(current)
                    current = None
                    collecting_constraints = False
                continue
            match = cls._OFFSET_LINE.match(line)
            if match and not collecting_constraints:
                if current is not None:
                    gadgets.append(current)
                current = {
                    "offset": int(match.group(1), 16),
                    "call": match.group(2).strip(),
                    "constraints": [],
                }
                collecting_constraints = False
                continue
            stripped = line.strip()
            if stripped.lower().startswith("constraints"):
                collecting_constraints = True
                continue
            if collecting_constraints and current is not None:
                current["constraints"].append(stripped)
        if current is not None:
            gadgets.append(current)
        return gadgets

    def find_one_gadgets(
        self,
        *,
        libc_path: str,
        timeout_sec: float = DEFAULT_ONE_GADGET_TIMEOUT_SEC,
    ) -> Dict[str, Any]:
        """Return every one-gadget candidate for a libc via the Ruby tool."""
        if timeout_sec <= 0:
            raise LibcIdentifierError("timeout_sec must be positive")
        try:
            mtime = os.path.getmtime(libc_path)
        except OSError as exc:
            raise LibcIdentifierError(
                f"libc '{libc_path}' is not readable: {exc}"
            ) from exc
        rc, stdout, stderr = self._one_gadget_runner(libc_path, timeout_sec)
        if rc == 127:
            return build_backend_response(
                "find_one_gadgets",
                parsed={
                    "schema": ONE_GADGET_SCHEMA,
                    "libc_path": os.path.abspath(libc_path),
                    "libc_mtime": mtime,
                    "supported": False,
                    "verdict": "unsupported",
                    "reason": (
                        stderr.strip()
                        or "one_gadget executable is not on PATH; install with 'gem install one_gadget'"
                    ),
                    "gadgets": [],
                },
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        if rc != 0:
            return build_backend_response(
                "find_one_gadgets",
                parsed={
                    "schema": ONE_GADGET_SCHEMA,
                    "libc_path": os.path.abspath(libc_path),
                    "libc_mtime": mtime,
                    "supported": True,
                    "verdict": "failed",
                    "reason": (stderr or stdout).strip()
                    or f"one_gadget exited with rc={rc}",
                    "gadgets": [],
                },
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        gadgets = self.parse_one_gadget_output(stdout)
        return build_backend_response(
            "find_one_gadgets",
            parsed={
                "schema": ONE_GADGET_SCHEMA,
                "libc_path": os.path.abspath(libc_path),
                "libc_mtime": mtime,
                "supported": True,
                "verdict": "parsed" if gadgets else "empty",
                "gadget_count": len(gadgets),
                "gadgets": gadgets,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


# ---------------------------------------------------------------------------
# Stage 7.3 -- typed ROP chain construction and sigreturn frame builder
# ---------------------------------------------------------------------------
class ChainBuilderError(RuntimeError):
    """Raised when a chain build request is malformed or a gadget is missing."""


class ChainBuilder:
    """Compose typed ROP chains from a live gadget index and libc metadata.

    Supported ``kind`` values today:

    * ``ret2libc``    -- x86_64 SysV: load rdi/rsi/rdx via ``pop <reg>``
      gadgets then transfer control to a target libc symbol (typically
      ``system``, ``execve`` or ``one_gadget``).  Stack alignment is
      computed, not assumed: the chain is entered via ``ret`` with rsp at
      8 (mod 16) on the first slot, every argument pops exactly two slots,
      and the target is entered by the final pop -- so rsp lands on the
      16-byte phase required by ``movaps``-using functions exactly when the
      total pop count is even.  A pure pop-args chain pops ``2*n + 1``
      slots (always odd), so ``align_stack=True`` appends a single bare
      ``ret`` to make the count even; the parity check is explicit so a
      chain shape that already leaves an even pop count skips the ``ret``.
    * ``ret2syscall`` -- x86_64 Linux: load rax with a syscall number,
      then set rdi/rsi/rdx/r10/r8/r9 (SYSCALL replaces rcx with r10)
      via pop gadgets, then execute a ``syscall ; ret`` gadget.

    ``ret2csu``, ``ret2dlresolve`` and stack-pivot flavors are documented
    as deliberate follow-ups: they need multi-gadget solver plumbing
    that is scoped to a later pass.  What ships here is what a real CTF
    ret2libc / ret2syscall exploit uses first.

    The sibling ``inspect_sigreturn_frame`` builds a typed x86_64
    sigreturn frame -- the payload a ``mov rax, 0xf ; syscall`` gadget
    hands to the kernel to force arbitrary register state before a
    single ``ret``.  We wrap :class:`pwnlib.rop.srop.SigreturnFrame` so
    the layout stays authoritative but expose it under the typed backend
    envelope every other S6/S7 tool uses.
    """

    def __init__(self, gadgets: Optional[GadgetIndex] = None) -> None:
        self.gadgets = gadgets if gadgets is not None else GadgetIndex()

    # ------------------------------------------------------------------
    # Chain layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonical_arch(arch: Optional[str]) -> str:
        return GadgetIndex._canonical_arch(arch)

    @staticmethod
    def _pointer_bytes(arch: Optional[str]) -> int:
        bytes_ = GadgetIndex._pointer_bytes_for_arch(arch)
        if bytes_ is None:
            raise ChainBuilderError(
                f"cannot build a ROP chain without a known pointer width for arch={arch!r}"
            )
        return bytes_

    def _find_pop_reg(
        self,
        *,
        binary_path: str,
        register: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        """Locate a ``pop <register> ; ret`` gadget (shortest wins)."""
        # We require the *simplest* possible gadget so an accidental side
        # effect (extra pop / mov) cannot corrupt an unrelated register
        # the caller was about to load themselves.
        pattern = rf"^\s*pop\s+{re.escape(register)}\s*;\s*ret\s*$"
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[pattern],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    def _find_bare_ret(
        self,
        *,
        binary_path: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[r"^\s*ret\s*$"],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    def _find_syscall_ret(
        self,
        *,
        binary_path: str,
        arch: str,
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Optional[Dict[str, Any]]:
        response = self.gadgets.find_gadget_sequence(
            binary_path=binary_path,
            steps=[r"^\s*syscall\s*;\s*ret\s*$"],
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=True,
        )
        step = (response.get("parsed", {}).get("steps") or [None])[0]
        return step["gadget"] if step and step.get("matched") else None

    @staticmethod
    def _encode_entries(
        entries: Sequence[Mapping[str, Any]], pointer_bytes: int
    ) -> bytes:
        buf = bytearray()
        for entry in entries:
            kind = entry.get("kind")
            if kind == "gadget":
                value = int(entry["address"])
            elif kind == "target":
                value = int(entry["address"])
            elif kind == "immediate":
                value = int(entry["value"])
            elif kind == "raw_bytes":
                raw = entry.get("bytes") or b""
                if isinstance(raw, str):
                    raw = bytes.fromhex(raw)
                buf.extend(raw)
                continue
            else:
                raise ChainBuilderError(f"unsupported chain entry kind: {kind!r}")
            try:
                buf.extend(int(value).to_bytes(pointer_bytes, "little", signed=False))
            except OverflowError as exc:
                raise ChainBuilderError(
                    f"chain entry value {value:#x} does not fit "
                    f"{pointer_bytes * 8}-bit pointer"
                ) from exc
        return bytes(buf)

    # ------------------------------------------------------------------
    # ret2libc
    # ------------------------------------------------------------------
    def _ret2libc_amd64(
        self,
        *,
        binary_path: str,
        target_address: int,
        args: Sequence[int],
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
        align_stack: bool,
        target_symbol: Optional[str],
    ) -> Dict[str, Any]:
        if len(args) > len(_SYSV_AMD64_ARG_REGS):
            raise ChainBuilderError(
                f"ret2libc supports up to {len(_SYSV_AMD64_ARG_REGS)} args, "
                f"got {len(args)}"
            )
        entries: List[Dict[str, Any]] = []
        assumptions: List[str] = [
            "x86_64 SysV: first args land in rdi, rsi, rdx, rcx, r8, r9",
            "The target function must reach a 16-byte-aligned rsp on entry",
        ]
        missing: List[str] = []
        arg_details: List[Dict[str, Any]] = []
        for index, value in enumerate(args):
            register = _SYSV_AMD64_ARG_REGS[index]
            gadget = self._find_pop_reg(
                binary_path=binary_path,
                register=register,
                arch="x86_64",
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
            arg_details.append(
                {
                    "index": index,
                    "register": register,
                    "value": int(value),
                    "gadget": gadget,
                }
            )
            if gadget is None:
                missing.append(f"pop {register} ; ret")
                continue
            entries.append(
                {
                    "kind": "gadget",
                    "address": gadget["address"],
                    "instructions": gadget["instruction_text"],
                    "purpose": f"load {register}",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(value),
                    "purpose": f"{register} = {value:#x}",
                }
            )
        if align_stack:
            # SysV alignment model (documented so a caller can verify the
            # emitted chain, not just trust that a ``ret`` exists):
            #
            #   * The chain is entered via ``ret`` from the overflowed frame.
            #     When that pivot ``ret`` executes, rsp points at the first
            #     chain slot and is 8 (mod 16) -- the standard epilogue state
            #     (the ABI leaves the saved return-address slot at 8 mod 16).
            #   * Each argument contributes exactly two 8-byte pops: the
            #     ``pop <reg>`` consumes the immediate and the gadget's own
            #     ``ret`` consumes the next slot.  A bare ``ret`` contributes
            #     one pop.
            #   * The target is entered by the final pop, so rsp must be 8
            #     (mod 16) there -- i.e. the total pop count from the pivot
            #     slot through the target pop must be EVEN.
            #   * A pure pop-args chain pops ``2*len(args) + 1`` slots, which
            #     is always odd, so a bare ``ret`` is appended to reach an
            #     even count and land rsp on the correct phase.  The parity
            #     check is written explicitly rather than as an unconditional
            #     insert, so a future chain-shape change (a leading pivot or
            #     extra-grooming gadget that adds slots) re-derives the
            #     alignment instead of blindly appending a ``ret`` that would
            #     flip an already-correct phase.
            pop_count = 2 * len(args) + 1
            if pop_count % 2 == 1:
                ret_gadget = self._find_bare_ret(
                    binary_path=binary_path,
                    arch="x86_64",
                    image_base=image_base,
                    bad_bytes=bad_bytes,
                )
                if ret_gadget is None:
                    assumptions.append(
                        "no bare 'ret' gadget available for stack alignment; "
                        "the target may crash on movaps if rsp is not 16-byte aligned"
                    )
                else:
                    entries.append(
                        {
                            "kind": "gadget",
                            "address": ret_gadget["address"],
                            "instructions": ret_gadget["instruction_text"],
                            "purpose": "align rsp to 16 bytes before the target call",
                        }
                    )
            else:
                assumptions.append(
                    "chain pop count is already even; rsp is on the correct "
                    "16-byte phase at target entry without an alignment ret"
                )
        entries.append(
            {
                "kind": "target",
                "address": int(target_address),
                "symbol": target_symbol,
                "purpose": (
                    f"call {target_symbol}" if target_symbol else "call target"
                ),
            }
        )
        pointer_bytes = 8
        try:
            raw_bytes = self._encode_entries(entries, pointer_bytes)
        except ChainBuilderError:
            # Missing gadgets already recorded; skip encoding.
            raw_bytes = b""
        verdict = "built" if not missing else "missing_gadgets"
        return {
            "schema": ROP_CHAIN_SCHEMA,
            "kind": "ret2libc",
            "arch": "x86_64",
            "pointer_bytes": pointer_bytes,
            "target": {
                "address": int(target_address),
                "symbol": target_symbol,
            },
            "args": arg_details,
            "align_stack": align_stack,
            "entries": entries,
            "raw_bytes": raw_bytes.hex(),
            "raw_bytes_length": len(raw_bytes),
            "missing_gadgets": missing,
            "assumptions": assumptions,
            "verdict": verdict,
            "reason": (
                None
                if not missing
                else "one or more required pop <reg> gadgets were not found"
            ),
        }

    # ------------------------------------------------------------------
    # ret2syscall
    # ------------------------------------------------------------------
    def _ret2syscall_amd64(
        self,
        *,
        binary_path: str,
        syscall_number: int,
        args: Sequence[int],
        image_base: Optional[int],
        bad_bytes: Optional[Sequence[Any]],
    ) -> Dict[str, Any]:
        if len(args) > len(_LINUX_AMD64_SYSCALL_ARG_REGS):
            raise ChainBuilderError(
                f"ret2syscall supports up to {len(_LINUX_AMD64_SYSCALL_ARG_REGS)} args, "
                f"got {len(args)}"
            )
        entries: List[Dict[str, Any]] = []
        assumptions: List[str] = [
            "Linux x86_64 syscall ABI: rax = number; args in rdi, rsi, rdx, r10, r8, r9",
            "SYSCALL clobbers rcx and r11 -- caller must not rely on them post-syscall",
        ]
        missing: List[str] = []
        arg_details: List[Dict[str, Any]] = []
        pop_rax = self._find_pop_reg(
            binary_path=binary_path,
            register="rax",
            arch="x86_64",
            image_base=image_base,
            bad_bytes=bad_bytes,
        )
        if pop_rax is None:
            missing.append("pop rax ; ret")
        else:
            entries.append(
                {
                    "kind": "gadget",
                    "address": pop_rax["address"],
                    "instructions": pop_rax["instruction_text"],
                    "purpose": "load rax (syscall number)",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(syscall_number),
                    "purpose": f"rax = {syscall_number}",
                }
            )
        for index, value in enumerate(args):
            register = _LINUX_AMD64_SYSCALL_ARG_REGS[index]
            gadget = self._find_pop_reg(
                binary_path=binary_path,
                register=register,
                arch="x86_64",
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
            arg_details.append(
                {
                    "index": index,
                    "register": register,
                    "value": int(value),
                    "gadget": gadget,
                }
            )
            if gadget is None:
                missing.append(f"pop {register} ; ret")
                continue
            entries.append(
                {
                    "kind": "gadget",
                    "address": gadget["address"],
                    "instructions": gadget["instruction_text"],
                    "purpose": f"load {register}",
                }
            )
            entries.append(
                {
                    "kind": "immediate",
                    "value": int(value),
                    "purpose": f"{register} = {value:#x}",
                }
            )
        syscall_gadget = self._find_syscall_ret(
            binary_path=binary_path,
            arch="x86_64",
            image_base=image_base,
            bad_bytes=bad_bytes,
        )
        if syscall_gadget is None:
            missing.append("syscall ; ret")
        else:
            entries.append(
                {
                    "kind": "gadget",
                    "address": syscall_gadget["address"],
                    "instructions": syscall_gadget["instruction_text"],
                    "purpose": "trigger syscall",
                }
            )
        pointer_bytes = 8
        try:
            raw_bytes = self._encode_entries(entries, pointer_bytes)
        except ChainBuilderError:
            raw_bytes = b""
        verdict = "built" if not missing else "missing_gadgets"
        return {
            "schema": ROP_CHAIN_SCHEMA,
            "kind": "ret2syscall",
            "arch": "x86_64",
            "pointer_bytes": pointer_bytes,
            "target": {
                "syscall_number": int(syscall_number),
            },
            "args": arg_details,
            "entries": entries,
            "raw_bytes": raw_bytes.hex(),
            "raw_bytes_length": len(raw_bytes),
            "missing_gadgets": missing,
            "assumptions": assumptions,
            "verdict": verdict,
            "reason": (
                None if not missing else "one or more required gadgets were not found"
            ),
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def build_rop_chain(
        self,
        *,
        binary_path: str,
        kind: str,
        arch: str = "x86_64",
        image_base: Optional[int] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        target_address: Optional[int] = None,
        target_symbol: Optional[str] = None,
        args: Optional[Sequence[int]] = None,
        align_stack: bool = True,
        syscall_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Assemble a typed ROP chain for the requested exploit shape."""
        canonical_arch = self._canonical_arch(arch)
        if canonical_arch != "x86_64":
            parsed = {
                "schema": ROP_CHAIN_SCHEMA,
                "kind": kind,
                "arch": arch,
                "supported": False,
                "verdict": "unsupported_arch",
                "reason": (
                    f"chain construction currently supports x86_64 only; "
                    f"got arch={arch!r}"
                ),
                "entries": [],
                "raw_bytes": "",
                "raw_bytes_length": 0,
                "assumptions": [],
            }
            return build_backend_response(
                "build_rop_chain",
                parsed=parsed,
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )

        args_seq = list(args or [])
        if kind == "ret2libc":
            if target_address is None:
                raise ChainBuilderError(
                    "ret2libc requires a resolved target_address "
                    "(compute libc_base + symbol_offset via identify_libc first)"
                )
            parsed = self._ret2libc_amd64(
                binary_path=binary_path,
                target_address=int(target_address),
                args=args_seq,
                image_base=image_base,
                bad_bytes=bad_bytes,
                align_stack=align_stack,
                target_symbol=target_symbol,
            )
        elif kind == "ret2syscall":
            if syscall_number is None:
                raise ChainBuilderError("ret2syscall requires a syscall_number")
            parsed = self._ret2syscall_amd64(
                binary_path=binary_path,
                syscall_number=int(syscall_number),
                args=args_seq,
                image_base=image_base,
                bad_bytes=bad_bytes,
            )
        else:
            raise ChainBuilderError(
                f"unsupported chain kind {kind!r}; expected 'ret2libc' or 'ret2syscall'"
            )
        return build_backend_response(
            "build_rop_chain",
            parsed=parsed,
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # SROP frame builder
    # ------------------------------------------------------------------
    _SIGRETURN_SUPPORTED_ARCHES: Tuple[str, ...] = ("x86_64", "i386")

    def inspect_sigreturn_frame(
        self,
        *,
        arch: str = "x86_64",
        registers: Optional[Mapping[str, int]] = None,
    ) -> Dict[str, Any]:
        """Build a typed sigreturn frame for the requested target arch.

        Delegates the byte-level layout to pwntools' SigreturnFrame so
        the offsets stay authoritative, but presents the result under
        the typed backend envelope every other stage uses: the raw
        bytes, the frame size, and every register slot's byte offset
        so a caller can hand-patch fields after the initial build.
        """
        canonical_arch = self._canonical_arch(arch)
        if canonical_arch not in self._SIGRETURN_SUPPORTED_ARCHES:
            return build_backend_response(
                "inspect_sigreturn_frame",
                parsed={
                    "schema": SIGRETURN_FRAME_SCHEMA,
                    "arch": arch,
                    "supported": False,
                    "verdict": "unsupported_arch",
                    "reason": (
                        f"sigreturn frame builder currently supports "
                        f"{self._SIGRETURN_SUPPORTED_ARCHES}; got arch={arch!r}"
                    ),
                    "frame_size": 0,
                    "frame_bytes": "",
                    "register_offsets": {},
                    "registers": {},
                    "syscall_register": None,
                    "syscall_number": None,
                },
                raw_responses=[],
                state="STATIC",
                parser_version=PARSER_VERSION,
                warnings=None,
            )
        from pwnlib.context import context as _pwn_context  # type: ignore[import-untyped]
        from pwnlib.rop.srop import SigreturnFrame  # type: ignore[import-untyped]  # local: pwntools is optional

        pwn_arch = "amd64" if canonical_arch == "x86_64" else "i386"
        # pwntools' SigreturnFrame is context-sensitive (arch + bits
        # come from the process-wide pwnlib.context) and blows up on
        # a stale bits value from a previous i386 call.  Scope the
        # context per call so this method is thread-safe and idempotent.
        pwn_bits = 64 if pwn_arch == "amd64" else 32
        # pwntools' ``context.kernel`` refers to the *kernel CPU arch*,
        # not the OS.  For a 32-bit i386 process running on a modern
        # 64-bit Linux kernel the sigcontext layout is the amd64 one;
        # amd64 processes ignore the field.  The context is scoped so
        # this method stays thread-safe.
        with _pwn_context.local(
            arch=pwn_arch,
            bits=pwn_bits,
            endian="little",
            kernel="amd64",
        ):
            frame = SigreturnFrame(arch=pwn_arch)
        assigned: Dict[str, int] = {}
        if registers:
            for name, value in registers.items():
                if name not in frame.registers.values():
                    raise ChainBuilderError(
                        f"register '{name}' is not part of a {canonical_arch} "
                        "sigreturn frame"
                    )
                # SigreturnFrame supports attribute assignment for every
                # register in ``registers.values()``.
                setattr(frame, name, int(value))
                assigned[name] = int(value)
        raw = bytes(frame)
        offsets: Dict[str, int] = {
            name: offset for offset, name in frame.registers.items()
        }
        # rt_sigreturn syscall number differs per arch: 15 on x86_64,
        # 173 on i386.  Surfaced so a caller building the trigger chain
        # does not have to look it up.
        syscall_number = 15 if canonical_arch == "x86_64" else 173
        return build_backend_response(
            "inspect_sigreturn_frame",
            parsed={
                "schema": SIGRETURN_FRAME_SCHEMA,
                "arch": canonical_arch,
                "supported": True,
                "verdict": "built",
                "frame_size": len(raw),
                "frame_bytes": raw.hex(),
                "register_offsets": offsets,
                "registers": assigned,
                "syscall_register": "rax" if canonical_arch == "x86_64" else "eax",
                "syscall_number": syscall_number,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    # ------------------------------------------------------------------
    # Chain constraints explanation
    # ------------------------------------------------------------------
    @staticmethod
    def explain_chain_constraints(
        *,
        chain: Sequence[Mapping[str, Any]],
        arch: str = "x86_64",
    ) -> Dict[str, Any]:
        """Summarize per-slot side effects and cross-slot register aliasing.

        Walks ``chain`` (as returned by :meth:`build_rop_chain`'s
        ``entries`` list) and reports which registers each gadget writes
        or clobbers, which slots hold immediates the chain treats as
        register values, and any register a later slot depends on that
        an earlier slot clobbers.  The output is intentionally advisory
        -- the goal is to help a caller notice a chain layout that is
        obviously wrong before they hand it to the target process.
        """
        canonical_arch = ChainBuilder._canonical_arch(arch)
        slots: List[Dict[str, Any]] = []
        register_state: Dict[str, str] = {}
        risky: List[Dict[str, Any]] = []
        for index, entry in enumerate(chain):
            kind = entry.get("kind")
            slot: Dict[str, Any] = {
                "index": index,
                "kind": kind,
                "purpose": entry.get("purpose"),
            }
            if kind == "gadget":
                # analyze the gadget text via GadgetIndex helpers
                instructions = str(entry.get("instructions") or "")
                lines = [
                    line.strip() for line in instructions.split(";") if line.strip()
                ]
                reads: List[str] = []
                writes: List[str] = []
                clobbers: List[str] = []
                for text in lines:
                    parsed = GadgetIndex._parse_instruction(text)
                    if parsed is None:
                        continue
                    mnemonic, operands = parsed
                    effects = GadgetIndex._instruction_side_effects(
                        mnemonic, operands, arch
                    )
                    reads.extend(effects["reads"])
                    writes.extend(effects["writes"])
                    clobbers.extend(effects["clobbers"])
                slot["reads"] = list(dict.fromkeys(reads))
                slot["writes"] = list(dict.fromkeys(writes))
                slot["clobbers"] = list(dict.fromkeys(clobbers))
                # Track: any previously-loaded arg register that this
                # gadget writes will get overwritten -- that is only a
                # problem if the *next* target read expects it to still
                # hold the earlier value.  We surface the write for the
                # caller to reason about; we do not decide correctness.
                for reg in slot["writes"] + slot["clobbers"]:
                    if reg in register_state and register_state[reg] != slot["purpose"]:
                        risky.append(
                            {
                                "slot_index": index,
                                "register": reg,
                                "previously_loaded_by": register_state[reg],
                                "now_overwritten_by": slot["purpose"]
                                or f"slot {index}",
                            }
                        )
                for reg in slot["writes"]:
                    register_state[reg] = slot["purpose"] or f"slot {index}"
            elif kind == "immediate":
                slot["value"] = entry.get("value")
            elif kind == "target":
                slot["address"] = entry.get("address")
                slot["symbol"] = entry.get("symbol")
            slots.append(slot)
        return build_backend_response(
            "explain_chain_constraints",
            parsed={
                "schema": CHAIN_CONSTRAINTS_SCHEMA,
                "arch": canonical_arch,
                "slot_count": len(slots),
                "slots": slots,
                "register_state_after_chain": register_state,
                "risky_overwrites": risky,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


class ChainValidatorError(RuntimeError):
    """Raised when a chain validation input is malformed."""


class ChainValidator:
    """Static validation and export for typed ROP chains.

    Validation is deliberately static: we check every gadget address
    lands in an executable mapping, verify no encoded pointer contains
    a forbidden byte, and surface a register-clobber summary drawn from
    the S7.3 :meth:`ChainBuilder.explain_chain_constraints` analyzer.
    Dynamic sandbox validation (execute the chain against a real
    debugger inferior) requires a live target and is documented as a
    future S7 extension rather than faked here.

    Export supports three formats:

    * ``raw``     -- hex-encoded little-endian pointer blob
    * ``python``  -- ``struct.pack``-flavored snippet suitable for
                     a Python 3 script (no pwntools dependency)
    * ``pwntools`` -- a ``p64(...)`` payload string a caller can
                      paste into a pwntools exploit template
    """

    def __init__(self, builder: Optional[ChainBuilder] = None) -> None:
        self._builder = builder if builder is not None else ChainBuilder()

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mapping_contains(mapping: Mapping[str, Any], address: int) -> bool:
        start = parse_int(mapping.get("start"))
        end = parse_int(mapping.get("end"))
        if start is None or end is None:
            return False
        return start <= address < end

    @staticmethod
    def _mapping_is_executable(mapping: Mapping[str, Any]) -> bool:
        # Accept either a boolean field or a POSIX-style "r-xp" string.
        if mapping.get("executable") is True:
            return True
        perms = str(mapping.get("permissions") or mapping.get("perms") or "")
        return "x" in perms.lower()

    @classmethod
    def _classify_address(
        cls,
        address: int,
        executable_mappings: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        matches: List[Mapping[str, Any]] = [
            m for m in executable_mappings if cls._mapping_contains(m, address)
        ]
        if not matches:
            return {"in_mapping": False, "executable": False, "mapping": None}
        # If multiple overlap, prefer the executable one.
        executable_hits = [m for m in matches if cls._mapping_is_executable(m)]
        chosen = executable_hits[0] if executable_hits else matches[0]
        return {
            "in_mapping": True,
            "executable": cls._mapping_is_executable(chosen),
            "mapping": {
                "start": parse_int(chosen.get("start")),
                "end": parse_int(chosen.get("end")),
                "file": chosen.get("file"),
                "permissions": chosen.get("permissions") or chosen.get("perms"),
            },
        }

    # ------------------------------------------------------------------
    # Public typed operations
    # ------------------------------------------------------------------
    def validate_rop_chain(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        executable_mappings: Optional[Sequence[Mapping[str, Any]]] = None,
        bad_bytes: Optional[Sequence[Any]] = None,
        arch: str = "x86_64",
    ) -> Dict[str, Any]:
        """Return a static pass / fail / unknown verdict for ``chain``.

        Checks (each surfaces per-slot evidence):

        * ``exec_mapping``  -- every gadget/target address lies in an
          executable mapping (skipped with ``unknown`` when the caller
          omits ``executable_mappings``)
        * ``bad_bytes``     -- every gadget/target/immediate address,
          when packed little-endian for the arch's pointer width, is
          free of every forbidden byte
        * ``register_clobber`` -- borrowed from
          :meth:`ChainBuilder.explain_chain_constraints`; surfaces
          risky overwrites but does not fail the chain by itself
          because a plausible chain may deliberately reload a register.
        """
        canonical_arch = self._builder._canonical_arch(arch)
        try:
            pointer_bytes = self._builder._pointer_bytes(canonical_arch)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        normalized_bad_bytes = GadgetIndex._normalize_bad_bytes(bad_bytes)
        executable_mappings_list = list(executable_mappings or [])
        exec_check: Dict[str, Any] = {
            "name": "exec_mapping",
            "verdict": "unknown" if not executable_mappings_list else "pass",
            "issues": [],
            "checked_slots": [],
        }
        bad_byte_check: Dict[str, Any] = {
            "name": "bad_bytes",
            "verdict": "pass",
            "issues": [],
            "bad_bytes": normalized_bad_bytes,
        }
        for index, entry in enumerate(chain):
            kind = entry.get("kind")
            address: Optional[int] = None
            if kind in ("gadget", "target"):
                address = parse_int(entry.get("address"))
            elif kind == "immediate":
                address = parse_int(entry.get("value"))
            if address is None:
                continue
            if kind in ("gadget", "target") and executable_mappings_list:
                classification = self._classify_address(
                    address, executable_mappings_list
                )
                classification["slot_index"] = index
                classification["address"] = address
                exec_check["checked_slots"].append(classification)
                if not classification["in_mapping"]:
                    exec_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "reason": "address is outside every provided mapping",
                        }
                    )
                elif not classification["executable"]:
                    exec_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "reason": "address is in a non-executable mapping",
                        }
                    )
            if normalized_bad_bytes:
                if GadgetIndex._address_contains_bad_byte(
                    address, pointer_bytes, normalized_bad_bytes
                ):
                    bad_byte_check["issues"].append(
                        {
                            "slot_index": index,
                            "address": address,
                            "kind": kind,
                            "reason": "encoded pointer contains a forbidden byte",
                        }
                    )
        if exec_check["issues"]:
            exec_check["verdict"] = "fail"
        if bad_byte_check["issues"]:
            bad_byte_check["verdict"] = "fail"

        clobber_response = self._builder.explain_chain_constraints(
            chain=chain, arch=canonical_arch
        )
        clobber_parsed = clobber_response["parsed"]
        clobber_check = {
            "name": "register_clobber",
            "verdict": ("warn" if clobber_parsed.get("risky_overwrites") else "pass"),
            "risky_overwrites": clobber_parsed.get("risky_overwrites") or [],
        }

        checks = [exec_check, bad_byte_check, clobber_check]
        verdicts = {c["verdict"] for c in checks}
        if "fail" in verdicts:
            overall = "fail"
        elif "unknown" in verdicts:
            overall = "unknown"
        else:
            overall = "pass"
        return build_backend_response(
            "validate_rop_chain",
            parsed={
                "schema": CHAIN_VALIDATION_SCHEMA,
                "arch": canonical_arch,
                "pointer_bytes": pointer_bytes,
                "slot_count": len(chain),
                "verdict": overall,
                "checks": checks,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def export_chain(
        self,
        *,
        chain: Sequence[Mapping[str, Any]],
        format: str = "raw",
        arch: str = "x86_64",
        variable_name: str = "chain",
    ) -> Dict[str, Any]:
        """Serialize a chain into a raw / python / pwntools payload."""
        if format not in CHAIN_EXPORT_FORMATS:
            raise ChainValidatorError(
                f"unsupported export format {format!r}; expected one of "
                f"{CHAIN_EXPORT_FORMATS}"
            )
        canonical_arch = self._builder._canonical_arch(arch)
        try:
            pointer_bytes = self._builder._pointer_bytes(canonical_arch)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        try:
            raw_bytes = self._builder._encode_entries(chain, pointer_bytes)
        except ChainBuilderError as exc:
            raise ChainValidatorError(str(exc)) from exc
        text: str
        if format == "raw":
            text = raw_bytes.hex()
        elif format == "python":
            struct_fmt = "<Q" if pointer_bytes == 8 else "<I"
            lines = ["import struct", "", f'{variable_name} = b""']
            for entry in chain:
                kind = entry.get("kind")
                if kind == "raw_bytes":
                    payload = entry.get("bytes") or b""
                    if isinstance(payload, str):
                        payload = bytes.fromhex(payload)
                    lines.append(f"{variable_name} += bytes.fromhex({payload.hex()!r})")
                    continue
                raw_value = (
                    entry.get("address")
                    if kind in ("gadget", "target")
                    else entry.get("value")
                )
                value = parse_int(raw_value)
                if value is None:
                    raise ChainValidatorError(
                        f"chain entry #{len(lines)} is missing a numeric "
                        f"address/value for kind={kind!r}"
                    )
                purpose = entry.get("purpose") or kind
                lines.append(
                    f"{variable_name} += struct.pack({struct_fmt!r}, {value:#x})"
                    f"  # {purpose}"
                )
            text = "\n".join(lines)
        else:  # pwntools
            pack_fn = "p64" if pointer_bytes == 8 else "p32"
            lines = [
                "from pwn import *",
                "",
                f"context.arch = {'amd64' if canonical_arch == 'x86_64' else canonical_arch!r}",
                "",
                f'{variable_name} = b""',
            ]
            for entry in chain:
                kind = entry.get("kind")
                if kind == "raw_bytes":
                    payload = entry.get("bytes") or b""
                    if isinstance(payload, str):
                        payload = bytes.fromhex(payload)
                    lines.append(f"{variable_name} += bytes.fromhex({payload.hex()!r})")
                    continue
                raw_value = (
                    entry.get("address")
                    if kind in ("gadget", "target")
                    else entry.get("value")
                )
                value = parse_int(raw_value)
                if value is None:
                    raise ChainValidatorError(
                        f"chain entry #{len(lines)} is missing a numeric "
                        f"address/value for kind={kind!r}"
                    )
                purpose = entry.get("purpose") or kind
                lines.append(f"{variable_name} += {pack_fn}({value:#x})  # {purpose}")
            text = "\n".join(lines)
        return build_backend_response(
            "export_chain",
            parsed={
                "schema": CHAIN_EXPORT_SCHEMA,
                "arch": canonical_arch,
                "pointer_bytes": pointer_bytes,
                "format": format,
                "slot_count": len(chain),
                "raw_bytes_length": len(raw_bytes),
                "raw_bytes": raw_bytes.hex(),
                "text": text,
            },
            raw_responses=[],
            state="STATIC",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


class RopTools:
    """Facade grouping every S7 typed backend under one session attribute."""

    def __init__(
        self,
        gadgets: Optional[GadgetIndex] = None,
        libc: Optional[LibcIdentifier] = None,
        chains: Optional[ChainBuilder] = None,
        validator: Optional[ChainValidator] = None,
    ) -> None:
        self.gadgets = gadgets if gadgets is not None else GadgetIndex()
        self.libc = libc if libc is not None else LibcIdentifier()
        self.chains = chains if chains is not None else ChainBuilder(self.gadgets)
        self.validator = (
            validator if validator is not None else ChainValidator(self.chains)
        )

    # ---- S7.1 gadget indexing ----------------------------------------
    def find_gadgets(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.find_gadgets(**kwargs)

    def find_gadget_sequence(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.find_gadget_sequence(**kwargs)

    def inspect_gadget(self, **kwargs: Any) -> Dict[str, Any]:
        return self.gadgets.inspect_gadget(**kwargs)

    # ---- S7.2 libc identification ------------------------------------
    def identify_libc(self, **kwargs: Any) -> Dict[str, Any]:
        return self.libc.identify_libc(**kwargs)

    def find_one_gadgets(self, **kwargs: Any) -> Dict[str, Any]:
        return self.libc.find_one_gadgets(**kwargs)

    # ---- S7.3 chain construction -------------------------------------
    def build_rop_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.build_rop_chain(**kwargs)

    def inspect_sigreturn_frame(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.inspect_sigreturn_frame(**kwargs)

    def explain_chain_constraints(self, **kwargs: Any) -> Dict[str, Any]:
        return self.chains.explain_chain_constraints(**kwargs)

    # ---- S7.4 chain validation + export ------------------------------
    def validate_rop_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.validator.validate_rop_chain(**kwargs)

    def export_chain(self, **kwargs: Any) -> Dict[str, Any]:
        return self.validator.export_chain(**kwargs)


__all__ = [
    "GadgetIndex",
    "GadgetIndexError",
    "GadgetLoader",
    "LibcIdentifier",
    "LibcIdentifierError",
    "ChainBuilder",
    "ChainBuilderError",
    "ChainValidator",
    "ChainValidatorError",
    "SymbolExtractor",
    "OneGadgetRunner",
    "RopTools",
    "GADGET_LIST_SCHEMA",
    "GADGET_SEQUENCE_SCHEMA",
    "GADGET_INSPECTION_SCHEMA",
    "LIBC_IDENTIFICATION_SCHEMA",
    "ONE_GADGET_SCHEMA",
    "ROP_CHAIN_SCHEMA",
    "SIGRETURN_FRAME_SCHEMA",
    "CHAIN_CONSTRAINTS_SCHEMA",
    "CHAIN_VALIDATION_SCHEMA",
    "CHAIN_EXPORT_SCHEMA",
    "CHAIN_EXPORT_FORMATS",
    "DEFAULT_MAX_GADGETS",
    "MAX_MAX_GADGETS",
    "DEFAULT_MAX_INSTRUCTIONS",
    "MAX_SEQUENCE_STEPS",
    "DEFAULT_ONE_GADGET_TIMEOUT_SEC",
    "ropper_gadget_loader",
]
