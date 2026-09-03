"""Typed ROP gadget indexing, S7.1 (split from rop.py)."""

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
from paomcp.tools.backends.rop_schemas import (
    GADGET_INSPECTION_SCHEMA,
    GADGET_LIST_SCHEMA,
    GADGET_SEQUENCE_SCHEMA,
    PARSER_VERSION,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_GADGETS = 100
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


__all__ = [
    "GadgetIndex",
    "GadgetIndexError",
    "GadgetLoader",
    "ropper_gadget_loader",
    "_extract_instruction",
    "_analyze_gadget_batch",
    "DEFAULT_MAX_GADGETS",
    "MAX_MAX_GADGETS",
    "DEFAULT_MAX_INSTRUCTIONS",
    "MAX_SEQUENCE_STEPS",
    "PARALLEL_ANALYSIS_THRESHOLD",
    "PARALLEL_ANALYSIS_CHUNK",
    "MAX_ANALYSIS_WORKERS",
    "PARALLEL_ANALYSIS_TIMEOUT_SEC",
]
