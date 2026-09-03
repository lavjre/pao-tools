"""Libc identification and one-gadget lookup, S7.2 (split from rop.py)."""

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
    DEFAULT_ONE_GADGET_TIMEOUT_SEC,
    LIBC_IDENTIFICATION_SCHEMA,
    ONE_GADGET_SCHEMA,
    PARSER_VERSION,
)

logger = logging.getLogger(__name__)


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


__all__ = [
    "LibcIdentifier",
    "LibcIdentifierError",
    "SymbolExtractor",
    "OneGadgetRunner",
]
