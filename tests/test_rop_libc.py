"""Tests for libc identification and one-gadget lookup."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Tuple

import pytest

from paomcp.tools.backends.rop import (
    LIBC_IDENTIFICATION_SCHEMA,
    ONE_GADGET_SCHEMA,
    LibcIdentifier,
    LibcIdentifierError,
)


def _tempfile(name: str = "libc.so.6") -> str:
    handle = tempfile.NamedTemporaryFile(
        prefix=name.replace(".", "_") + "_", delete=False
    )
    handle.write(b"\x7fELF")  # not a real ELF; extractor is faked in tests
    handle.close()
    return handle.name


def _fake_extractor(
    symbols: Dict[str, Dict[str, int]],
) -> Callable[[str], Dict[str, int]]:
    """Return an extractor that matches by basename, tolerant of tempfile munging.

    ``_tempfile`` builds prefixes like ``libc-2.35.so`` → ``libc-2_35_so_``,
    so the extractor normalizes both the fixture key and the actual
    basename to underscore form before checking containment.
    """

    def _canonicalize(text: str) -> str:
        return text.replace(".", "_").lower()

    def _extract(path: str) -> Dict[str, int]:
        basename = _canonicalize(os.path.basename(path))
        for stub, entries in symbols.items():
            if _canonicalize(stub) in basename:
                return dict(entries)
        raise RuntimeError(f"unknown fixture libc: {path}")

    return _extract


def _fake_one_gadget_runner(
    rc: int, stdout: str, stderr: str = ""
) -> Callable[[str, float], Tuple[int, str, str]]:
    def _run(binary_path: str, timeout_sec: float) -> Tuple[int, str, str]:
        return rc, stdout, stderr

    return _run


# ---------------------------------------------------------------------------
# Pure scoring
# ---------------------------------------------------------------------------
class TestScoreCandidate:
    def test_consistent_when_all_leaks_agree_on_base(self) -> None:
        libc_symbols = {"puts": 0x84420, "read": 0x115280, "system": 0x50D70}
        base = 0x7F00_0000_0000
        leaks = {name: base + off for name, off in libc_symbols.items()}
        score = LibcIdentifier.score_candidate("/lib/libc.so.6", libc_symbols, leaks)
        assert score["verdict"] == "consistent"
        assert score["base"] == base
        assert len(score["matched"]) == 3
        assert score["unmatched"] == []

    def test_inconsistent_when_bases_differ(self) -> None:
        libc_symbols = {"puts": 0x84420, "read": 0x115280}
        leaks = {"puts": 0x7FFF_1234_4420, "read": 0x7FFF_1234_5280}  # different bases
        score = LibcIdentifier.score_candidate("/lib/libc.so.6", libc_symbols, leaks)
        assert score["verdict"] == "inconsistent"
        assert len(score["distinct_bases"]) == 2

    def test_insufficient_when_only_one_leak_matches(self) -> None:
        libc_symbols = {"puts": 0x84420}
        base = 0x7F00_0000_0000
        leaks = {"puts": base + 0x84420, "system": 0x7FFF_1234_0D70}
        score = LibcIdentifier.score_candidate("/lib/libc.so.6", libc_symbols, leaks)
        assert score["verdict"] == "insufficient"
        assert score["base"] == base
        assert score["unmatched"] == ["system"]

    def test_no_matches_verdict_when_zero_symbols_hit(self) -> None:
        libc_symbols = {"puts": 0x84420}
        leaks = {"nonexistent": 0x1000}
        score = LibcIdentifier.score_candidate("/lib/libc.so.6", libc_symbols, leaks)
        assert score["verdict"] == "no_matches"
        assert score["base"] is None
        assert score["matched"] == []


class TestAggregateVerdict:
    def test_single_consistent_with_3_plus_matches_is_confirmed(self) -> None:
        candidate = {
            "verdict": "consistent",
            "matched": [{"symbol": "puts"}, {"symbol": "read"}, {"symbol": "system"}],
        }
        verdict, best, tie = LibcIdentifier.aggregate_verdict([candidate])
        assert verdict == "confirmed"
        assert best is candidate
        assert tie == [candidate]

    def test_single_consistent_with_two_matches_is_likely(self) -> None:
        candidate = {
            "verdict": "consistent",
            "matched": [{"symbol": "puts"}, {"symbol": "read"}],
        }
        verdict, best, _ = LibcIdentifier.aggregate_verdict([candidate])
        assert verdict == "likely"
        assert best is candidate

    def test_multiple_consistent_is_ambiguous(self) -> None:
        c1 = {"verdict": "consistent", "matched": [{}, {}]}
        c2 = {"verdict": "consistent", "matched": [{}, {}]}
        verdict, best, tie = LibcIdentifier.aggregate_verdict([c1, c2])
        assert verdict == "ambiguous"
        assert best is None
        assert len(tie) == 2

    def test_only_insufficient_candidates_is_insufficient(self) -> None:
        candidates = [{"verdict": "insufficient", "matched": [{}]}]
        verdict, best, _ = LibcIdentifier.aggregate_verdict(candidates)
        assert verdict == "insufficient"
        assert best is None

    def test_only_inconsistent_or_unreadable_is_rejected(self) -> None:
        candidates = [
            {"verdict": "inconsistent", "matched": [{}, {}]},
            {"verdict": "unreadable"},
        ]
        verdict, best, tie = LibcIdentifier.aggregate_verdict(candidates)
        assert verdict == "rejected"
        assert best is None and tie == []


# ---------------------------------------------------------------------------
# Public typed operation
# ---------------------------------------------------------------------------
class TestIdentifyLibc:
    def test_confirmed_selects_the_matching_candidate(self) -> None:
        real = _tempfile("libc-2.35.so")
        decoy = _tempfile("libc-2.31.so")
        real_symbols = {"puts": 0x84420, "read": 0x115280, "system": 0x50D70}
        base = 0x7F00_0000_0000
        extractor = _fake_extractor(
            {
                "libc-2.35.so": real_symbols,
                "libc-2.31.so": {"puts": 0x84200, "read": 0x114280, "system": 0x50A00},
            }
        )
        try:
            identifier = LibcIdentifier(symbol_extractor=extractor)
            response = identifier.identify_libc(
                leaks={name: base + off for name, off in real_symbols.items()},
                candidate_libcs=[decoy, real],
            )
            parsed = response["parsed"]
            assert parsed["schema"] == LIBC_IDENTIFICATION_SCHEMA
            assert parsed["verdict"] == "confirmed"
            assert parsed["selected"]["libc_path"].endswith(os.path.basename(real))
            assert parsed["selected"]["base"] == base
        finally:
            os.unlink(real)
            os.unlink(decoy)

    def test_ambiguous_returns_tie_break(self) -> None:
        libc_a = _tempfile("libc-2.35.so")
        libc_b = _tempfile("libc-2.35-x86.so")
        symbols = {"puts": 0x84420, "read": 0x115280}
        base = 0x7F00_0000_0000
        extractor = _fake_extractor(
            {"libc-2.35.so": symbols, "libc-2.35-x86.so": symbols}
        )
        try:
            identifier = LibcIdentifier(symbol_extractor=extractor)
            response = identifier.identify_libc(
                leaks={name: base + off for name, off in symbols.items()},
                candidate_libcs=[libc_a, libc_b],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "ambiguous"
            assert parsed["selected"] is None
            assert len(parsed["tie_break"]) == 2
        finally:
            os.unlink(libc_a)
            os.unlink(libc_b)

    def test_insufficient_when_only_one_leak_matches(self) -> None:
        libc = _tempfile("libc-2.35.so")
        extractor = _fake_extractor({"libc-2.35.so": {"puts": 0x84420}})
        base = 0x7F00_0000_0000
        try:
            identifier = LibcIdentifier(symbol_extractor=extractor)
            response = identifier.identify_libc(
                leaks={"puts": base + 0x84420, "system": 0x7FFF_1234_0D70},
                candidate_libcs=[libc],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "insufficient"
        finally:
            os.unlink(libc)

    def test_unreadable_candidate_is_recorded_not_raised(self) -> None:
        real = _tempfile("libc-2.35.so")
        symbols = {"puts": 0x84420, "read": 0x115280}
        base = 0x7F00_0000_0000
        extractor = _fake_extractor({"libc-2.35.so": symbols})
        try:
            identifier = LibcIdentifier(symbol_extractor=extractor)
            response = identifier.identify_libc(
                leaks={name: base + off for name, off in symbols.items()},
                candidate_libcs=["/tmp/does-not-exist-pao", real],
            )
            parsed = response["parsed"]
            assert parsed["verdict"] == "likely"
            unreadable = [
                c for c in parsed["candidates"] if c["verdict"] == "unreadable"
            ]
            assert len(unreadable) == 1
            assert "error" in unreadable[0]
        finally:
            os.unlink(real)

    def test_empty_leaks_raise(self) -> None:
        identifier = LibcIdentifier(symbol_extractor=_fake_extractor({}))
        with pytest.raises(LibcIdentifierError, match="non-empty mapping"):
            identifier.identify_libc(leaks={}, candidate_libcs=["/tmp/x"])

    def test_empty_candidates_raise(self) -> None:
        identifier = LibcIdentifier(symbol_extractor=_fake_extractor({}))
        with pytest.raises(LibcIdentifierError, match="non-empty sequence"):
            identifier.identify_libc(leaks={"puts": 1}, candidate_libcs=[])

    def test_invalid_leak_address_raises(self) -> None:
        identifier = LibcIdentifier(symbol_extractor=_fake_extractor({}))
        with pytest.raises(LibcIdentifierError, match="non-negative integer"):
            identifier.identify_libc(
                leaks={"puts": "not-a-number"},
                candidate_libcs=["/tmp/x"],
            )


class TestParseOneGadgetOutput:
    _SAMPLE = (
        '0x50a37 execve("/bin/sh", r15, r12)\n'
        "constraints:\n"
        "  [r15] == NULL || r15 == NULL\n"
        "  [r12] == NULL || r12 == NULL\n"
        "\n"
        '0x50a41 execve("/bin/sh", r15, rdx)\n'
        "constraints:\n"
        "  [r15] == NULL || r15 == NULL\n"
        "  [rdx] == NULL || rdx == NULL\n"
    )

    def test_parses_offset_and_call(self) -> None:
        parsed = LibcIdentifier.parse_one_gadget_output(self._SAMPLE)
        assert len(parsed) == 2
        assert parsed[0]["offset"] == 0x50A37
        assert "execve" in parsed[0]["call"]
        assert parsed[0]["constraints"] == [
            "[r15] == NULL || r15 == NULL",
            "[r12] == NULL || r12 == NULL",
        ]

    def test_empty_input_returns_empty_list(self) -> None:
        assert LibcIdentifier.parse_one_gadget_output("") == []


class TestFindOneGadgets:
    def test_missing_tool_returns_unsupported_verdict(self) -> None:
        libc = _tempfile("libc.so.6")
        runner = _fake_one_gadget_runner(127, "", "one_gadget not on PATH")
        try:
            identifier = LibcIdentifier(one_gadget_runner=runner)
            response = identifier.find_one_gadgets(libc_path=libc)
            parsed = response["parsed"]
            assert parsed["schema"] == ONE_GADGET_SCHEMA
            assert parsed["supported"] is False
            assert parsed["verdict"] == "unsupported"
            assert "one_gadget" in parsed["reason"]
        finally:
            os.unlink(libc)

    def test_parsed_when_tool_returns_gadgets(self) -> None:
        libc = _tempfile("libc.so.6")
        sample = (
            '0x50a37 execve("/bin/sh", r15, r12)\n' "constraints:\n" "  [r15] == NULL\n"
        )
        runner = _fake_one_gadget_runner(0, sample, "")
        try:
            identifier = LibcIdentifier(one_gadget_runner=runner)
            response = identifier.find_one_gadgets(libc_path=libc)
            parsed = response["parsed"]
            assert parsed["verdict"] == "parsed"
            assert parsed["gadget_count"] == 1
            assert parsed["gadgets"][0]["offset"] == 0x50A37
        finally:
            os.unlink(libc)

    def test_failed_when_tool_returns_nonzero(self) -> None:
        libc = _tempfile("libc.so.6")
        runner = _fake_one_gadget_runner(1, "", "cannot analyze libc")
        try:
            identifier = LibcIdentifier(one_gadget_runner=runner)
            response = identifier.find_one_gadgets(libc_path=libc)
            parsed = response["parsed"]
            assert parsed["verdict"] == "failed"
            assert "cannot analyze" in parsed["reason"]
        finally:
            os.unlink(libc)

    def test_missing_libc_raises(self) -> None:
        identifier = LibcIdentifier(
            one_gadget_runner=_fake_one_gadget_runner(0, "", ""),
        )
        with pytest.raises(LibcIdentifierError, match="not readable"):
            identifier.find_one_gadgets(libc_path="/tmp/does-not-exist-pao")

    def test_bad_timeout_raises(self) -> None:
        libc = _tempfile("libc.so.6")
        try:
            identifier = LibcIdentifier(
                one_gadget_runner=_fake_one_gadget_runner(0, "", ""),
            )
            with pytest.raises(
                LibcIdentifierError, match="timeout_sec must be positive"
            ):
                identifier.find_one_gadgets(libc_path=libc, timeout_sec=0)
        finally:
            os.unlink(libc)
