"""Tests for Stage 9.5 typed research-report generator."""

from __future__ import annotations

import json

import pytest

from paomcp.tools.backends.research import ResearchRegistry
from paomcp.tools.backends.research_report import (
    REPORT_FORMATS,
    RESEARCH_REPORT_SCHEMA,
    ReportError,
    ResearchReport,
)


def _seed(registry: ResearchRegistry) -> str:
    sid = registry.create_session_response(
        name="ret2libc pwn",
        description="CTF level 4",
        metadata={"binary": "/pwn/vuln", "argv": ["./vuln"]},
    )["parsed"]["session"]["session_id"]
    registry.add_hypothesis_response(
        session_id=sid,
        statement="bug is stack overflow at ret",
        state="supported",
        rationale="cyclic offset 40",
    )
    registry.add_hypothesis_response(
        session_id=sid,
        statement="ret2libc chain to system(/bin/sh)",
        state="proposed",
        rationale="libc leaked via puts",
    )
    registry.add_hypothesis_response(
        session_id=sid,
        statement="race on second thread",
        state="unknown",
    )
    registry.add_note_response(
        session_id=sid, body="crash reproduces every run", tags=["fact"]
    )
    registry.add_note_response(
        session_id=sid,
        body="guessed offset 40 from RBP pattern",
        tags=["heuristic"],
    )
    registry.link_evidence_response(
        session_id=sid,
        kind="crash",
        ref="crash-abc",
        description="segv at ret",
    )
    registry.link_evidence_response(
        session_id=sid,
        kind="heap_snapshot",
        ref="heap-1",
        description="clean heap",
    )
    registry.link_evidence_response(
        session_id=sid,
        kind="artifact",
        ref="art-1",
        description="core dump",
    )
    registry.link_evidence_response(
        session_id=sid,
        kind="tool_result",
        ref="op-42",
        description="ropper output",
        metadata={"inference": True},
    )
    registry.create_snapshot_response(
        session_id=sid,
        name="pre-crash",
        slots={"registers": {"rip": 0x400200}},
    )
    registry.append_timeline_event_response(
        session_id=sid, kind="reproduction", summary="reproduced with cyclic 100"
    )
    return sid


class TestFormatDispatch:
    def test_default_formats_are_markdown_and_json(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        result = rpt.generate_report(session_id=sid)
        assert set(result["formats"]) == {"markdown", "json"}
        assert "markdown" in result["rendered"]
        assert "json" in result["rendered"]

    def test_html_format_included_when_requested(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        result = rpt.generate_report(
            session_id=sid, formats=["json", "markdown", "html"]
        )
        assert set(result["formats"]) == {"json", "markdown", "html"}
        assert result["rendered"]["html"].startswith("<!DOCTYPE html>")

    def test_unknown_format_rejected(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        with pytest.raises(ReportError, match="unknown report formats"):
            rpt.generate_report(session_id=sid, formats=["latex"])

    def test_report_formats_vocabulary_documented(self) -> None:
        assert set(REPORT_FORMATS) == {"json", "markdown", "html"}


class TestJsonPayloadShape:
    def test_facts_heuristics_hypotheses_are_distinct(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        result = rpt.generate_report(session_id=sid, formats=["json"])
        payload = json.loads(result["rendered"]["json"])
        assert payload["schema"] == RESEARCH_REPORT_SCHEMA
        # Facts contains ONLY direct observations
        crash_refs = {c["ref"] for c in payload["facts"]["crashes"]}
        heap_refs = {h["ref"] for h in payload["facts"]["heap_snapshots"]}
        assert crash_refs == {"crash-abc"}
        assert heap_refs == {"heap-1"}
        # A note tagged "heuristic" surfaces under heuristics, not facts
        heuristic_bodies = {n["body"] for n in payload["heuristics"]["notes"]}
        assert any("guessed offset 40" in body for body in heuristic_bodies)
        # A tool_result whose metadata carries inference=True is a
        # heuristic even though its kind is a factual class
        heuristic_evidence_refs = {e["ref"] for e in payload["heuristics"]["evidence"]}
        assert "op-42" in heuristic_evidence_refs
        # Every hypothesis is present with its state
        states = {h["state"] for h in payload["hypotheses"]["all"]}
        assert states == {"supported", "proposed", "unknown"}

    def test_rop_hypotheses_are_surfaced(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        payload = json.loads(
            rpt.generate_report(session_id=sid, formats=["json"])["rendered"]["json"]
        )
        rop = payload["hypotheses"]["rop"]
        assert len(rop) == 1
        assert "ret2libc" in rop[0]["statement"]

    def test_reproduction_reports_supplied_and_missing(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        payload = json.loads(
            rpt.generate_report(session_id=sid, formats=["json"])["rendered"]["json"]
        )
        repro = payload["reproduction"]
        assert repro["supplied"]["binary"] == "/pwn/vuln"
        assert repro["supplied"]["argv"] == ["./vuln"]
        # env and working_dir were not supplied in the fixture
        assert "env" in repro["missing"]
        assert "working_dir" in repro["missing"]

    def test_known_unknowns_names_real_gaps(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        payload = json.loads(
            rpt.generate_report(session_id=sid, formats=["json"])["rendered"]["json"]
        )
        text = " | ".join(payload["known_unknowns"])
        assert "state=unknown" in text  # race hypothesis surfaced


class TestNoCrashSession:
    def test_missing_crash_becomes_a_known_unknown(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="s")["parsed"]["session"][
            "session_id"
        ]
        registry.add_hypothesis_response(
            session_id=sid, statement="something", rationale="anything"
        )
        rpt = ResearchReport(registry)
        payload = json.loads(
            rpt.generate_report(session_id=sid, formats=["json"])["rendered"]["json"]
        )
        assert any("no crash evidence linked" in u for u in payload["known_unknowns"])

    def test_proposed_without_rationale_becomes_a_known_unknown(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="s")["parsed"]["session"][
            "session_id"
        ]
        registry.add_hypothesis_response(
            session_id=sid, statement="propose but no rationale"
        )
        rpt = ResearchReport(registry)
        payload = json.loads(
            rpt.generate_report(session_id=sid, formats=["json"])["rendered"]["json"]
        )
        assert any("has no rationale recorded" in u for u in payload["known_unknowns"])


class TestMarkdownRendering:
    def test_markdown_carries_every_top_level_section(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        md = rpt.generate_report(session_id=sid, formats=["markdown"])["rendered"][
            "markdown"
        ]
        for header in (
            "# Research report",
            "## Hypotheses",
            "## Facts",
            "## Heuristics",
            "## ROP hypotheses",
            "## Reproduction guide",
            "## Timeline",
            "## Known unknowns",
        ):
            assert header in md, f"missing section: {header}"

    def test_markdown_lists_hypothesis_state_and_rationale(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        md = rpt.generate_report(session_id=sid, formats=["markdown"])["rendered"][
            "markdown"
        ]
        assert "state = supported" in md
        assert "cyclic offset 40" in md

    def test_empty_session_markdown_still_renders(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="empty")["parsed"]["session"][
            "session_id"
        ]
        rpt = ResearchReport(registry)
        md = rpt.generate_report(session_id=sid, formats=["markdown"])["rendered"][
            "markdown"
        ]
        assert "No hypotheses recorded." in md
        assert "No named gaps in the record." not in md  # unknowns exist
        assert "no crash evidence linked" in md


class TestHtmlSafety:
    def test_html_escapes_session_name_and_body(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="pwn <script>alert(1)</script>")[
            "parsed"
        ]["session"]["session_id"]
        registry.add_note_response(session_id=sid, body="also <em>injected</em>")
        rpt = ResearchReport(registry)
        html_out = rpt.generate_report(session_id=sid, formats=["html"])["rendered"][
            "html"
        ]
        assert "<script>" not in html_out
        assert "&lt;script&gt;" in html_out
        # Body is escaped inside the <pre> wrapper too
        assert "&lt;em&gt;" in html_out


class TestResponseWrapper:
    def test_response_uses_typed_envelope(self) -> None:
        registry = ResearchRegistry()
        sid = _seed(registry)
        rpt = ResearchReport(registry)
        response = rpt.generate_report_response(session_id=sid, formats=["markdown"])
        assert response["success"] is True
        assert response["command"] == "generate_final_report"
        assert response["parsed"]["schema"] == RESEARCH_REPORT_SCHEMA
        assert "markdown" in response["parsed"]["rendered"]
