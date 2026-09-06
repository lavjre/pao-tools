"""Opt-in allocation-event tracing (P1 minimal).

Breakpoint-based malloc/free call-count tracing. ``start_alloc_trace`` plants
persistent breakpoints on the allocator entry points (``malloc``/``free`` by
default, overridable), ``list_alloc_events`` reports per-breakpoint hit counts
plus a bounded synthetic event log (one entry per observed hit-count delta, so
a runaway loop cannot grow state without bound), and ``stop_alloc_trace``
removes the breakpoints and returns a summary.

This is call-count tracing, not per-allocation arg capture: full arg/stack
capture needs stop-event polling that would stall the session. The manager is
explicit about that limit in every response. Disabled by default; zero
overhead when no trace is active.
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from paomcp.tools.backends.foundation import LiveInferiorGuard, build_backend_response
from paomcp.tools.backends.heap_schemas import PARSER_VERSION

ALLOC_TRACE_START_SCHEMA = "pao.alloc-trace-start.v1"
ALLOC_TRACE_LIST_SCHEMA = "pao.alloc-trace-list.v1"
ALLOC_TRACE_STOP_SCHEMA = "pao.alloc-trace-stop.v1"

DEFAULT_TRACE_SYMBOLS = ("malloc", "free")
MAX_TRACES = 4
MAX_EVENTS_PER_TRACE = 10_000


class AllocTraceManager:
    """Owns breakpoint-backed alloc traces for one debug session."""

    def __init__(
        self,
        gdb: Any,
        inspection: Any,
        debugger: Any,
        session_id: str,
        *,
        discovery: Any,
    ) -> None:
        self.gdb = gdb
        self.inspection = inspection
        self.debugger = debugger
        self.session_id = session_id
        self.discovery = discovery
        self._lock = threading.RLock()
        self._traces: Dict[str, Dict[str, Any]] = {}

    def _require_live(self) -> None:
        LiveInferiorGuard.require_live(
            self.gdb.get_last_stop_record(),
            self.gdb.get_inferior_pid(),
            detail=(
                "alloc tracing requires a live, stopped inferior; run the "
                "target to a stop before starting a trace"
            ),
        )

    def _hit_counts(self) -> Dict[int, int]:
        try:
            listing = self.debugger.list_breakpoints()
        except Exception:
            return {}
        parsed = listing.get("parsed") if isinstance(listing, dict) else None
        out: Dict[int, int] = {}
        bps = (parsed or {}).get("breakpoints") or []
        for bp in bps:
            try:
                num = int(str(bp.get("number", "")))
            except (TypeError, ValueError):
                continue
            try:
                hits = int(bp.get("hit_count", 0) or 0)
            except (TypeError, ValueError):
                hits = 0
            out[num] = hits
        return out

    def start_alloc_trace(
        self,
        *,
        symbols: Optional[List[str]] = None,
        max_events: int = MAX_EVENTS_PER_TRACE,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        self._require_live()
        wanted = list(symbols or list(DEFAULT_TRACE_SYMBOLS))
        if not wanted or len(wanted) > 8:
            raise ValueError("symbols must name 1-8 breakpoint locations")
        if max_events <= 0 or max_events > MAX_EVENTS_PER_TRACE:
            raise ValueError(f"max_events must be 1-{MAX_EVENTS_PER_TRACE}")
        with self._lock:
            if len(self._traces) >= MAX_TRACES:
                raise RuntimeError(f"at most {MAX_TRACES} concurrent alloc traces")
            trace_id = uuid.uuid4().hex[:12]
            bps: List[Dict[str, Any]] = []
            for sym in wanted:
                created = self.debugger.create_breakpoint(location=sym)
                if not created.get("success"):
                    for prev in bps:
                        try:
                            self.debugger.delete_breakpoint(int(prev["number"]))
                        except Exception:
                            pass
                    return build_backend_response(
                        f"start_alloc_trace {sym}",
                        parsed={
                            "schema": ALLOC_TRACE_START_SCHEMA,
                            "supported": False,
                            "verdict": "unresolvable",
                            "reason": str(created.get("error") or "breakpoint failed"),
                            "symbols": wanted,
                        },
                        raw_responses=[created],
                        state=self.gdb.get_state(),
                        parser_version=PARSER_VERSION,
                    )
                record = (created.get("parsed") or {}).get("breakpoint") or {}
                bps.append(
                    {
                        "number": int(str(record.get("number", "0"))),
                        "symbol": sym,
                        "last_hits": 0,
                    }
                )
            trace = {
                "trace_id": trace_id,
                "symbols": wanted,
                "breakpoints": bps,
                "max_events": max_events,
                "events": [],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stopped": False,
            }
            self._traces[trace_id] = trace
        return build_backend_response(
            "start_alloc_trace",
            parsed={
                "schema": ALLOC_TRACE_START_SCHEMA,
                "supported": True,
                "verdict": "started",
                "trace_id": trace_id,
                "symbols": wanted,
                "breakpoints": bps,
                "max_events": max_events,
                "limit_note": (
                    "call-count tracing via breakpoint hit counts; "
                    "per-alloc args/stacks are not captured in P1"
                ),
            },
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
        )

    def list_alloc_events(
        self,
        *,
        trace_id: str,
        limit: int = 100,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        self._require_live()
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                raise ValueError(f"unknown trace_id: {trace_id!r}")
            hits = self._hit_counts()
            for bp in trace["breakpoints"]:
                num = int(bp["number"])
                now_hits = int(hits.get(num, 0))
                delta = now_hits - int(bp.get("last_hits", 0))
                if delta > 0:
                    room = int(trace["max_events"]) - len(trace["events"])
                    for _ in range(min(delta, room)):
                        trace["events"].append(
                            {
                                "symbol": bp["symbol"],
                                "breakpoint": num,
                                "observed_at": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                    bp["last_hits"] = now_hits
                    if len(trace["events"]) >= int(trace["max_events"]):
                        trace["events"] = trace["events"][-int(trace["max_events"]) :]
            events = list(trace["events"])[-max(1, limit) :]
            parsed = {
                "schema": ALLOC_TRACE_LIST_SCHEMA,
                "supported": True,
                "verdict": "listed",
                "trace_id": trace_id,
                "symbols": trace["symbols"],
                "breakpoints": [
                    {
                        "number": int(b["number"]),
                        "symbol": b["symbol"],
                        "hit_count": int(b.get("last_hits", 0)),
                    }
                    for b in trace["breakpoints"]
                ],
                "event_count": len(trace["events"]),
                "events": events,
                "stopped": bool(trace.get("stopped")),
                "limit_note": "P1 reports hit-count deltas, not args/stacks",
            }
        return build_backend_response(
            "list_alloc_events",
            parsed=parsed,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
        )

    def stop_alloc_trace(
        self,
        *,
        trace_id: str,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        self._require_live()
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                raise ValueError(f"unknown trace_id: {trace_id!r}")
            deleted: List[int] = []
            for bp in trace["breakpoints"]:
                try:
                    res = self.debugger.delete_breakpoint(int(bp["number"]))
                    if res.get("success"):
                        deleted.append(int(bp["number"]))
                except Exception:
                    pass
            trace["stopped"] = True
            summary = {
                "schema": ALLOC_TRACE_STOP_SCHEMA,
                "supported": True,
                "verdict": "stopped",
                "trace_id": trace_id,
                "deleted_breakpoints": deleted,
                "event_count": len(trace["events"]),
                "symbols": trace["symbols"],
            }
            self._traces.pop(trace_id, None)
        return build_backend_response(
            "stop_alloc_trace",
            parsed=summary,
            raw_responses=[],
            state=self.gdb.get_state(),
            parser_version=PARSER_VERSION,
        )


__all__ = [
    "AllocTraceManager",
    "ALLOC_TRACE_START_SCHEMA",
    "ALLOC_TRACE_LIST_SCHEMA",
    "ALLOC_TRACE_STOP_SCHEMA",
]
