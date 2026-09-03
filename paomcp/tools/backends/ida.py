"""IDA Pro MCP bridge (mrexodia/ida-pro-mcp + idalib-mcp headless).

This backend does not reimplement IDA analysis. It talks to an already
running IDA Pro MCP server over HTTP and exposes a small token-lean
subset of its tools through pao-tools with the shared response envelope.

Transports probed (in order):
- GUI/legacy SSE: ``PAO_IDA_MCP_URL`` (default ``http://127.0.0.1:13337/mcp``)
- Headless idalib supervisor: ``PAO_IDALIB_MCP_URL`` (default ``http://127.0.0.1:8745/mcp``)

When neither answers, every method degrades to ``ready: False`` with
reasons instead of raising at startup — same contract as GhidraDecompiler.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional
import requests

from paomcp.tools.backends.foundation import build_backend_response

logger = logging.getLogger(__name__)

DEFAULT_IDA_MCP_URL = os.environ.get("PAO_IDA_MCP_URL", "http://127.0.0.1:13337/mcp")
DEFAULT_IDALIB_MCP_URL = os.environ.get(
    "PAO_IDALIB_MCP_URL", "http://127.0.0.1:8745/mcp"
)
DEFAULT_IDA_TIMEOUT_SEC = float(os.environ.get("PAO_IDA_TIMEOUT_SEC", "30"))
MAX_IDA_TIMEOUT_SEC = 120.0
DEFAULT_IDA_LIMIT = 100
MAX_IDA_LIMIT = 1000

# Small curated subset: keeps pao-tools lean instead of proxying
# all 60+ IDA tools verbatim (token cost on every list_tools call).
# dbg_* require ?ext=dbg on the endpoint (handled by _call_dbg).
PASSTHROUGH_METHODS = frozenset(
    {
        "decompile",
        "disasm",
        "lookup_funcs",
        "list_funcs",
        "list_globals",
        "imports",
        "xrefs_to",
        "xrefs_to_field",
        "callees",
        "get_bytes",
        "get_string",
        "get_int",
        "find_bytes",
        "find_insns",
        "find_regex",
        "basic_blocks",
        "py_eval",
        "analyze_funcs",
        "int_convert",
        "idb_open",
        "idb_list",
        "idb_close",
        "idb_save",
        "rename",
        "set_comments",
        "server_health",
        "dbg_start",
        "dbg_exit",
        "dbg_continue",
        "dbg_run_to",
        "dbg_step_into",
        "dbg_step_over",
        "dbg_bps",
        "dbg_add_bp",
        "dbg_delete_bp",
        "dbg_toggle_bp",
        "dbg_regs",
        "dbg_regs_all",
        "dbg_gpregs",
        "dbg_regs_named",
        "dbg_stacktrace",
        "dbg_read",
        "dbg_write",
    }
)

DBG_METHODS = frozenset({m for m in PASSTHROUGH_METHODS if m.startswith("dbg_")})


def _clamp_limit(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_IDA_LIMIT
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return DEFAULT_IDA_LIMIT
    return max(1, min(ivalue, MAX_IDA_LIMIT))


def _clamp_timeout(value: Optional[float]) -> float:
    if value is None:
        return DEFAULT_IDA_TIMEOUT_SEC
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return DEFAULT_IDA_TIMEOUT_SEC
    if fvalue <= 0:
        return DEFAULT_IDA_TIMEOUT_SEC
    return min(fvalue, MAX_IDA_TIMEOUT_SEC)


class IdaProMCP:
    """Thin JSON-RPC client for a running ida-pro-mcp / idalib-mcp server."""

    def __init__(
        self,
        mcp_url: Optional[str] = None,
        idalib_url: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> None:
        self.mcp_url = mcp_url or DEFAULT_IDA_MCP_URL
        self.idalib_url = idalib_url or DEFAULT_IDALIB_MCP_URL
        self.timeout_sec = _clamp_timeout(timeout_sec)
        self.model_schema_version = "1"
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()

    def _session_client(self) -> requests.Session:
        """Shared keep-alive session (one TCP/TLS handshake, not one per call)."""
        session = self._session
        if session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = requests.Session()
                session = self._session
        return session

    def close(self) -> None:
        """Drop the keep-alive session; the bridge degrades, never raises."""
        with self._session_lock:
            session, self._session = self._session, None
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - shutdown path
                logger.debug("IDA session close failed", exc_info=True)

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._session_client().post(
            url, json=payload, timeout=self.timeout_sec
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("IDA MCP returned a non-object JSON-RPC response")
        return data

    def _call(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if method not in PASSTHROUGH_METHODS:
            return build_backend_response(
                f"ida/{method}",
                parsed=None,
                raw_responses=[],
                state="idle",
                success=False,
                error=f"IDA method '{method}' is not in the curated bridge subset",
                error_type="ValueError",
            )
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": method, "arguments": params or {}},
        }
        errors: List[str] = []
        urls: tuple = (self.mcp_url, self.idalib_url)
        if method in DBG_METHODS:
            # Debugger tools are hidden unless ?ext=dbg is present.
            urls = tuple(
                u if "ext=" in u else (u + ("&ext=dbg" if "?" in u else "?ext=dbg"))
                for u in urls
            )
        for url in urls:
            try:
                data = self._post(url, payload)
                return build_backend_response(
                    f"ida/{method}",
                    parsed=data.get("result", data),
                    raw_responses=[],
                    state="idle",
                    extra={"ida_endpoint": url},
                )
            except Exception as exc:  # noqa: BLE001 - probe both endpoints
                errors.append(f"{url}: {exc}")
                logger.debug("IDA MCP call via %s failed: %s", url, exc)
        return build_backend_response(
            f"ida/{method}",
            parsed=None,
            raw_responses=[],
            state="idle",
            success=False,
            error="No IDA MCP endpoint answered",
            error_type="IdaUnavailableError",
            extra={"attempts": errors},
        )

    def status(self) -> Dict[str, Any]:
        """Probe both endpoints; never raises, mirrors Ghidra status shape."""
        probes = []
        for label, url in (("ida_gui", self.mcp_url), ("idalib", self.idalib_url)):
            try:
                data = self._post(
                    url,
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/list",
                        "params": {},
                    },
                )
                tools = data.get("result", {}).get("tools", data.get("tools", []))
                probes.append(
                    {
                        "endpoint": label,
                        "url": url,
                        "reachable": True,
                        "tool_count": len(tools) if isinstance(tools, list) else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - degraded status path
                probes.append(
                    {
                        "endpoint": label,
                        "url": url,
                        "reachable": False,
                        "error": str(exc),
                    }
                )
        ready = any(p["reachable"] for p in probes)
        reasons: List[str] = []
        if not ready:
            reasons.append(
                "No IDA MCP endpoint answered; start IDA with ida-pro-mcp "
                "or run `uv run idalib-mcp --port 8745` for headless"
            )
        return {
            "backend": "ida-pro-mcp",
            "ready": ready,
            "model_schema_version": self.model_schema_version,
            "mcp_url": self.mcp_url,
            "idalib_url": self.idalib_url,
            "timeout_sec": self.timeout_sec,
            "methods": sorted(PASSTHROUGH_METHODS),
            "probes": probes,
            "reasons": reasons,
        }

    # Typed helpers (token-lean defaults, database-scoped like idalib-mcp).

    def decompile(self, addr: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"addr": addr}
        if database:
            params["database"] = database
        return self._call("decompile", params)

    def disasm(self, addr: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addr": addr}
        if database:
            params["database"] = database
        return self._call("disasm", params)

    def list_funcs(
        self,
        queries: Optional[Any] = None,
        database: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"limit": _clamp_limit(limit)}
        if queries is not None:
            params["queries"] = queries
        if database:
            params["database"] = database
        return self._call("list_funcs", params)

    def lookup(self, queries: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"queries": queries}
        if database:
            params["database"] = database
        return self._call("lookup_funcs", params)

    def xrefs_to(self, addrs: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addrs": addrs}
        if database:
            params["database"] = database
        return self._call("xrefs_to", params)

    def get_bytes(self, addrs: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addrs": addrs}
        if database:
            params["database"] = database
        return self._call("get_bytes", params)

    def get_string(self, addrs: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addrs": addrs}
        if database:
            params["database"] = database
        return self._call("get_string", params)

    def find_bytes(
        self,
        patterns: Any,
        database: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "patterns": patterns,
            "limit": _clamp_limit(limit),
            "offset": max(0, int(offset)),
        }
        if database:
            params["database"] = database
        return self._call("find_bytes", params)

    def py_eval(self, code: str, database: Optional[str] = None) -> Dict[str, Any]:
        if not code or not code.strip():
            return build_backend_response(
                "ida/py_eval",
                parsed=None,
                raw_responses=[],
                state="idle",
                success=False,
                error="code cannot be empty",
                error_type="ValueError",
            )
        params: Dict[str, Any] = {"code": code}
        if database:
            params["database"] = database
        return self._call("py_eval", params)

    def idb_open(
        self,
        input_path: str,
        mode: str = "prefer_headless",
        preferred_session_id: str = "",
    ) -> Dict[str, Any]:
        if mode not in (
            "prefer_headless",
            "force_headless",
            "prefer_gui",
            "force_gui",
        ):
            return build_backend_response(
                "ida/idb_open",
                parsed=None,
                raw_responses=[],
                state="idle",
                success=False,
                error=f"Unknown idb mode '{mode}'",
                error_type="ValueError",
            )
        return self._call(
            "idb_open",
            {
                "input_path": input_path,
                "mode": mode,
                "preferred_session_id": preferred_session_id,
            },
        )

    def idb_list(self) -> Dict[str, Any]:
        return self._call("idb_list", {})

    def idb_close(self, database: str, save: bool = True) -> Dict[str, Any]:
        return self._call("idb_close", {"database": database, "save": save})

    def rename(
        self, batch: Dict[str, Any], database: Optional[str] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"batch": batch}
        if database:
            params["database"] = database
        return self._call("rename", params)

    def set_comments(
        self, items: Any, database: Optional[str] = None
    ) -> Dict[str, Any]:
        params = {"items": items}
        if database:
            params["database"] = database
        return self._call("set_comments", params)

    # Debugger extension (?ext=dbg). All forward database for idalib parity.

    def dbg_start(self, database: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if database:
            params["database"] = database
        return self._call("dbg_start", params)

    def dbg_exit(self, database: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if database:
            params["database"] = database
        return self._call("dbg_exit", params)

    def dbg_continue(self, database: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if database:
            params["database"] = database
        return self._call("dbg_continue", params)

    def dbg_run_to(self, addr: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addr": addr}
        if database:
            params["database"] = database
        return self._call("dbg_run_to", params)

    def dbg_step(
        self, kind: str = "into", database: Optional[str] = None
    ) -> Dict[str, Any]:
        method = "dbg_step_into" if kind == "into" else "dbg_step_over"
        params = {}
        if database:
            params["database"] = database
        return self._call(method, params)

    def dbg_bps(self, database: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if database:
            params["database"] = database
        return self._call("dbg_bps", params)

    def dbg_add_bp(self, addrs: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"addrs": addrs}
        if database:
            params["database"] = database
        return self._call("dbg_add_bp", params)

    def dbg_regs(
        self,
        names: Optional[Any] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        if names:
            return self._call(
                "dbg_regs_named",
                {"names": names, **({"database": database} if database else {})},
            )
        params = {}
        if database:
            params["database"] = database
        return self._call("dbg_regs", params)

    def dbg_stacktrace(self, database: Optional[str] = None) -> Dict[str, Any]:
        params = {}
        if database:
            params["database"] = database
        return self._call("dbg_stacktrace", params)

    def dbg_read(self, regions: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"regions": regions}
        if database:
            params["database"] = database
        return self._call("dbg_read", params)

    def dbg_write(self, regions: Any, database: Optional[str] = None) -> Dict[str, Any]:
        params = {"regions": regions}
        if database:
            params["database"] = database
        return self._call("dbg_write", params)


__all__ = [
    "IdaProMCP",
    "PASSTHROUGH_METHODS",
    "DBG_METHODS",
    "DEFAULT_IDA_MCP_URL",
    "DEFAULT_IDALIB_MCP_URL",
]
