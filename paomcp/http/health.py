import logging
from typing import Any, Optional

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from paomcp.services import AppServices

logger = logging.getLogger(__name__)


def register_health_routes(
    mcp: FastMCP, services: Optional[AppServices] = None
) -> None:
    """Register ``/healthz`` and ``/readyz``.

    FastMCP 3.1's streamable-http lifespan discards the user lifespan
    result, so ``app.state.services`` is never populated by the server
    itself. The services graph is therefore wired into the routes at
    construction time; the lifespan-result and app-state probes remain as
    fallbacks for embedding apps that run the factory without services.
    """

    def _resolve_services(request: Request) -> Optional[Any]:
        if services is not None:
            return services
        lifespan_result = getattr(mcp, "_lifespan_result", None)
        if isinstance(lifespan_result, dict):
            candidate = lifespan_result.get("services")
            if candidate is not None:
                return candidate
        scope = getattr(request, "scope", {}) or {}
        for key in ("lifespan_context", "state", "app_state"):
            candidate = scope.get(key) if isinstance(scope, dict) else None
            if isinstance(candidate, dict) and candidate.get("services"):
                return candidate["services"]
        app = getattr(request, "app", None)
        state = getattr(app, "state", None)
        return getattr(state, "services", None)

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: can this server do debugger work? (liveness: /healthz).

        ``ready`` tracks the default debug session only. Ghidra and IDA are
        optional accelerators: their state is reported under ``checks`` with
        reasons, but a degraded accelerator never flips readiness.
        """
        try:
            resolved = _resolve_services(request)
            if resolved is None:
                return JSONResponse(
                    {
                        "ready": False,
                        "reasons": ["services not initialized"],
                    },
                    status_code=503,
                )
            reasons = []
            default_id = resolved.default_session_id
            if default_id is None:
                reasons.append("no default debug session (debugger unavailable)")
            ghidra = resolved.decompiler.status()
            ida_status = resolved.ida.status()
            if not ghidra.get("ready"):
                reasons.append("ghidra not ready (offline decomp unavailable)")
            if not ida_status.get("ready"):
                reasons.append("ida not ready (bridge degraded)")
            capacity = resolved.session_registry.status()
            ready = default_id is not None
            return JSONResponse(
                {
                    "ready": ready,
                    "checks": {
                        "debugger_ready": ready,
                        "ghidra_ready": bool(ghidra.get("ready")),
                        "ida_ready": bool(ida_status.get("ready")),
                        "sessions": {
                            k: capacity[k]
                            for k in (
                                "active",
                                "occupied",
                                "available",
                                "max_sessions",
                            )
                            if k in capacity
                        },
                    },
                    "reasons": reasons,
                },
                status_code=200 if ready else 503,
            )
        except Exception:
            logger.exception("Readiness check failed")
            return JSONResponse(
                {"ready": False, "reasons": ["readiness check failed"]},
                status_code=503,
            )
