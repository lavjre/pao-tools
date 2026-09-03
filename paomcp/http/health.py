from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


def register_health_routes(mcp: FastMCP) -> None:
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
            services = None
            scope = getattr(request, "scope", {}) or {}
            for key in ("lifespan_context", "state", "app_state"):
                candidate = scope.get(key) if isinstance(scope, dict) else None
                if isinstance(candidate, dict) and candidate.get("services"):
                    services = candidate["services"]
                    break
            app = getattr(request, "app", None)
            if services is None and app is not None:
                state = getattr(app, "state", None)
                services = getattr(state, "services", None)
            if services is None:
                return JSONResponse(
                    {
                        "ready": False,
                        "reasons": ["services not initialized"],
                    },
                    status_code=503,
                )
            reasons = []
            default_id = services.default_session_id
            if default_id is None:
                reasons.append("no default debug session (debugger unavailable)")
            ghidra = services.decompiler.status()
            ida_status = services.ida.status()
            if not ghidra.get("ready"):
                reasons.append("ghidra not ready (offline decomp unavailable)")
            if not ida_status.get("ready"):
                reasons.append("ida not ready (bridge degraded)")
            capacity = services.session_registry.status()
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
        except Exception as exc:
            return JSONResponse(
                {"ready": False, "reasons": [str(exc)]}, status_code=503
            )
