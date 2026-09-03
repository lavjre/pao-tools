from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse


def register_health_routes(mcp: FastMCP) -> None:
    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @mcp.custom_route("/readyz", methods=["GET"])
    async def readyz(request: Request) -> JSONResponse:
        """Readiness: Ghidra + IDA + sessions (liveness stays /healthz)."""
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
                        "ready": True,
                        "checks": {"liveness": "ok"},
                        "reasons": [
                            "services not visible to HTTP scope; "
                            "use get_decompiler_status/get_ida_status tools "
                            "for deep checks"
                        ],
                    }
                )
            ghidra = services.decompiler.status()
            ida_status = services.ida.status()
            sessions = services.session_registry.status()
            reasons = []
            if not ghidra.get("ready"):
                reasons.append("ghidra not ready (offline decomp unavailable)")
            if not ida_status.get("ready"):
                reasons.append("ida not ready (bridge degraded)")
            return JSONResponse(
                {
                    "ready": True,
                    "checks": {
                        "ghidra_ready": bool(ghidra.get("ready")),
                        "ida_ready": bool(ida_status.get("ready")),
                        "sessions": sessions,
                    },
                    "reasons": reasons,
                }
            )
        except Exception as exc:
            return JSONResponse(
                {"ready": False, "reasons": [str(exc)]}, status_code=503
            )
