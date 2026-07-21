from typing import Any, Dict

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from pwnomcp.tools.common import catch_errors, get_services


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def get_retdec_status(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get the current RetDec decompilation status, lazily initializing as needed."""
        services = get_services(ctx)
        analyzer = services.retdec_analyzer
        if not analyzer._initialized:
            await analyzer.initialize()
        status = analyzer.get_status()
        return status

    @mcp.tool()
    @catch_errors()
    async def get_decompiled_code(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return RetDec decompiled C code if available, or a status describing why not."""
        services = get_services(ctx)
        analyzer = services.retdec_analyzer
        if not analyzer._initialized:
            await analyzer.initialize()
        code = analyzer.get_decompiled_code()
        if code:
            return {"status": "success", "decompiled_code": code}

        status = analyzer.get_status()
        return {
            "status": "unavailable",
            "reason": status.get("status"),
            "details": status,
        }
