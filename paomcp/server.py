import logging
from typing import Optional

from fastmcp import FastMCP

from paomcp.http.health import register_health_routes
from paomcp.lifespan import create_lifespan
from paomcp.services import AppServices
from paomcp.tools import register_all_tools

# Logging is centralized in paomcp.logger; do not call basicConfig here so
# embedding apps and tests keep control of their handlers.
logger = logging.getLogger(__name__)


def create_mcp(
    services: Optional[AppServices] = None, profile: Optional[str] = None
) -> FastMCP:
    mcp = FastMCP(
        name="pao-tools",
        lifespan=create_lifespan(services=services),
    )
    register_health_routes(mcp)
    register_all_tools(mcp, profile=profile)
    return mcp


mcp = create_mcp()
