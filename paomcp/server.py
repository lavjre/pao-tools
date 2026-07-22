import logging
from typing import Optional

from fastmcp import FastMCP

from paomcp.http.health import register_health_routes
from paomcp.lifespan import create_lifespan
from paomcp.services import AppServices
from paomcp.tools import register_all_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def create_mcp(services: Optional[AppServices] = None) -> FastMCP:
    mcp = FastMCP(
        name="pao-tools",
        lifespan=create_lifespan(services=services),
    )
    register_health_routes(mcp)
    register_all_tools(mcp)
    return mcp


mcp = create_mcp()
