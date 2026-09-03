import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import uvicorn
from starlette.middleware import Middleware

from paomcp.http.attach import create_attach_app
from paomcp.http.auth import BearerAuthMiddleware, check_remote_bind
from paomcp.server import create_mcp
from paomcp.services import close_services, create_services

logger = logging.getLogger(__name__)


@dataclass
class AttachServerHandle:
    server: uvicorn.Server
    thread: threading.Thread


def _start_attach_server(
    host: Optional[str], port: Optional[int], attach_app
) -> Optional[AttachServerHandle]:
    if not host or not port:
        logger.info("Attach server disabled (host or port missing)")
        return None

    attach_config = uvicorn.Config(attach_app, host=host, port=port, log_level="info")
    attach_server = uvicorn.Server(attach_config)

    def _serve_attach() -> None:
        try:
            logger.info("Attach server listening on %s:%s", host, port)
            attach_server.run()
        except Exception:
            logger.exception("Attach server crashed")

    thread = threading.Thread(
        target=_serve_attach, name="pao-tools-attach-server", daemon=True
    )
    thread.start()

    return AttachServerHandle(server=attach_server, thread=thread)


def _stop_attach_server(handle: Optional[AttachServerHandle]) -> None:
    if not handle:
        return

    logger.info("Stopping attach server...")
    handle.server.should_exit = True
    handle.server.force_exit = True
    if handle.thread.is_alive():
        handle.thread.join(timeout=5)


def run_stdio() -> None:
    logger.info("Initializing Pao Tools server (stdio mode)...")
    services = create_services()
    mcp = create_mcp(services=services)
    try:
        logger.info("Starting MCP server in stdio mode...")
        mcp.run(transport="stdio")
    finally:
        close_services(services)


def run_http(
    host: str = "127.0.0.1",
    port: int = 5500,
    attach_host: str = "127.0.0.1",
    attach_port: int = 5501,
    http_path: str = "/mcp",
) -> None:
    logger.info(
        "Starting MCP server on %s:%s (path %s) and attach server on %s:%s",
        host,
        port,
        http_path,
        attach_host,
        attach_port,
    )

    # Loopback binds need nothing; any non-loopback bind fails closed here
    # unless PAO_ALLOW_REMOTE=1 and PAO_MCP_TOKEN are both set. Auth is
    # scoped per listener so a remote attach port never weakens loopback MCP.
    token_mcp = check_remote_bind(host, what="MCP server")
    token_attach = check_remote_bind(attach_host, what="attach server")

    services = create_services()
    mcp = create_mcp(services=services)
    attach_app = create_attach_app(services)
    if token_attach is not None:
        attach_app.add_middleware(BearerAuthMiddleware, token=token_attach)
    attach_handle = _start_attach_server(attach_host, attach_port, attach_app)

    try:
        logger.info("Launching FastMCP server via mcp.run(transport='http')...")
        extra_kwargs: Dict[str, Any] = (
            {"middleware": [Middleware(BearerAuthMiddleware, token=token_mcp)]}
            if token_mcp is not None
            else {}
        )
        mcp.run(transport="http", host=host, port=port, path=http_path, **extra_kwargs)
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down Pao Tools server")
    finally:
        _stop_attach_server(attach_handle)
        close_services(services)
