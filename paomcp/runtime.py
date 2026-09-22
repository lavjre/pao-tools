import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

import uvicorn
from starlette.middleware import Middleware

from paomcp.http.attach import create_attach_app
from paomcp.http.auth import BearerAuthMiddleware, check_remote_bind
from paomcp.logger import configure_logging
from paomcp.server import create_mcp
from paomcp.services import AppServices, close_services, create_services
from paomcp.tools.profiles import profile_name_from_env

logger = logging.getLogger(__name__)

# In-flight attach handlers drive GDB through the Python thread pool, where a
# single command can legitimately take 7s+. Shutdown therefore drains the
# attach server first and only closes the service graph once no handler can
# still be mid-call; both waits stay bounded.
_ATTACH_DRAIN_TIMEOUT_SECONDS = 30.0
_ATTACH_FORCE_TIMEOUT_SECONDS = 5.0


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

    attach_config = uvicorn.Config(
        attach_app,
        host=host,
        port=port,
        log_level="info",
        timeout_graceful_shutdown=int(_ATTACH_DRAIN_TIMEOUT_SECONDS),
    )
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


def _stop_attach_server(
    handle: Optional[AttachServerHandle],
    *,
    drain_timeout: float = _ATTACH_DRAIN_TIMEOUT_SECONDS,
    force_timeout: float = _ATTACH_FORCE_TIMEOUT_SECONDS,
) -> bool:
    """Stop the attach server and return whether its handlers fully drained.

    A graceful ``should_exit`` first drains in-flight requests; if the
    bounded drain expires, force the exit and report ``False`` so the caller
    defers service shutdown instead of closing the debugger pipes under a
    handler that may still be mid-GDB-call.
    """
    if not handle:
        return True

    logger.info("Stopping attach server (draining for up to %ss)...", drain_timeout)
    handle.server.should_exit = True
    if handle.thread.is_alive():
        handle.thread.join(timeout=drain_timeout)
    if handle.thread.is_alive():
        logger.warning(
            "Attach server still draining after %ss; forcing exit", drain_timeout
        )
        handle.server.force_exit = True
        handle.thread.join(timeout=force_timeout)
    if handle.thread.is_alive():
        logger.warning("Attach server thread is still alive; not closing services")
        return False
    return True


def run_stdio() -> None:
    configure_logging()
    profile = profile_name_from_env()
    logger.info("Initializing Pao Tools server (stdio mode)...")
    logger.info("Tool profile: %s", profile or "full")
    services = create_services()
    try:
        mcp = create_mcp(services=services, profile=profile)
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
    configure_logging()
    profile = profile_name_from_env()
    logger.info(
        "Starting MCP server on %s:%s (path %s) and attach server on %s:%s",
        host,
        port,
        http_path,
        attach_host,
        attach_port,
    )
    logger.info("Tool profile: %s", profile or "full")

    # Loopback binds need nothing; any non-loopback bind fails closed here
    # unless PAO_ALLOW_REMOTE=1 and PAO_MCP_TOKEN are both set. Auth is
    # scoped per listener so a remote attach port never weakens loopback MCP.
    token_mcp = check_remote_bind(host, what="MCP server")
    token_attach = check_remote_bind(attach_host, what="attach server")

    services: Optional[AppServices] = None
    attach_handle: Optional[AttachServerHandle] = None
    try:
        services = create_services()
        mcp = create_mcp(services=services, profile=profile)
        attach_app = create_attach_app(services)
        if token_attach is not None:
            attach_app.add_middleware(BearerAuthMiddleware, token=token_attach)
        attach_handle = _start_attach_server(attach_host, attach_port, attach_app)

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
        drained = _stop_attach_server(attach_handle)
        if services is not None:
            if drained:
                close_services(services)
            else:
                logger.warning(
                    "Attach handlers did not drain; deferring service shutdown "
                    "instead of closing debugger pipes under an active handler"
                )
