"""Alternate ASGI entrypoint (``uvicorn paomcp.asgi:app``).

Prefer ``python -m paomcp`` (runtime.py), which enforces the
loopback-by-default bind gate. This module fails closed the same way:
non-loopback serving requires ``PAO_ALLOW_REMOTE=1`` plus
``PAO_MCP_TOKEN``, applied here as bearer middleware.
"""

import logging
import os

from starlette.middleware import Middleware

from paomcp.http.auth import BearerAuthMiddleware, check_remote_bind
from paomcp.server import create_mcp

logger = logging.getLogger(__name__)

_ASGI_HOST = os.environ.get("PAO_ASGI_HOST", "127.0.0.1")
_ASGI_PROFILE = os.environ.get("PAO_PROFILE", "full")
_token = check_remote_bind(_ASGI_HOST, what="ASGI app")
logger.info("ASGI app tool profile: %s", _ASGI_PROFILE)
# Built via the factory (not the import-time full global) so PAO_PROFILE
# actually narrows the listed surface here, matching run_http/run_stdio.
mcp = create_mcp(profile=_ASGI_PROFILE)
_middleware = (
    [Middleware(BearerAuthMiddleware, token=_token)] if _token is not None else []
)

if _token is not None:
    logger.warning("ASGI app serving non-loopback with bearer auth enabled")

app = mcp.http_app(path="/mcp", middleware=_middleware or None)
