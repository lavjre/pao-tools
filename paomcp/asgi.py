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
from paomcp.server import mcp

logger = logging.getLogger(__name__)

_ASGI_HOST = os.environ.get("PAO_ASGI_HOST", "127.0.0.1")
_token = check_remote_bind(_ASGI_HOST, what="ASGI app")
_middleware = (
    [Middleware(BearerAuthMiddleware, token=_token)] if _token is not None else []
)

if _token is not None:
    logger.warning("ASGI app serving non-loopback with bearer auth enabled")

app = mcp.http_app(path="/mcp", middleware=_middleware or None)
