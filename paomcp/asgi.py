"""Alternate ASGI entrypoint (``uvicorn paomcp.asgi:app``).

Prefer ``python -m paomcp`` (runtime.py), which enforces the
loopback-by-default bind gate before binding. This module cannot see the
address uvicorn will actually bind, so it enforces the same policy per
request instead: loopback clients stay tokenless, non-loopback clients are
refused unless ``PAO_ALLOW_REMOTE=1`` is set and they present the
``PAO_MCP_TOKEN`` bearer. A declared bind host (``PAO_ASGI_HOST`` no longer
exists) cannot widen access.
"""

import logging

from starlette.middleware import Middleware

from paomcp.http.auth import RemoteBindGateMiddleware
from paomcp.logger import configure_logging
from paomcp.server import create_mcp
from paomcp.tools.profiles import profile_name_from_env

logger = logging.getLogger(__name__)

configure_logging()

profile = profile_name_from_env()
logger.info("ASGI app tool profile: %s", profile or "full")
# Built via the factory (not the import-time full global) so PAO_PROFILE
# actually narrows the listed surface here, matching run_http/run_stdio.
mcp = create_mcp(profile=profile)
app = mcp.http_app(path="/mcp", middleware=[Middleware(RemoteBindGateMiddleware)])
