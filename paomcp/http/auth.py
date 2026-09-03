"""Loopback-by-default binding gates and bearer auth for HTTP surfaces.

Single-user research tool: loopback needs no token, anything else refuses
to bind unless ``PAO_ALLOW_REMOTE=1`` **and** ``PAO_MCP_TOKEN`` is set, in
which case every route except ``/healthz`` requires the bearer token.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_ALLOW_REMOTE_VALUES = {"1", "true", "yes", "on"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_OPEN_PATHS = {"/healthz"}


def allow_remote_enabled() -> bool:
    """Return whether non-loopback binds are explicitly permitted."""
    return os.environ.get("PAO_ALLOW_REMOTE", "").lower() in _ALLOW_REMOTE_VALUES


def configured_token() -> Optional[str]:
    """Return the operator bearer token, or None when unset/blank."""
    token = os.environ.get("PAO_MCP_TOKEN", "")
    return token if token.strip() else None


def is_loopback_host(host: Optional[str]) -> bool:
    """Return whether a bind host is loopback-only."""
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


def check_remote_bind(host: Optional[str], *, what: str) -> Optional[str]:
    """Enforce the loopback-by-default policy for one listener.

    Returns the bearer token that must guard the listener (None on
    loopback). Raises RuntimeError instead of binding an unauthenticated
    RCE surface to the network.
    """
    if is_loopback_host(host):
        return None
    if not allow_remote_enabled():
        raise RuntimeError(
            f"Refusing to bind {what} to non-loopback host {host!r}: set "
            "PAO_ALLOW_REMOTE=1 to allow remote binds (loopback needs nothing)"
        )
    token = configured_token()
    if token is None:
        raise RuntimeError(
            f"Refusing to bind {what} to non-loopback host {host!r} without "
            "PAO_MCP_TOKEN: remote binds require a bearer token"
        )
    logger.warning(
        "Binding %s to non-loopback host %s with bearer auth enabled", what, host
    )
    return token


class BearerAuthMiddleware:
    """Pure-ASGI bearer check; ``/healthz`` stays open for orchestrators."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[Any]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") in _OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        presented = headers.get("authorization", "")
        scheme, _, credential = presented.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            credential, self.token
        ):
            body = b'{"detail":"missing or invalid bearer token"}'
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
