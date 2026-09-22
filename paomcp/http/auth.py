"""Loopback-by-default binding gates and bearer auth for HTTP surfaces.

Single-user research tool: loopback needs no token, anything else refuses
to bind unless ``PAO_ALLOW_REMOTE=1`` **and** ``PAO_MCP_TOKEN`` is set, in
which case every route except ``/healthz`` requires the bearer token.

``check_remote_bind`` enforces the policy at bind time (``python -m
paomcp``). ``RemoteBindGateMiddleware`` enforces the same policy at request
time for ASGI entrypoints (``uvicorn paomcp.asgi:app``) that cannot know
which address uvicorn will actually bind: loopback clients stay tokenless,
non-loopback clients need ``PAO_ALLOW_REMOTE=1`` plus a valid bearer, and
no env-declared bind address is trusted.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import secrets
import socket
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_ALLOW_REMOTE_VALUES = {"1", "true", "yes", "on"}
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_OPEN_PATHS = {"/healthz"}


def allow_remote_enabled() -> bool:
    """Return whether non-loopback binds are explicitly permitted."""
    return os.environ.get("PAO_ALLOW_REMOTE", "").lower() in _ALLOW_REMOTE_VALUES


def configured_token() -> Optional[str]:
    """Return the operator bearer token, or None when unset/blank.

    The value is stripped so a token with accidental padding still matches
    the credential presented by clients.
    """
    token = os.environ.get("PAO_MCP_TOKEN", "")
    stripped = token.strip()
    return stripped if stripped else None


def _is_loopback_ip(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def is_loopback_host(host: Optional[str]) -> bool:
    """Return whether a bind host only ever resolves to loopback addresses.

    Literal loopback names and addresses take a fast path. Other names are
    resolved with :func:`socket.getaddrinfo` and count as loopback only when
    *every* resolved address is loopback; resolution failures fail closed.
    """
    text = (host or "").strip().lower()
    if not text:
        return False
    if text in _LOOPBACK_HOSTS:
        return True
    if _is_loopback_ip(text):
        return True
    try:
        infos = socket.getaddrinfo(text, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    addresses: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if sockaddr and isinstance(sockaddr[0], str):
            addresses.add(sockaddr[0])
    return bool(addresses) and all(_is_loopback_ip(addr) for addr in addresses)


def is_loopback_address(host: Optional[str]) -> bool:
    """Return whether a peer address is loopback; never performs DNS."""
    text = (host or "").strip()
    if not text:
        return False
    if text.lower().startswith("::ffff:"):
        text = text[7:]
    return text.lower() == "localhost" or _is_loopback_ip(text)


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


def _constant_time_equals(presented: str, expected: str) -> bool:
    """Compare credentials as bytes so non-ASCII headers cannot raise."""
    return secrets.compare_digest(
        presented.encode("utf-8", "surrogateescape"),
        expected.encode("utf-8", "surrogateescape"),
    )


def _presented_bearer(scope: Dict[str, Any]) -> Optional[str]:
    headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }
    presented = headers.get("authorization", "")
    scheme, _, credential = presented.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return credential


async def _send_unauthorized(
    send: Callable[[Any], Awaitable[Any]], detail: str
) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
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


class BearerAuthMiddleware:
    """Pure-ASGI bearer check; ``/healthz`` stays open for orchestrators.

    Used by ``runtime.run_http`` after the bind-time gate has already
    established that the listener is non-loopback.
    """

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
        credential = _presented_bearer(scope)
        if credential is None or not _constant_time_equals(credential, self.token):
            await _send_unauthorized(send, "missing or invalid bearer token")
            return
        await self.app(scope, receive, send)


class RemoteBindGateMiddleware:
    """Request-time loopback-by-default gate for ASGI embedding.

    Loopback clients stay tokenless; a non-loopback client is refused with
    401 unless ``PAO_ALLOW_REMOTE=1`` is set and it presents the configured
    ``PAO_MCP_TOKEN`` bearer. The decision reads the real peer address from
    the ASGI scope, so a declared bind host (e.g. ``PAO_ASGI_HOST``) cannot
    widen access. In-process scopes without a ``client`` tuple are treated
    as loopback because real transports always supply one.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Dict[str, Any],
        receive: Callable[[], Awaitable[Any]],
        send: Callable[[Any], Awaitable[Any]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") in _OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        host = client[0] if isinstance(client, (tuple, list)) and client else None
        if host is None or is_loopback_address(host):
            await self.app(scope, receive, send)
            return
        if not allow_remote_enabled():
            await _send_unauthorized(
                send, "remote access disabled: set PAO_ALLOW_REMOTE=1 and PAO_MCP_TOKEN"
            )
            return
        token = configured_token()
        credential = _presented_bearer(scope)
        if (
            token is None
            or credential is None
            or not _constant_time_equals(credential, token)
        ):
            await _send_unauthorized(send, "missing or invalid bearer token")
            return
        await self.app(scope, receive, send)
