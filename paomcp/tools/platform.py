"""MCP tool for unified platform capability discovery."""

import threading
from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.backends.platform import (
    DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
    PlatformCapabilitiesCache,
)
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_blocking,
    run_session_action,
)

_SERVICE_CACHE_ATTRIBUTE = "_pao_platform_capabilities_cache"
_SERVICE_CACHE_CREATION_LOCK = threading.Lock()


def _service_cache(services: Any) -> PlatformCapabilitiesCache:
    """Return the platform cache owned by one AppServices lifetime."""
    cached = getattr(services, _SERVICE_CACHE_ATTRIBUTE, None)
    if isinstance(cached, PlatformCapabilitiesCache):
        return cached
    with _SERVICE_CACHE_CREATION_LOCK:
        cached = getattr(services, _SERVICE_CACHE_ATTRIBUTE, None)
        if isinstance(cached, PlatformCapabilitiesCache):
            return cached
        cached = PlatformCapabilitiesCache()
        setattr(services, _SERVICE_CACHE_ATTRIBUTE, cached)
        return cached


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def get_platform_capabilities(
        session_id: Optional[str] = None,
        refresh: bool = False,
        probe: bool = False,
        probe_timeout: float = 2.0,
        probe_total_timeout: float = DEFAULT_PROBE_TOTAL_TIMEOUT_SECONDS,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get one typed manifest for the installed analysis platform.

        Args:
            session_id: Optional existing debugger session. When supplied, its cached
                GDB/pwndbg compatibility result is used. Required only when a specific
                session must be refreshed.
            refresh: Re-query GDB/pwndbg and rebuild the platform manifest.
            probe: Run bounded, non-interactive version probes for resolved helper
                executables. No subprocess is launched when false.
            probe_timeout: Per-executable timeout, greater than 0 and at most 10
                seconds.
            probe_total_timeout: Deadline for the complete parallel probe set,
                greater than 0 and at most 60 seconds.
        """
        services = get_services(ctx)
        debugger_capabilities = services.debugger_compatibility
        resolved_session_id: Optional[str] = None

        if session_id is not None or refresh:
            selected_session_id = (
                session_id
                or services.session_registry.default_session_id
                or services.default_session_id
            )
            session = resolve_debug_session(
                services,
                session_id=selected_session_id,
                create_if_missing=False,
            )
            debugger_capabilities = await run_session_action(
                session,
                lambda: session.tools.get_debugger_capabilities(refresh),
                sync_pid=False,
            )
            resolved_session_id = session.session_id
            if refresh:
                services.debugger_compatibility = debugger_capabilities

        decompiler_status = services.decompiler.status()
        cache_scope = (
            f"session:{resolved_session_id}"
            if resolved_session_id is not None
            else "startup"
        )
        backend = _service_cache(services).get(
            cache_scope,
            debugger_capabilities,
            decompiler_status,
            refresh=refresh,
        )
        cancellation_event = services.operation_registry.cancellation_event()
        result = await run_blocking(
            lambda: backend.get_capabilities(
                refresh=refresh,
                probe=probe,
                probe_timeout=probe_timeout,
                probe_total_timeout=probe_total_timeout,
                cancellation_event=cancellation_event,
            )
        )
        if resolved_session_id is not None:
            result["session_id"] = resolved_session_id
        return result
