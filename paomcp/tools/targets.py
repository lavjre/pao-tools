"""MCP tools for typed target-provider lifecycle (Stage 8)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def _resolve_session_id(services: Any, session_id: Optional[str]) -> Optional[str]:
    """Return the caller-provided session id, or fall back to the default."""
    if session_id is None or session_id == "":
        try:
            registry = services.session_registry
        except AttributeError:
            return None
        return getattr(registry, "default_session_id", None)
    return session_id


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def create_target(
        kind: str,
        config: Optional[Mapping[str, Any]] = None,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Create a typed target descriptor for a future attach / connect.

        ``kind`` is one of: ``local_process``, ``attach_pid``,
        ``gdbserver``, ``tcp_remote``, ``ssh``, ``docker``, ``qemu_user``,
        ``qemu_system``.  All eight providers are registered; actual
        availability depends on their runtime executable, daemon, credentials,
        image, kernel, rootfs, or remote endpoint.

        ``config`` is a per-kind dict.  Common keys:

        * ``local_process``: ``binary_path`` (required), ``argv``, ``env``,
          ``working_dir``, ``stdin_bytes``.
        * ``attach_pid``: ``pid`` (required, positive int).
        * ``gdbserver`` / ``tcp_remote``: host and port plus the provider's
          process or connection settings.
        * ``ssh``: ``host``, ``ssh_user``, key/password authentication and
          optional ``ssh_known_hosts_path``. Unknown host keys are rejected.
        * ``docker``: container/image and command settings.
        * ``qemu_user`` / ``qemu_system``: emulator plus binary, rootfs,
          kernel, machine and GDB-listen settings as applicable.

        Optional metadata (informational, never overrides typed
        detection): ``arch``, ``platform``, ``notes``, ``name``.  Every
        target is bound to the resolved ``session_id`` (falls back to
        the default debug session when omitted) and counts against the
        per-session target cap.

        Creating a target does not start it -- call ``start_target``
        before ``attach_debugger``.
        """
        services = get_services(ctx)
        resolved_session = _resolve_session_id(services, session_id)
        return services.target_registry.create_target_response(
            kind=kind,
            config=dict(config) if config else None,
            session_id=resolved_session,
        )

    @mcp.tool()
    @catch_errors()
    async def list_targets(
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Enumerate targets registered for a session (or all sessions).

        Args:
            session_id: Filter to targets owned by this session.  Omit
                (``None``) to list every registered target regardless
                of ownership.
        """
        services = get_services(ctx)
        return services.target_registry.list_targets_response(session_id=session_id)

    @mcp.tool()
    @catch_errors()
    async def get_target_status(
        target_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Return a target's typed lifecycle state and metadata.

        For local-process targets we refresh the subprocess liveness
        from the OS before reporting so a target that has since exited
        shows ``state = "stopped"`` with the observed ``exit_code``
        rather than a stale ``running``.
        """
        services = get_services(ctx)
        return services.target_registry.get_status_response(target_id=target_id)

    @mcp.tool()
    @catch_errors()
    async def get_target_capabilities(
        target_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Report which lifecycle operations this target's provider supports.

        Every provider surfaces the same capability keys (``start``,
        ``stop``, ``attach_debugger``, ``file_sync``, ``reconnect``,
        ``port_forward``) so a caller can branch honestly.
        """
        services = get_services(ctx)
        return services.target_registry.get_capabilities_response(target_id=target_id)

    @mcp.tool()
    @catch_errors()
    async def start_target(
        target_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Bring a target to a runnable state.

        Provider-specific behavior may spawn a process/container/emulator,
        verify an existing PID or endpoint, or establish an SSH/GDB-remote
        transport.  Inspect ``get_target_capabilities`` before branching.
        """
        services = get_services(ctx)
        return services.target_registry.start_target_response(target_id=target_id)

    @mcp.tool()
    @catch_errors()
    async def stop_target(
        target_id: str,
        timeout_sec: float = 10.0,
        remove: bool = False,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Stop or detach a target (kind-specific semantics).

        * ``local_process``: SIGTERM, then SIGKILL after ``timeout_sec``.
        * ``attach_pid``:    detach (never kills the caller's inferior).
        * remote/container/QEMU kinds: stop only provider-owned resources;
          externally owned endpoints are disconnected rather than killed.

        When ``remove`` is true the target is dropped from the registry
        after a successful stop, freeing its session-cap slot.
        """
        services = get_services(ctx)
        return services.target_registry.stop_target_response(
            target_id=target_id,
            timeout_sec=timeout_sec,
            remove=remove,
        )

    @mcp.tool()
    @catch_errors()
    async def connect_target(
        target_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Bring a remote-kind target's transport up (alias for start).

        For remote providers this probes, spawns, or opens the provider
        transport without attaching GDB.  Prefer this over ``start_target``
        for remote workflows so the verb matches the intent.
        """
        services = get_services(ctx)
        return services.target_registry.connect_target_response(target_id=target_id)

    @mcp.tool()
    @catch_errors()
    async def disconnect_target(
        target_id: str,
        timeout_sec: float = 10.0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Tear down the target's transport without freeing its registry slot.

        Semantics mirror ``stop_target`` for remote kinds -- SIGTERM the
        owned gdbserver, or (for ``tcp_remote``) simply detach without
        touching the remote process.  Follow with ``reconnect_target``
        if you plan to reuse the same target id.
        """
        services = get_services(ctx)
        return services.target_registry.disconnect_target_response(
            target_id=target_id, timeout_sec=timeout_sec
        )

    @mcp.tool()
    @catch_errors()
    async def reconnect_target(
        target_id: str,
        timeout_sec: float = 10.0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Disconnect then re-connect a target in one call.

        Useful when a remote gdbserver's session drifts (target
        crashed, connection stalled) but the spec is still valid.
        """
        services = get_services(ctx)
        return services.target_registry.reconnect_target_response(
            target_id=target_id, timeout_sec=timeout_sec
        )

    @mcp.tool()
    @catch_errors()
    async def sync_target_files(
        target_id: str,
        source: str,
        destination: str,
        direction: str = "download",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Transfer one file between the caller and the remote target.

        ``direction`` is ``"download"`` (remote ``source`` -> local
        ``destination``) or ``"upload"`` (local ``source`` -> remote
        ``destination``).  Only providers whose ``capabilities.file_sync``
        is true accept this operation; SSH uses SFTP and Docker uses
        ``docker cp``.  Other providers fail with a typed error rather
        than silently no-op-ing.
        """
        services = get_services(ctx)
        return services.target_registry.sync_target_files_response(
            target_id=target_id,
            source=source,
            destination=destination,
            direction=direction,
        )

    @mcp.tool()
    @catch_errors()
    async def fetch_runtime_libraries(
        target_id: str,
        destination_dir: str,
        candidate_paths: Optional[list] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Pull the target's libc / dynamic loader for offline analysis.

        For an ``ssh`` target this SFTPs each candidate path (default
        set covers glibc + musl on both x86_64 and i386) into
        ``destination_dir`` when the remote file exists.  The response
        distinguishes ``fetched`` from ``missing`` paths so a caller
        knows which candidates the target actually has.  Pair with
        ``identify_libc`` (S7.2) to score leaks against the pulled
        libc.
        """
        services = get_services(ctx)
        return services.target_registry.fetch_runtime_libraries_response(
            target_id=target_id,
            destination_dir=destination_dir,
            candidate_paths=candidate_paths,
        )

    @mcp.tool()
    @catch_errors()
    async def attach_debugger(
        target_id: str,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Attach the debugger session's GDB to the target.

        The target's own ``session_id`` field records ownership; the
        ``session_id`` argument here selects which debugger session's
        ``GdbController`` gets the attach.  Falls back to the default
        debug session when omitted.  After a successful attach the
        target moves to ``state = "attached"``.
        """
        services = get_services(ctx)
        resolved_session = _resolve_session_id(services, session_id)
        session = resolve_debug_session(
            services, session_id=resolved_session, create_if_missing=False
        )

        def _do_attach() -> Dict[str, Any]:
            return services.target_registry.attach_debugger_response(
                target_id=target_id, gdb=session.gdb
            )

        result = await run_session_action(session, _do_attach)
        result["session_id"] = session.session_id
        return result


__all__ = ["register"]
