"""MCP tools for retrieving large response artifacts."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services, run_blocking


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def list_artifacts(
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List stored large-response artifacts owned by one exact session.

        Args:
            session_id: Debug session owner. Omit it to list only unscoped artifacts.
        """
        services = get_services(ctx)
        artifacts = await run_blocking(
            lambda: services.artifact_store.list_owned(session_id)
        )
        return {
            "success": True,
            "session_id": session_id,
            "count": len(artifacts),
            "artifacts": [artifact.to_dict() for artifact in artifacts],
        }

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def get_artifact_info(
        artifact_id: str,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get size, hash, source tool, and session metadata for an artifact."""
        services = get_services(ctx)
        artifact = await run_blocking(
            lambda: services.artifact_store.get_for_session(artifact_id, session_id)
        )
        return {
            "success": True,
            "session_id": session_id,
            "artifact": artifact.to_dict(),
        }

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def get_artifact_store_status(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get artifact usage, free disk space, worker state, and retention policy."""
        services = get_services(ctx)
        status = await run_blocking(services.artifact_store.status)
        return {"success": True, **status}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def set_artifact_pinned(
        artifact_id: str,
        pinned: bool = True,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Pin or unpin an artifact against automatic TTL/quota cleanup."""
        services = get_services(ctx)
        artifact = await run_blocking(
            lambda: services.artifact_store.set_pinned_for_session(
                artifact_id, session_id, pinned
            )
        )
        return {"success": True, "artifact": artifact.to_dict()}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def delete_artifact(
        artifact_id: str,
        force: bool = False,
        session_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Delete an artifact. Pinned artifacts require force=true."""
        services = get_services(ctx)
        artifact = await run_blocking(
            lambda: services.artifact_store.delete_for_session(
                artifact_id, session_id, force=force
            )
        )
        return {"success": True, "deleted": artifact.to_dict()}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def cleanup_artifacts(
        dry_run: bool = True,
        idempotency_key: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Preview or apply the configured TTL, quota, and disk cleanup policy."""
        services = get_services(ctx)
        result = await run_blocking(
            lambda: services.artifact_store.cleanup(dry_run=dry_run)
        )
        return {"success": True, **result}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def read_artifact(
        artifact_id: str,
        offset: int = 0,
        length: Optional[int] = None,
        format: str = "auto",
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read artifact content by byte range without losing stored data.

        Args:
            artifact_id: Identifier returned by an artifact-delivered tool response.
            offset: Zero-based byte offset.
            length: Bytes to return. Defaults to the configured chunk size and is
                capped by ``PAO_ARTIFACT_MAX_READ_BYTES``. Use repeated calls with
                ``next_offset`` for streaming.
            format: auto (text->content only, binary->base64 only), text,
                base64, or both (legacy dual-field, costs ~2.3x tokens).
        """
        services = get_services(ctx)
        effective_length = (
            services.artifact_store.default_chunk_bytes if length is None else length
        )
        result = await run_blocking(
            lambda: services.artifact_store.read_for_session(
                artifact_id,
                session_id,
                offset=offset,
                length=effective_length,
                format=format,
            )
        )
        return {"success": True, "session_id": session_id, **result}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def verify_artifact(
        artifact_id: str,
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Recompute an artifact's size and SHA-256 integrity manifest."""
        services = get_services(ctx)
        result = await run_blocking(
            lambda: services.artifact_store.verify(artifact_id, session_id)
        )
        return {"success": True, "session_id": session_id, **result}
