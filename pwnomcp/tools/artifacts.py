"""MCP tools for retrieving large response artifacts."""

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from pwnomcp.tools.common import catch_errors, get_services, run_blocking


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def list_artifacts(
        session_id: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List stored large-response artifacts.

        Args:
            session_id: Optional debug session filter. Omit it to list every artifact.
        """
        services = get_services(ctx)
        artifacts = await run_blocking(
            lambda: services.artifact_store.list(session_id=session_id)
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
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Get size, hash, source tool, and session metadata for an artifact."""
        services = get_services(ctx)
        artifact = await run_blocking(lambda: services.artifact_store.get(artifact_id))
        return {"success": True, "artifact": artifact.to_dict()}

    @mcp.tool()
    @catch_errors(auto_artifact=False)
    async def read_artifact(
        artifact_id: str,
        offset: int = 0,
        length: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Read artifact content by byte range without losing stored data.

        Args:
            artifact_id: Identifier returned by an artifact-delivered tool response.
            offset: Zero-based byte offset.
            length: Bytes to return. Defaults to the configured chunk size. There is
                no enforced maximum; use repeated calls with next_offset for streaming.
        """
        services = get_services(ctx)
        effective_length = (
            services.artifact_store.default_chunk_bytes if length is None else length
        )
        result = await run_blocking(
            lambda: services.artifact_store.read(
                artifact_id,
                offset=offset,
                length=effective_length,
            )
        )
        return {"success": True, **result}
