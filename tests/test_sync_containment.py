"""Tool-layer containment tests for workspace-scoped file operations.

sync_target_files / fetch_runtime_libraries accept local paths from the
caller.  Those paths must stay inside the workspace so an untrusted MCP
caller cannot use the providers as an arbitrary local read/write
primitive.  The containment lives at the MCP tool boundary (resolve_
workspace_path), so these tests drive the published tools through an
in-process client.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from fastmcp import Client

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationRegistry
from paomcp.server import create_mcp
from paomcp.tools.backends.targets import TargetRegistry


def _services(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(tmp_path / "workspace")),
        operation_registry=OperationRegistry(),
        artifact_store=ArtifactStore(str(tmp_path / "artifacts")),
        target_registry=TargetRegistry(),
    )


def _envelope(result) -> dict:
    """Extract the ToolResponse dict from a CallToolResult."""
    for content in result.content:
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    return {}


async def _call(mcp, tool: str, arguments: dict) -> dict:
    async with Client(mcp) as client:
        result = await client.call_tool(tool, arguments, raise_on_error=False)
        return _envelope(result)


@pytest.mark.asyncio
async def test_sync_download_rejects_out_of_workspace_destination(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "sync_target_files",
        {
            "target_id": "whatever",
            "source": "/remote/file",
            "destination": "/etc/passwd",
            "direction": "download",
        },
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_sync_upload_rejects_out_of_workspace_source(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "sync_target_files",
        {
            "target_id": "whatever",
            "source": str(tmp_path / "etc" / "secret"),
            "destination": "/remote/file",
            "direction": "upload",
        },
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_sync_relative_destination_resolves_into_workspace(tmp_path) -> None:
    # A relative destination is legal; it must be rewritten under the
    # workspace before it reaches the registry (proven by the "not found"
    # target error, which only fires after path resolution succeeds).
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "sync_target_files",
        {
            "target_id": "nope",
            "source": "/remote/file",
            "destination": "downloads/out.bin",
            "direction": "download",
        },
    )
    assert response["success"] is False
    assert "nope" in response["error"]["message"]


@pytest.mark.asyncio
async def test_fetch_runtime_libraries_rejects_out_of_workspace_dir(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "fetch_runtime_libraries",
        {
            "target_id": "whatever",
            "destination_dir": "/tmp",
        },
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()
