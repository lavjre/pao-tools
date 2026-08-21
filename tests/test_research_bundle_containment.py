"""Tool-layer containment tests for research bundle file operations.

export_research_session / import_research_session / verify_research_bundle
accept local paths from the caller.  Those paths must stay inside the
workspace so an untrusted MCP caller cannot use the bundle surface as an
arbitrary local read/write primitive (the same boundary already applied to
fetch_repo and sync_target_files).  The containment lives at the MCP tool
boundary (resolve_workspace_path), so these tests drive the published tools
through an in-process client and assert the rejection fires before any
backend work.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from fastmcp import Client

from paomcp.server import create_mcp


def _services(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(tmp_path / "workspace")),
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
async def test_export_rejects_out_of_workspace_destination(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "export_research_session",
        {"session_id": "whatever", "destination": "/etc/passwd"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_import_rejects_out_of_workspace_source(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "import_research_session",
        {"source_path": "/etc/shadow"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_import_rejects_out_of_workspace_artifact_dir(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "import_research_session",
        {
            "source_path": "bundles/in.zip",
            "artifact_destination_dir": "/tmp",
        },
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_verify_rejects_out_of_workspace_source(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    response = await _call(
        mcp,
        "verify_research_bundle",
        {"source_path": "/var/log/syslog"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()
