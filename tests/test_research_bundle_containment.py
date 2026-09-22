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
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastmcp import Client

from paomcp.server import create_mcp


def _services(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_paths=SimpleNamespace(workspace_root=str(tmp_path / "workspace")),
    )


def _symlink_escape(tmp_path, name: str) -> Path:
    """Create workspace/<name> -> <tmp_path>/outside and return the outside dir."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    link = workspace / name
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    return outside


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


def test_contained_path_rejects_realpath_escape(tmp_path, monkeypatch) -> None:
    from paomcp.tools import research as research_tools

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_realpath = os.path.realpath

    def fake_realpath(path):
        if os.path.basename(str(path)) == "bundle.zip":
            return str(outside / "bundle.zip")
        return real_realpath(path)

    monkeypatch.setattr(research_tools.os.path, "realpath", fake_realpath)

    with pytest.raises(ValueError, match="symlink escape"):
        research_tools._resolve_contained_workspace_path(
            "bundle.zip",
            workspace_root=str(workspace),
            require_exists=False,
            kind="destination",
        )


@pytest.mark.asyncio
async def test_export_rejects_symlinked_destination_escape(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    _symlink_escape(tmp_path, "escape")
    response = await _call(
        mcp,
        "export_research_session",
        {"session_id": "whatever", "destination": "escape/bundle.zip"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_import_rejects_symlinked_source_escape(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    outside = _symlink_escape(tmp_path, "escape")
    (outside / "bundle.zip").write_bytes(b"not-a-bundle")
    response = await _call(
        mcp,
        "import_research_session",
        {"source_path": "escape/bundle.zip"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_import_rejects_symlinked_artifact_dir_escape(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    (tmp_path / "workspace").mkdir(exist_ok=True)
    (tmp_path / "workspace" / "in.zip").write_bytes(b"not-a-bundle")
    _symlink_escape(tmp_path, "extract")
    response = await _call(
        mcp,
        "import_research_session",
        {"source_path": "in.zip", "artifact_destination_dir": "extract"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()


@pytest.mark.asyncio
async def test_verify_rejects_symlinked_source_escape(tmp_path) -> None:
    mcp = create_mcp(services=_services(tmp_path))
    outside = _symlink_escape(tmp_path, "escape")
    (outside / "bundle.zip").write_bytes(b"not-a-bundle")
    response = await _call(
        mcp,
        "verify_research_bundle",
        {"source_path": "escape/bundle.zip"},
    )
    assert response["success"] is False
    assert "workspace" in response["error"]["message"].lower()
