"""Tests for Claude Code plugin packaging (Feature A).

Asserts the three manifests ship a coherent, installable plugin:

* ``.claude-plugin/plugin.json``        -- the plugin manifest
* ``.claude-plugin/marketplace.json``   -- in-repo marketplace
* ``.mcp.json`` (repo root)             -- stdio MCP server that launches the
                                           server from ``${CLAUDE_PLUGIN_ROOT}``

These are pure JSON/shape tests -- they never validate against the live
``claude`` CLI (that is an explicit, manual ``claude plugin validate .`` step).
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = REPO_ROOT / ".claude-plugin"


def _load(relative: str):
    with open(REPO_ROOT / relative, encoding="utf-8") as fh:
        return json.load(fh)


def test_plugin_manifest_required_fields():
    manifest = _load(".claude-plugin/plugin.json")
    for key in (
        "name",
        "displayName",
        "version",
        "description",
        "author",
        "repository",
        "license",
    ):
        assert key in manifest, f"plugin.json missing {key!r}"

    assert manifest["name"] == "pao-tools"
    assert manifest["version"] == "0.3.0"
    assert manifest["repository"] == "https://github.com/lavjre/pao-tools"
    assert (
        isinstance(manifest["description"], str) and len(manifest["description"]) > 40
    )
    # author is the standard {name, url?} shape.
    assert isinstance(manifest["author"], dict) and "name" in manifest["author"]


def test_marketplace_manifest_shape():
    marketplace = _load(".claude-plugin/marketplace.json")
    assert "name" in marketplace
    assert "owner" in marketplace and "name" in marketplace["owner"]
    plugins = marketplace["plugins"]
    assert isinstance(plugins, list) and len(plugins) >= 1
    entry = plugins[0]
    for key in ("name", "source", "description"):
        assert key in entry, f"marketplace plugin entry missing {key!r}"
    assert entry["name"] == "pao-tools"
    assert entry["source"] == "."


def test_marketplace_name_not_reserved():
    # Reserved marketplace names must be avoided so `claude plugin marketplace
    # add` does not collide with built-in marketplaces.
    RESERVED = {"claude", "anthropic", "community", "official", "example"}
    marketplace = _load(".claude-plugin/marketplace.json")
    name = marketplace["name"].lower()
    assert name not in RESERVED, f"marketplace name {name!r} is reserved"
    assert name == "pao-tools"


def test_mcp_json_launches_server_from_plugin_root():
    mcp = _load(".mcp.json")
    servers = mcp["mcpServers"]
    assert "pao-tools" in servers
    server = servers["pao-tools"]

    assert server["type"] == "stdio"
    assert server["command"] == "uv"

    args = server["args"]
    # Must resolve the project relative to the plugin root, then run stdio.
    assert "--directory" in args
    root_idx = args.index("--directory")
    assert args[root_idx + 1] == "${CLAUDE_PLUGIN_ROOT}"
    assert "paomcp" in args and "--stdio" in args
    assert args[-1] == "--stdio"

    env = server["env"]
    assert env.get("PWNDBG_NO_AUTOUPDATE") == "1"
    assert env.get("PYTHONUNBUFFERED") == "1"
    assert env.get("PAO_RUNTIME_ROOT") == "${CLAUDE_PLUGIN_ROOT}/runtime"


def test_plugin_and_skill_files_present():
    assert (PLUGIN_DIR / "plugin.json").is_file()
    assert (PLUGIN_DIR / "marketplace.json").is_file()
    skill = REPO_ROOT / "skills" / "heap-exploitation-triage" / "SKILL.md"
    assert skill.is_file(), "skills/heap-exploitation-triage/SKILL.md must ship"
    assert "## Workflow" in skill.read_text(encoding="utf-8")


def test_profile_skills_ship_with_workflows():
    for skill in ("pwn-exploit-dev", "reverse-engineering", "bounty-hunting"):
        path = REPO_ROOT / "skills" / skill / "SKILL.md"
        assert path.is_file(), f"skills/{skill}/SKILL.md must ship"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\nname: " + skill), skill
        assert "description:" in text.split("---")[1], skill
        assert "## Workflow" in text, skill
        assert "## Rules" in text, skill
