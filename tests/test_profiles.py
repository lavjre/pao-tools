import pytest

from paomcp.server import create_mcp
from paomcp.tools.profiles import (
    ALL_MODULES,
    PROFILES,
    profile_from_env,
    resolve_profile,
)


async def _tool_names(profile=None):
    mcp = create_mcp(profile=profile)
    tools = await mcp.list_tools(run_middleware=False)
    return {tool.name for tool in tools}


def test_resolve_profile_defaults_and_validates():
    assert resolve_profile(None) is None
    assert resolve_profile("full") is None
    assert resolve_profile(" PWN ") is not None
    with pytest.raises(ValueError, match="unknown tool profile"):
        resolve_profile("nope")


def test_profile_from_env(monkeypatch):
    monkeypatch.delenv("PAO_PROFILE", raising=False)
    assert profile_from_env() is None
    monkeypatch.setenv("PAO_PROFILE", "rev")
    assert profile_from_env() is not None


def test_profile_module_sets_cover_all_modules():
    union = set()
    for name, modules in PROFILES.items():
        if name == "full":
            assert modules is None
            continue
        assert modules <= ALL_MODULES, name
        union |= set(modules)
    assert union == set(ALL_MODULES)


@pytest.mark.asyncio
async def test_full_profile_has_all_tools():
    assert len(await _tool_names()) == 199
    assert len(await _tool_names("full")) == 199


@pytest.mark.asyncio
async def test_profiles_shrink_list_tools():
    full = await _tool_names()
    counts = {}
    for profile in ("pwn", "rev", "bounty"):
        names = await _tool_names(profile)
        counts[profile] = len(names)
        assert names < full, profile
        assert len(names) > 50, profile
    # rev is the leanest mission surface, bounty the broadest non-full one.
    assert counts["rev"] < counts["pwn"] < counts["bounty"]


@pytest.mark.asyncio
async def test_profiles_keep_mission_tools():
    pwn = await _tool_names("pwn")
    assert {"run_until_crash", "build_rop_chain", "inspect_tcache"} <= pwn
    assert "ida_decompile" not in pwn
    rev = await _tool_names("rev")
    assert {"decompile_binary", "ida_decompile", "analyze_macho"} <= rev
    assert "build_rop_chain" not in rev
    assert "run_until_crash" not in rev
    bounty = await _tool_names("bounty")
    assert {"generate_bounty_report", "run_fuzz_batch", "analyze_kernel_oops"} <= bounty
    assert "poison_tcache_link" not in bounty


@pytest.mark.asyncio
async def test_every_profile_keeps_response_envelope():
    for profile in ("pwn", "rev", "bounty"):
        mcp = create_mcp(profile=profile)
        tools = await mcp.list_tools(run_middleware=False)
        for tool in tools:
            assert tool.output_schema is not None, (profile, tool.name)
            assert set(tool.output_schema.get("properties", {})) == {
                "api_version",
                "success",
                "tool",
                "session_id",
                "data",
                "error",
                "warnings",
                "meta",
            }, (profile, tool.name)
