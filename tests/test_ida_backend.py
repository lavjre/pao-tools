from paomcp.tools.backends.ida import (
    IdaProMCP,
    PASSTHROUGH_METHODS,
    _clamp_limit,
    _clamp_timeout,
)


def test_clamps():
    assert _clamp_limit(None) == 100
    assert _clamp_limit(0) == 1
    assert _clamp_limit(5000) == 1000
    assert _clamp_timeout(None) > 0
    assert _clamp_timeout(-5) > 0
    assert _clamp_timeout(9999) == 120.0


def test_blocked_method_rejected_without_network():
    ida = IdaProMCP(
        mcp_url="http://127.0.0.1:1/mcp", idalib_url="http://127.0.0.1:1/mcp"
    )
    result = ida._call("nonexistent_method_xyz", {})
    assert result["success"] is False
    assert "not in the curated" in result["error"]


def test_status_degraded_without_server():
    ida = IdaProMCP(
        mcp_url="http://127.0.0.1:1/mcp", idalib_url="http://127.0.0.1:1/mcp"
    )
    status = ida.status()
    assert status["backend"] == "ida-pro-mcp"
    assert status["ready"] is False
    assert status["reasons"]


def test_py_eval_rejects_empty():
    ida = IdaProMCP()
    result = ida.py_eval("   ")
    assert result["success"] is False


def test_idb_open_rejects_bad_mode():
    ida = IdaProMCP()
    result = ida.idb_open("/workspace/x", mode="bogus")
    assert result["success"] is False


def test_curated_subset_covers_core_surface():
    for name in (
        "decompile",
        "disasm",
        "list_funcs",
        "py_eval",
        "idb_open",
        "idb_list",
        "rename",
        "dbg_continue",
        "dbg_regs",
        "dbg_read",
    ):
        assert name in PASSTHROUGH_METHODS


def test_dbg_step_selects_method(monkeypatch):
    ida = IdaProMCP(
        mcp_url="http://127.0.0.1:1/mcp", idalib_url="http://127.0.0.1:1/mcp"
    )
    seen = {}

    def fake_call(method, params=None):
        seen["method"] = method
        return {"success": True, "method": method}

    monkeypatch.setattr(ida, "_call", fake_call)
    ida.dbg_step("into")
    assert seen["method"] == "dbg_step_into"
    ida.dbg_step("over")
    assert seen["method"] == "dbg_step_over"


def test_dbg_call_appends_ext_param():
    ida = IdaProMCP(
        mcp_url="http://127.0.0.1:13337/mcp",
        idalib_url="http://127.0.0.1:8745/mcp",
    )
    # _call builds dbg URLs with ?ext=dbg; verify via monkeypatched _post
    captured = []

    def fake_post(url, payload):
        captured.append(url)
        raise RuntimeError("stop after first")

    ida._post = fake_post  # type: ignore[method-assign]
    ida.dbg_bps()
    assert captured
    assert all("ext=dbg" in u for u in captured)


def test_session_is_shared_and_close_resets():
    ida = IdaProMCP()
    first = ida._session_client()
    assert ida._session_client() is first
    ida.close()
    assert ida._session is None
    # A new session is lazily created after close; close is idempotent.
    assert ida._session_client() is not first
    ida.close()
    ida.close()


def test_publish_crash_builds_set_cmt_snippet(monkeypatch):
    ida = IdaProMCP()
    seen = {}

    def fake_eval(code, database=None):
        seen["code"] = code
        seen["database"] = database
        return {"success": True, "parsed": {"result": "ok"}}

    monkeypatch.setattr(ida, "py_eval", fake_eval)
    result = ida.publish_crash(
        "0x401000", "SIGSEGV", database="db", classification="bad_ptr", note='a"b\nc'
    )
    assert result["success"] is True
    assert "idc.set_cmt(0x401000" in seen["code"]
    assert "SIGSEGV" in seen["code"]
    assert seen["database"] == "db"
    assert result["parsed"]["published"]["address"] == 0x401000


def test_publish_rejects_bad_address_and_empty():
    ida = IdaProMCP()
    bad = ida.publish_crash("nope", "SIGSEGV")
    assert bad["success"] is False
    import pytest

    with pytest.raises(ValueError, match="signal"):
        ida.publish_crash(0x1000, "  ")
    with pytest.raises(ValueError, match="text"):
        ida.publish_note(0x1000, "  ")
