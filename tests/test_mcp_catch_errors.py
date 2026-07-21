import pytest

from pwnomcp.tools.common import build_tool_response, catch_errors


@pytest.mark.asyncio
async def test_catch_errors_returns_unified_error_response():
    @catch_errors()
    async def boom(session_id: str):
        raise ValueError("boom")

    result = await boom(session_id="chal-a")

    assert result["api_version"] == "1.0"
    assert result["success"] is False
    assert result["tool"] == "boom"
    assert result["session_id"] == "chal-a"
    assert result["data"] is None
    assert result["error"] == {
        "code": "VALUE_ERROR",
        "message": "boom",
        "type": "ValueError",
        "details": None,
    }
    assert result["warnings"] == []
    assert result["meta"]["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_catch_errors_normalizes_success_response():
    @catch_errors()
    async def inspect_target():
        return {
            "success": True,
            "session_id": "chal-a",
            "stdout": "complete output",
        }

    result = await inspect_target()

    assert result["success"] is True
    assert result["tool"] == "inspect_target"
    assert result["session_id"] == "chal-a"
    assert result["data"] == {"stdout": "complete output"}
    assert result["error"] is None


def test_build_tool_response_preserves_full_output():
    full_output = "A" * 1_000_000

    result = build_tool_response(
        "run_command", {"success": True, "stdout": full_output}
    )

    assert result["data"]["stdout"] == full_output
    assert len(result["data"]["stdout"]) == 1_000_000
