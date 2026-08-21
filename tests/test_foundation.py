"""Unit tests for the reusable backend foundation primitives."""

import pytest

from paomcp.tools.backends.foundation import (
    LiveInferiorError,
    LiveInferiorGuard,
    build_backend_response,
    mapping_for,
    merge_warnings,
    parse_int,
)


def test_build_backend_response_success_shape():
    response = build_backend_response(
        "demo",
        parsed={"value": 1},
        raw_responses=[],
        state="stopped",
    )
    assert response == {
        "success": True,
        "command": "demo",
        "state": "stopped",
        "parser_version": "1",
        "parsed": {"value": 1},
        "raw_responses": [],
    }


def test_build_backend_response_failure_includes_error_and_type():
    response = build_backend_response(
        "demo",
        parsed={},
        raw_responses=[],
        state="idle",
        success=False,
        error="boom",
        error_type="DemoError",
        warnings=["heads up"],
    )
    assert response["success"] is False
    assert response["error"] == "boom"
    assert response["type"] == "DemoError"
    assert response["warnings"] == ["heads up"]


def test_build_backend_response_extra_cannot_override_reserved_keys():
    with pytest.raises(ValueError, match="reserved envelope key"):
        build_backend_response(
            "demo",
            parsed={},
            raw_responses=[],
            state="idle",
            extra={"state": "hacked"},
        )


def test_build_backend_response_extra_merges_new_keys():
    response = build_backend_response(
        "demo",
        parsed={},
        raw_responses=[],
        state="stopped",
        extra={"usable": True, "complete": False},
    )
    assert response["usable"] is True
    assert response["complete"] is False


@pytest.mark.parametrize(
    "value,expected",
    [
        ("0x40", 0x40),
        ("(void *) 0x7fffdead", 0x7FFFDEAD),
        ("123", 123),
        ("-5", -5),
        (None, None),
        ("not-a-number", None),
        ("", None),
        # Bare hex without a 0x prefix (GDB/crash tooling renders pointers
        # this way).  Ambiguous decimals parse as decimal via base-0 and never
        # reach the hex fallback.
        ("40113a", 0x40113A),
        ("7fffdead", 0x7FFFDEAD),
        ("12345", 12345),
        ("abcdef", 0xABCDEF),
        ("0x", None),
    ],
)
def test_parse_int(value, expected):
    assert parse_int(value) == expected


def test_mapping_for_containment():
    mappings = [
        {"start": "0x400000", "end": "0x401000"},
        {"start": "0x7ff000", "end": "0x800000"},
    ]
    assert mapping_for(0x400500, mappings)["start"] == "0x400000"
    # End is exclusive: 0x401000 is not in the first range, but is below the
    # second range's start, so it matches nothing.
    assert mapping_for(0x401000, mappings) is None
    assert mapping_for(0x7FF500, mappings)["start"] == "0x7ff000"
    assert mapping_for(0x300000, mappings) is None
    assert mapping_for(None, mappings) is None


def test_mapping_for_skips_unparseable_entries():
    mappings = [{"start": "bogus", "end": "0x1"}, {"start": "0x10", "end": "0x20"}]
    assert mapping_for(0x15, mappings)["start"] == "0x10"


def test_live_inferior_guard_is_live():
    live = {"is_live": True}
    assert LiveInferiorGuard.is_live(live, 123) is True
    assert LiveInferiorGuard.is_live(live, None) is False
    assert LiveInferiorGuard.is_live({"is_live": False}, 123) is False
    assert LiveInferiorGuard.is_live(None, 123) is False


def test_live_inferior_guard_require_live_raises():
    with pytest.raises(LiveInferiorError, match="no longer"):
        LiveInferiorGuard.require_live({"is_live": False}, 123)
    record = LiveInferiorGuard.require_live({"is_live": True, "x": 1}, 123)
    assert record["x"] == 1


def test_merge_warnings_appends():
    response = {"warnings": ["a"]}
    merge_warnings(response, ["b", "c"])
    assert response["warnings"] == ["a", "b", "c"]
    merge_warnings(response, None)
    assert response["warnings"] == ["a", "b", "c"]
