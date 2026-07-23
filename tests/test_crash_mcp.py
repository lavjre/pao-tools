import pytest

from paomcp.server import create_mcp


@pytest.mark.asyncio
async def test_s5_public_tools_and_input_constraints_are_published():
    tools = await create_mcp().list_tools(run_middleware=False)
    by_name = {tool.name: tool for tool in tools}

    assert {
        "run_until_crash",
        "analyze_crash",
        "find_cyclic_offset",
        "capture_core_dump",
        "classify_crash",
        "compare_crashes",
        "generate_crash_report",
    } <= by_name.keys()

    run_properties = by_name["run_until_crash"].parameters["properties"]
    assert run_properties["mode"]["enum"] == ["auto", "run", "continue", "wait"]

    cyclic_properties = by_name["find_cyclic_offset"].parameters["properties"]
    assert cyclic_properties["n"]["minimum"] == 2
    assert cyclic_properties["n"]["maximum"] == 8
    assert cyclic_properties["max_length"]["minimum"] == 1
    assert cyclic_properties["max_stack_scan_bytes"]["minimum"] == 0
    assert cyclic_properties["max_matches"]["maximum"] == 256
    assert cyclic_properties["endian"]["enum"] == ["little", "big"]
    assert cyclic_properties["value"]["anyOf"][0]["maxLength"] == 132
    assert cyclic_properties["alphabet"]["minLength"] == 2
    assert cyclic_properties["alphabet"]["maxLength"] == 256

    report_properties = by_name["generate_crash_report"].parameters["properties"]
    assert report_properties["report_format"]["enum"] == ["json", "markdown"]
    assert report_properties["stack_bytes"]["anyOf"][0]["minimum"] == 1
    assert report_properties["include_mappings"]["default"] is None


@pytest.mark.asyncio
async def test_s5_identity_inputs_are_required_for_classification_and_comparison():
    tools = await create_mcp().list_tools(run_middleware=False)
    by_name = {tool.name: tool for tool in tools}

    classify_required = set(by_name["classify_crash"].parameters["required"])
    compare_required = set(by_name["compare_crashes"].parameters["required"])

    assert {"expected_crash_id", "session_id"} <= classify_required
    assert {"left_crash_id", "right_crash_id", "session_id"} <= compare_required
