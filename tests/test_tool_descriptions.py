import pytest

from paomcp.server import create_mcp


@pytest.mark.asyncio
async def test_python_execution_tool_guidance_in_descriptions():
    mcp = create_mcp()
    tools = await mcp.list_tools(run_middleware=False)
    descriptions = {tool.name: (tool.description or "").lower() for tool in tools}

    assert "info registers" in descriptions["execute"]
    assert "current gdb state" in descriptions["execute"]

    assert "python code dynamically" in descriptions["execute_python_code"]
    assert (
        "only persist files in /workspace when the user explicitly asks"
        in descriptions["execute_python_code"]
    )

    assert "script_path" in descriptions["execute_python_script"]
    assert (
        "timeout: seconds to wait before termination"
        in descriptions["execute_python_script"]
    )

    assert "do not use this to run the target binary" in descriptions["run_command"]
    assert "stdout/stderr/exit code" in descriptions["run_command"]
    assert "under /workspace" in descriptions["run_command"]
    assert "execute_python_code" in descriptions["run_command"]
    assert "paocli" in descriptions["run_command"]
    assert "use this for build and helper commands" in descriptions["run_command"]

    assert "under /workspace" in descriptions["spawn_process"]
    assert "do not use this to run the target binary" in descriptions["spawn_process"]
    assert "set_file + run" in descriptions["spawn_process"]
    assert "under /workspace" in descriptions["set_file"]
    assert "under /workspace" in descriptions["execute_python_script"]
    assert "under /workspace" in descriptions["fetch_repo"]
    assert "branch/tag/commit" in descriptions["fetch_repo"]
    assert "equivalent to --start" in descriptions["run"]
    assert "does not append a newline" in descriptions["sendinput"]
    assert "0xdeadbeef" in descriptions["get_memory"]

    assert "paocli_attach_result" in descriptions["paocli"]
    assert "persistent" in descriptions["paocli"]
    assert "/workspace scripts" in descriptions["paocli"]
    assert "binary_path" in descriptions["paocli"]
    assert "locally with ghidra" in descriptions["decompile_binary"]
    assert "no external" in descriptions["decompile_binary"]
    assert "availability" in descriptions["get_decompiler_status"]


@pytest.mark.asyncio
async def test_all_mcp_tools_publish_unified_response_schema():
    mcp = create_mcp()
    tools = await mcp.list_tools(run_middleware=False)
    required_properties = {
        "api_version",
        "success",
        "tool",
        "session_id",
        "data",
        "error",
        "warnings",
        "meta",
    }

    assert len(tools) == 157
    tool_names = {tool.name for tool in tools}
    assert {"paocli", "paocli_stop", "list_paocli_sessions"} <= tool_names
    assert {"decompile_binary", "get_decompiler_status"} <= tool_names
    assert {"detect_allocator", "get_allocator_capabilities"} <= tool_names
    assert {"inspect_chunk", "validate_chunk"} <= tool_names
    assert {"list_arenas", "inspect_arena"} <= tool_names
    assert {"inspect_fastbins", "inspect_tcache", "inspect_bins"} <= tool_names
    assert {"classify_heap_corruption", "validate_free"} <= tool_names
    assert {
        "analyze_heap_technique",
        "detect_uaf",
        "heap_technique_catalog",
    } <= tool_names
    assert "poison_tcache_link" in tool_names
    assert {"inspect_slub_cache", "decode_slub_freelist"} <= tool_names
    assert {
        "capture_heap_snapshot",
        "list_heap_snapshots",
        "get_heap_snapshot",
        "diff_heap_snapshots",
        "link_snapshot_to_crash",
    } <= tool_names
    assert {
        "find_gadgets",
        "find_gadget_sequence",
        "inspect_gadget",
        "confirm_chain",
    } <= tool_names
    assert {"identify_libc", "find_one_gadgets"} <= tool_names
    assert {
        "build_rop_chain",
        "inspect_sigreturn_frame",
        "explain_chain_constraints",
    } <= tool_names
    assert {"validate_rop_chain", "export_chain"} <= tool_names
    assert {
        "create_target",
        "list_targets",
        "get_target_status",
        "get_target_capabilities",
        "start_target",
        "stop_target",
        "attach_debugger",
    } <= tool_names
    assert {
        "connect_target",
        "disconnect_target",
        "reconnect_target",
    } <= tool_names
    assert {"sync_target_files", "fetch_runtime_libraries"} <= tool_names
    assert {
        "create_research_session",
        "get_research_session",
        "list_research_sessions",
        "add_research_note",
        "link_evidence",
        "add_hypothesis",
        "update_hypothesis_state",
    } <= tool_names
    assert {
        "create_named_snapshot",
        "list_snapshots",
        "compare_snapshots",
    } <= tool_names
    assert {
        "append_timeline_event",
        "get_research_timeline",
    } <= tool_names
    assert {
        "export_research_session",
        "import_research_session",
        "verify_research_bundle",
    } <= tool_names
    assert "generate_final_report" in tool_names
    legacy_prefix = "".join(("pwn", "cli"))
    assert all(legacy_prefix not in name for name in tool_names)
    removed_backend = "".join(("ret", "dec"))
    assert all(removed_backend not in name for name in tool_names)
    assert {
        "analyze_elf",
        "checksec",
        "inspect_got_plt",
        "memory_maps",
        "list_threads",
        "get_process_info",
        "inspect_libc",
        "inspect_address",
        "list_breakpoints",
        "delete_breakpoint",
        "toggle_breakpoint",
        "set_watchpoint",
        "read_registers",
        "write_register",
        "evaluate_expression",
        "select_thread",
        "search_memory",
        "write_memory",
        "disassemble",
        "step_until_event",
        "get_last_stop",
        "run_until_crash",
        "capture_crash_snapshot",
        "analyze_crash",
        "classify_crash",
        "compare_crashes",
        "find_cyclic_offset",
        "capture_core_dump",
        "generate_crash_report",
    } <= tool_names
    for tool in tools:
        assert tool.output_schema is not None, tool.name
        properties = set(tool.output_schema.get("properties", {}))
        assert properties == required_properties, tool.name


@pytest.mark.asyncio
async def test_high_impact_mutations_expose_idempotency_keys():
    tools = await create_mcp().list_tools(run_middleware=False)
    by_name = {tool.name: tool for tool in tools}
    mutations = {
        "create_debug_session",
        "close_debug_session",
        "execute",
        "set_file",
        "attach",
        "run",
        "step_control",
        "gdb_interrupt",
        "finish",
        "jump",
        "return_from_function",
        "until",
        "set_breakpoint",
        "create_breakpoint",
        "delete_breakpoint",
        "toggle_breakpoint",
        "set_watchpoint",
        "delete_watchpoint",
        "toggle_watchpoint",
        "write_register",
        "select_thread",
        "select_frame",
        "write_memory",
        "revert_memory_patch",
        "poison_tcache_link",
        "step_until_event",
        "run_until_crash",
        "capture_crash_snapshot",
        "analyze_crash",
        "classify_crash",
        "compare_crashes",
        "capture_core_dump",
        "generate_crash_report",
        "run_command",
        "spawn_process",
        "kill_process",
        "decompile_binary",
        "set_artifact_pinned",
        "delete_artifact",
        "cleanup_artifacts",
    }

    for name in mutations:
        properties = by_name[name].parameters.get("properties", {})
        assert "idempotency_key" in properties, name
