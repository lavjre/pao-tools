"""Manual Stage 4 smoke test for the project's pinned Docker runtime."""

import json

from paomcp.state.session import SessionState
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.pwndbg import PwndbgTools


def require_success(name, result):
    if not result.get("success"):
        raise RuntimeError(f"{name} failed: {result}")
    return result


def first_register_value(result):
    registers = result.get("parsed", {}).get("registers", [])
    if not registers:
        raise RuntimeError(f"Missing register value: {result}")
    return registers[0]["value"]


def main():
    gdb = GdbController()
    state = SessionState(session_id="stage4-smoke")
    tools = PwndbgTools(gdb, state)
    summary = {}
    try:
        require_success("set_file", tools.set_file("/bin/ls"))
        require_success("starti", tools.execute("starti"))

        created = require_success("set_breakpoint", tools.set_breakpoint("*$pc"))
        if not created.get("parsed", {}).get("breakpoint"):
            raise RuntimeError(f"set_breakpoint returned no typed record: {created}")
        breakpoint_number = int(created["parsed"]["breakpoint"]["number"])
        summary["breakpoints"] = require_success(
            "list_breakpoints", tools.list_breakpoints()
        )["parsed"]["count"]
        require_success(
            "toggle_breakpoint", tools.toggle_breakpoint(breakpoint_number, False)
        )
        require_success("delete_breakpoint", tools.delete_breakpoint(breakpoint_number))

        watched = require_success(
            "set_watchpoint", tools.set_watchpoint("*(long*)$rsp", "write")
        )
        watchpoint_number = int(watched["parsed"]["watchpoint"]["number"])
        require_success("delete_watchpoint", tools.delete_breakpoint(watchpoint_number))
        summary["watchpoint"] = "created-and-deleted"

        registers = require_success(
            "read_registers", tools.read_registers(["rip", "rsp", "rax"])
        )
        summary["registers"] = registers["parsed"]["count"]
        summary["pc"] = require_success(
            "evaluate_expression", tools.evaluate_expression_typed("$pc")
        )["parsed"]["value"]
        summary["thread"] = require_success("select_thread", tools.select_thread(1))[
            "parsed"
        ]["thread_id"]

        searched = require_success(
            "search_memory",
            tools.search_memory(
                "7f454c46", value_type="bytes", hex_encoded=True, limit=5
            ),
        )
        if searched["parsed"]["count"] < 1:
            raise RuntimeError(f"search_memory returned no ELF match: {searched}")
        summary["search_matches"] = searched["parsed"]["count"]

        disassembly = require_success(
            "disassemble", tools.disassemble(start="$pc", byte_count=32)
        )
        if disassembly["parsed"]["count"] < 1:
            raise RuntimeError(f"disassemble returned no instructions: {disassembly}")
        summary["instructions"] = disassembly["parsed"]["count"]

        rax = first_register_value(tools.read_registers(["rax"]))
        require_success("write_register", tools.write_register("rax", rax))
        summary["register_write"] = "same-value-verified"

        stack_read = require_success("read_stack", gdb.read_memory_bytes("$rsp", 1))
        stack_byte = DebuggerOperations._memory_contents(stack_read)
        if not stack_byte:
            raise RuntimeError(f"Could not read stack byte: {stack_read}")
        require_success("write_memory", tools.write_memory("$rsp", stack_byte, True))
        summary["memory_write"] = "same-byte-verified"

        advanced = require_success(
            "step_until_event", tools.step_until_event("next_call")
        )
        summary["advanced_step_state"] = advanced["state"]

        print(json.dumps({"success": True, "summary": summary}, sort_keys=True))
    finally:
        gdb.close()


if __name__ == "__main__":
    main()
