import pytest

from paomcp.tools.backends.debugger import DebuggerOperations


def response(payload=None, *, success=True, state="stopped", console=False):
    return {
        "success": success,
        "state": state,
        "responses": [
            {
                "type": "console" if console else "result",
                "message": None if console else ("done" if success else "error"),
                "payload": payload,
            }
        ],
    }


class FakeGdb:
    def __init__(self):
        self.calls = []
        self.register_value = "0x10"

    def get_state(self):
        return "stopped"

    @staticmethod
    def quote_mi_argument(value):
        return f'"{value}"'

    def list_breakpoints(self):
        self.calls.append("-break-list")
        return response(
            {
                "BreakpointTable": {
                    "body": [
                        {
                            "number": "1",
                            "type": "breakpoint",
                            "disp": "keep",
                            "enabled": "y",
                            "addr": "0x401000",
                            "func": "main",
                            "line": "7",
                            "times": "2",
                            "original-location": "main",
                        }
                    ]
                }
            }
        )

    def delete_breakpoint(self, number):
        self.calls.append(f"-break-delete {number}")
        return response({})

    def enable_breakpoint(self, number):
        self.calls.append(f"-break-enable {number}")
        return response({})

    def disable_breakpoint(self, number):
        self.calls.append(f"-break-disable {number}")
        return response({})

    def set_watchpoint(self, expression, access):
        self.calls.append(("watch", expression, access))
        return response({"wpt": {"number": "2", "exp": expression}})

    def execute_mi_command(self, command, timeout_sec=7.0):
        self.calls.append(command)
        if command == "-data-list-register-names":
            return response({"register-names": ["rax", "rbx", "rip"]})
        if command.startswith("-data-list-register-values x"):
            numbers = command.split()[2:]
            if not numbers:
                numbers = ["0", "1", "2"]
            return response(
                {
                    "register-values": [
                        {"number": number, "value": self.register_value}
                        for number in numbers
                    ]
                }
            )
        if command.startswith('-data-evaluate-expression "$rax='):
            self.register_value = command.split("=", 1)[1].removesuffix('"')
            return response({"value": "void"})
        if command.startswith("-data-evaluate-expression"):
            return response({"value": "0x401000 <main>"})
        if command == '-data-write-memory-bytes "$rsp" 4142':
            return response({})
        if command == '-data-read-memory-bytes "$rsp" 2':
            return response(
                {"memory": [{"begin": "0x7fffffffe000", "contents": "4142"}]}
            )
        raise AssertionError(f"Unexpected MI command: {command}")

    def get_register_values_mi(self, register_numbers=None, timeout_sec=7.0):
        suffix = ""
        if register_numbers:
            suffix = " " + " ".join(str(number) for number in register_numbers)
        return self.execute_mi_command(
            f"-data-list-register-values x{suffix}", timeout_sec=timeout_sec
        )

    def select_thread(self, thread_id):
        self.calls.append(f"-thread-select {thread_id}")
        return response({"new-thread-id": str(thread_id), "frame": {"func": "worker"}})

    def execute_command(self, command, timeout_sec=7.0):
        self.calls.append(command)
        if command.startswith("search "):
            return response(
                "Searching for byte: b'\\x7fELF'\n"
                "target          0x400000 0x10102464c457f\n"
                "0x500000 0x7f454c46\n",
                console=True,
            )
        if command == 'nextcall "malloc.*"':
            result = response("", console=True)
            result["responses"].append(
                {
                    "type": "notify",
                    "message": "stopped",
                    "payload": {"reason": "breakpoint-hit", "thread-id": "1"},
                }
            )
            return result
        raise AssertionError(f"Unexpected console command: {command}")

    def disassemble_range(self, start, byte_count, timeout_sec=7.0):
        self.calls.append(("disassemble", start, byte_count))
        return response(
            {
                "asm_insns": [
                    {
                        "address": "0x401000",
                        "func-name": "main",
                        "offset": "0",
                        "inst": "push rbp",
                    }
                ]
            }
        )


def test_breakpoint_crud_and_watchpoint_are_typed():
    adapter = DebuggerOperations(FakeGdb(), timeout_seconds=3)

    listed = adapter.list_breakpoints()
    deleted = adapter.delete_breakpoint(1)
    toggled = adapter.toggle_breakpoint(1, False)
    watched = adapter.set_watchpoint("*(long*)$rsp", "write")

    assert listed["parsed"]["breakpoints"][0]["function"] == "main"
    assert listed["parsed"]["breakpoints"][0]["hit_count"] == 2
    assert deleted["parsed"] == {"number": 1, "deleted": True}
    assert toggled["parsed"] == {"number": 1, "enabled": False}
    assert watched["parsed"]["watchpoint"]["number"] == "2"


def test_breakpoint_insert_prefers_mi_result_over_async_notification():
    adapter = DebuggerOperations(FakeGdb(), timeout_seconds=3)
    result = {
        "success": True,
        "state": "stopped",
        "responses": [
            {
                "type": "notify",
                "message": "thread-selected",
                "payload": {"id": "1"},
            },
            {
                "type": "result",
                "message": "done",
                "payload": {
                    "bkpt": {
                        "number": "4",
                        "enabled": "y",
                        "addr": "0x401000",
                        "original-location": "*$pc",
                    }
                },
            },
        ],
    }

    decorated = adapter.decorate_created_breakpoint(result)

    assert decorated["parsed"]["breakpoint"]["number"] == "4"


def test_register_read_write_and_expression_evaluation():
    adapter = DebuggerOperations(FakeGdb(), timeout_seconds=3)

    registers = adapter.read_registers(["rax", "$rip"])
    expression = adapter.evaluate_expression("$rip+4")
    write = adapter.write_register("rax", "0x20")

    assert [item["name"] for item in registers["parsed"]["registers"]] == [
        "rax",
        "rip",
    ]
    assert expression["parsed"]["value"] == "0x401000 <main>"
    assert write["success"] is True
    assert write["parsed"]["before"][0]["value"] == "0x10"
    assert write["parsed"]["after"][0]["value"] == "0x20"


def test_thread_memory_search_write_and_disassembly():
    gdb = FakeGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    selected = adapter.select_thread(3)
    searched = adapter.search_memory(
        "7f454c46", value_type="bytes", hex_encoded=True, limit=10
    )
    written = adapter.write_memory("$rsp", "0x4142", verify=True, allow_unsafe=True)
    disassembly = adapter.disassemble("$pc", 32)

    assert selected["parsed"]["thread_id"] == "3"
    assert searched["parsed"]["count"] == 2
    assert searched["parsed"]["matches"][0]["module"] == "target"
    assert 'search --no-save -t bytes -x -l 10 "7f454c46"' in gdb.calls
    assert written["parsed"]["verified"] is True
    assert disassembly["parsed"]["instructions"][0]["instruction"] == "push rbp"


def test_advanced_step_and_input_validation():
    gdb = FakeGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    result = adapter.step_until_event("next_call", symbol_regex="malloc.*")

    assert result["parsed"]["stop"]["reason"] == "breakpoint-hit"
    with pytest.raises(ValueError):
        adapter.search_memory("x\nquit")
    with pytest.raises(ValueError):
        adapter.write_memory("$rsp", "abc")
    with pytest.raises(ValueError):
        adapter.step_until_event("instruction", mnemonic="mov;quit")
    with pytest.raises(ValueError):
        adapter.select_thread(0)


class FakeLldbController:
    """LldbController-shaped double: execution results carry the typed payload
    at the top level instead of inside a ``result`` response."""

    def __init__(self):
        self.calls = []

    def get_state(self):
        return "stopped"

    def get_inferior_epoch(self):
        return 1

    @staticmethod
    def quote_mi_argument(value):
        return f'"{value}"'

    def execute_mi_command(self, command, timeout_sec=7.0):
        self.calls.append(command)
        if command == "-data-list-register-names":
            return {
                "command": command,
                "success": True,
                "state": "stopped",
                "responses": [
                    {
                        "type": "result",
                        "message": "done",
                        "payload": {"register-names": ["rax", "rip"]},
                    }
                ],
            }
        raise AssertionError(f"Unexpected MI command: {command}")

    def get_register_values_mi(self, register_numbers=None, timeout_sec=7.0):
        self.calls.append("-data-list-register-values")
        return {
            "command": "register read",
            "success": True,
            "state": "stopped",
            "responses": [
                {"type": "console", "payload": "rax = 0x0000000000401000"},
                {"type": "console", "payload": "rip = 0x0000000000401004"},
            ],
            "payload": {
                "register-values": [
                    {"number": 0, "value": "0x0000000000401000"},
                    {"number": 1, "value": "0x0000000000401004"},
                ]
            },
        }

    def list_breakpoints(self):
        self.calls.append("breakpoint list")
        table = [
            {
                "number": "1",
                "type": "breakpoint",
                "disp": "keep",
                "enabled": "y",
                "addr": "0x401000",
                "func": "main",
            }
        ]
        return {
            "command": "breakpoint list",
            "success": True,
            "state": "stopped",
            "responses": [
                {"type": "console", "payload": "1: name = 'main', locations = 1"}
            ],
            "payload": {
                "BreakpointTable": {"body": table},
                "breakpoints": table,
                "count": len(table),
            },
        }

    def list_stack_frames_mi(self, low=0, high=None, timeout_sec=7.0):
        self.calls.append("thread backtrace")
        stack = [
            {"level": "0", "addr": "0x401000", "func": "main", "args": []},
            {"level": "1", "addr": "0x400900", "func": "_start", "args": []},
        ]
        return {
            "command": "thread backtrace",
            "success": True,
            "state": "stopped",
            "responses": [
                {
                    "type": "console",
                    "payload": "frame #0: 0x0000000000401000 a.out`main",
                }
            ],
            "payload": {"stack": stack},
        }


def test_lldb_top_level_payloads_feed_typed_reads():
    adapter = DebuggerOperations(FakeLldbController(), timeout_seconds=3)

    listed = adapter.list_breakpoints()
    registers = adapter.read_registers()
    frames = adapter.stack_frames()

    assert listed["success"] is True
    assert listed["parsed"]["count"] == 1
    assert listed["parsed"]["breakpoints"][0]["function"] == "main"
    assert registers["success"] is True
    assert [item["name"] for item in registers["parsed"]["registers"]] == [
        "rax",
        "rip",
    ]
    assert frames["success"] is True
    assert frames["parsed"]["count"] == 2
    assert frames["parsed"]["frames"][0]["function"] == "main"
