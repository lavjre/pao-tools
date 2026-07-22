from copy import deepcopy
import time

import pytest

from paomcp.operations import OperationCancelledError
from paomcp.state.session import SessionState
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.pwndbg import PwndbgTools


def mi_response(payload=None, *, success=True, notify_first=False):
    responses = []
    if notify_first:
        responses.append(
            {
                "type": "notify",
                "message": "thread-selected",
                "payload": {"id": "1"},
            }
        )
    responses.append(
        {
            "type": "result",
            "message": "done" if success else "error",
            "payload": payload,
        }
    )
    return {"success": success, "state": "stopped", "responses": responses}


class CompletionGdb:
    def __init__(self):
        self.calls = []
        self.memory = "aabb"
        self.frame_inspection_succeeds = True
        self.frame_listing_succeeds = True
        self.raw_exec_evidence = False

    def initialize(self):
        return {"status": "initialized"}

    def get_state(self):
        return "stopped"

    def get_inferior_pid(self):
        return None

    def execute_command(self, command, timeout_sec=7.0):
        self.calls.append(("raw", command, timeout_sec))
        responses = []
        if self.raw_exec_evidence:
            responses.append(
                {
                    "type": "console",
                    "payload": "process 100 is executing new program: /workspace/next\n",
                }
            )
        responses.append({"type": "result", "message": "done", "payload": {}})
        return {"success": True, "state": "stopped", "responses": responses}

    @staticmethod
    def quote_mi_argument(value):
        return f'"{value}"'

    def set_breakpoint(self, location, condition=None, temporary=False):
        self.calls.append(("break", location, condition, temporary))
        return mi_response(
            {
                "bkpt": {
                    "number": "3",
                    "type": "breakpoint",
                    "disp": "del" if temporary else "keep",
                    "enabled": "y",
                    "addr": "0x401000",
                    "cond": condition,
                    "original-location": location,
                }
            },
            notify_first=True,
        )

    def list_breakpoints(self):
        self.calls.append("-break-list")
        return mi_response(
            {
                "BreakpointTable": {
                    "body": [
                        {
                            "number": "1",
                            "type": "breakpoint",
                            "disp": "keep",
                            "enabled": "y",
                            "addr": "0x401000",
                        },
                        {
                            "number": "2",
                            "type": "hw watchpoint",
                            "disp": "keep",
                            "enabled": "y",
                            "what": "*$rsp",
                        },
                    ]
                }
            },
            notify_first=True,
        )

    def delete_breakpoint(self, number):
        self.calls.append(f"-break-delete {number}")
        return mi_response({})

    def enable_breakpoint(self, number):
        self.calls.append(f"-break-enable {number}")
        return mi_response({})

    def disable_breakpoint(self, number):
        self.calls.append(f"-break-disable {number}")
        return mi_response({})

    def list_stack_frames_mi(self, low=0, high=None, timeout_sec=7.0):
        self.calls.append(("frames", low, high, timeout_sec))
        return mi_response(
            {
                "stack": [
                    {
                        "level": "0",
                        "addr": "0x401000",
                        "func": "main",
                        "fullname": "/workspace/main.c",
                        "line": "7",
                        "arch": "i386:x86-64",
                        "args": [{"name": "argc", "value": "1"}],
                    },
                    {"level": "1", "addr": "0x400900", "func": "_start"},
                ]
            },
            success=self.frame_listing_succeeds,
            notify_first=True,
        )

    def select_frame(self, level, timeout_sec=7.0):
        self.calls.append((f"-stack-select-frame {level}", timeout_sec))
        return mi_response({})

    def get_selected_frame_mi(self, timeout_sec=7.0):
        self.calls.append(("-stack-info-frame", timeout_sec))
        return mi_response(
            {"frame": {"level": "1", "addr": "0x400900", "func": "_start"}},
            success=self.frame_inspection_succeeds,
            notify_first=True,
        )

    def list_threads_mi(self, timeout_sec=7.0):
        self.calls.append(("-thread-info", timeout_sec))
        return mi_response(
            {
                "threads": [
                    {
                        "id": "1",
                        "target-id": "Thread 100.100",
                        "thread-group": "i1",
                        "state": "stopped",
                        "core": "3",
                        "frame": {
                            "level": "0",
                            "addr": "0x401000",
                            "func": "main",
                        },
                    }
                ],
                "current-thread-id": "1",
            },
            notify_first=True,
        )

    def list_thread_groups_mi(self, recurse=1, timeout_sec=7.0):
        self.calls.append(("inferiors", recurse, timeout_sec))
        return mi_response(
            {
                "groups": [
                    {
                        "id": "i1",
                        "type": "process",
                        "pid": "100",
                        "executable": "/workspace/target",
                        "user-selected": "1",
                        "cores": ["3"],
                    }
                ]
            }
        )

    def execute_mi_command(self, command, timeout_sec=7.0):
        self.calls.append(command)
        if command == "-data-list-register-names":
            return mi_response(
                {"register-names": ["rax", "rip", "eflags", "xmm0", "ymm1"]}
            )
        if command.startswith("-data-list-register-values x"):
            numbers = command.split()[2:] or ["0", "1", "2", "3", "4"]
            return mi_response(
                {
                    "register-values": [
                        {"number": number, "value": f"0x{int(number) + 10:x}"}
                        for number in numbers
                    ]
                }
            )
        if command.startswith("-data-read-memory-bytes"):
            return mi_response(
                {"memory": [{"begin": "0x500000", "contents": self.memory}]}
            )
        if command.startswith("-data-write-memory-bytes"):
            self.memory = command.rsplit(" ", 1)[1]
            return mi_response({})
        raise AssertionError(f"Unexpected MI command: {command}")

    def get_register_values_mi(self, register_numbers=None, timeout_sec=7.0):
        suffix = ""
        if register_numbers:
            suffix = " " + " ".join(str(number) for number in register_numbers)
        return self.execute_mi_command(
            f"-data-list-register-values x{suffix}", timeout_sec=timeout_sec
        )

    def disassemble_range(self, start, byte_count, timeout_sec=7.0):
        return mi_response(
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


def mapping_result(*, writable=True, success=True, end="0x501000"):
    return {
        "success": success,
        "parsed": {
            "count": 1,
            "mappings": [
                {
                    "start": "0x500000",
                    "end": end,
                    "permissions": "rw-p" if writable else "r--p",
                    "writable": writable,
                    "file": "[stack]",
                }
            ],
        },
        "raw_responses": [{"type": "console", "payload": "vmmap evidence"}],
    }


def test_temporary_conditional_breakpoint_prefers_result_after_async_notify():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    result = adapter.create_breakpoint("main", "$rax == 1", temporary=True)

    assert result["success"] is True
    assert result["parsed"]["breakpoint"]["number"] == "3"
    assert result["parsed"]["breakpoint"]["temporary"] is True
    assert result["parsed"]["requested"]["condition"] == "$rax == 1"
    assert gdb.calls[0] == ("break", "main", "$rax == 1", True)
    with pytest.raises(ValueError):
        adapter.create_breakpoint("main; quit")


def test_watchpoint_specific_crud_rejects_a_code_breakpoint_id():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    listed = adapter.list_watchpoints()
    rejected = adapter.delete_watchpoint(1)
    toggled = adapter.toggle_watchpoint(2, False)

    assert listed["parsed"]["count"] == 1
    assert listed["parsed"]["watchpoints"][0]["expression"] == "*$rsp"
    assert rejected["type"] == "WatchpointNotFoundError"
    assert "-break-delete 1" not in gdb.calls
    assert toggled["success"] is True
    assert "-break-disable 2" in gdb.calls


def test_full_stack_frames_frame_selection_and_readback_failure():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    frames = adapter.stack_frames()
    selected = adapter.select_frame(1)
    gdb.frame_inspection_succeeds = False
    failed = adapter.select_frame(1)

    assert gdb.calls[0] == ("frames", 0, None, 3)
    assert frames["parsed"]["count"] == 2
    assert frames["parsed"]["frames"][0]["arguments"][0]["name"] == "argc"
    assert selected["parsed"]["frame"]["function"] == "_start"
    assert failed["success"] is False
    assert failed["type"] == "FrameSelectionError"
    with pytest.raises(ValueError):
        adapter.stack_frames(4, 3)


def test_vector_register_group_threads_and_inferiors_are_typed():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    vectors = adapter.read_register_group("vector")
    threads = adapter.threads()
    inferiors = adapter.inferiors(recurse=2)

    assert [item["name"] for item in vectors["parsed"]["registers"]] == [
        "xmm0",
        "ymm1",
    ]
    assert threads["parsed"]["current_thread_id"] == "1"
    assert threads["parsed"]["threads"][0]["frame"]["function"] == "main"
    assert inferiors["parsed"]["inferiors"][0]["pid"] == "100"
    assert ("-thread-info", 3) in gdb.calls
    assert ("inferiors", 2, 3) in gdb.calls
    with pytest.raises(ValueError):
        adapter.read_register_group("secret")


def test_memory_write_requires_writable_mapping_unless_explicitly_overridden():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    rejected = adapter.write_memory(
        "0x500000", "ccdd", mapping_result=mapping_result(writable=False)
    )
    crossing = adapter.write_memory(
        "0x500000",
        "ccdd",
        mapping_result=mapping_result(end="0x500001"),
    )
    written = adapter.write_memory(
        "0x500000",
        "ccdd",
        mapping_result=mapping_result(writable=False),
        allow_unsafe=True,
    )

    assert rejected["type"] == "MemoryProtectionError"
    assert crossing["type"] == "MemoryProtectionError"
    assert "crosses the end" in crossing["error"]
    assert not any(
        isinstance(call, str) and call.startswith("-data-write-memory-bytes")
        for call in gdb.calls[:1]
    )
    assert written["success"] is True
    assert written["parsed"]["original_hex"] == "aabb"
    assert written["parsed"]["mapping_validation"]["override_used"] is True
    assert written["raw_responses"]["mapping"]


def test_memory_write_enforces_configured_payload_cap_before_gdb_mutation():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3, max_memory_write_bytes=1)

    with pytest.raises(ValueError, match="1-byte limit"):
        adapter.write_memory(
            "0x500000", "aabb", mapping_result=mapping_result(writable=True)
        )

    assert gdb.memory == "aabb"
    assert not any(
        isinstance(call, str) and call.startswith("-data-write-memory-bytes")
        for call in gdb.calls
    )


def test_snapshot_is_partial_on_component_failure_and_diff_is_typed():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)
    before = adapter.capture_snapshot(
        register_names=["rax", "rip"], mapping_result=mapping_result()
    )
    after = adapter.capture_snapshot(
        register_names=["rax", "rip"], mapping_result=mapping_result()
    )
    after["parsed"]["snapshot"]["registers"][0]["value"] = "0x99"
    after["parsed"]["snapshot"]["inferiors"][0]["pid"] = "101"
    gdb.frame_listing_succeeds = False
    partial = adapter.capture_snapshot(mapping_result=mapping_result())

    before_envelope = {
        "api_version": "1.0",
        "success": True,
        "tool": "capture_debugger_snapshot",
        "data": {key: value for key, value in before.items() if key != "success"},
    }
    after_envelope = {
        "api_version": "1.0",
        "success": True,
        "tool": "capture_debugger_snapshot",
        "data": {key: value for key, value in after.items() if key != "success"},
    }
    diff = adapter.diff_snapshots(before_envelope, after_envelope)

    assert before["complete"] is True
    assert before["usable"] is True
    assert before["core_complete"] is True
    assert before["raw_responses"]["frames"]
    assert partial["success"] is True
    assert partial["usable"] is True
    assert partial["core_complete"] is False
    assert partial["complete"] is False
    assert partial["parsed"]["component_status"]["frames"] is False
    assert diff["parsed"]["changed"] is True
    assert diff["parsed"]["sections"]["registers"]["change_count"] == 1
    assert diff["parsed"]["sections"]["inferiors"]["change_count"] == 1
    assert diff["raw_responses"]["before"]
    with pytest.raises(ValueError):
        adapter.diff_snapshots({"not": "a snapshot"}, {})

    wrong_version = dict(before["parsed"]["snapshot"])
    wrong_version["snapshot_version"] = "0"
    with pytest.raises(ValueError, match="snapshot_version"):
        adapter.diff_snapshots(wrong_version, after)

    missing_frames = dict(before["parsed"]["snapshot"])
    missing_frames.pop("frames")
    with pytest.raises(ValueError, match="missing canonical fields: frames"):
        adapter.diff_snapshots(missing_frames, after)


def test_snapshot_diff_counts_scalar_changes_and_rejects_identity_collisions():
    adapter = DebuggerOperations(CompletionGdb(), timeout_seconds=3)
    captured = adapter.capture_snapshot(mapping_result=mapping_result())
    before = captured["parsed"]["snapshot"]
    after = deepcopy(before)
    after["state"] = "running"
    after["current_thread_id"] = "2"

    scalar_diff = adapter.diff_snapshots(before, after)

    assert scalar_diff["parsed"]["changed"] is True
    assert scalar_diff["parsed"]["change_count"] == 2
    assert scalar_diff["parsed"]["record_change_count"] == 0
    assert scalar_diff["parsed"]["scalar_change_count"] == 2
    assert scalar_diff["parsed"]["state"]["changed"] is True
    assert scalar_diff["parsed"]["current_thread_id"]["changed"] is True

    duplicate = deepcopy(before)
    duplicate["registers"].append(deepcopy(duplicate["registers"][0]))
    with pytest.raises(ValueError, match="duplicate identity"):
        adapter.diff_snapshots(duplicate, after)

    missing_identity = deepcopy(before)
    missing_identity["instructions"][0].pop("address")
    with pytest.raises(ValueError, match="missing identity field 'address'"):
        adapter.diff_snapshots(missing_identity, after)


def test_snapshot_deadline_rejects_mapping_only_success_and_marks_timeouts():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)

    result = adapter.capture_snapshot(
        mapping_result=mapping_result(), deadline=time.monotonic() - 1
    )

    assert result["success"] is False
    assert result["usable"] is False
    assert result["core_complete"] is False
    assert result["complete"] is False
    assert result["parsed"]["component_status"]["mappings"] is True
    assert result["parsed"]["component_status"]["registers"] is False
    assert result["type"] == "TimeoutError"
    assert result["timed_out"] is True
    assert not gdb.calls


def test_snapshot_checks_cooperative_cancellation_between_components():
    gdb = CompletionGdb()
    adapter = DebuggerOperations(gdb, timeout_seconds=3)
    checks = 0

    def cancel_after_registers():
        nonlocal checks
        checks += 1
        if checks == 2:
            raise OperationCancelledError("cancel test")

    with pytest.raises(OperationCancelledError, match="cancel test"):
        adapter.capture_snapshot(cancellation_check=cancel_after_registers)

    assert checks == 2
    assert "-data-list-register-names" in gdb.calls
    assert not any(
        isinstance(call, tuple) and call[0] == "frames" for call in gdb.calls
    )


def test_gdb_controller_returns_structured_timeout_instead_of_raising():
    class TimeoutWriter:
        def write(self, command, timeout_sec):
            raise TimeoutError(f"deadline for {command}")

    controller = GdbController.__new__(GdbController)
    controller.controller = TimeoutWriter()
    controller._state = "stopped"

    result = controller.execute_mi_command("-thread-info", timeout_sec=2.5)

    assert result["success"] is False
    assert result["type"] == "TimeoutError"
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 2.5


class FakeInspection:
    timeout_seconds = 3

    def __init__(self, writable=True):
        self.writable = writable

    def address_info(self, address):
        return {
            "success": True,
            "parsed": {"address": address},
            "raw_responses": [],
        }

    def memory_maps(self, address_or_module=None):
        return mapping_result(writable=self.writable)


def test_pwndbg_patch_history_is_session_local_and_revertible():
    gdb = CompletionGdb()
    tools = PwndbgTools(gdb, SessionState())
    tools.inspection = FakeInspection()

    written = tools.write_memory("0x500000", "ccdd")
    patch_id = written["parsed"]["patch_id"]
    patches = tools.list_memory_patches()
    reverted = tools.revert_memory_patch(patch_id)

    assert patch_id == "patch-000001"
    assert patches["parsed"]["patches"][0]["original_hex"] == "aabb"
    assert reverted["success"] is True
    assert reverted["parsed"]["reverted"] is True
    assert gdb.memory == "aabb"
    assert tools.list_memory_patches(active_only=True)["parsed"]["count"] == 0
    with pytest.raises(ValueError):
        tools.revert_memory_patch(patch_id)

    other_session = PwndbgTools(CompletionGdb(), SessionState())
    assert other_session.list_memory_patches()["parsed"]["count"] == 0


def test_patch_quota_never_evicts_active_patches_and_prunes_inactive_history():
    gdb = CompletionGdb()
    tools = PwndbgTools(
        gdb,
        SessionState(),
        max_memory_patches=1,
        max_memory_patch_bytes=4,
    )
    tools.inspection = FakeInspection()

    first = tools.write_memory("0x500000", "ccdd")
    blocked = tools.write_memory("0x500000", "eeff")

    assert first["success"] is True
    assert blocked["success"] is False
    assert blocked["type"] == "MemoryPatchQuotaError"
    assert blocked["parsed"]["write_applied"] is False
    assert gdb.memory == "ccdd"
    assert tools.list_memory_patches(active_only=True)["parsed"]["count"] == 1

    tools.revert_memory_patch(first["parsed"]["patch_id"])
    tools.inspection = FakeInspection(writable=False)
    rejected = tools.write_memory("0x500000", "eeff")

    assert rejected["success"] is False
    assert rejected["parsed"]["retention_pruned"] == []
    assert tools.list_memory_patches()["parsed"]["count"] == 1

    tools.inspection = FakeInspection(writable=True)
    second = tools.write_memory("0x500000", "eeff")
    listed = tools.list_memory_patches()

    assert second["success"] is True
    assert second["parsed"]["retention_pruned"] == ["patch-000001"]
    assert listed["parsed"]["count"] == 1
    assert listed["parsed"]["patches"][0]["patch_id"] == "patch-000002"
    assert listed["parsed"]["quota"]["retained_patch_bytes"] == 4


def test_patch_byte_quota_rejects_write_before_mutation():
    gdb = CompletionGdb()
    tools = PwndbgTools(
        gdb,
        SessionState(),
        max_memory_patches=4,
        max_memory_patch_bytes=3,
    )
    tools.inspection = FakeInspection()

    result = tools.write_memory("0x500000", "ccdd")

    assert result["success"] is False
    assert result["type"] == "MemoryPatchQuotaError"
    assert result["parsed"]["patch_quota"]["required_patch_bytes"] == 4
    assert gdb.memory == "aabb"


def test_raw_inspection_preserves_patches_but_lifecycle_and_exec_invalidate_them():
    gdb = CompletionGdb()
    tools = PwndbgTools(gdb, SessionState())
    tools.inspection = FakeInspection()
    first = tools.write_memory("0x500000", "ccdd")

    tools.execute("info registers")
    assert tools.list_memory_patches(active_only=True)["parsed"]["count"] == 1

    tools.execute("run")
    first_patch = tools.list_memory_patches()["parsed"]["patches"][0]
    assert first_patch["active"] is False
    assert first_patch["invalidated"] is True
    assert "raw debugger lifecycle command" in first_patch["invalidation_reason"]

    second_tools = PwndbgTools(CompletionGdb(), SessionState())
    second_tools.inspection = FakeInspection()
    second_tools.write_memory("0x500000", "ccdd")
    second_tools.gdb.raw_exec_evidence = True
    second_tools.execute("continue")
    second_patch = second_tools.list_memory_patches()["parsed"]["patches"][0]
    assert second_patch["active"] is False
    assert second_patch["invalidation_reason"] == "inferior executed a new program"
