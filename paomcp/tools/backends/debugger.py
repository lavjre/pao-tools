"""Typed GDB/MI and pwndbg operations for debugger and memory control."""

from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from paomcp.tools.backends.gdb import GdbController

ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
HEX_PATTERN = r"0x[0-9a-fA-F]+"
SAFE_EXPRESSION_PATTERN = re.compile(
    r"^[A-Za-z0-9_.$+\-*/%&|^~<>()\[\],: @!='?]{1,512}$"
)
SAFE_REGISTER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
SAFE_MAPPING_PATTERN = re.compile(r"^[A-Za-z0-9_.$+*/:\-\[\]()]{1,160}$")
SAFE_MNEMONIC_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,31}$")
SEARCH_TYPES = {
    "byte",
    "short",
    "word",
    "dword",
    "qword",
    "pointer",
    "string",
    "bytes",
    "asm",
}
STEP_EVENTS = {
    "next_call",
    "next_jump",
    "next_return",
    "next_syscall",
    "step_syscall",
    "instruction",
    "address",
}
PARSER_VERSION = "1"
SNAPSHOT_VERSION = "1"
DEFAULT_MAX_MEMORY_WRITE_BYTES = 4 * 1024 * 1024
MAX_CONFIGURED_MEMORY_WRITE_BYTES = 256 * 1024 * 1024
SNAPSHOT_RECORD_SECTIONS = (
    "registers",
    "frames",
    "instructions",
    "threads",
    "inferiors",
    "mappings",
)
SNAPSHOT_REQUIRED_FIELDS = frozenset(
    {
        "snapshot_version",
        "captured_at",
        "state",
        "registers",
        "frames",
        "instructions",
        "threads",
        "current_thread_id",
        "inferiors",
        "mappings",
    }
)
SNAPSHOT_CORE_COMPONENTS = (
    "registers",
    "frames",
    "disassembly",
    "threads",
    "inferiors",
)
SNAPSHOT_IDENTITY_FIELDS = {
    "registers": ("name",),
    "frames": ("level",),
    "instructions": ("address",),
    "threads": ("id",),
    "inferiors": ("id",),
    "mappings": ("start", "end", "file"),
}


class DebuggerOperations:
    """Provide validated typed debugger operations over one GDB controller."""

    def __init__(
        self,
        gdb: GdbController,
        timeout_seconds: Optional[float] = None,
        max_memory_write_bytes: Optional[int] = None,
    ):
        self.gdb = gdb
        configured_timeout = timeout_seconds
        if configured_timeout is None:
            configured_timeout = float(
                os.environ.get("PAO_PWNDBG_INSPECTION_TIMEOUT_SECONDS", "60")
            )
        if configured_timeout <= 0:
            raise ValueError("Debugger operation timeout must be greater than zero")
        self.timeout_seconds = configured_timeout
        configured_write_limit = max_memory_write_bytes
        if configured_write_limit is None:
            configured_write_limit = int(
                os.environ.get(
                    "PAO_MAX_MEMORY_WRITE_BYTES",
                    str(DEFAULT_MAX_MEMORY_WRITE_BYTES),
                )
            )
        if not 1 <= configured_write_limit <= MAX_CONFIGURED_MEMORY_WRITE_BYTES:
            raise ValueError(
                "Memory write limit must be between 1 and "
                f"{MAX_CONFIGURED_MEMORY_WRITE_BYTES} bytes"
            )
        self.max_memory_write_bytes = configured_write_limit
        self._register_names: Optional[List[str]] = None
        self._register_names_epoch: Optional[int] = None

    def clear_register_cache(self) -> None:
        """Clear architecture-specific register metadata after changing files."""
        self._register_names = None
        self._register_names_epoch = None

    def snapshot_deadline(self) -> float:
        """Return the monotonic deadline for one compound snapshot operation."""
        return time.monotonic() + self.timeout_seconds

    def _remaining_timeout(self, deadline: Optional[float] = None) -> float:
        """Return a bounded command timeout from an optional compound deadline."""
        if deadline is None:
            return self.timeout_seconds
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Debugger snapshot exceeded its {self.timeout_seconds}s deadline"
            )
        return min(self.timeout_seconds, remaining)

    @staticmethod
    def _payloads(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        payloads = []
        for response in result.get("responses", []):
            payload = response.get("payload")
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    @classmethod
    def _first_payload(cls, result: Dict[str, Any]) -> Dict[str, Any]:
        for response in result.get("responses", []):
            if response.get("type") != "result":
                continue
            payload = response.get("payload")
            if isinstance(payload, dict):
                return payload
        payloads = cls._payloads(result)
        return payloads[0] if payloads else {}

    @staticmethod
    def _raw_output(result: Dict[str, Any]) -> str:
        output = []
        for response in result.get("responses", []):
            payload = response.get("payload")
            if isinstance(payload, str):
                output.append(payload)
        return ANSI_PATTERN.sub("", "".join(output)).replace("\r\n", "\n")

    @classmethod
    def _error_message(cls, result: Dict[str, Any], command: str) -> str:
        for payload in cls._payloads(result):
            message = payload.get("msg")
            if message:
                return str(message)
        if result.get("error"):
            return str(result["error"])
        raw_output = cls._raw_output(result)
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        return lines[-1] if lines else f"Debugger command failed: {command}"

    def _typed_result(
        self,
        command: str,
        result: Dict[str, Any],
        parsed: Dict[str, Any],
        *,
        error_type: str = "DebuggerCommandError",
    ) -> Dict[str, Any]:
        success = bool(result.get("success"))
        response = {
            "success": success,
            "command": command,
            "state": result.get("state", self.gdb.get_state()),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": parsed,
            "raw_output": self._raw_output(result),
            "raw_responses": result.get("responses", []),
        }
        if not success:
            response["error"] = self._error_message(result, command)
            response["type"] = result.get("type", error_type)
            if result.get("timed_out"):
                response["timed_out"] = True
                response["effective_timeout_seconds"] = result.get(
                    "timeout_seconds", self.timeout_seconds
                )
        return response

    @staticmethod
    def _validate_expression(expression: str, *, kind: str) -> str:
        if not SAFE_EXPRESSION_PATTERN.fullmatch(expression):
            raise ValueError(f"Unsafe {kind}: '{expression}'")
        return expression

    @staticmethod
    def _validate_register(name: str) -> str:
        normalized = name.lstrip("$")
        if not SAFE_REGISTER_PATTERN.fullmatch(normalized):
            raise ValueError(f"Unsafe register name: '{name}'")
        return normalized

    @staticmethod
    def _quote_cli_argument(value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("CLI argument contains a forbidden control character")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @staticmethod
    def _positive_number(value: int, *, name: str, maximum: int) -> int:
        if not 1 <= value <= maximum:
            raise ValueError(f"{name} must be between 1 and {maximum}")
        return value

    @staticmethod
    def _breakpoint_record(item: Dict[str, Any]) -> Dict[str, Any]:
        hit_count = item.get("times")
        line = item.get("line")
        breakpoint_type = str(item.get("type", ""))
        is_watchpoint = "watchpoint" in breakpoint_type.lower() or any(
            key in item for key in ("what", "exp")
        )
        return {
            "number": str(item.get("number", "")),
            "type": item.get("type"),
            "disposition": item.get("disp"),
            "enabled": item.get("enabled") == "y",
            "address": item.get("addr"),
            "function": item.get("func"),
            "file": item.get("fullname") or item.get("file"),
            "line": int(str(line)) if str(line or "").isdigit() else None,
            "condition": item.get("cond"),
            "hit_count": (
                int(str(hit_count)) if str(hit_count or "").isdigit() else None
            ),
            "original_location": item.get("original-location"),
            "thread_groups": item.get("thread-groups", []),
            "expression": item.get("what") or item.get("exp"),
            "is_watchpoint": is_watchpoint,
            "temporary": item.get("disp") == "del",
        }

    def create_breakpoint(
        self,
        location: str,
        condition: Optional[str] = None,
        temporary: bool = False,
    ) -> Dict[str, Any]:
        """Create a validated persistent or temporary code breakpoint."""
        location = self._validate_expression(location, kind="breakpoint location")
        if condition is not None:
            condition = self._validate_expression(
                condition, kind="breakpoint condition"
            )
        result = self.gdb.set_breakpoint(
            location, condition=condition, temporary=temporary
        )
        response = self.decorate_created_breakpoint(result)
        response["command"] = result.get("command", "-break-insert")
        response["parsed"]["requested"] = {
            "location": location,
            "condition": condition,
            "temporary": temporary,
        }
        return response

    def decorate_created_breakpoint(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add a typed breakpoint record to an existing MI insert result."""
        payload = self._first_payload(result)
        item = payload.get("bkpt")
        result["parser_version"] = PARSER_VERSION
        result["timeout_seconds"] = self.timeout_seconds
        result["raw_output"] = self._raw_output(result)
        result["raw_responses"] = result.get("responses", [])
        result["parsed"] = {
            "breakpoint": (
                self._breakpoint_record(item) if isinstance(item, dict) else None
            )
        }
        if not result.get("success") and "error" not in result:
            result["error"] = self._error_message(
                result, result.get("command", "break")
            )
            result["type"] = "BreakpointError"
        return result

    def list_breakpoints(self) -> Dict[str, Any]:
        result = self.gdb.list_breakpoints()
        payload = self._first_payload(result)
        table = payload.get("BreakpointTable", {})
        body = table.get("body", []) if isinstance(table, dict) else []
        records = [
            self._breakpoint_record(item) for item in body if isinstance(item, dict)
        ]
        return self._typed_result(
            "-break-list", result, {"count": len(records), "breakpoints": records}
        )

    def delete_breakpoint(self, number: int) -> Dict[str, Any]:
        self._positive_number(number, name="breakpoint number", maximum=1_000_000)
        result = self.gdb.delete_breakpoint(number)
        return self._typed_result(
            f"-break-delete {number}",
            result,
            {"number": number, "deleted": bool(result.get("success"))},
            error_type="BreakpointError",
        )

    def toggle_breakpoint(self, number: int, enable: bool) -> Dict[str, Any]:
        self._positive_number(number, name="breakpoint number", maximum=1_000_000)
        result = (
            self.gdb.enable_breakpoint(number)
            if enable
            else self.gdb.disable_breakpoint(number)
        )
        action = "enable" if enable else "disable"
        return self._typed_result(
            f"-break-{action} {number}",
            result,
            {"number": number, "enabled": enable},
            error_type="BreakpointError",
        )

    def set_watchpoint(self, expression: str, access: str = "write") -> Dict[str, Any]:
        expression = self._validate_expression(expression, kind="watch expression")
        if access not in {"write", "read", "access"}:
            raise ValueError("Watchpoint access must be write, read, or access")
        result = self.gdb.set_watchpoint(expression, access)
        payload = self._first_payload(result)
        watchpoint = None
        watchpoint_type = None
        for key in ("wpt", "hw-rwpt", "hw-awpt", "bkpt"):
            if isinstance(payload.get(key), dict):
                watchpoint = payload[key]
                watchpoint_type = key
                break
        parsed = {
            "watchpoint": (
                {
                    "number": str(watchpoint.get("number", "")),
                    "expression": watchpoint.get("exp", expression),
                    "access": access,
                    "type": watchpoint_type,
                }
                if watchpoint
                else None
            )
        }
        option = {"write": "", "read": "-r ", "access": "-a "}[access]
        return self._typed_result(
            f"-break-watch {option}{expression}",
            result,
            parsed,
            error_type="WatchpointError",
        )

    @staticmethod
    def _watchpoint_records(
        breakpoints: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [item for item in breakpoints if item.get("is_watchpoint")]

    def list_watchpoints(self) -> Dict[str, Any]:
        """List only hardware/software watchpoints from the breakpoint table."""
        listed = self.list_breakpoints()
        breakpoints = listed.get("parsed", {}).get("breakpoints", [])
        watchpoints = self._watchpoint_records(breakpoints)
        listed["command"] = "list_watchpoints"
        listed["parsed"] = {
            "count": len(watchpoints),
            "watchpoints": watchpoints,
        }
        if not listed.get("success"):
            listed["type"] = "WatchpointError"
        return listed

    def _watchpoint_action(
        self, number: int, *, delete: bool = False, enable: Optional[bool] = None
    ) -> Dict[str, Any]:
        self._positive_number(number, name="watchpoint number", maximum=1_000_000)
        listed = self.list_watchpoints()
        if not listed.get("success"):
            return listed
        match = next(
            (
                item
                for item in listed["parsed"]["watchpoints"]
                if item.get("number") == str(number)
            ),
            None,
        )
        if match is None:
            return {
                "success": False,
                "command": "delete_watchpoint" if delete else "toggle_watchpoint",
                "state": self.gdb.get_state(),
                "timeout_seconds": self.timeout_seconds,
                "parser_version": PARSER_VERSION,
                "parsed": {"number": number, "watchpoint": None},
                "raw_output": listed.get("raw_output", ""),
                "raw_responses": {"list": listed.get("raw_responses", [])},
                "error": f"Breakpoint #{number} is not a watchpoint",
                "type": "WatchpointNotFoundError",
            }
        if delete:
            action = self.gdb.delete_breakpoint(number)
            command = f"-break-delete {number}"
            parsed = {"number": number, "deleted": bool(action.get("success"))}
        else:
            assert enable is not None
            action = (
                self.gdb.enable_breakpoint(number)
                if enable
                else self.gdb.disable_breakpoint(number)
            )
            verb = "enable" if enable else "disable"
            command = f"-break-{verb} {number}"
            parsed = {"number": number, "enabled": enable}
        response = self._typed_result(
            command, action, parsed, error_type="WatchpointError"
        )
        response["raw_responses"] = {
            "list": listed.get("raw_responses", []),
            "action": action.get("responses", []),
        }
        response["watchpoint"] = match
        return response

    def delete_watchpoint(self, number: int) -> Dict[str, Any]:
        """Delete a watchpoint after verifying its typed breakpoint kind."""
        return self._watchpoint_action(number, delete=True)

    def toggle_watchpoint(self, number: int, enable: bool) -> Dict[str, Any]:
        """Enable/disable a watchpoint after verifying its typed breakpoint kind."""
        return self._watchpoint_action(number, enable=enable)

    def _load_register_names(
        self, refresh: bool = False, deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        # Register layout follows the inferior: a new exec/attach/run bumps
        # the epoch, so a cached table from a previous epoch is stale.
        try:
            epoch = self.gdb.get_inferior_epoch()
        except Exception:
            epoch = None
        if (
            self._register_names is not None
            and not refresh
            and (epoch is None or epoch == self._register_names_epoch)
        ):
            return {"success": True, "names": self._register_names, "responses": []}
        result = self.gdb.execute_mi_command(
            "-data-list-register-names",
            timeout_sec=self._remaining_timeout(deadline),
        )
        payload = self._first_payload(result)
        names = payload.get("register-names", [])
        if result.get("success") and isinstance(names, list):
            self._register_names = [str(name) for name in names]
            self._register_names_epoch = epoch
        return {
            "success": bool(result.get("success")),
            "names": self._register_names or [],
            "responses": result.get("responses", []),
            "result": result,
        }

    def read_registers(
        self,
        names: Optional[Sequence[str]] = None,
        refresh_names: bool = False,
        *,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        name_result = self._load_register_names(
            refresh=refresh_names, deadline=deadline
        )
        if not name_result["success"]:
            result = name_result["result"]
            return self._typed_result(
                "read_registers", result, {"count": 0, "registers": []}
            )
        register_names = name_result["names"]
        requested_names = None
        register_numbers = None
        if names:
            requested_names = [self._validate_register(name) for name in names]
            number_by_name = {
                name: number for number, name in enumerate(register_names) if name
            }
            missing = [name for name in requested_names if name not in number_by_name]
            if missing:
                raise ValueError("Unknown registers: " + ", ".join(missing))
            register_numbers = [number_by_name[name] for name in requested_names]

        value_result = self.gdb.get_register_values_mi(
            register_numbers, timeout_sec=self._remaining_timeout(deadline)
        )
        payload = self._first_payload(value_result)
        raw_values = payload.get("register-values", [])
        registers = []
        if isinstance(raw_values, list):
            for item in raw_values:
                if (
                    not isinstance(item, dict)
                    or not str(item.get("number", "")).isdigit()
                ):
                    continue
                number = int(item["number"])
                name = register_names[number] if number < len(register_names) else ""
                if name:
                    registers.append(
                        {"number": number, "name": name, "value": item.get("value")}
                    )
        response = self._typed_result(
            "read_registers",
            value_result,
            {
                "count": len(registers),
                "requested": requested_names,
                "registers": registers,
            },
        )
        response["raw_responses"] = {
            "names": name_result.get("responses", []),
            "values": value_result.get("responses", []),
        }
        return response

    def evaluate_expression(self, expression: str) -> Dict[str, Any]:
        expression = self._validate_expression(expression, kind="GDB expression")
        # Safe by construction: SAFE_EXPRESSION_PATTERN rejects `"` and `\`,
        # so the validated expression cannot break out of the MI quoting.
        command = f'-data-evaluate-expression "{expression}"'
        result = self.gdb.execute_mi_command(command, timeout_sec=self.timeout_seconds)
        payload = self._first_payload(result)
        return self._typed_result(
            command, result, {"expression": expression, "value": payload.get("value")}
        )

    def write_register(self, name: str, value: str) -> Dict[str, Any]:
        name = self._validate_register(name)
        value = self._validate_expression(value, kind="register value")
        before = self.read_registers([name])
        if not before.get("success"):
            return before
        assignment = self.evaluate_expression(f"${name}={value}")
        after = self.read_registers([name]) if assignment.get("success") else None
        success = bool(assignment.get("success") and after and after.get("success"))
        response = {
            "success": success,
            "command": f"write_register {name}",
            "state": self.gdb.get_state(),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": {
                "name": name,
                "requested_value": value,
                "before": before.get("parsed", {}).get("registers", []),
                "after": (
                    after.get("parsed", {}).get("registers", []) if after else []
                ),
            },
            "raw_responses": {
                "before": before.get("raw_responses", []),
                "assignment": assignment.get("raw_responses", []),
                "after": after.get("raw_responses", []) if after else [],
            },
        }
        if not success:
            response["error"] = assignment.get("error", "Register write failed")
            response["type"] = "RegisterWriteError"
        return response

    def select_thread(self, thread_id: int) -> Dict[str, Any]:
        self._positive_number(thread_id, name="thread_id", maximum=1_000_000)
        result = self.gdb.select_thread(thread_id)
        payload = self._first_payload(result)
        return self._typed_result(
            f"-thread-select {thread_id}",
            result,
            {
                "thread_id": payload.get("new-thread-id", str(thread_id)),
                "frame": payload.get("frame"),
            },
            error_type="ThreadSelectionError",
        )

    @staticmethod
    def _frame_record(frame: Dict[str, Any]) -> Dict[str, Any]:
        level = frame.get("level")
        line = frame.get("line")
        args = frame.get("args", [])
        return {
            "level": int(str(level)) if str(level or "").isdigit() else level,
            "address": frame.get("addr"),
            "function": frame.get("func"),
            "file": frame.get("fullname") or frame.get("file"),
            "line": int(str(line)) if str(line or "").isdigit() else None,
            "architecture": frame.get("arch"),
            "arguments": args if isinstance(args, list) else [],
            "from": frame.get("from"),
        }

    def stack_frames(
        self,
        low: int = 0,
        high: Optional[int] = None,
        *,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return every requested frame without applying an implicit limit."""
        if low < 0 or low > 1_000_000:
            raise ValueError("low must be between 0 and 1000000")
        if high is None and low != 0:
            raise ValueError("low must be 0 when high is omitted")
        if high is not None and (high < low or high > 1_000_000):
            raise ValueError("high must be greater than or equal to low and <= 1000000")
        result = self.gdb.list_stack_frames_mi(
            low=low,
            high=high,
            timeout_sec=self._remaining_timeout(deadline),
        )
        payload = self._first_payload(result)
        raw_frames = payload.get("stack", [])
        frames = [
            self._frame_record(item) for item in raw_frames if isinstance(item, dict)
        ]
        command = "-stack-list-frames"
        if high is not None:
            command += f" {low} {high}"
        return self._typed_result(
            command,
            result,
            {"count": len(frames), "low": low, "high": high, "frames": frames},
            error_type="StackFrameError",
        )

    def select_frame(
        self, level: int, *, deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        """Select one frame and read it back so success is evidence-backed."""
        if level < 0 or level > 1_000_000:
            raise ValueError("frame level must be between 0 and 1000000")
        operation_deadline = deadline or self.snapshot_deadline()
        selected = self.gdb.select_frame(
            level, timeout_sec=self._remaining_timeout(operation_deadline)
        )
        inspected = None
        frame = None
        if selected.get("success"):
            inspected = self.gdb.get_selected_frame_mi(
                timeout_sec=self._remaining_timeout(operation_deadline)
            )
            payload = self._first_payload(inspected)
            item = payload.get("frame")
            if isinstance(item, dict):
                frame = self._frame_record(item)
        success = bool(
            selected.get("success") and inspected and inspected.get("success")
        )
        response: Dict[str, Any] = {
            "success": success,
            "command": f"-stack-select-frame {level}",
            "state": self.gdb.get_state(),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": {"requested_level": level, "frame": frame},
            "raw_responses": {
                "select": selected.get("responses", []),
                "inspect": inspected.get("responses", []) if inspected else [],
            },
        }
        if not success:
            failed = (
                inspected if inspected and not inspected.get("success") else selected
            )
            response["error"] = self._error_message(
                failed, f"-stack-select-frame {level}"
            )
            response["type"] = "FrameSelectionError"
        return response

    @classmethod
    def _thread_record(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        frame = item.get("frame")
        return {
            "id": str(item.get("id", "")),
            "target_id": item.get("target-id"),
            "thread_group": item.get("thread-group"),
            "name": item.get("name"),
            "details": item.get("details"),
            "state": item.get("state"),
            "core": item.get("core"),
            "frame": cls._frame_record(frame) if isinstance(frame, dict) else None,
        }

    def threads(self, *, deadline: Optional[float] = None) -> Dict[str, Any]:
        """Return typed GDB threads, independent of pwndbg rendering."""
        result = self.gdb.list_threads_mi(timeout_sec=self._remaining_timeout(deadline))
        payload = self._first_payload(result)
        raw_threads = payload.get("threads", [])
        threads = [
            self._thread_record(item) for item in raw_threads if isinstance(item, dict)
        ]
        return self._typed_result(
            "-thread-info",
            result,
            {
                "count": len(threads),
                "current_thread_id": payload.get("current-thread-id"),
                "threads": threads,
            },
            error_type="ThreadInfoError",
        )

    @classmethod
    def _thread_group_record(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        children = item.get("children", [])
        return {
            "id": str(item.get("id", "")),
            "type": item.get("type"),
            "pid": item.get("pid"),
            "description": item.get("description"),
            "executable": item.get("executable"),
            "user_selected": item.get("user-selected") == "1",
            "cores": item.get("cores", []),
            "children": [
                cls._thread_group_record(child)
                for child in children
                if isinstance(child, dict)
            ],
        }

    def inferiors(
        self, recurse: int = 1, *, deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        """List GDB thread groups/inferiors using the MI hierarchy."""
        if not 0 <= recurse <= 16:
            raise ValueError("recurse must be between 0 and 16")
        result = self.gdb.list_thread_groups_mi(
            recurse=recurse,
            timeout_sec=self._remaining_timeout(deadline),
        )
        payload = self._first_payload(result)
        raw_groups = payload.get("groups", [])
        groups = [
            self._thread_group_record(item)
            for item in raw_groups
            if isinstance(item, dict)
        ]
        return self._typed_result(
            f"-list-thread-groups --recurse {recurse}",
            result,
            {"count": len(groups), "recurse": recurse, "inferiors": groups},
            error_type="InferiorInfoError",
        )

    @staticmethod
    def _register_group_for(name: str) -> str:
        normalized = name.lower()
        if re.match(r"^(?:xmm|ymm|zmm|v|q)\d+$", normalized):
            return "vector"
        if re.match(r"^(?:st|mm|f|d)\d+$", normalized) or normalized in {
            "fpsr",
            "fpcr",
            "fctrl",
            "fstat",
            "ftag",
            "fiseg",
            "fioff",
            "foseg",
            "fooff",
            "fop",
            "mxcsr",
        }:
            return "float"
        if normalized in {
            "eflags",
            "cpsr",
            "xpsr",
            "cs",
            "ss",
            "ds",
            "es",
            "fs",
            "gs",
            "orig_rax",
        }:
            return "system"
        return "general"

    def read_register_group(
        self, group: str, refresh_names: bool = False
    ) -> Dict[str, Any]:
        """Read an architecture-derived register group, including vector values."""
        if group not in {"general", "vector", "float", "system", "all"}:
            raise ValueError(
                "register group must be general, vector, float, system, or all"
            )
        names_result = self._load_register_names(refresh=refresh_names)
        if not names_result.get("success"):
            return self._typed_result(
                "read_register_group", names_result["result"], {"group": group}
            )
        names = [name for name in names_result["names"] if name]
        selected = (
            names
            if group == "all"
            else [name for name in names if self._register_group_for(name) == group]
        )
        if not selected:
            return {
                "success": True,
                "command": f"read_register_group {group}",
                "state": self.gdb.get_state(),
                "timeout_seconds": self.timeout_seconds,
                "parser_version": PARSER_VERSION,
                "parsed": {"group": group, "count": 0, "registers": []},
                "raw_responses": {"names": names_result.get("responses", [])},
            }
        response = self.read_registers(selected)
        response["command"] = f"read_register_group {group}"
        response["parsed"]["group"] = group
        raw_responses = response.setdefault("raw_responses", {})
        if isinstance(raw_responses, dict):
            raw_responses["group_names"] = names_result.get("responses", [])
        return response

    @staticmethod
    def parse_search(raw_output: str) -> Dict[str, Any]:
        matches = []
        with_module = re.compile(rf"^\s*(.+?)\s{{2,}}({HEX_PATTERN})\s+(.+?)\s*$")
        without_module = re.compile(rf"^\s*({HEX_PATTERN})\s+(.+?)\s*$")
        for line in raw_output.splitlines():
            match = with_module.match(line)
            if match:
                module, address, value = match.groups()
                matches.append(
                    {
                        "module": module.strip(),
                        "address": address.lower(),
                        "value": value,
                    }
                )
                continue
            match = without_module.match(line)
            if match:
                address, value = match.groups()
                matches.append(
                    {"module": None, "address": address.lower(), "value": value}
                )
        return {"count": len(matches), "matches": matches}

    def search_memory(
        self,
        value: str,
        value_type: str = "bytes",
        mapping: Optional[str] = None,
        hex_encoded: bool = False,
        executable: bool = False,
        writable: bool = False,
        limit: Optional[int] = None,
        aligned: Optional[int] = None,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        if value_type not in SEARCH_TYPES:
            raise ValueError("Unsupported search type: " + value_type)
        parts = ["search", "--no-save", "-t", value_type]
        if hex_encoded:
            parts.append("-x")
        if executable:
            parts.append("-e")
        if writable:
            parts.append("-w")
        if limit is not None:
            self._positive_number(limit, name="limit", maximum=1_000_000)
            parts.extend(["-l", str(limit)])
        if aligned is not None:
            self._positive_number(aligned, name="aligned", maximum=1 << 30)
            parts.extend(["-a", str(aligned)])
        if step is not None:
            self._positive_number(step, name="step", maximum=1 << 48)
            parts.extend(["-s", hex(step)])
        parts.append(self._quote_cli_argument(value))
        if mapping:
            if not SAFE_MAPPING_PATTERN.fullmatch(mapping):
                raise ValueError(f"Unsafe mapping filter: '{mapping}'")
            parts.append(self._quote_cli_argument(mapping))
        command = " ".join(parts)
        result = self.gdb.execute_command(command, timeout_sec=self.timeout_seconds)
        return self._typed_result(
            command,
            result,
            self.parse_search(self._raw_output(result)),
            error_type="MemorySearchError",
        )

    @classmethod
    def _memory_contents(cls, result: Dict[str, Any]) -> Optional[str]:
        payload = cls._first_payload(result)
        memory = payload.get("memory", [])
        if isinstance(memory, list) and memory and isinstance(memory[0], dict):
            contents = memory[0].get("contents")
            return str(contents).lower() if contents is not None else None
        return None

    def normalize_memory_write(self, data_hex: str) -> tuple[str, int]:
        """Validate one write payload and return normalized hex plus byte count."""
        payload = data_hex[2:] if data_hex.startswith("0x") else data_hex
        if len(payload) > self.max_memory_write_bytes * 2:
            raise ValueError(
                f"Memory write exceeds the {self.max_memory_write_bytes}-byte limit"
            )
        normalized = payload.lower()
        if (
            not normalized
            or len(normalized) % 2
            or not re.fullmatch(r"[0-9a-f]+", normalized)
        ):
            raise ValueError("data_hex must contain an even number of hex digits")
        byte_count = len(normalized) // 2
        if byte_count > self.max_memory_write_bytes:
            raise ValueError(
                f"Memory write exceeds the {self.max_memory_write_bytes}-byte limit"
            )
        return normalized, byte_count

    @staticmethod
    def _mapping_validation(
        mapping_result: Optional[Dict[str, Any]],
        allow_unsafe: bool,
        address: str,
        byte_count: int,
    ) -> Dict[str, Any]:
        if mapping_result is None:
            return {
                "checked": True,
                "mapped": False,
                "writable": False,
                "override_used": bool(allow_unsafe),
                "mapping": None,
                "resolved_address": None,
                "range_checked": False,
                "range_within_mapping": None,
            }
        parsed = mapping_result.get("parsed", {})
        mappings = parsed.get("mappings", []) if isinstance(parsed, dict) else []
        valid_mappings = [item for item in mappings if isinstance(item, dict)]
        resolved_address = None
        try:
            resolved_address = int(address, 0)
        except ValueError:
            pass
        mapping = None
        range_checked = resolved_address is not None
        range_within_mapping = None
        if mapping_result.get("success") and resolved_address is not None:
            for candidate in valid_mappings:
                try:
                    start = int(str(candidate.get("start")), 16)
                    end = int(str(candidate.get("end")), 16)
                except (TypeError, ValueError):
                    continue
                if start <= resolved_address < end:
                    mapping = candidate
                    range_within_mapping = resolved_address + byte_count <= end
                    break
        mapped = mapping is not None and range_checked and range_within_mapping is True
        writable = bool(mapping.get("writable")) if mapping else False
        return {
            "checked": True,
            "mapped": mapped,
            "writable": writable,
            "override_used": bool(allow_unsafe and (not mapped or not writable)),
            "mapping": mapping,
            "resolved_address": (
                hex(resolved_address) if resolved_address is not None else None
            ),
            "range_checked": range_checked,
            "range_within_mapping": range_within_mapping,
        }

    def write_memory(
        self,
        address: str,
        data_hex: str,
        verify: bool = True,
        mapping_result: Optional[Dict[str, Any]] = None,
        allow_unsafe: bool = False,
    ) -> Dict[str, Any]:
        address = self._validate_expression(address, kind="memory address")
        normalized, byte_count = self.normalize_memory_write(data_hex)
        quoted_address = self.gdb.quote_mi_argument(address)
        command = f"-data-write-memory-bytes {quoted_address} {normalized}"
        mapping_validation = self._mapping_validation(
            mapping_result,
            allow_unsafe,
            address,
            byte_count,
        )
        if (
            mapping_validation["checked"]
            and (not mapping_validation["mapped"] or not mapping_validation["writable"])
            and not allow_unsafe
        ):
            if mapping_validation.get("range_within_mapping") is False:
                reason = "Write range crosses the end of its mapped region"
            elif not mapping_validation["mapped"]:
                reason = "Address is not present in a mapped region"
            else:
                reason = "Address belongs to a non-writable mapping"
            return {
                "success": False,
                "command": command,
                "state": self.gdb.get_state(),
                "timeout_seconds": self.timeout_seconds,
                "parser_version": PARSER_VERSION,
                "parsed": {
                    "address": address,
                    "byte_count": byte_count,
                    "data_hex": normalized,
                    "mapping_validation": mapping_validation,
                },
                "raw_responses": {
                    "mapping": (
                        mapping_result.get("raw_responses", [])
                        if mapping_result
                        else []
                    )
                },
                "error": reason + "; pass allow_unsafe=true to override",
                "type": "MemoryProtectionError",
            }
        before_result = self.gdb.execute_mi_command(
            f"-data-read-memory-bytes {quoted_address} {byte_count}",
            timeout_sec=self.timeout_seconds,
        )
        original = self._memory_contents(before_result)
        write_result = self.gdb.execute_mi_command(
            command, timeout_sec=self.timeout_seconds
        )
        read_result = None
        actual = None
        if write_result.get("success") and verify:
            read_result = self.gdb.execute_mi_command(
                f"-data-read-memory-bytes {quoted_address} {byte_count}",
                timeout_sec=self.timeout_seconds,
            )
            actual = self._memory_contents(read_result)
        verified = actual == normalized if verify and actual is not None else None
        success = bool(
            write_result.get("success")
            and (
                not verify or (read_result and read_result.get("success") and verified)
            )
        )
        response = {
            "success": success,
            "command": command,
            "state": self.gdb.get_state(),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": {
                "address": address,
                "byte_count": byte_count,
                "data_hex": normalized,
                "verify_requested": verify,
                "verified": verified,
                "actual_hex": actual,
                "original_hex": original,
                "revertible": original is not None,
                "write_applied": bool(write_result.get("success")),
                "verification_uncertain": bool(
                    write_result.get("success") and verify and not success
                ),
                "mapping_validation": mapping_validation,
            },
            "raw_responses": {
                "mapping": (
                    mapping_result.get("raw_responses", []) if mapping_result else []
                ),
                "before": before_result.get("responses", []),
                "write": write_result.get("responses", []),
                "verify": read_result.get("responses", []) if read_result else [],
            },
        }
        if not success:
            source = (
                read_result
                if read_result and not read_result.get("success")
                else write_result
            )
            response["error"] = self._error_message(source, command)
            response["type"] = "MemoryWriteError"
        return response

    def disassemble(
        self,
        start: str = "$pc",
        byte_count: int = 64,
        *,
        deadline: Optional[float] = None,
    ) -> Dict[str, Any]:
        start = self._validate_expression(start, kind="disassembly start")
        self._positive_number(byte_count, name="byte_count", maximum=16 * 1024 * 1024)
        result = self.gdb.disassemble_range(
            start,
            byte_count,
            timeout_sec=self._remaining_timeout(deadline),
        )
        payload = self._first_payload(result)
        raw_instructions = payload.get("asm_insns", [])
        instructions = []
        if isinstance(raw_instructions, list):
            for item in raw_instructions:
                if not isinstance(item, dict):
                    continue
                instructions.append(
                    {
                        "address": item.get("address"),
                        "function": item.get("func-name"),
                        "offset": (
                            int(item["offset"])
                            if str(item.get("offset", "")).isdigit()
                            else item.get("offset")
                        ),
                        "instruction": item.get("inst"),
                    }
                )
        return self._typed_result(
            f"disassemble {start} +{byte_count}",
            result,
            {
                "start": start,
                "byte_count": byte_count,
                "count": len(instructions),
                "instructions": instructions,
            },
            error_type="DisassemblyError",
        )

    @classmethod
    def _stop_details(cls, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        stop = None
        for response in result.get("responses", []):
            if (
                response.get("type") == "notify"
                and response.get("message") == "stopped"
            ):
                payload = response.get("payload")
                if isinstance(payload, dict):
                    stop = payload
        return stop

    def step_until_event(
        self,
        event: str,
        symbol_regex: Optional[str] = None,
        mnemonic: Optional[str] = None,
        operands: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        if event not in STEP_EVENTS:
            raise ValueError("Unsupported step event: " + event)
        if event == "next_call":
            command = "nextcall"
            if symbol_regex:
                command += " " + self._quote_cli_argument(symbol_regex)
        elif event == "next_jump":
            command = "nextjmp"
        elif event == "next_return":
            command = "nextret"
        elif event == "next_syscall":
            command = "nextsyscall"
        elif event == "step_syscall":
            command = "stepsyscall"
        elif event == "instruction":
            if not mnemonic or not SAFE_MNEMONIC_PATTERN.fullmatch(mnemonic):
                raise ValueError("instruction event requires a safe mnemonic")
            command = f"stepuntilasm {mnemonic}"
            if operands:
                self._validate_expression(operands, kind="instruction operands")
                command += " " + self._quote_cli_argument(operands)
        else:
            if not target:
                raise ValueError("address event requires target")
            target = self._validate_expression(target, kind="step target")
            command = f"xuntil {target}"
        result = self.gdb.execute_command(command, timeout_sec=self.timeout_seconds)
        return self._typed_result(
            command,
            result,
            {"event": event, "stop": self._stop_details(result)},
            error_type="AdvancedStepError",
        )

    @staticmethod
    def _component_parsed(result: Dict[str, Any], key: str) -> Any:
        parsed = result.get("parsed", {})
        return parsed.get(key) if isinstance(parsed, dict) else None

    def _snapshot_timeout_component(
        self, component: str, error: Exception
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "command": f"capture_debugger_snapshot:{component}",
            "state": self.gdb.get_state(),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": {},
            "raw_responses": [],
            "error": str(error),
            "type": "TimeoutError",
            "timed_out": True,
        }

    def capture_snapshot(
        self,
        register_names: Optional[Sequence[str]] = None,
        disassembly_bytes: int = 64,
        mapping_result: Optional[Dict[str, Any]] = None,
        *,
        deadline: Optional[float] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture a typed snapshot under one deadline with cancellation points."""
        self._positive_number(
            disassembly_bytes,
            name="disassembly_bytes",
            maximum=16 * 1024 * 1024,
        )
        operation_deadline = (
            deadline if deadline is not None else self.snapshot_deadline()
        )
        components: Dict[str, Dict[str, Any]] = {}
        component_actions: Sequence[tuple[str, Callable[[], Dict[str, Any]]]] = (
            (
                "registers",
                lambda: self.read_registers(
                    register_names, deadline=operation_deadline
                ),
            ),
            ("frames", lambda: self.stack_frames(deadline=operation_deadline)),
            (
                "disassembly",
                lambda: self.disassemble(
                    "$pc", disassembly_bytes, deadline=operation_deadline
                ),
            ),
            ("threads", lambda: self.threads(deadline=operation_deadline)),
            ("inferiors", lambda: self.inferiors(deadline=operation_deadline)),
        )
        for name, action in component_actions:
            if cancellation_check is not None:
                cancellation_check()
            try:
                self._remaining_timeout(operation_deadline)
                components[name] = action()
            except TimeoutError as exc:
                components[name] = self._snapshot_timeout_component(name, exc)
            if cancellation_check is not None:
                cancellation_check()
        if mapping_result is not None:
            components["mappings"] = mapping_result
        status = {name: bool(item.get("success")) for name, item in components.items()}
        usable = any(status.get(name, False) for name in SNAPSHOT_CORE_COMPONENTS)
        core_complete = all(
            status.get(name, False) for name in SNAPSHOT_CORE_COMPONENTS
        )
        core_timed_out = all(
            bool(components.get(name, {}).get("timed_out"))
            for name in SNAPSHOT_CORE_COMPONENTS
        )
        complete = core_complete and (
            mapping_result is None or status.get("mappings", False)
        )
        snapshot = {
            "snapshot_version": SNAPSHOT_VERSION,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "state": self.gdb.get_state(),
            "registers": self._component_parsed(components["registers"], "registers")
            or [],
            "frames": self._component_parsed(components["frames"], "frames") or [],
            "instructions": self._component_parsed(
                components["disassembly"], "instructions"
            )
            or [],
            "threads": self._component_parsed(components["threads"], "threads") or [],
            "current_thread_id": self._component_parsed(
                components["threads"], "current_thread_id"
            ),
            "inferiors": self._component_parsed(components["inferiors"], "inferiors")
            or [],
            "mappings": (
                self._component_parsed(components["mappings"], "mappings") or []
                if "mappings" in components
                else []
            ),
        }
        response: Dict[str, Any] = {
            "success": usable,
            "usable": usable,
            "core_complete": core_complete,
            "complete": complete,
            "command": "capture_debugger_snapshot",
            "state": self.gdb.get_state(),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": {
                "snapshot": snapshot,
                "component_status": status,
                "usable": usable,
                "core_complete": core_complete,
                "complete": complete,
            },
            "raw_responses": {
                name: item.get("raw_responses", item.get("responses", []))
                for name, item in components.items()
            },
        }
        failed = [name for name, ok in status.items() if not ok]
        if failed:
            response["warnings"] = ["Failed snapshot components: " + ", ".join(failed)]
        if not usable:
            if core_timed_out:
                response["error"] = (
                    f"Debugger snapshot exceeded its {self.timeout_seconds}s deadline"
                )
                response["type"] = "TimeoutError"
                response["timed_out"] = True
            else:
                response["error"] = "No core debugger snapshot component succeeded"
                response["type"] = "DebuggerSnapshotError"
        return response

    @classmethod
    def _snapshot_payload(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap a ToolResponse, backend response, or inner snapshot value."""
        current: Dict[str, Any] = value
        for _ in range(6):
            data = current.get("data")
            if isinstance(data, dict):
                current = data
                continue
            parsed = current.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("snapshot"), dict):
                current = parsed["snapshot"]
                continue
            snapshot = current.get("snapshot")
            if isinstance(snapshot, dict):
                current = snapshot
                continue
            break
        return current

    @staticmethod
    def _snapshot_evidence(value: Dict[str, Any]) -> Any:
        """Read raw evidence from either the public envelope or backend result."""
        current = value
        for _ in range(4):
            if "raw_responses" in current:
                return current["raw_responses"]
            data = current.get("data")
            if not isinstance(data, dict):
                break
            current = data
        return []

    @classmethod
    def _validate_snapshot(cls, value: Dict[str, Any], label: str) -> Dict[str, Any]:
        snapshot = cls._snapshot_payload(value)
        if snapshot.get("snapshot_version") != SNAPSHOT_VERSION:
            raise ValueError(f"{label} snapshot_version must be '{SNAPSHOT_VERSION}'")
        missing = sorted(SNAPSHOT_REQUIRED_FIELDS - set(snapshot))
        if missing:
            raise ValueError(
                f"{label} snapshot is missing canonical fields: " + ", ".join(missing)
            )
        if not isinstance(snapshot.get("captured_at"), str) or not snapshot.get(
            "captured_at"
        ):
            raise ValueError(f"{label} snapshot captured_at must be a string")
        if not isinstance(snapshot.get("state"), str):
            raise ValueError(f"{label} snapshot state must be a string")
        for section in SNAPSHOT_RECORD_SECTIONS:
            records = snapshot.get(section)
            if not isinstance(records, list) or any(
                not isinstance(item, dict) for item in records
            ):
                raise ValueError(
                    f"{label} snapshot section '{section}' must be a list of objects"
                )
            identity_fields = SNAPSHOT_IDENTITY_FIELDS[section]
            seen: set[tuple[str, ...]] = set()
            for index, item in enumerate(records):
                identity: List[str] = []
                for field in identity_fields:
                    if field not in item:
                        raise ValueError(
                            f"{label} snapshot section '{section}' record {index} "
                            f"is missing identity field '{field}'"
                        )
                    raw_value = item[field]
                    if isinstance(raw_value, (dict, list, set, tuple)):
                        raise ValueError(
                            f"{label} snapshot section '{section}' record {index} "
                            f"identity field '{field}' must be scalar"
                        )
                    if raw_value is None and not (
                        section == "mappings" and field == "file"
                    ):
                        raise ValueError(
                            f"{label} snapshot section '{section}' record {index} "
                            f"identity field '{field}' must not be null"
                        )
                    normalized = "" if raw_value is None else str(raw_value)
                    if not normalized and not (
                        section == "mappings" and field == "file"
                    ):
                        raise ValueError(
                            f"{label} snapshot section '{section}' record {index} "
                            f"identity field '{field}' must not be empty"
                        )
                    identity.append(normalized)
                identity_key = tuple(identity)
                if identity_key in seen:
                    rendered = ", ".join(
                        f"{field}={identity[position]!r}"
                        for position, field in enumerate(identity_fields)
                    )
                    raise ValueError(
                        f"{label} snapshot section '{section}' has duplicate "
                        f"identity: {rendered}"
                    )
                seen.add(identity_key)
        current_thread_id = snapshot.get("current_thread_id")
        if current_thread_id is not None and not isinstance(current_thread_id, str):
            raise ValueError(
                f"{label} snapshot current_thread_id must be a string or null"
            )
        return snapshot

    @staticmethod
    def _diff_records(
        before: Sequence[Dict[str, Any]],
        after: Sequence[Dict[str, Any]],
        key_fields: Sequence[str],
    ) -> Dict[str, Any]:
        def record_key(item: Dict[str, Any]) -> tuple[str, ...]:
            return tuple(
                "" if item.get(field) is None else str(item.get(field))
                for field in key_fields
            )

        before_by_key = {record_key(item): item for item in before}
        after_by_key = {record_key(item): item for item in after}
        before_keys = set(before_by_key)
        after_keys = set(after_by_key)
        added = [after_by_key[key] for key in sorted(after_keys - before_keys)]
        removed = [before_by_key[key] for key in sorted(before_keys - after_keys)]
        changed = [
            {
                "key": {
                    field: key[position] for position, field in enumerate(key_fields)
                },
                "before": before_by_key[key],
                "after": after_by_key[key],
            }
            for key in sorted(before_keys & after_keys)
            if before_by_key[key] != after_by_key[key]
        ]
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
            "change_count": len(added) + len(removed) + len(changed),
        }

    @classmethod
    def diff_snapshots(
        cls, before_value: Dict[str, Any], after_value: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Diff two caller-supplied snapshots without hidden server state."""
        before = cls._validate_snapshot(before_value, "before")
        after = cls._validate_snapshot(after_value, "after")

        def records(snapshot: Dict[str, Any], section: str) -> List[Dict[str, Any]]:
            value = snapshot.get(section, [])
            if not isinstance(value, list) or any(
                not isinstance(item, dict) for item in value
            ):
                raise ValueError(
                    f"Snapshot section '{section}' must be a list of objects"
                )
            return value

        register_diff = cls._diff_records(
            records(before, "registers"), records(after, "registers"), ["name"]
        )
        frame_diff = cls._diff_records(
            records(before, "frames"), records(after, "frames"), ["level"]
        )
        instruction_diff = cls._diff_records(
            records(before, "instructions"),
            records(after, "instructions"),
            ["address"],
        )
        thread_diff = cls._diff_records(
            records(before, "threads"), records(after, "threads"), ["id"]
        )
        inferior_diff = cls._diff_records(
            records(before, "inferiors"), records(after, "inferiors"), ["id"]
        )
        mapping_diff = cls._diff_records(
            records(before, "mappings"),
            records(after, "mappings"),
            ["start", "end", "file"],
        )
        sections = {
            "registers": register_diff,
            "frames": frame_diff,
            "instructions": instruction_diff,
            "threads": thread_diff,
            "inferiors": inferior_diff,
            "mappings": mapping_diff,
        }
        scalar_changes = {
            "state": {
                "changed": before.get("state") != after.get("state"),
                "before": before.get("state"),
                "after": after.get("state"),
            },
            "current_thread_id": {
                "changed": before.get("current_thread_id")
                != after.get("current_thread_id"),
                "before": before.get("current_thread_id"),
                "after": after.get("current_thread_id"),
            },
        }
        record_change_count = sum(item["change_count"] for item in sections.values())
        scalar_change_count = sum(
            1 for item in scalar_changes.values() if item["changed"]
        )
        total = record_change_count + scalar_change_count
        return {
            "success": True,
            "command": "diff_debugger_snapshots",
            "parser_version": PARSER_VERSION,
            "parsed": {
                "changed": total > 0,
                "change_count": total,
                "record_change_count": record_change_count,
                "scalar_change_count": scalar_change_count,
                "state": scalar_changes["state"],
                "current_thread_id": scalar_changes["current_thread_id"],
                "scalar_changes": scalar_changes,
                "sections": sections,
            },
            "raw_responses": {
                "before": cls._snapshot_evidence(before_value),
                "after": cls._snapshot_evidence(after_value),
            },
        }
