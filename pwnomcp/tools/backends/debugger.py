"""Typed GDB/MI and pwndbg operations for debugger and memory control."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

from pwnomcp.tools.backends.gdb import GdbController

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


class DebuggerOperations:
    """Provide validated typed debugger operations over one GDB controller."""

    def __init__(
        self,
        gdb: GdbController,
        timeout_seconds: Optional[float] = None,
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
        self._register_names: Optional[List[str]] = None

    def clear_register_cache(self) -> None:
        """Clear architecture-specific register metadata after changing files."""
        self._register_names = None

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
            response["type"] = error_type
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
        }

    def decorate_created_breakpoint(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add a typed breakpoint record to an existing MI insert result."""
        payload = self._first_payload(result)
        item = payload.get("bkpt")
        result["parser_version"] = PARSER_VERSION
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

    def _load_register_names(self, refresh: bool = False) -> Dict[str, Any]:
        if self._register_names is not None and not refresh:
            return {"success": True, "names": self._register_names, "responses": []}
        result = self.gdb.execute_mi_command(
            "-data-list-register-names", timeout_sec=self.timeout_seconds
        )
        payload = self._first_payload(result)
        names = payload.get("register-names", [])
        if result.get("success") and isinstance(names, list):
            self._register_names = [str(name) for name in names]
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
    ) -> Dict[str, Any]:
        name_result = self._load_register_names(refresh=refresh_names)
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
            register_numbers, timeout_sec=self.timeout_seconds
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

    def write_memory(
        self, address: str, data_hex: str, verify: bool = True
    ) -> Dict[str, Any]:
        address = self._validate_expression(address, kind="memory address")
        normalized = data_hex.removeprefix("0x").lower()
        if (
            not normalized
            or len(normalized) % 2
            or not re.fullmatch(r"[0-9a-f]+", normalized)
        ):
            raise ValueError("data_hex must contain an even number of hex digits")
        command = f"-data-write-memory-bytes {address} {normalized}"
        write_result = self.gdb.execute_mi_command(
            command, timeout_sec=self.timeout_seconds
        )
        read_result = None
        actual = None
        if write_result.get("success") and verify:
            read_result = self.gdb.execute_mi_command(
                f"-data-read-memory-bytes {address} {len(normalized) // 2}",
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
                "byte_count": len(normalized) // 2,
                "data_hex": normalized,
                "verify_requested": verify,
                "verified": verified,
                "actual_hex": actual,
            },
            "raw_responses": {
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

    def disassemble(self, start: str = "$pc", byte_count: int = 64) -> Dict[str, Any]:
        start = self._validate_expression(start, kind="disassembly start")
        self._positive_number(byte_count, name="byte_count", maximum=16 * 1024 * 1024)
        result = self.gdb.disassemble_range(
            start, byte_count, timeout_sec=self.timeout_seconds
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
