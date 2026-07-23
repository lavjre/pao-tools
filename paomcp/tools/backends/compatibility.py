"""Typed discovery for the pwndbg runtime used by pao-tools."""

from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional

from paomcp.tools.backends.gdb import GdbController

ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SAFE_COMMAND_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
SAFE_FILTER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _./+-]{0,79}$")
COMMAND_LINE_PATTERN = re.compile(
    r"^\s*([a-z][a-z0-9_-]*)(?:\s+\[[^\]]+\])?" r"(?:(?:\s+--\s+)|(?:\s{2,}))(.+?)\s*$"
)
PARSER_VERSION = "1"

REQUIRED_COMMAND_GROUPS: Dict[str, List[str]] = {
    "core": ["context", "regs", "telescope", "vmmap"],
    "elf": ["checksec", "elfsections", "got", "plt"],
    "runtime": ["auxv", "libcinfo", "linkmap", "procinfo", "threads", "xinfo"],
    "memory": ["hexdump", "search", "strings", "leakfind"],
    "crash": ["cyclic", "nearpc", "retaddr", "stack"],
    "heap": ["arena", "bins", "fastbins", "heap", "malloc-chunk", "tcachebins"],
    "rop": ["onegadget", "rop", "ropper", "sigreturn"],
    "stepping": ["nextcall", "nextjmp", "nextret", "nextsyscall", "stepuntilasm"],
}


class PwndbgCompatibility:
    """Discover versions, commands, categories, and expected feature groups."""

    def __init__(
        self,
        gdb: GdbController,
        expected_version: Optional[str] = None,
    ):
        self.gdb = gdb
        self.expected_version = expected_version or os.environ.get(
            "PAO_PWNDBG_EXPECTED_VERSION"
        )
        self._cached_capabilities: Optional[Dict[str, Any]] = None

    @staticmethod
    def _raw_output(result: Dict[str, Any]) -> str:
        parts: List[str] = []
        for response in result.get("responses", []):
            payload = response.get("payload")
            if isinstance(payload, str):
                parts.append(payload)
        return ANSI_PATTERN.sub("", "".join(parts)).replace("\r\n", "\n")

    @staticmethod
    def _validate(value: str, *, command: bool = False) -> str:
        pattern = SAFE_COMMAND_PATTERN if command else SAFE_FILTER_PATTERN
        if not pattern.fullmatch(value):
            kind = "command" if command else "filter"
            raise ValueError(f"Unsafe pwndbg {kind}: '{value}'")
        return value

    @staticmethod
    def _parse_commands(raw_output: str) -> List[Dict[str, str]]:
        commands: Dict[str, Dict[str, str]] = {}
        for line in raw_output.splitlines():
            match = COMMAND_LINE_PATTERN.match(line)
            if not match:
                continue
            name, description = match.groups()
            if name in {"usage", "pwndbg"}:
                continue
            commands[name] = {"name": name, "description": description.strip()}
        return [commands[name] for name in sorted(commands)]

    def list_commands(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List pwndbg commands, optionally filtered by category or text."""
        if category and query:
            raise ValueError("Use either category or query, not both")
        command = "pwndbg"
        if category:
            command += f' -c "{self._validate(category)}"'
        elif query:
            command += f" {self._validate(query)}"
        result = self.gdb.execute_command(command)
        raw_output = self._raw_output(result)
        commands = self._parse_commands(raw_output)
        return {
            "success": bool(result.get("success")),
            "command": command,
            "parser_version": PARSER_VERSION,
            "parsed": {"count": len(commands), "commands": commands},
            "raw_output": raw_output,
            "raw_responses": result.get("responses", []),
        }

    def list_categories(self) -> Dict[str, Any]:
        """List category names reported by the installed pwndbg."""
        command = "pwndbg --list-categories"
        result = self.gdb.execute_command(command)
        raw_output = self._raw_output(result)
        categories = []
        for line in raw_output.splitlines():
            value = line.strip().lstrip("-*").strip()
            if (
                value
                and value != command
                and len(value) <= 80
                and not value.lower().startswith("usage")
            ):
                categories.append(value)
        return {
            "success": bool(result.get("success")),
            "command": command,
            "parsed": {"categories": sorted(set(categories))},
            "raw_output": raw_output,
            "raw_responses": result.get("responses", []),
        }

    def get_help(self, command_name: str) -> Dict[str, Any]:
        """Return help for one validated GDB/pwndbg command."""
        command_name = self._validate(command_name, command=True)
        command = f"help {command_name}"
        result = self.gdb.execute_command(command)
        return {
            "success": bool(result.get("success")),
            "command": command,
            "command_name": command_name,
            "raw_output": self._raw_output(result),
            "raw_responses": result.get("responses", []),
        }

    @staticmethod
    def _parse_versions(raw_output: str) -> Dict[str, Optional[str]]:
        pwndbg_match = re.search(
            r"(?im)^\s*Pwndbg(?:\s+version)?\s*[: ]\s*([^\s,]+)", raw_output
        )
        gdb_match = re.search(
            r"(?im)^\s*(?:Gdb\s*:|GNU gdb[^\d]*)([^\r\n]+)", raw_output
        )
        return {
            "pwndbg": pwndbg_match.group(1) if pwndbg_match else None,
            "gdb": gdb_match.group(1).strip() if gdb_match else None,
        }

    def get_capabilities(self, refresh: bool = False) -> Dict[str, Any]:
        """Return a cached compatibility report for the current runtime."""
        if self._cached_capabilities is not None and not refresh:
            return self._cached_capabilities
        command_result = self.list_commands()
        category_result = self.list_categories()
        version_result = self.gdb.execute_command("version")
        core_help_result = self.gdb.execute_command("help generate-core-file")
        version_output = self._raw_output(version_result)
        core_help_output = self._raw_output(core_help_result)
        versions = self._parse_versions(version_output)
        names = {item["name"] for item in command_result["parsed"].get("commands", [])}
        groups = {
            group: {
                "available": [name for name in expected if name in names],
                "missing": [name for name in expected if name not in names],
            }
            for group, expected in REQUIRED_COMMAND_GROUPS.items()
        }
        critical_missing = groups["core"]["missing"]
        version_matches = (
            self.expected_version is None or versions["pwndbg"] == self.expected_version
        )
        warnings = []
        if not version_matches:
            warnings.append(
                f"Expected pwndbg {self.expected_version}, found {versions['pwndbg']}"
            )
        if critical_missing:
            warnings.append("Missing core commands: " + ", ".join(critical_missing))
        generate_core_file_available = bool(
            core_help_result.get("success")
            and core_help_output.strip()
            and "undefined command" not in core_help_output.lower()
        )
        if not generate_core_file_available:
            warnings.append(
                "GDB built-in generate-core-file is unavailable; typed core capture "
                "will be disabled"
            )
        report = {
            "success": bool(
                command_result["success"] and version_result.get("success")
            ),
            "ready": bool(names and not critical_missing and version_matches),
            "parser_version": PARSER_VERSION,
            "expected_pwndbg_version": self.expected_version,
            "version_matches": version_matches,
            "versions": versions,
            "command_count": len(names),
            "categories": category_result["parsed"]["categories"],
            "command_groups": groups,
            "gdb_features": {
                "generate_core_file": {
                    "available": generate_core_file_available,
                    "source": "gdb-cli",
                    "command": "generate-core-file",
                }
            },
            "executables": {
                name: shutil.which(name)
                for name in ("pwndbg", "gdb", "gdbserver", "ropper", "ROPgadget")
            },
            "warnings": warnings,
            "evidence": {
                "version": version_output,
                "commands": command_result["raw_output"],
                "categories": category_result["raw_output"],
                "generate_core_file_help": core_help_output,
            },
        }
        self._cached_capabilities = report
        return report
