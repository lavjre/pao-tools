"""Typed parsers for ELF and runtime inspection commands exposed by pwndbg."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from pwnomcp.tools.backends.gdb import GdbController

ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
HEX_PATTERN = r"0x[0-9a-fA-F]+"
SAFE_GDB_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_.$+*/:\-\[\]()]{1,160}$")
SAFE_SYMBOL_FILTER_PATTERN = re.compile(r"^[A-Za-z0-9_@.*?+\-]{1,128}$")
PARSER_VERSION = "1"


class PwndbgInspection:
    """Execute and parse stable ELF/process inspection commands."""

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
            raise ValueError("Inspection timeout must be greater than zero")
        self.timeout_seconds = configured_timeout

    @staticmethod
    def _raw_output(result: Dict[str, Any]) -> str:
        parts: List[str] = []
        for response in result.get("responses", []):
            payload = response.get("payload")
            if isinstance(payload, str):
                parts.append(payload)
        return ANSI_PATTERN.sub("", "".join(parts)).replace("\r\n", "\n")

    @staticmethod
    def _quote_gdb_argument(value: str) -> str:
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError("GDB argument contains a forbidden control character")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    @staticmethod
    def _validate_gdb_value(value: str, *, kind: str) -> str:
        if not SAFE_GDB_VALUE_PATTERN.fullmatch(value):
            raise ValueError(f"Unsafe {kind}: '{value}'")
        return value

    @staticmethod
    def _validate_symbol_filter(value: str) -> str:
        if not SAFE_SYMBOL_FILTER_PATTERN.fullmatch(value):
            raise ValueError(f"Unsafe symbol filter: '{value}'")
        return value

    @staticmethod
    def _command_error(raw_output: str, command: str) -> str:
        lines = [
            line.strip()
            for line in raw_output.splitlines()
            if line.strip() and line.strip() != command
        ]
        return lines[-1] if lines else f"pwndbg command failed: {command}"

    def _result(
        self,
        command: str,
        result: Dict[str, Any],
        parsed: Dict[str, Any],
    ) -> Dict[str, Any]:
        raw_output = self._raw_output(result)
        success = bool(result.get("success"))
        response = {
            "success": success,
            "command": command,
            "state": result.get("state", self.gdb.get_state()),
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": parsed,
            "raw_output": raw_output,
            "raw_responses": result.get("responses", []),
        }
        if not success:
            response["error"] = self._command_error(raw_output, command)
            response["type"] = "PwndbgCommandError"
        return response

    def _execute(self, command: str, parser: Any) -> Dict[str, Any]:
        result = self.gdb.execute_command(command, timeout_sec=self.timeout_seconds)
        raw_output = self._raw_output(result)
        return self._result(command, result, parser(raw_output))

    @staticmethod
    def _enabled_status(value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        normalized = value.lower()
        negative = ("no ", "disabled", "not found", "without")
        if any(marker in normalized for marker in negative):
            return False
        positive = ("enabled", "found", "full", "partial", "pie")
        if any(marker in normalized for marker in positive):
            return True
        return None

    @classmethod
    def parse_checksec(cls, raw_output: str) -> Dict[str, Any]:
        fields: Dict[str, str] = {}
        known = {
            "file",
            "arch",
            "relro",
            "stack",
            "nx",
            "pie",
            "fortify",
            "shstk",
            "ibt",
        }
        for line in raw_output.splitlines():
            match = re.match(r"^\s*([A-Za-z]+):\s*(.+?)\s*$", line)
            if match and match.group(1).lower() in known:
                fields[match.group(1).lower()] = match.group(2).strip()

        relro_status = fields.get("relro")
        relro_level = "unknown"
        if relro_status:
            normalized_relro = relro_status.lower()
            if "full" in normalized_relro:
                relro_level = "full"
            elif "partial" in normalized_relro:
                relro_level = "partial"
            elif "no relro" in normalized_relro:
                relro_level = "none"

        mitigations = {
            "relro": {"status": relro_status, "level": relro_level},
            "canary": {
                "status": fields.get("stack"),
                "enabled": cls._enabled_status(fields.get("stack")),
            },
            "nx": {
                "status": fields.get("nx"),
                "enabled": cls._enabled_status(fields.get("nx")),
            },
            "pie": {
                "status": fields.get("pie"),
                "enabled": cls._enabled_status(fields.get("pie")),
            },
            "fortify": {
                "status": fields.get("fortify"),
                "enabled": cls._enabled_status(fields.get("fortify")),
            },
            "shadow_stack": {
                "status": fields.get("shstk"),
                "enabled": cls._enabled_status(fields.get("shstk")),
            },
            "ibt": {
                "status": fields.get("ibt"),
                "enabled": cls._enabled_status(fields.get("ibt")),
            },
        }
        return {
            "file": fields.get("file"),
            "arch": fields.get("arch"),
            "mitigations": mitigations,
            "details": fields,
        }

    @staticmethod
    def parse_elf_sections(raw_output: str) -> Dict[str, Any]:
        section_pattern = re.compile(
            rf"^\s*({HEX_PATTERN})\s+({HEX_PATTERN})\s+([RrWwXx-]{{3}})"
            r"\s+((?:0x)?[0-9a-fA-F]+)\s+(\S+)\s*$"
        )
        sections = []
        for line in raw_output.splitlines():
            match = section_pattern.match(line)
            if not match:
                continue
            start, end, permissions, size, name = match.groups()
            sections.append(
                {
                    "name": name,
                    "start": start.lower(),
                    "end": end.lower(),
                    "size": size.lower(),
                    "permissions": permissions.upper(),
                    "readable": permissions[0].lower() == "r",
                    "writable": permissions[1].lower() == "w",
                    "executable": permissions[2].lower() == "x",
                }
            )
        return {"count": len(sections), "sections": sections}

    @staticmethod
    def parse_got(raw_output: str) -> Dict[str, Any]:
        object_path = None
        protection = None
        reported_count = None
        entries = []
        for line in raw_output.splitlines():
            object_match = re.match(r"^State of the GOT of\s+(.+?):\s*$", line.strip())
            if object_match:
                object_path = object_match.group(1).strip()
                continue
            protection_match = re.match(
                r"^GOT protection:\s*(.+?)\s*\|\s*Found\s+(\d+)\s+GOT entries",
                line.strip(),
            )
            if protection_match:
                protection = protection_match.group(1).strip()
                reported_count = int(protection_match.group(2))
                continue
            entry_match = re.match(
                rf"^\s*\[({HEX_PATTERN})\]\s+(\S+)\s+->\s+(.+?)\s*$", line
            )
            if entry_match:
                address, symbol, target = entry_match.groups()
                entries.append(
                    {
                        "address": address.lower(),
                        "symbol": symbol,
                        "target": target.strip(),
                    }
                )
        return {
            "object": object_path,
            "protection": protection,
            "reported_count": reported_count,
            "count": len(entries),
            "entries": entries,
        }

    @staticmethod
    def parse_plt(raw_output: str) -> Dict[str, Any]:
        section_pattern = re.compile(
            rf"^Section\s+(\S+)\s+({HEX_PATTERN})\s+-\s+({HEX_PATTERN}):\s*$"
        )
        entry_pattern = re.compile(rf"^\s*({HEX_PATTERN}):\s+(.+?)\s*$")
        sections: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for line in raw_output.splitlines():
            section_match = section_pattern.match(line.strip())
            if section_match:
                name, start, end = section_match.groups()
                current = {
                    "name": name,
                    "start": start.lower(),
                    "end": end.lower(),
                    "entries": [],
                }
                sections.append(current)
                continue
            entry_match = entry_pattern.match(line)
            if not entry_match:
                continue
            address, symbol = entry_match.groups()
            if current is None:
                current = {"name": None, "start": None, "end": None, "entries": []}
                sections.append(current)
            current["entries"].append(
                {"address": address.lower(), "symbol": symbol.strip()}
            )
        return {
            "section_count": len(sections),
            "entry_count": sum(len(section["entries"]) for section in sections),
            "sections": sections,
        }

    @staticmethod
    def _normalize_hex(value: str) -> str:
        return value.lower() if value.lower().startswith("0x") else f"0x{value.lower()}"

    @classmethod
    def _parse_mapping_line(cls, line: str) -> Optional[Dict[str, Any]]:
        mapping_pattern = re.compile(
            rf"^\s*[►>]?\s*({HEX_PATTERN})\s+({HEX_PATTERN})\s+([rwxps-]{{4}})"
            r"\s+((?:0x)?[0-9a-fA-F]+)\s+((?:0x)?[0-9a-fA-F]+)\s+(.+?)\s*$",
            re.IGNORECASE,
        )
        match = mapping_pattern.match(line)
        if not match:
            return None
        start, end, permissions, size, offset, file_name = match.groups()
        permissions = permissions.lower()
        return {
            "start": start.lower(),
            "end": end.lower(),
            "size": cls._normalize_hex(size),
            "offset": cls._normalize_hex(offset),
            "permissions": permissions,
            "readable": permissions[0] == "r",
            "writable": permissions[1] == "w",
            "executable": permissions[2] == "x",
            "private": permissions[3] == "p",
            "shared": permissions[3] == "s",
            "file": file_name.strip(),
        }

    @classmethod
    def parse_vmmap(cls, raw_output: str) -> Dict[str, Any]:
        mappings = []
        for line in raw_output.splitlines():
            mapping = cls._parse_mapping_line(line)
            if mapping is not None:
                mappings.append(mapping)
        return {"count": len(mappings), "mappings": mappings}

    @staticmethod
    def parse_threads(raw_output: str) -> Dict[str, Any]:
        threads = []
        shown = None
        total = None
        for line in raw_output.splitlines():
            summary_match = re.match(r"^Showing\s+(\d+)\s+of\s+(\d+)\s+threads", line)
            if summary_match:
                shown = int(summary_match.group(1))
                total = int(summary_match.group(2))
                continue
            columns = re.split(r"\s{2,}", line.strip())
            if len(columns) < 4 or not columns[0].isdigit():
                continue
            threads.append(
                {
                    "global_num": int(columns[0]),
                    "name": columns[1],
                    "status": columns[2],
                    "pc": columns[3],
                    "symbol": columns[4] if len(columns) > 4 else None,
                }
            )
        return {
            "shown": shown if shown is not None else len(threads),
            "total": total if total is not None else len(threads),
            "threads": threads,
        }

    @staticmethod
    def _unquote(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    @classmethod
    def parse_procinfo(cls, raw_output: str) -> Dict[str, Any]:
        fields: Dict[str, Any] = {}
        file_descriptors: Dict[str, str] = {}
        integer_fields = {"pid", "tid", "ppid"}
        list_fields = {"uid", "gid", "groups"}
        for line in raw_output.splitlines():
            match = re.match(r"^\s*([a-z]+(?:\[\d+\])?)\s+(.+?)\s*$", line)
            if not match:
                continue
            key, value = match.groups()
            if key.startswith("fd["):
                file_descriptors[key[3:-1]] = value
            elif key in integer_fields and value.isdigit():
                fields[key] = int(value)
            elif key in list_fields:
                fields[key] = [int(item) for item in re.findall(r"\d+", value)]
            else:
                fields[key] = cls._unquote(value)
        fields["file_descriptors"] = file_descriptors
        return fields

    @classmethod
    def parse_libcinfo(cls, raw_output: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "libc": None,
            "version": None,
            "linked": None,
            "urls": {},
            "mappings": {},
            "symbolication": {},
        }
        section = None
        pending_mapping = None
        for line in raw_output.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("libc: "):
                result["libc"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("libc version: "):
                result["version"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("linked: "):
                result["linked"] = stripped.split(":", 1)[1].strip()
                continue
            if stripped.startswith("URLs"):
                section = "urls"
                continue
            if stripped == "Mappings:":
                section = "mappings"
                continue
            if stripped == "Symbolication:":
                section = "symbolication"
                continue

            mapping_match = re.match(
                rf"^(libc|ld) is at:\s*({HEX_PATTERN})\s*$", stripped
            )
            if section == "mappings" and mapping_match:
                name, base = mapping_match.groups()
                result["mappings"][name] = {"base": base.lower(), "path": None}
                pending_mapping = name
                continue
            if section == "mappings" and pending_mapping and stripped.startswith("/"):
                result["mappings"][pending_mapping]["path"] = stripped
                pending_mapping = None
                continue
            if section == "urls":
                url_match = re.match(r"^([^:]+):\s+(.+?)\s*$", stripped)
                if url_match:
                    key = url_match.group(1).strip().replace(" ", "_")
                    value = url_match.group(2).strip()
                    result["urls"][key] = None if value == "not available" else value
                continue
            if section == "symbolication":
                symbol_match = re.match(r"^([^:]+):\s+(yes|no)\s*$", stripped)
                if symbol_match:
                    key = symbol_match.group(1).strip().replace(" ", "_")
                    result["symbolication"][key] = symbol_match.group(2) == "yes"
        return result

    @staticmethod
    def parse_linkmap(raw_output: str) -> Dict[str, Any]:
        entries = []
        entry_pattern = re.compile(
            rf"^\s*({HEX_PATTERN})\s+(.+?)\s{{2,}}({HEX_PATTERN})\s+({HEX_PATTERN})\s*$"
        )
        for line in raw_output.splitlines():
            match = entry_pattern.match(line)
            if not match:
                continue
            node, objfile, load_bias, dynamic_segment = match.groups()
            entries.append(
                {
                    "node": node.lower(),
                    "objfile": objfile.strip(),
                    "load_bias": load_bias.lower(),
                    "dynamic_segment": dynamic_segment.lower(),
                }
            )
        return {"count": len(entries), "entries": entries}

    @classmethod
    def parse_xinfo(cls, raw_output: str) -> Dict[str, Any]:
        address = None
        mapping = None
        offsets: Dict[str, Dict[str, str]] = {}
        sections = []
        header_match = re.search(
            rf"Extended information for virtual address\s+({HEX_PATTERN})", raw_output
        )
        if header_match:
            address = header_match.group(1).lower()
        for line in raw_output.splitlines():
            if mapping is None:
                mapping = cls._parse_mapping_line(line)
                if mapping is not None:
                    continue
            offset_match = re.match(
                rf"^\s*(Mapped Area|File \(Base\)|File \(Segment\)|File \(Disk\))"
                rf"\s+({HEX_PATTERN})\s*=\s*(.+?)\s+\+\s+({HEX_PATTERN})\s*$",
                line,
            )
            if offset_match:
                label, inspected, base, offset = offset_match.groups()
                offsets[label.lower().replace(" ", "_")] = {
                    "address": inspected.lower(),
                    "base": base.strip(),
                    "offset": offset.lower(),
                }
                continue
            section_match = re.match(
                rf"^\s*(\.[^\s]+)\s+({HEX_PATTERN})\s*=\s*({HEX_PATTERN})"
                rf"\s+\+\s+({HEX_PATTERN})\s*$",
                line,
            )
            if section_match:
                name, inspected, base, offset = section_match.groups()
                sections.append(
                    {
                        "name": name,
                        "address": inspected.lower(),
                        "base": base.lower(),
                        "offset": offset.lower(),
                    }
                )
        return {
            "address": address,
            "mapping": mapping,
            "offsets": offsets,
            "sections": sections,
        }

    def checksec(self, binary_path: str) -> Dict[str, Any]:
        command = f"checksec -f {self._quote_gdb_argument(binary_path)}"
        return self._execute(command, self.parse_checksec)

    def elf_sections(self, no_rebase: bool = True) -> Dict[str, Any]:
        command = "elfsections --no-rebase" if no_rebase else "elfsections"
        return self._execute(command, self.parse_elf_sections)

    def got(
        self,
        symbol_filter: Optional[str] = None,
        include_readonly: bool = False,
        all_objects: bool = False,
    ) -> Dict[str, Any]:
        parts = ["got"]
        if include_readonly:
            parts.append("--show-readonly")
        if all_objects:
            parts.append("--all")
        if symbol_filter:
            parts.append(self._validate_symbol_filter(symbol_filter))
        command = " ".join(parts)
        return self._execute(command, self.parse_got)

    def plt(self, all_symbols: bool = False) -> Dict[str, Any]:
        command = "plt --all-symbols" if all_symbols else "plt"
        return self._execute(command, self.parse_plt)

    def memory_maps(
        self,
        address_or_module: Optional[str] = None,
        writable: bool = False,
        executable: bool = False,
        gaps: bool = False,
    ) -> Dict[str, Any]:
        parts = ["vmmap"]
        if writable:
            parts.append("--writable")
        if executable:
            parts.append("--executable")
        if gaps:
            parts.append("--gaps")
        if address_or_module:
            parts.append(
                self._validate_gdb_value(
                    address_or_module, kind="address/module filter"
                )
            )
        command = " ".join(parts)
        return self._execute(command, self.parse_vmmap)

    def threads(self, limit: Optional[int] = None) -> Dict[str, Any]:
        if limit is not None and not 1 <= limit <= 10000:
            raise ValueError("Thread limit must be between 1 and 10000")
        command = "threads" if limit is None else f"threads {limit}"
        return self._execute(command, self.parse_threads)

    def process_info(self) -> Dict[str, Any]:
        return self._execute("procinfo", self.parse_procinfo)

    def libc_info(self) -> Dict[str, Any]:
        return self._execute("libcinfo", self.parse_libcinfo)

    def link_map(self) -> Dict[str, Any]:
        return self._execute("linkmap", self.parse_linkmap)

    def address_info(self, address: str = "$pc") -> Dict[str, Any]:
        address = self._validate_gdb_value(address, kind="address expression")
        command = f"xinfo {address}"
        return self._execute(command, self.parse_xinfo)

    def _aggregate(
        self,
        command: str,
        components: Dict[str, Dict[str, Any]],
        parsed: Dict[str, Any],
        state: str,
    ) -> Dict[str, Any]:
        failed = [name for name, item in components.items() if not item.get("success")]
        succeeded = [name for name, item in components.items() if item.get("success")]
        response: Dict[str, Any] = {
            "success": bool(succeeded),
            "complete": not failed,
            "command": command,
            "state": state,
            "timeout_seconds": self.timeout_seconds,
            "parser_version": PARSER_VERSION,
            "parsed": parsed,
            "component_status": {
                name: bool(item.get("success")) for name, item in components.items()
            },
            "evidence": {
                name: item.get("raw_output", "") for name, item in components.items()
            },
            "raw_responses": {
                name: item.get("raw_responses", []) for name, item in components.items()
            },
        }
        if failed and succeeded:
            response["warnings"] = [
                "Failed inspection components: " + ", ".join(failed)
            ]
        elif failed:
            response["error"] = "All inspection components failed: " + ", ".join(failed)
            response["type"] = "PwndbgInspectionError"
        return response

    def analyze_elf(self, binary_path: str) -> Dict[str, Any]:
        components = {
            "checksec": self.checksec(binary_path),
            "sections": self.elf_sections(no_rebase=True),
            "plt": self.plt(),
        }
        return self._aggregate(
            "analyze_elf",
            components,
            {
                "binary_path": binary_path,
                "security": components["checksec"]["parsed"],
                "sections": components["sections"]["parsed"],
                "plt": components["plt"]["parsed"],
            },
            self.gdb.get_state(),
        )

    def inspect_got_plt(
        self,
        symbol_filter: Optional[str] = None,
        include_got: bool = True,
        include_plt: bool = True,
        include_readonly: bool = False,
        all_objects: bool = False,
        all_plt_symbols: bool = False,
    ) -> Dict[str, Any]:
        if not include_got and not include_plt:
            raise ValueError("At least one of include_got or include_plt must be true")
        validated_filter = (
            self._validate_symbol_filter(symbol_filter) if symbol_filter else None
        )
        components: Dict[str, Dict[str, Any]] = {}
        if include_got:
            components["got"] = self.got(
                symbol_filter=validated_filter,
                include_readonly=include_readonly,
                all_objects=all_objects,
            )
        if include_plt:
            components["plt"] = self.plt(all_symbols=all_plt_symbols)
            if validated_filter:
                needle = validated_filter.lower()
                parsed_plt = components["plt"]["parsed"]
                filtered_sections = []
                for section in parsed_plt["sections"]:
                    entries = [
                        entry
                        for entry in section["entries"]
                        if needle in entry["symbol"].lower()
                    ]
                    if entries:
                        filtered_sections.append({**section, "entries": entries})
                parsed_plt["sections"] = filtered_sections
                parsed_plt["section_count"] = len(filtered_sections)
                parsed_plt["entry_count"] = sum(
                    len(section["entries"]) for section in filtered_sections
                )
        return self._aggregate(
            "inspect_got_plt",
            components,
            {name: item["parsed"] for name, item in components.items()},
            self.gdb.get_state(),
        )

    def inspect_libc(self, include_link_map: bool = True) -> Dict[str, Any]:
        components = {"libc": self.libc_info()}
        if include_link_map:
            components["link_map"] = self.link_map()
        return self._aggregate(
            "inspect_libc",
            components,
            {name: item["parsed"] for name, item in components.items()},
            self.gdb.get_state(),
        )
