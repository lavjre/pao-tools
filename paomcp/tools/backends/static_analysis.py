"""Typed, cacheable static analysis for workspace ELF binaries."""

from __future__ import annotations

import copy
import hashlib
import json
import mmap
import os
import re
import shutil
import sys
import threading
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import elftools
from elftools.common.exceptions import ELFError
from elftools.elf.descriptions import describe_reloc_type
from elftools.elf.dynamic import DynamicSection, DynamicSegment
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection, RelrRelocationSection
from elftools.elf.segments import NoteSegment
from elftools.elf.sections import NoteSection, SymbolTableSection

from paomcp.operations import OperationCancelledError

ANALYZER_NAME = "pyelftools"
ANALYZER_VERSION = str(getattr(elftools, "__version__", "unknown"))
PARSER_VERSION = "1"
DEFAULT_CACHE_ENTRIES = 8
DEFAULT_CACHE_BYTES = 64 * 1024 * 1024
HASH_CHUNK_BYTES = 1024 * 1024
STREAM_STRING_CHUNK_BYTES = 64 * 1024
STREAM_CAPACITY_CHECK_BYTES = 1024 * 1024


def _hex(value: Any) -> str:
    """Return an integer-like ELF value without losing JSON precision."""
    return hex(int(value))


def _text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _primitive(value: Any) -> Any:
    """Convert pyelftools containers into JSON-safe primitive values."""
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return str(value)


class StaticBinaryAnalyzer:
    """Analyze immutable ELF content with a bounded thread-safe LRU cache."""

    def __init__(
        self,
        max_cache_entries: Optional[int] = None,
        max_cache_bytes: Optional[int] = None,
    ) -> None:
        configured = max_cache_entries
        if configured is None:
            configured = int(
                os.environ.get(
                    "PAO_STATIC_ANALYSIS_CACHE_ENTRIES", str(DEFAULT_CACHE_ENTRIES)
                )
            )
        if configured <= 0:
            raise ValueError("Static-analysis cache size must be greater than zero")
        configured_bytes = max_cache_bytes
        if configured_bytes is None:
            configured_bytes = int(
                os.environ.get(
                    "PAO_STATIC_ANALYSIS_CACHE_BYTES", str(DEFAULT_CACHE_BYTES)
                )
            )
        if configured_bytes <= 0:
            raise ValueError(
                "Static-analysis cache byte budget must be greater than zero"
            )
        self.max_cache_entries = configured
        self.max_cache_bytes = configured_bytes
        self._core_cache_budget = max(1, configured_bytes // 2)
        self._strings_cache_budget = max(1, configured_bytes - self._core_cache_budget)
        self._cache: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
        self._cache_weights: Dict[Tuple[str, str, str], int] = {}
        self._cache_bytes = 0
        self._strings_cache: (
            "OrderedDict[Tuple[str, str, str, int], List[Dict[str, Any]]]"
        ) = OrderedDict()
        self._strings_cache_weights: Dict[Tuple[str, str, str, int], int] = {}
        self._strings_cache_bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _deep_size(value: Any, seen: Optional[set[int]] = None) -> int:
        """Estimate retained Python heap size without serializing another full copy."""
        if seen is None:
            seen = set()
        identity = id(value)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(value)
        if isinstance(value, dict):
            size += sum(
                StaticBinaryAnalyzer._deep_size(key, seen)
                + StaticBinaryAnalyzer._deep_size(item, seen)
                for key, item in value.items()
            )
        elif isinstance(value, (list, tuple, set, frozenset)):
            size += sum(StaticBinaryAnalyzer._deep_size(item, seen) for item in value)
        return size

    @staticmethod
    def _validate_limit(limit: Optional[int]) -> Optional[int]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero when provided")
        return limit

    @staticmethod
    def _validate_min_string_length(min_length: int) -> int:
        if min_length <= 0 or min_length > 65536:
            raise ValueError("min_length must be between 1 and 65536")
        return min_length

    @staticmethod
    def _require_file(binary_path: str) -> str:
        resolved = os.path.abspath(binary_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"binary_path not found: '{resolved}'")
        return resolved

    @staticmethod
    def _fingerprint(
        binary_path: str,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        StaticBinaryAnalyzer._raise_if_cancelled(cancellation_event)
        before = os.stat(binary_path)
        digest = hashlib.sha256()
        with open(binary_path, "rb") as binary_file:
            magic = binary_file.read(4)
            if magic != b"\x7fELF":
                raise ValueError(f"binary_path is not an ELF file: '{binary_path}'")
            digest.update(magic)
            while True:
                block = binary_file.read(HASH_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
                StaticBinaryAnalyzer._raise_if_cancelled(cancellation_event)
        StaticBinaryAnalyzer._raise_if_cancelled(cancellation_event)
        stat = os.stat(binary_path)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise RuntimeError("binary_path changed while it was being fingerprinted")
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": str(stat.st_size),
            "modified_ns": str(stat.st_mtime_ns),
            "_device": str(stat.st_dev),
            "_inode": str(stat.st_ino),
        }

    @staticmethod
    def _public_fingerprint(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        """Strip filesystem identity fields used only for stability checks."""
        return {
            key: value for key, value in fingerprint.items() if not key.startswith("_")
        }

    @staticmethod
    def _cache_key(sha256: str) -> Tuple[str, str, str]:
        return sha256, ANALYZER_VERSION, PARSER_VERSION

    def _cache_get(self, key: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            self._cache.move_to_end(key)
            return copy.deepcopy(cached)

    def _cache_put(self, key: Tuple[str, str, str], value: Dict[str, Any]) -> None:
        weight = self._deep_size(value)
        if weight > self._core_cache_budget:
            return
        with self._lock:
            previous_weight = self._cache_weights.pop(key, 0)
            self._cache_bytes -= previous_weight
            self._cache[key] = copy.deepcopy(value)
            self._cache_weights[key] = weight
            self._cache_bytes += weight
            self._cache.move_to_end(key)
            while (
                len(self._cache) > self.max_cache_entries
                or self._cache_bytes > self._core_cache_budget
            ):
                evicted_key, _ = self._cache.popitem(last=False)
                self._cache_bytes -= self._cache_weights.pop(evicted_key, 0)

    def _strings_cache_get(
        self, key: Tuple[str, str, str, int], limit: Optional[int]
    ) -> Optional[Tuple[List[Dict[str, Any]], int]]:
        with self._lock:
            cached = self._strings_cache.get(key)
            if cached is None:
                return None
            self._strings_cache.move_to_end(key)
            delivered = cached if limit is None else cached[:limit]
            return copy.deepcopy(delivered), len(cached)

    def _strings_cache_put(
        self,
        key: Tuple[str, str, str, int],
        value: List[Dict[str, Any]],
    ) -> None:
        weight = self._deep_size(value)
        if weight > self._strings_cache_budget:
            return
        with self._lock:
            previous_weight = self._strings_cache_weights.pop(key, 0)
            self._strings_cache_bytes -= previous_weight
            self._strings_cache[key] = copy.deepcopy(value)
            self._strings_cache_weights[key] = weight
            self._strings_cache_bytes += weight
            self._strings_cache.move_to_end(key)
            while (
                len(self._strings_cache) > self.max_cache_entries
                or self._strings_cache_bytes > self._strings_cache_budget
            ):
                evicted_key, _ = self._strings_cache.popitem(last=False)
                self._strings_cache_bytes -= self._strings_cache_weights.pop(
                    evicted_key, 0
                )

    @staticmethod
    def _header(elf: ELFFile) -> Dict[str, Any]:
        header = elf.header
        ident = header["e_ident"]
        return {
            "class_bits": elf.elfclass,
            "endianness": "little" if elf.little_endian else "big",
            "os_abi": str(ident["EI_OSABI"]),
            "abi_version": int(ident["EI_ABIVERSION"]),
            "type": str(header["e_type"]),
            "machine": str(header["e_machine"]),
            "version": str(header["e_version"]),
            "entry": _hex(header["e_entry"]),
            "program_header_offset": _hex(header["e_phoff"]),
            "section_header_offset": _hex(header["e_shoff"]),
            "flags": _hex(header["e_flags"]),
            "header_size_bytes": int(header["e_ehsize"]),
            "program_header_entry_size_bytes": int(header["e_phentsize"]),
            "program_header_count": int(header["e_phnum"]),
            "section_header_entry_size_bytes": int(header["e_shentsize"]),
            "section_header_count": int(header["e_shnum"]),
            "section_name_table_index": int(header["e_shstrndx"]),
        }

    @staticmethod
    def _segment_permissions(flags: int) -> Dict[str, bool]:
        return {
            "readable": bool(flags & 4),
            "writable": bool(flags & 2),
            "executable": bool(flags & 1),
        }

    @classmethod
    def _program_headers(cls, elf: ELFFile) -> List[Dict[str, Any]]:
        records = []
        for index, segment in enumerate(elf.iter_segments()):
            flags = int(segment["p_flags"])
            record: Dict[str, Any] = {
                "index": index,
                "type": str(segment["p_type"]),
                "flags": _hex(flags),
                **cls._segment_permissions(flags),
                "offset": _hex(segment["p_offset"]),
                "virtual_address": _hex(segment["p_vaddr"]),
                "physical_address": _hex(segment["p_paddr"]),
                "file_size": _hex(segment["p_filesz"]),
                "memory_size": _hex(segment["p_memsz"]),
                "alignment": _hex(segment["p_align"]),
            }
            if str(segment["p_type"]) == "PT_INTERP":
                try:
                    record["interpreter"] = segment.get_interp_name()
                except (AttributeError, ELFError):
                    record["interpreter"] = None
            records.append(record)
        return records

    @staticmethod
    def _section_permissions(flags: int) -> Dict[str, bool]:
        return {
            "allocated": bool(flags & 0x2),
            "writable": bool(flags & 0x1),
            "executable": bool(flags & 0x4),
        }

    @classmethod
    def _sections(cls, elf: ELFFile) -> List[Dict[str, Any]]:
        records = []
        for index, section in enumerate(elf.iter_sections()):
            flags = int(section["sh_flags"])
            records.append(
                {
                    "index": index,
                    "name": section.name,
                    "type": str(section["sh_type"]),
                    "flags": _hex(flags),
                    **cls._section_permissions(flags),
                    "address": _hex(section["sh_addr"]),
                    "offset": _hex(section["sh_offset"]),
                    "size": _hex(section["sh_size"]),
                    "entry_size": _hex(section["sh_entsize"]),
                    "alignment": _hex(section["sh_addralign"]),
                    "link": int(section["sh_link"]),
                    "info": int(section["sh_info"]),
                }
            )
        return records

    @staticmethod
    def _symbol_record(
        table_name: str,
        index: int,
        symbol: Any,
    ) -> Dict[str, Any]:
        entry = symbol.entry
        section_index = entry["st_shndx"]
        return {
            "table": table_name,
            "index": index,
            "name": symbol.name,
            "value": _hex(entry["st_value"]),
            "size": _hex(entry["st_size"]),
            "binding": str(entry["st_info"]["bind"]),
            "type": str(entry["st_info"]["type"]),
            "visibility": str(entry["st_other"]["visibility"]),
            "section_index": (
                int(section_index)
                if isinstance(section_index, int)
                else str(section_index)
            ),
        }

    @classmethod
    def _symbols(cls, elf: ELFFile) -> Dict[str, Any]:
        all_symbols: List[Dict[str, Any]] = []
        imports: List[Dict[str, Any]] = []
        exports: List[Dict[str, Any]] = []
        import_keys = set()
        export_keys = set()
        for section in elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            is_dynamic = section.name == ".dynsym"
            for index, symbol in enumerate(section.iter_symbols()):
                record = cls._symbol_record(section.name, index, symbol)
                all_symbols.append(record)
                if not symbol.name:
                    continue
                undefined = record["section_index"] == "SHN_UNDEF"
                key = (record["name"], record["type"], record["binding"])
                if undefined and is_dynamic and key not in import_keys:
                    imports.append(copy.deepcopy(record))
                    import_keys.add(key)
                elif (
                    is_dynamic
                    and record["binding"] in ("STB_GLOBAL", "STB_WEAK")
                    and record["visibility"] in ("STV_DEFAULT", "STV_PROTECTED")
                    and key not in export_keys
                ):
                    exports.append(copy.deepcopy(record))
                    export_keys.add(key)
        return {
            "count": len(all_symbols),
            "symbols": all_symbols,
            "import_count": len(imports),
            "imports": imports,
            "export_count": len(exports),
            "exports": exports,
        }

    @classmethod
    def _relocations(cls, elf: ELFFile) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for section in elf.iter_sections():
            if isinstance(section, RelrRelocationSection):
                for index, relocation in enumerate(section.iter_relocations()):
                    records.append(
                        {
                            "section": section.name,
                            "index": index,
                            "offset": _hex(relocation.entry["r_offset"]),
                            "type": "SHT_RELR",
                            "type_name": "relative",
                            "symbol_index": None,
                            "symbol": None,
                        }
                    )
                continue
            if not isinstance(section, RelocationSection):
                continue
            symbol_table = None
            linked_index = int(section["sh_link"])
            if linked_index:
                candidate = elf.get_section(linked_index)
                if isinstance(candidate, SymbolTableSection):
                    symbol_table = candidate
            for index, relocation in enumerate(section.iter_relocations()):
                entry = relocation.entry
                symbol_index = int(entry["r_info_sym"])
                symbol_name = None
                if (
                    symbol_table is not None
                    and symbol_index < symbol_table.num_symbols()
                ):
                    symbol_name = symbol_table.get_symbol(symbol_index).name or None
                relocation_type = int(entry["r_info_type"])
                try:
                    type_name = describe_reloc_type(relocation_type, elf)
                except (KeyError, TypeError, ValueError):
                    type_name = str(relocation_type)
                record: Dict[str, Any] = {
                    "section": section.name,
                    "index": index,
                    "offset": _hex(entry["r_offset"]),
                    "type": relocation_type,
                    "type_name": type_name,
                    "symbol_index": symbol_index,
                    "symbol": symbol_name,
                }
                if "r_addend" in entry:
                    record["addend"] = int(entry["r_addend"])
                records.append(record)
        return records

    @staticmethod
    def _dynamic(elf: ELFFile) -> Dict[str, Any]:
        needed: List[str] = []
        rpath: List[str] = []
        runpath: List[str] = []
        soname = None
        tags: List[Dict[str, Any]] = []
        sources: List[Any] = [
            section
            for section in elf.iter_sections()
            if isinstance(section, DynamicSection)
        ]
        if not sources:
            sources = [
                segment
                for segment in elf.iter_segments()
                if isinstance(segment, DynamicSegment)
            ]
        for source in sources:
            source_name = getattr(source, "name", None) or "PT_DYNAMIC"
            for index, tag in enumerate(source.iter_tags()):
                tag_name = str(tag.entry.d_tag)
                record: Dict[str, Any] = {
                    "source": source_name,
                    "index": index,
                    "tag": tag_name,
                    "value": _hex(tag.entry.d_val),
                }
                if tag_name == "DT_NEEDED":
                    value = str(tag.needed)
                    needed.append(value)
                    record["string"] = value
                elif tag_name == "DT_RPATH":
                    value = str(tag.rpath)
                    rpath.extend(value.split(":"))
                    record["string"] = value
                elif tag_name == "DT_RUNPATH":
                    value = str(tag.runpath)
                    runpath.extend(value.split(":"))
                    record["string"] = value
                elif tag_name == "DT_SONAME":
                    soname = str(tag.soname)
                    record["string"] = soname
                tags.append(record)
        return {
            "needed": needed,
            "rpath": rpath,
            "runpath": runpath,
            "soname": soname,
            "tag_count": len(tags),
            "tags": tags,
        }

    @staticmethod
    def _note_description(value: Any) -> str:
        if isinstance(value, bytes):
            return value.hex()
        return re.sub(r"[^0-9a-fA-F]", "", str(value)).lower()

    @classmethod
    def _build_id(cls, elf: ELFFile) -> Optional[str]:
        for section in elf.iter_sections():
            if not isinstance(section, NoteSection):
                continue
            for note in section.iter_notes():
                if str(note["n_type"]) == "NT_GNU_BUILD_ID":
                    return cls._note_description(note["n_desc"])
        for segment in elf.iter_segments():
            if not isinstance(segment, NoteSegment):
                continue
            for note in segment.iter_notes():
                if str(note["n_type"]) == "NT_GNU_BUILD_ID":
                    return cls._note_description(note["n_desc"])
        return None

    @staticmethod
    def _debug_link(elf: ELFFile) -> Optional[Dict[str, Any]]:
        section = elf.get_section_by_name(".gnu_debuglink")
        if section is None:
            return None
        data = section.data()
        terminator = data.find(b"\x00")
        if terminator < 0:
            return {"file": data.decode("utf-8", errors="replace"), "crc32": None}
        file_name = data[:terminator].decode("utf-8", errors="replace")
        crc_offset = (terminator + 4) & ~3
        crc = None
        if crc_offset + 4 <= len(data):
            crc_bytes = data[crc_offset : crc_offset + 4]
            crc_value = (
                int.from_bytes(crc_bytes, "little")
                if elf.little_endian
                else int.from_bytes(crc_bytes, "big")
            )
            crc = _hex(crc_value)
        return {"file": file_name, "crc32": crc}

    @staticmethod
    def _die_attr(die: Any, name: str) -> Any:
        attribute = die.attributes.get(name)
        return _primitive(attribute.value) if attribute is not None else None

    @classmethod
    def _dwarf(cls, elf: ELFFile) -> Dict[str, Any]:
        sections = [
            {
                "name": section.name,
                "offset": _hex(section["sh_offset"]),
                "size": _hex(section["sh_size"]),
            }
            for section in elf.iter_sections()
            if section.name.startswith((".debug_", ".zdebug_"))
        ]
        present = elf.has_dwarf_info(strict=True)
        compilation_units: List[Dict[str, Any]] = []
        warning = None
        if present:
            try:
                dwarf_info = elf.get_dwarf_info(follow_links=False)
                for cu in dwarf_info.iter_CUs():
                    top_die = cu.get_top_DIE()
                    record = {
                        "offset": _hex(cu.cu_offset),
                        "unit_length": _hex(cu["unit_length"]),
                        "version": int(cu["version"]),
                        "address_size_bytes": int(cu["address_size"]),
                        "name": cls._die_attr(top_die, "DW_AT_name"),
                        "compilation_directory": cls._die_attr(
                            top_die, "DW_AT_comp_dir"
                        ),
                        "producer": cls._die_attr(top_die, "DW_AT_producer"),
                        "language": cls._die_attr(top_die, "DW_AT_language"),
                    }
                    if "unit_type" in cu.header:
                        record["unit_type"] = str(cu["unit_type"])
                    compilation_units.append(record)
            except (ELFError, KeyError, TypeError, ValueError) as exc:
                warning = f"DWARF metadata could not be decoded: {exc}"
        result: Dict[str, Any] = {
            "present": present,
            "section_count": len(sections),
            "sections": sections,
            "compilation_unit_count": len(compilation_units),
            "compilation_units": compilation_units,
        }
        if warning:
            result["warning"] = warning
        return result

    @classmethod
    def _parse_core(
        cls,
        binary_path: str,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        try:
            with open(binary_path, "rb") as binary_file:
                elf = ELFFile(binary_file)
                cls._raise_if_cancelled(cancellation_event)
                header = cls._header(elf)
                program_headers = cls._program_headers(elf)
                cls._raise_if_cancelled(cancellation_event)
                sections = cls._sections(elf)
                cls._raise_if_cancelled(cancellation_event)
                symbol_data = cls._symbols(elf)
                cls._raise_if_cancelled(cancellation_event)
                relocations = cls._relocations(elf)
                cls._raise_if_cancelled(cancellation_event)
                dynamic = cls._dynamic(elf)
                cls._raise_if_cancelled(cancellation_event)
                dwarf = cls._dwarf(elf)
                cls._raise_if_cancelled(cancellation_event)
                return {
                    "elf_header": header,
                    "program_headers": {
                        "count": len(program_headers),
                        "entries": program_headers,
                    },
                    "sections": {"count": len(sections), "entries": sections},
                    **symbol_data,
                    "relocations": {
                        "count": len(relocations),
                        "entries": relocations,
                    },
                    "dynamic": dynamic,
                    "build_id": cls._build_id(elf),
                    "debug_link": cls._debug_link(elf),
                    "dwarf": dwarf,
                }
        except ELFError as exc:
            raise ValueError(f"Invalid or unsupported ELF file: {exc}") from exc

    def _core_for(
        self,
        binary_path: str,
        fingerprint: Dict[str, Any],
        *,
        refresh: bool,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Tuple[Dict[str, Any], bool, Tuple[str, str, str]]:
        key = self._cache_key(str(fingerprint["sha256"]))
        before = os.stat(binary_path)
        if (
            str(before.st_dev),
            str(before.st_ino),
            before.st_size,
            before.st_mtime_ns,
        ) != (
            fingerprint["_device"],
            fingerprint["_inode"],
            int(str(fingerprint["size_bytes"])),
            int(str(fingerprint["modified_ns"])),
        ):
            raise RuntimeError("binary_path changed after it was fingerprinted")
        if not refresh:
            cached = self._cache_get(key)
            if cached is not None:
                self._raise_if_cancelled(cancellation_event)
                return cached, True, key
        parsed = self._parse_core(binary_path, cancellation_event)
        after_fingerprint = self._fingerprint(binary_path, cancellation_event)
        if after_fingerprint["sha256"] != fingerprint["sha256"]:
            raise RuntimeError("binary_path changed while static analysis was running")
        self._cache_put(key, parsed)
        return parsed, False, key

    def _all_strings(
        self,
        binary_path: str,
        fingerprint: Dict[str, Any],
        *,
        min_length: int,
        limit: Optional[int],
        refresh: bool,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Tuple[List[Dict[str, Any]], int, bool]:
        key = (
            str(fingerprint["sha256"]),
            ANALYZER_VERSION,
            PARSER_VERSION,
            min_length,
        )
        before = os.stat(binary_path)
        if (
            str(before.st_dev),
            str(before.st_ino),
            before.st_size,
            before.st_mtime_ns,
        ) != (
            fingerprint["_device"],
            fingerprint["_inode"],
            int(str(fingerprint["size_bytes"])),
            int(str(fingerprint["modified_ns"])),
        ):
            raise RuntimeError("binary_path changed after it was fingerprinted")
        if not refresh:
            cached = self._strings_cache_get(key, limit)
            if cached is not None:
                self._raise_if_cancelled(cancellation_event)
                cached_records, cached_total_count = cached
                return cached_records, cached_total_count, True

        pattern = re.compile(rb"[\x20-\x7e]{%d,}" % min_length)
        records: List[Dict[str, Any]] = []
        total_count = 0
        with open(binary_path, "rb") as binary_file:
            with mmap.mmap(binary_file.fileno(), 0, access=mmap.ACCESS_READ) as view:
                for match in pattern.finditer(view):
                    self._raise_if_cancelled(cancellation_event)
                    total_count += 1
                    if limit is not None and len(records) >= limit:
                        continue
                    value = match.group(0)
                    records.append(
                        {
                            "offset": _hex(match.start()),
                            "length": len(value),
                            "encoding": "ascii",
                            "value": value.decode("ascii"),
                        }
                    )
        after_fingerprint = self._fingerprint(binary_path, cancellation_event)
        if after_fingerprint["sha256"] != fingerprint["sha256"]:
            raise RuntimeError("binary_path changed while strings were being scanned")
        if limit is None:
            self._strings_cache_put(key, records)
        return records, total_count, False

    @staticmethod
    def _limited(
        records: List[Dict[str, Any]],
        limit: Optional[int],
        *,
        total_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        delivered = records if limit is None else records[:limit]
        resolved_total = len(records) if total_count is None else total_count
        return {
            "total_count": resolved_total,
            "returned_count": len(delivered),
            "limited_by_caller": limit is not None and resolved_total > len(delivered),
            "limit": limit,
            "entries": delivered,
        }

    def extract_strings(
        self,
        binary_path: str,
        *,
        min_length: int = 4,
        limit: Optional[int] = None,
        refresh: bool = False,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Extract printable ASCII strings with exact file offsets."""
        min_length = self._validate_min_string_length(min_length)
        limit = self._validate_limit(limit)
        binary_path = self._require_file(binary_path)
        fingerprint = self._fingerprint(binary_path, cancellation_event)
        strings, total_count, cache_hit = self._all_strings(
            binary_path,
            fingerprint,
            min_length=min_length,
            limit=limit,
            refresh=refresh,
            cancellation_event=cancellation_event,
        )
        return {
            "success": True,
            "analyzer": ANALYZER_NAME,
            "analyzer_version": ANALYZER_VERSION,
            "parser_version": PARSER_VERSION,
            "binary_path": binary_path,
            "fingerprint": self._public_fingerprint(fingerprint),
            "cache": {
                "hit": cache_hit,
                "refreshed": refresh,
                "max_entries": self.max_cache_entries,
                "max_bytes": self.max_cache_bytes,
            },
            "min_length": min_length,
            "strings": self._limited(strings, limit, total_count=total_count),
        }

    @staticmethod
    def _raise_if_cancelled(
        cancellation_event: Optional[threading.Event],
    ) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise OperationCancelledError("Static analysis was cancelled")

    @classmethod
    def _write_streamed_string_record(
        cls,
        output_file: Any,
        view: mmap.mmap,
        start: int,
        end: int,
        cancellation_event: Optional[threading.Event] = None,
        capacity_check: Optional[Callable[[], None]] = None,
    ) -> None:
        """Write one NDJSON record without copying a potentially huge match."""
        output_file.write(
            '{"offset":'
            + json.dumps(_hex(start))
            + ',"length":'
            + str(end - start)
            + ',"encoding":"ascii","value":"'
        )
        cursor = start
        while cursor < end:
            cls._raise_if_cancelled(cancellation_event)
            chunk_end = min(end, cursor + STREAM_STRING_CHUNK_BYTES)
            chunk = view[cursor:chunk_end].decode("ascii")
            output_file.write(json.dumps(chunk, ensure_ascii=True)[1:-1])
            if capacity_check is not None:
                capacity_check()
            cursor = chunk_end
        output_file.write('"}\n')

    def stream_strings_to_file(
        self,
        binary_path: str,
        output_path: str,
        *,
        min_length: int = 4,
        cancellation_event: Optional[threading.Event] = None,
        minimum_free_bytes: int = 0,
    ) -> Dict[str, Any]:
        """Stream every printable string as NDJSON using bounded working memory."""
        min_length = self._validate_min_string_length(min_length)
        binary_path = self._require_file(binary_path)
        fingerprint = self._fingerprint(binary_path, cancellation_event)
        self._raise_if_cancelled(cancellation_event)
        pattern = re.compile(rb"[\x20-\x7e]{%d,}" % min_length)
        total_count = 0
        last_capacity_check = 0
        try:
            with (
                open(binary_path, "rb") as binary_file,
                open(output_path, "w", encoding="utf-8", newline="\n") as output_file,
                mmap.mmap(binary_file.fileno(), 0, access=mmap.ACCESS_READ) as view,
            ):

                def ensure_output_capacity() -> None:
                    nonlocal last_capacity_check
                    current_size = output_file.tell()
                    if current_size - last_capacity_check < STREAM_CAPACITY_CHECK_BYTES:
                        return
                    output_file.flush()
                    free_bytes = shutil.disk_usage(os.path.dirname(output_path)).free
                    if free_bytes < minimum_free_bytes + current_size:
                        raise OSError(
                            "Insufficient free disk for lossless string artifact"
                        )
                    last_capacity_check = current_size

                for match in pattern.finditer(view):
                    self._raise_if_cancelled(cancellation_event)
                    self._write_streamed_string_record(
                        output_file,
                        view,
                        match.start(),
                        match.end(),
                        cancellation_event,
                        ensure_output_capacity,
                    )
                    total_count += 1
                    ensure_output_capacity()
                output_file.flush()
                os.fsync(output_file.fileno())
        except Exception:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
            raise
        after_fingerprint = self._fingerprint(binary_path, cancellation_event)
        if after_fingerprint["sha256"] != fingerprint["sha256"]:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass
            raise RuntimeError("binary_path changed while strings were being scanned")
        return {
            "success": True,
            "analyzer": ANALYZER_NAME,
            "analyzer_version": ANALYZER_VERSION,
            "parser_version": PARSER_VERSION,
            "binary_path": binary_path,
            "fingerprint": self._public_fingerprint(fingerprint),
            "cache": {
                "hit": False,
                "refreshed": False,
                "max_entries": self.max_cache_entries,
                "max_bytes": self.max_cache_bytes,
                "streamed": True,
            },
            "min_length": min_length,
            "total_count": total_count,
            "output_path": output_path,
            "output_size_bytes": os.path.getsize(output_path),
        }

    def analyze(
        self,
        binary_path: str,
        *,
        include_strings: bool = True,
        min_string_length: int = 4,
        string_limit: Optional[int] = None,
        refresh: bool = False,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Return a complete typed ELF model and optional printable strings."""
        min_string_length = self._validate_min_string_length(min_string_length)
        string_limit = self._validate_limit(string_limit)
        binary_path = self._require_file(binary_path)
        fingerprint = self._fingerprint(binary_path, cancellation_event)
        parsed, cache_hit, key = self._core_for(
            binary_path,
            fingerprint,
            refresh=refresh,
            cancellation_event=cancellation_event,
        )
        public_fingerprint = self._public_fingerprint(fingerprint)
        parsed["fingerprint"] = copy.deepcopy(public_fingerprint)
        warnings = []
        dwarf_warning = parsed.get("dwarf", {}).get("warning")
        if dwarf_warning:
            warnings.append(dwarf_warning)
        strings_cache_hit = None
        if include_strings:
            strings, total_count, strings_cache_hit = self._all_strings(
                binary_path,
                fingerprint,
                min_length=min_string_length,
                limit=string_limit,
                refresh=refresh,
                cancellation_event=cancellation_event,
            )
            parsed["strings"] = {
                "min_length": min_string_length,
                **self._limited(strings, string_limit, total_count=total_count),
            }
        return {
            "success": True,
            "analyzer": ANALYZER_NAME,
            "analyzer_version": ANALYZER_VERSION,
            "parser_version": PARSER_VERSION,
            "binary_path": binary_path,
            "fingerprint": public_fingerprint,
            "cache": {
                "hit": cache_hit,
                "strings_hit": strings_cache_hit,
                "refreshed": refresh,
                "key": ":".join(key),
                "max_entries": self.max_cache_entries,
                "max_bytes": self.max_cache_bytes,
                "retained_bytes": self._cache_bytes + self._strings_cache_bytes,
            },
            "parsed": parsed,
            "warnings": warnings,
        }

    @staticmethod
    def _mapping_diff(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        changed = {}
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                changed[key] = {"left": left.get(key), "right": right.get(key)}
        return changed

    @classmethod
    def _keyed_records_diff(
        cls,
        left_records: Iterable[Dict[str, Any]],
        right_records: Iterable[Dict[str, Any]],
        *,
        key_name: str,
    ) -> Dict[str, Any]:
        left: Dict[str, List[Dict[str, Any]]] = {}
        right: Dict[str, List[Dict[str, Any]]] = {}
        for record in left_records:
            left.setdefault(str(record[key_name]), []).append(record)
        for record in right_records:
            right.setdefault(str(record[key_name]), []).append(record)
        common = sorted(set(left) & set(right))
        changed = {}
        for key in common:
            if left[key] == right[key]:
                continue
            if len(left[key]) == 1 and len(right[key]) == 1:
                changed[key] = cls._mapping_diff(left[key][0], right[key][0])
            else:
                changed[key] = {"left": left[key], "right": right[key]}
        return {
            "added": [
                record
                for key in sorted(set(right) - set(left))
                for record in right[key]
            ],
            "removed": [
                record for key in sorted(set(left) - set(right)) for record in left[key]
            ],
            "changed": changed,
        }

    @staticmethod
    def _record_multiset_diff(
        left_records: Iterable[Dict[str, Any]],
        right_records: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Diff record multisets without discarding duplicate ELF entries."""
        left_serialized = [
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in left_records
        ]
        right_serialized = [
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            for record in right_records
        ]
        left_counts = Counter(left_serialized)
        right_counts = Counter(right_serialized)
        added: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        for serialized, count in sorted((right_counts - left_counts).items()):
            added.extend(json.loads(serialized) for _ in range(count))
        for serialized, count in sorted((left_counts - right_counts).items()):
            removed.extend(json.loads(serialized) for _ in range(count))
        return {"added": added, "removed": removed}

    @staticmethod
    def _names(records: Iterable[Dict[str, Any]]) -> List[str]:
        return sorted({str(record["name"]) for record in records if record.get("name")})

    @staticmethod
    def _set_diff(left: Iterable[str], right: Iterable[str]) -> Dict[str, List[str]]:
        left_set = set(left)
        right_set = set(right)
        return {
            "added": sorted(right_set - left_set),
            "removed": sorted(left_set - right_set),
            "unchanged": sorted(left_set & right_set),
        }

    def compare(
        self,
        left_path: str,
        right_path: str,
        *,
        refresh: bool = False,
        cancellation_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Compare two complete ELF models without truncating differences."""
        self._raise_if_cancelled(cancellation_event)
        left = self.analyze(
            left_path,
            include_strings=False,
            refresh=refresh,
            cancellation_event=cancellation_event,
        )
        right = self.analyze(
            right_path,
            include_strings=False,
            refresh=refresh,
            cancellation_event=cancellation_event,
        )
        self._raise_if_cancelled(cancellation_event)
        left_parsed = left["parsed"]
        right_parsed = right["parsed"]
        left_dynamic = left_parsed["dynamic"]
        right_dynamic = right_parsed["dynamic"]
        differences = {
            "fingerprint": self._mapping_diff(
                left["fingerprint"], right["fingerprint"]
            ),
            "elf_header": self._mapping_diff(
                left_parsed["elf_header"], right_parsed["elf_header"]
            ),
            "sections": self._keyed_records_diff(
                left_parsed["sections"]["entries"],
                right_parsed["sections"]["entries"],
                key_name="name",
            ),
            "program_headers": self._keyed_records_diff(
                left_parsed["program_headers"]["entries"],
                right_parsed["program_headers"]["entries"],
                key_name="index",
            ),
            "symbols": self._record_multiset_diff(
                left_parsed["symbols"], right_parsed["symbols"]
            ),
            "relocations": self._record_multiset_diff(
                left_parsed["relocations"]["entries"],
                right_parsed["relocations"]["entries"],
            ),
            "imports": self._set_diff(
                self._names(left_parsed["imports"]),
                self._names(right_parsed["imports"]),
            ),
            "exports": self._set_diff(
                self._names(left_parsed["exports"]),
                self._names(right_parsed["exports"]),
            ),
            "needed_libraries": self._set_diff(
                left_dynamic["needed"], right_dynamic["needed"]
            ),
            "rpath": self._set_diff(left_dynamic["rpath"], right_dynamic["rpath"]),
            "runpath": self._set_diff(
                left_dynamic["runpath"], right_dynamic["runpath"]
            ),
            "build_id": {
                "left": left_parsed["build_id"],
                "right": right_parsed["build_id"],
                "changed": left_parsed["build_id"] != right_parsed["build_id"],
            },
            "dwarf": {
                "left_present": left_parsed["dwarf"]["present"],
                "right_present": right_parsed["dwarf"]["present"],
                "left_compilation_units": left_parsed["dwarf"][
                    "compilation_unit_count"
                ],
                "right_compilation_units": right_parsed["dwarf"][
                    "compilation_unit_count"
                ],
            },
            "counts": {
                name: {
                    "left": left_parsed[name]["count"],
                    "right": right_parsed[name]["count"],
                }
                for name in ("sections", "program_headers", "relocations")
            },
        }
        self._raise_if_cancelled(cancellation_event)
        return {
            "success": True,
            "analyzer": ANALYZER_NAME,
            "analyzer_version": ANALYZER_VERSION,
            "parser_version": PARSER_VERSION,
            "identical": left["fingerprint"]["sha256"]
            == right["fingerprint"]["sha256"],
            "left": {
                "binary_path": left["binary_path"],
                "fingerprint": left["fingerprint"],
                "cache": left["cache"],
            },
            "right": {
                "binary_path": right["binary_path"],
                "fingerprint": right["fingerprint"],
                "cache": right["cache"],
            },
            "differences": differences,
        }
