# mypy: disable-error-code="attr-defined"
"""ELF core-file estimation and generation (split from crash.py)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from elftools.elf.sections import NoteSection
from pwnlib.util.cyclic import de_bruijn  # type: ignore[import-untyped]

from paomcp.artifacts import (
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    WriterNotQuiescentError,
)
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.foundation import (
    LiveInferiorError,
    LiveInferiorGuard,
    mapping_for,
    parse_int,
)
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

import logging

logger = logging.getLogger(__name__)

from paomcp.tools.backends.crash_schemas import (
    PARSER_VERSION,
    CRASH_SNAPSHOT_VERSION,
    DEFAULT_FATAL_SIGNALS,
    SAFE_SIGNAL_PATTERN,
    MAX_RUN_TIMEOUT_SECONDS,
    DEFAULT_RUN_TIMEOUT_SECONDS,
    DEFAULT_CORE_TIMEOUT_SECONDS,
    DEFAULT_MAX_CORE_DUMP_BYTES,
    MAX_CONFIGURED_CORE_DUMP_BYTES,
    MAX_STACK_BYTES,
    MAX_CYCLIC_LENGTH,
    MAX_CYCLIC_STACK_SCAN_BYTES,
    MAX_CYCLIC_SCAN_MATCHES,
    MAX_CYCLIC_VALUE_CHARS,
    MAX_CYCLIC_ALPHABET_CHARS,
    MAX_ELF_CORE_NOTE_ENTRIES,
    MAX_CRASH_HISTORY_ENTRIES,
    MAX_CRASH_HISTORY_BYTES,
    CRASH_CLASSIFICATION_SCHEMA,
    CRASH_FINGERPRINT_SCHEMA,
    CRASH_COMPARISON_SCHEMA,
)


def _record_error(message: str) -> Exception:
    """Build CrashRecordError with a deferred import (crash.py owns it).

    A top-level import would cycle: crash.py imports these mixins for the
    CrashTriage bases. Resolving at call time is always safe because the
    class (and its module) is fully loaded before any method runs.
    """

    from paomcp.tools.backends.crash import CrashRecordError  # noqa: PLC0415

    return CrashRecordError(message)


class CrashCoreMixin:
    """Mixin supplying CrashTriage.core behavior; never instantiated alone."""

    def estimate_core_size(self) -> Dict[str, Any]:
        """Estimate an upper bound from readable mappings before core generation."""
        mapping_result = self.inspection.memory_maps()
        mappings = mapping_result.get("parsed", {}).get("mappings", [])
        total = 1024 * 1024
        counted = 0
        saturated = False
        for item in mappings if isinstance(mappings, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                # Mappings may already carry ints (parsed) or hex strings.
                start_raw, end_raw = item.get("start"), item.get("end")
                start = (
                    int(start_raw)
                    if isinstance(start_raw, int)
                    else int(str(start_raw), 16)
                )
                end = (
                    int(end_raw) if isinstance(end_raw, int) else int(str(end_raw), 16)
                )
            except (TypeError, ValueError):
                continue
            mapping_size = end - start
            if mapping_size <= 0:
                continue
            counted += 1
            total = min(MAX_CONFIGURED_CORE_DUMP_BYTES, total + mapping_size)
            saturated = total == MAX_CONFIGURED_CORE_DUMP_BYTES
        return {
            "success": bool(mapping_result.get("success") and counted),
            "estimated_upper_bound_bytes": total,
            "mapping_count": counted,
            "estimate_saturated": saturated,
            "source": "pwndbg-cli",
            "raw_responses": mapping_result.get("raw_responses", []),
        }

    @staticmethod
    def _validate_elf_core(path: str) -> Dict[str, Any]:
        file_size = os.path.getsize(path)
        with open(path, "rb") as source:
            header = source.read(64)
        if len(header) < 24 or header[:4] != b"\x7fELF":
            raise CoreDumpGenerationError("Generated file is not an ELF file")
        elf_class = header[4]
        if elf_class not in {1, 2}:
            raise CoreDumpGenerationError("Generated ELF has an unsupported class")
        data_encoding = header[5]
        if data_encoding not in {1, 2}:
            raise CoreDumpGenerationError("Generated ELF has an unsupported byte order")
        if header[6] != 1:
            raise CoreDumpGenerationError(
                "Generated ELF has an unsupported ident version"
            )
        byteorder: Literal["little", "big"] = "little" if data_encoding == 1 else "big"
        elf_type = int.from_bytes(header[16:18], byteorder=byteorder)
        if elf_type != 4:
            raise CoreDumpGenerationError(
                f"Generated ELF type is {elf_type}, expected ET_CORE (4)"
            )
        if int.from_bytes(header[20:24], byteorder=byteorder) != 1:
            raise CoreDumpGenerationError("Generated ELF has an unsupported version")
        if elf_class == 2:  # ELF64
            ehsize = 64
            expected_phentsize = 56
            if len(header) < 64:
                raise CoreDumpGenerationError("Generated ELF64 header is truncated")
            e_phoff = int.from_bytes(header[32:40], byteorder=byteorder)
            e_ehsize = int.from_bytes(header[52:54], byteorder=byteorder)
            e_phentsize = int.from_bytes(header[54:56], byteorder=byteorder)
            e_phnum = int.from_bytes(header[56:58], byteorder=byteorder)
        else:  # ELF32
            ehsize = 52
            expected_phentsize = 32
            with open(path, "rb") as source:
                header32 = source.read(52)
            if len(header32) < 52:
                raise CoreDumpGenerationError("Generated ELF32 header is truncated")
            e_phoff = int.from_bytes(header32[28:32], byteorder=byteorder)
            e_ehsize = int.from_bytes(header32[40:42], byteorder=byteorder)
            e_phentsize = int.from_bytes(header32[42:44], byteorder=byteorder)
            e_phnum = int.from_bytes(header32[44:46], byteorder=byteorder)
        if e_ehsize != ehsize:
            raise CoreDumpGenerationError(
                f"Generated ELF header size is {e_ehsize}, expected {ehsize}"
            )
        if e_phentsize != expected_phentsize:
            raise CoreDumpGenerationError(
                "Generated core has an invalid program-header entry size"
            )
        if e_phnum < 1 or e_phoff < ehsize:
            raise CoreDumpGenerationError(
                "Generated core has no program-header table; the file is truncated"
            )
        required = e_phoff + e_phnum * e_phentsize
        if file_size < required:
            raise CoreDumpGenerationError(
                f"Generated core is {file_size} bytes but its program-header table "
                f"requires at least {required}; the file is truncated"
            )
        with open(path, "rb") as source:
            source.seek(e_phoff)
            program_headers = source.read(e_phnum * e_phentsize)
        types: List[int] = []
        load_segments = 0
        material_load_segments = 0
        note_segments = 0
        recognized_core_notes = 0
        note_entries = 0
        for index in range(e_phnum):
            entry = program_headers[index * e_phentsize : (index + 1) * e_phentsize]
            segment_type = int.from_bytes(entry[0:4], byteorder=byteorder)
            if elf_class == 2:
                segment_offset = int.from_bytes(entry[8:16], byteorder=byteorder)
                file_bytes = int.from_bytes(entry[32:40], byteorder=byteorder)
                memory_bytes = int.from_bytes(entry[40:48], byteorder=byteorder)
            else:
                segment_offset = int.from_bytes(entry[4:8], byteorder=byteorder)
                file_bytes = int.from_bytes(entry[16:20], byteorder=byteorder)
                memory_bytes = int.from_bytes(entry[20:24], byteorder=byteorder)
            if segment_offset > file_size or file_bytes > file_size - segment_offset:
                raise CoreDumpGenerationError(
                    f"Generated core program header {index} points outside the file"
                )
            if segment_type == 1:
                load_segments += 1
                if file_bytes > memory_bytes:
                    raise CoreDumpGenerationError(
                        f"Generated core PT_LOAD {index} has filesz greater than memsz"
                    )
                if file_bytes > 0 and memory_bytes > 0:
                    material_load_segments += 1
            elif segment_type == 4:
                note_segments += 1
                if file_bytes < 12:
                    raise CoreDumpGenerationError(
                        f"Generated core PT_NOTE {index} has no complete note header"
                    )
                note_end = segment_offset + file_bytes
                cursor = segment_offset
                with open(path, "rb") as source:
                    while cursor + 12 <= note_end:
                        note_entries += 1
                        if note_entries > MAX_ELF_CORE_NOTE_ENTRIES:
                            raise CoreDumpGenerationError(
                                "Generated core contains too many ELF note entries"
                            )
                        source.seek(cursor)
                        note_header = source.read(12)
                        if len(note_header) != 12:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} is truncated"
                            )
                        name_size = int.from_bytes(
                            note_header[0:4], byteorder=byteorder
                        )
                        description_size = int.from_bytes(
                            note_header[4:8], byteorder=byteorder
                        )
                        note_type = int.from_bytes(
                            note_header[8:12], byteorder=byteorder
                        )
                        if name_size == 0:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} contains an empty note name"
                            )
                        padded_name_size = (name_size + 3) & ~3
                        padded_description_size = (description_size + 3) & ~3
                        next_cursor = (
                            cursor + 12 + padded_name_size + padded_description_size
                        )
                        if next_cursor <= cursor or next_cursor > note_end:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} contains a malformed note"
                            )
                        # ELF core note names are tiny. Avoid allocating an
                        # attacker-controlled namesz while still advancing over
                        # a structurally valid, unrecognized note.
                        note_name = b""
                        if name_size <= 4096:
                            source.seek(cursor + 12)
                            note_name = source.read(name_size)
                            if len(note_name) != name_size:
                                raise CoreDumpGenerationError(
                                    f"Generated core PT_NOTE {index} is truncated"
                                )
                        if (
                            note_name.rstrip(b"\x00") == b"CORE"
                            and note_type == 1  # NT_PRSTATUS
                            and description_size > 0
                        ):
                            recognized_core_notes += 1
                        cursor = next_cursor
                    trailing = note_end - cursor
                    if trailing:
                        source.seek(cursor)
                        padding = source.read(trailing)
                        if trailing > 3 or padding != b"\x00" * trailing:
                            raise CoreDumpGenerationError(
                                f"Generated core PT_NOTE {index} has malformed trailing data"
                            )
            types.append(segment_type)
        if not note_segments or not load_segments:
            raise CoreDumpGenerationError(
                "Generated core must contain both PT_NOTE and PT_LOAD segments"
            )
        if not recognized_core_notes:
            raise CoreDumpGenerationError(
                "Generated core has no CORE/NT_PRSTATUS process-status note"
            )
        if not material_load_segments:
            raise CoreDumpGenerationError(
                "Generated core has no non-empty PT_LOAD memory segment"
            )
        return {
            "elf_class": 32 if elf_class == 1 else 64,
            "byte_order": byteorder,
            "elf_type": "ET_CORE",
            "program_header_count": e_phnum,
            "program_header_types": types,
            "note_segment_count": note_segments,
            "load_segment_count": load_segments,
            "material_load_segment_count": material_load_segments,
            "recognized_core_note_count": recognized_core_notes,
            "note_entry_count": note_entries,
            "size_bytes": file_size,
        }

    def generate_core_file(
        self,
        expected_crash_id: str,
        output_path: str,
        *,
        max_size_bytes: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        min_free_bytes: int = 0,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Use GDB CLI to generate one monitored core file at a trusted path."""
        record = self._require_crash_record(expected_crash_id, fatal_signals)
        try:
            LiveInferiorGuard.require_live(
                record,
                self.gdb.get_inferior_pid(),
                detail="Core capture requires the exact current stopped live crash",
            )
        except LiveInferiorError as exc:
            raise _record_error(str(exc)) from exc
        cap = (
            self.max_core_dump_bytes if max_size_bytes is None else int(max_size_bytes)
        )
        if not 1 <= cap <= self.max_core_dump_bytes:
            raise ValueError(
                f"max_size_bytes must be between 1 and {self.max_core_dump_bytes}"
            )
        timeout = self._timeout(
            timeout_seconds, self.default_core_timeout_seconds, name="timeout_seconds"
        )
        if min_free_bytes < 0:
            raise ValueError("min_free_bytes cannot be negative")
        if cancellation_check is not None:
            cancellation_check()

        # ``generate-core-file`` is one of GDB's raw-filename commands: the
        # complete remainder of the command line is the filename, including
        # spaces and quote characters.  Adding shell-style quotes makes those
        # quotes part of the actual filename.  Only command delimiters need to
        # be rejected here; the artifact store supplies this internal path and
        # GDB receives no shell interpolation.
        if any(character in output_path for character in ("\x00", "\r", "\n")):
            raise ValueError("Core output path contains a forbidden control character")
        output_path = os.path.abspath(output_path)
        command = "generate-core-file " + output_path
        monitor_done = threading.Event()
        violation: List[BaseException] = []
        last_reported = -1

        def signal_core_writer() -> None:
            """Interrupt a blocking GDB command without racing its MI reader."""
            signal_interrupt = getattr(self.gdb, "signal_interrupt", None)
            if callable(signal_interrupt):
                interrupted = signal_interrupt()
            else:
                # A legacy adapter without signal-only support cannot safely
                # consume MI from this monitor thread. Terminate the writer
                # boundary instead and force the registry to rebuild it.
                killer = getattr(self.gdb, "kill_process_group", None)
                interrupted = (
                    killer()
                    if callable(killer)
                    else {
                        "success": False,
                        "error": "No signal-only or process-kill capability",
                    }
                )
                self._invalidate_gdb("core monitor lacked signal-only interruption")
            if not interrupted.get("success", False):
                violation.append(
                    CoreDumpGenerationError(
                        "Could not signal the active GDB core writer: "
                        + str(interrupted.get("error", "unknown interrupt failure"))
                    )
                )

        def ensure_writer_quiescent() -> None:
            """Guarantee gdb has stopped writing before staging can be unlinked.

            After signalling, a cooperative SIGINT may not stop gdb mid-write of
            a large core: on POSIX the file would remain open and, once staging
            is unlinked, keep consuming disk through the deleted descriptor with
            the size monitor already gone.  We wait a bounded time for the gdb
            process to exit and, if it has not, SIGKILL its process group so the
            descriptor is closed and the inode released before cleanup runs.

            If neither the wait nor the kill can *confirm* the process is dead,
            we raise :class:`WriterNotQuiescentError` so the artifact store
            quarantines (rather than unlinks) the staging file — an unlink here
            would recreate the very deleted-fd disk leak we are preventing.
            """
            waiter = getattr(self.gdb, "wait_for_writer_quiescence", None)
            if callable(waiter):
                try:
                    if waiter(timeout_sec=2.0):
                        return
                except BaseException as exc:
                    raise WriterNotQuiescentError(
                        "Could not verify that gdb stopped writing the core file; "
                        "the staging file is quarantined"
                    ) from exc
            killer = getattr(self.gdb, "kill_process_group", None)
            if not callable(killer):
                raise WriterNotQuiescentError(
                    "The debugger adapter cannot confirm or force writer quiescence; "
                    "the staging file is quarantined"
                )
            try:
                outcome = killer()
            except BaseException as exc:
                raise WriterNotQuiescentError(
                    "The debugger writer could not be terminated; the staging file "
                    "is quarantined"
                ) from exc
            if not isinstance(outcome, dict):
                raise WriterNotQuiescentError(
                    "The debugger returned no verifiable writer-termination result; "
                    "the staging file is quarantined"
                )
            # kill_process_group only reports success once it has *confirmed* the
            # process is dead (poll() returned a code). Treat a non-success/
            # unconfirmed result as a quarantine condition.
            confirmed = outcome.get("success") and outcome.get(
                "confirmed_dead", outcome.get("success")
            )
            if not confirmed:
                raise WriterNotQuiescentError(
                    "Could not confirm gdb stopped writing the core file; the "
                    "staging file is quarantined instead of unlinked to avoid a "
                    "deleted-descriptor disk leak"
                )

        def monitor() -> None:
            nonlocal last_reported
            while not monitor_done.wait(0.05):
                try:
                    if cancellation_check is not None:
                        cancellation_check()
                    size = os.path.getsize(output_path)
                except FileNotFoundError:
                    size = 0
                except Exception as exc:
                    violation.append(exc)
                    signal_core_writer()
                    return
                if progress_callback is not None and size != last_reported:
                    try:
                        last_reported = size
                        progress_callback(size, cap, "Generating core file")
                    except Exception as exc:
                        violation.append(exc)
                if size > cap:
                    violation.append(
                        ArtifactSizeLimitError(
                            f"Core staging file exceeded its {cap}-byte hard limit"
                        )
                    )
                elif (
                    shutil.disk_usage(os.path.dirname(output_path)).free
                    < min_free_bytes
                ):
                    violation.append(
                        ArtifactStorageFullError(
                            "Core generation crossed the configured free-space reserve"
                        )
                    )
                if violation:
                    # Only signal here; the authoritative quiescence guarantee
                    # (and its quarantine-on-failure) runs once in the finally
                    # block on the command-owning thread, where an exception can
                    # actually propagate.
                    signal_core_writer()
                    return

        watcher = threading.Thread(
            target=monitor, name="core-size-monitor", daemon=True
        )
        watcher_started = False
        started = time.monotonic()
        timeout_interrupt: Optional[Dict[str, Any]] = None
        # The command is about to touch the output file. Until it cleanly
        # returns and the file is validated, we must assume gdb may still hold
        # the descriptor open, so every abnormal exit escalates to quiescence
        # before the caller is allowed to unlink staging.
        command_completed_cleanly = False
        fsize_state: Dict[str, Any] = {"applied": False}
        apply_limit = getattr(self.gdb, "apply_fsize_limit", None)
        restore_limit = getattr(self.gdb, "restore_fsize_limit", None)
        fsize_restored: Optional[bool] = None
        quiescence_error: Optional[BaseException] = None
        primary_exc: Optional[BaseException] = None
        restore_failure: Optional[BaseException] = None
        writer_started = False
        try:
            # Scope a hard RLIMIT_FSIZE onto gdb's OWN pid for the duration of the
            # core write only. The inferior was forked earlier and is unaffected,
            # so this cannot alter the debugged program's behavior; it only makes
            # the kernel abort an oversized core with zero overshoot.
            if not callable(apply_limit):
                raise CoreDumpGenerationError(
                    "Core capture requires kernel RLIMIT_FSIZE support"
                )
            fsize_state = apply_limit(cap)
            if not fsize_state.get("applied"):
                detail = fsize_state.get("error", "the kernel limit was not applied")
                raise CoreDumpGenerationError(
                    "Core capture refused because its kernel file-size limit "
                    f"could not be enforced: {detail}"
                )
            watcher.start()
            watcher_started = True
            writer_started = True
            result = self.gdb.execute_command(command, timeout_sec=timeout)
            if result.get("timed_out"):
                # The original write call has relinquished the MI response
                # stream, so it is now safe to signal and drain synchronously.
                timeout_interrupt = self.gdb.interrupt(timeout_sec=1.0)
            else:
                command_completed_cleanly = not result.get("timed_out")
        except BaseException as exc:
            primary_exc = exc
        finally:
            monitor_done.set()
            if watcher_started:
                watcher.join(timeout=2)
            # P1: confirm writer is gone BEFORE restoring the soft limit so the
            # cap stays in place while GDB may still hold the core descriptor.
            if writer_started and (
                (not command_completed_cleanly) or violation or primary_exc is not None
            ):
                try:
                    ensure_writer_quiescent()
                except WriterNotQuiescentError as exc:
                    quiescence_error = exc
                    self._invalidate_gdb("core writer could not be confirmed quiescent")
            # Restore soft limit only when writer is confirmed gone. If
            # quiescence failed, keep the cap in place — the file may still
            # be growing and will be quarantined; removing the cap now could
            # let it fill the disk before quarantine takes effect.
            if fsize_state.get("applied") and quiescence_error is None:
                if callable(restore_limit):
                    try:
                        fsize_restored = bool(restore_limit(fsize_state))
                    except BaseException as exc:
                        fsize_restored = False
                        restore_failure = exc
                else:
                    fsize_restored = False
                    restore_failure = CoreDumpGenerationError(
                        "The debugger adapter cannot restore its file-size limit"
                    )
                if not fsize_restored:
                    # A live session retaining the temporary cap is unsafe for
                    # later inferiors. Invalidate it, while preserving any earlier
                    # command/quiescence exception as the primary failure.
                    self._invalidate_gdb("failed to restore the gdb file-size limit")
                    if restore_failure is None:
                        restore_failure = CoreDumpGenerationError(
                            "The debugger adapter reported that the file-size limit "
                            "was not restored"
                        )
        # Quiescence failure (potential disk leak) takes precedence over every
        # other error, including the primary exception from execute_command.
        if quiescence_error is not None:
            raise quiescence_error
        if primary_exc is not None:
            raise primary_exc
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if watcher.is_alive():
            raise CoreDumpGenerationError(
                "Core monitor did not terminate; staging cleanup requires manual "
                "runtime validation"
            )
        if violation:
            raise violation[0]
        if cancellation_check is not None:
            cancellation_check()
        if result.get("timed_out"):
            interrupt_success = bool(
                timeout_interrupt and timeout_interrupt.get("success")
            )
            raise CoreDumpGenerationError(
                "GDB core generation timed out and was interrupted; partial output "
                f"was not committed (interrupt success={interrupt_success})"
            )
        if not result.get("success"):
            raise CoreDumpGenerationError(self.debugger._error_message(result, command))
        if not os.path.isfile(output_path):
            raise CoreDumpGenerationError(
                "GDB reported success but created no core file"
            )
        size_bytes = os.path.getsize(output_path)
        if size_bytes > cap:
            raise ArtifactSizeLimitError(
                f"Core file is {size_bytes} bytes, exceeding the {cap}-byte limit"
            )
        validated = self._validate_elf_core(output_path)
        if restore_failure is not None:
            raise CoreDumpGenerationError(
                "Failed to restore the debugger file-size limit: " f"{restore_failure}"
            )
        if progress_callback is not None:
            progress_callback(size_bytes, cap, "Core file generated")
        response: Dict[str, Any] = {
            "success": True,
            "command": command,
            "state": self.gdb.get_state(),
            "parser_version": PARSER_VERSION,
            "parsed": {
                "crash_id": expected_crash_id,
                "size_bytes": size_bytes,
                "max_size_bytes": cap,
                "elapsed_ms": elapsed_ms,
                "limit_enforcement": {
                    "kernel_rlimit_applied": True,
                    "kernel_rlimit_bytes": fsize_state.get("applied_soft"),
                    "kernel_rlimit_restored": fsize_restored,
                    "fallback": None,
                },
                "validation": validated,
                "source": "gdb-cli",
            },
            "raw_output": self.debugger._raw_output(result),
            "raw_responses": result.get("responses", []),
        }
        if fsize_restored is False:
            response["warnings"] = [
                "The debugger file-size limit could not be restored; the session "
                "was invalidated and must be rebuilt"
            ]
        return response


class CoreDumpGenerationError(RuntimeError):
    """Raised when GDB cannot produce a complete, validated ELF core file."""
