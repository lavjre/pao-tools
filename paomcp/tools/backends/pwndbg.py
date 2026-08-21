"""
Pwndbg tools for MCP server

Provides MCP tool implementations for GDB/pwndbg commands.
Each tool returns immediate results suitable for LLM interaction.
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

from paomcp.tools.backends.backend import BackendSpec
from paomcp.tools.backends.compatibility import PwndbgCompatibility
from paomcp.tools.backends.crash import CrashTriage
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.heap import HeapTools
from paomcp.tools.backends.inspection import PwndbgInspection

if TYPE_CHECKING:
    from paomcp.state.session import SessionState

logger = logging.getLogger(__name__)

DEFAULT_MAX_MEMORY_PATCHES = 256
DEFAULT_MAX_MEMORY_PATCH_BYTES = 32 * 1024 * 1024
MAX_CONFIGURED_MEMORY_PATCHES = 100_000
MAX_CONFIGURED_MEMORY_PATCH_BYTES = 1024 * 1024 * 1024
RAW_ADDRESS_SPACE_COMMANDS = frozenset(
    {
        "run",
        "r",
        "start",
        "starti",
        "attach",
        "detach",
        "kill",
        "disconnect",
        "file",
        "exec-file",
        "core",
        "core-file",
        "target",
        "load",
        "inferior",
        "add-inferior",
        "clone-inferior",
        "remove-inferiors",
        "-exec-run",
        "-exec-abort",
        "-target-attach",
        "-target-detach",
        "-target-disconnect",
        "-file-exec-file",
        "-file-exec-and-symbols",
        "-add-inferior",
        "-remove-inferiors",
    }
)
EXEC_TRANSITION_PATTERN = re.compile(
    r"\b(?:executing new program|exec(?:ution)? catchpoint)\b", re.IGNORECASE
)


def _bounded_environment_int(name: str, default: int, maximum: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


class PwndbgTools:
    """MCP tools for pwndbg interaction"""

    def __init__(
        self,
        gdb_controller: GdbController,
        session_state: SessionState,
        *,
        backend: Optional[BackendSpec] = None,
        max_memory_write_bytes: Optional[int] = None,
        max_memory_patches: Optional[int] = None,
        max_memory_patch_bytes: Optional[int] = None,
    ):
        """
        Initialize pwndbg tools

        Args:
            gdb_controller: GDB controller instance
            session_state: Session state manager
            backend: resolved debugger backend spec; ``None`` (or a
                ``gdb-mi3`` protocol) enables the full pwndbg surface, while
                an ``lldb-cli`` backend gates pwndbg-only inspection behind a
                typed "requires the gdb/pwndbg backend" verdict.
        """
        self.gdb = gdb_controller
        self.session = session_state
        self.backend = backend
        self.compatibility = PwndbgCompatibility(gdb_controller)
        self.inspection = PwndbgInspection(gdb_controller)
        self.debugger = DebuggerOperations(
            gdb_controller,
            timeout_seconds=self.inspection.timeout_seconds,
            max_memory_write_bytes=max_memory_write_bytes,
        )
        self.crash = CrashTriage(
            gdb_controller,
            self.debugger,
            self.inspection,
            session_state.session_id,
        )
        self.heap = HeapTools(
            gdb_controller,
            self.inspection,
            self.debugger,
            session_state.session_id,
            binary_path_provider=lambda: self.session.binary_path,
            backend=backend,
            patch_recorder=self,
        )
        self.max_memory_patches = (
            max_memory_patches
            if max_memory_patches is not None
            else _bounded_environment_int(
                "PAO_MAX_MEMORY_PATCHES",
                DEFAULT_MAX_MEMORY_PATCHES,
                MAX_CONFIGURED_MEMORY_PATCHES,
            )
        )
        self.max_memory_patch_bytes = (
            max_memory_patch_bytes
            if max_memory_patch_bytes is not None
            else _bounded_environment_int(
                "PAO_MAX_MEMORY_PATCH_BYTES",
                DEFAULT_MAX_MEMORY_PATCH_BYTES,
                MAX_CONFIGURED_MEMORY_PATCH_BYTES,
            )
        )
        if not 1 <= self.max_memory_patches <= MAX_CONFIGURED_MEMORY_PATCHES:
            raise ValueError(
                "Memory patch count limit must be between 1 and "
                f"{MAX_CONFIGURED_MEMORY_PATCHES}"
            )
        if not 1 <= self.max_memory_patch_bytes <= MAX_CONFIGURED_MEMORY_PATCH_BYTES:
            raise ValueError(
                "Memory patch byte limit must be between 1 and "
                f"{MAX_CONFIGURED_MEMORY_PATCH_BYTES}"
            )
        self._memory_patches: Dict[str, Dict[str, Any]] = {}
        self._memory_patch_sequence = 0
        self._inferior_generation = 0
        self._last_gdb_inferior_epoch = self._gdb_inferior_epoch()
        # Ensure GDB is initialized once lazily to avoid startup cost unless used
        try:
            self.gdb.initialize()
        except Exception:
            # Defer errors to first actual call to avoid failing construction
            logger.exception(
                "GDB initialize() failed during tool init; will retry on demand"
            )

    @property
    def _is_lldb(self) -> bool:
        backend = getattr(self, "backend", None)
        return backend is not None and getattr(backend, "protocol", None) == "lldb-cli"

    def _backend_gate(
        self, capability: str, command: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return a typed verdict dict when the pwndbg surface is unavailable."""
        if not self._is_lldb:
            return None
        backend_name = getattr(self.backend, "name", "lldb")
        return {
            "success": False,
            "command": command or capability,
            "state": self.gdb.get_state(),
            "type": "GdbOnlyBackendError",
            "error": (
                f"'{capability}' requires the gdb/pwndbg backend; "
                f"session uses {backend_name}"
            ),
        }

    def _advance_inferior_generation(self, reason: str) -> None:
        """Invalidate active patches whenever address-space identity can change."""
        self._inferior_generation += 1
        for patch in self._memory_patches.values():
            if patch.get("active"):
                patch["active"] = False
                patch["invalidated"] = True
                patch["invalidation_reason"] = reason
        current_epoch = self._gdb_inferior_epoch()
        if current_epoch is not None:
            self._last_gdb_inferior_epoch = current_epoch

    def _gdb_inferior_epoch(self) -> Optional[int]:
        getter = getattr(self.gdb, "get_inferior_epoch", None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except (AttributeError, TypeError, ValueError):
            return None

    def _sync_inferior_generation(self, reason: str) -> bool:
        """Invalidate patches when GDB reports an inferior start/exit transition."""
        current = self._gdb_inferior_epoch()
        if current is None:
            return False
        previous = self._last_gdb_inferior_epoch
        self._last_gdb_inferior_epoch = current
        if previous is None or current == previous:
            return False
        self._advance_inferior_generation(reason)
        return True

    @staticmethod
    def _raw_address_space_reason(command: str) -> Optional[str]:
        match = re.match(r"^\s*([^\s]+)", command)
        if match is None:
            return None
        root = match.group(1).lower()
        if root in RAW_ADDRESS_SPACE_COMMANDS:
            return f"raw debugger lifecycle command: {root}"
        return None

    @staticmethod
    def _result_indicates_exec(result: Dict[str, Any]) -> bool:
        for response in result.get("responses", []):
            if not isinstance(response, dict):
                continue
            payload = response.get("payload")
            if isinstance(payload, dict):
                reason = str(payload.get("reason", ""))
                if "exec" in reason.lower():
                    return True
            elif isinstance(payload, str) and EXEC_TRANSITION_PATTERN.search(payload):
                return True
        return False

    def _observe_inferior_transition(
        self, result: Dict[str, Any], explicit_reason: Optional[str] = None
    ) -> None:
        epoch_changed = self._sync_inferior_generation(
            explicit_reason or "GDB inferior lifecycle notification"
        )
        reason = explicit_reason
        if reason is None and self._result_indicates_exec(result):
            reason = "inferior executed a new program"
        if reason is not None and not epoch_changed:
            self._advance_inferior_generation(reason)

    def get_debugger_capabilities(self, refresh: bool = False) -> Dict[str, Any]:
        """Return typed pwndbg/GDB runtime compatibility information.

        On an lldb session this returns a degraded success report: the rich
        pwndbg surface is unavailable, but the caller still learns which
        backend the session drives and what backends exist on the host.
        """
        from paomcp.tools.backends.backend import available_backends

        if self._is_lldb:
            backend_name = getattr(self.backend, "name", "lldb")
            return {
                "success": True,
                "ready": False,
                "parser_version": "1",
                "backend": (
                    self.backend.to_dict() if self.backend else {"name": backend_name}
                ),
                "available_backends": available_backends(),
                "warning": (
                    "session drives the lldb backend; the rich pwndbg tool "
                    "surface (heap parsers, elf runtime, crash triage) requires "
                    "the gdb/pwndbg backend"
                ),
            }
        self.gdb.initialize()
        report = self.compatibility.get_capabilities(refresh=refresh)
        if isinstance(report, dict):
            report["backend"] = (
                self.backend.to_dict() if self.backend else {"name": "gdb"}
            )
            report["available_backends"] = available_backends()
        return report

    def list_pwndbg_commands(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List installed pwndbg commands with parsed and raw evidence."""
        gate = self._backend_gate("list_pwndbg_commands")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self.compatibility.list_commands(category=category, query=query)

    def get_pwndbg_command_help(self, command_name: str) -> Dict[str, Any]:
        """Return raw help for one validated pwndbg command."""
        gate = self._backend_gate("get_pwndbg_command_help")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self.compatibility.get_help(command_name)

    def _record_inspection(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize and record one typed inspection result."""
        state = result.get("state", self.gdb.get_state())
        self.session.update_state(state)
        self.session.record_command(result.get("command", "inspection"), result)
        return result

    def analyze_elf(self, binary_path: str) -> Dict[str, Any]:
        """Load and inspect an ELF using checksec, sections, and PLT parsers."""
        gate = self._backend_gate("analyze_elf")
        if gate is not None:
            return gate
        self.gdb.initialize()
        if not self.session.binary_loaded or self.session.binary_path != binary_path:
            load_result = self.set_file(binary_path)
            if not load_result.get("success"):
                return {
                    "success": False,
                    "command": "analyze_elf",
                    "state": self.gdb.get_state(),
                    "error": f"Failed to load ELF: {binary_path}",
                    "type": "ElfLoadError",
                    "load_result": load_result,
                }
        return self._record_inspection(self.inspection.analyze_elf(binary_path))

    def checksec(self, binary_path: str) -> Dict[str, Any]:
        """Return typed mitigation information for an ELF path."""
        gate = self._backend_gate("checksec")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(self.inspection.checksec(binary_path))

    def inspect_got_plt(
        self,
        symbol_filter: Optional[str] = None,
        include_got: bool = True,
        include_plt: bool = True,
        include_readonly: bool = False,
        all_objects: bool = False,
        all_plt_symbols: bool = False,
    ) -> Dict[str, Any]:
        """Inspect typed GOT/PLT entries for the selected binary/runtime."""
        gate = self._backend_gate("inspect_got_plt")
        if gate is not None:
            return gate
        self.gdb.initialize()
        result = self.inspection.inspect_got_plt(
            symbol_filter=symbol_filter,
            include_got=include_got,
            include_plt=include_plt,
            include_readonly=include_readonly,
            all_objects=all_objects,
            all_plt_symbols=all_plt_symbols,
        )
        return self._record_inspection(result)

    def memory_maps(
        self,
        address_or_module: Optional[str] = None,
        writable: bool = False,
        executable: bool = False,
        gaps: bool = False,
    ) -> Dict[str, Any]:
        """Return parsed virtual memory mappings."""
        gate = self._backend_gate("memory_maps")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(
            self.inspection.memory_maps(
                address_or_module=address_or_module,
                writable=writable,
                executable=executable,
                gaps=gaps,
            )
        )

    def list_threads(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Return parsed pwndbg thread information."""
        gate = self._backend_gate("list_threads")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(self.inspection.threads(limit=limit))

    def get_process_info(self) -> Dict[str, Any]:
        """Return parsed process metadata from procinfo."""
        gate = self._backend_gate("get_process_info")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(self.inspection.process_info())

    def inspect_libc(self, include_link_map: bool = True) -> Dict[str, Any]:
        """Return parsed libc metadata and optional dynamic link map."""
        gate = self._backend_gate("inspect_libc")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(
            self.inspection.inspect_libc(include_link_map=include_link_map)
        )

    def inspect_address(self, address: str = "$pc") -> Dict[str, Any]:
        """Return parsed mapping, offsets, and sections for an address."""
        gate = self._backend_gate("inspect_address")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_inspection(self.inspection.address_info(address))

    def execute(self, command: str) -> Dict[str, Any]:
        """Execute arbitrary GDB/pwndbg command and return raw responses"""
        logger.info(f"Execute tool: {command}")
        self.gdb.initialize()
        result = self.gdb.execute_command(command)
        if result.get("success"):
            self._observe_inferior_transition(
                result, explicit_reason=self._raw_address_space_reason(command)
            )
        else:
            self._sync_inferior_generation("GDB inferior lifecycle notification")
        self.session.update_state(result["state"])
        self.session.record_command(command, result)
        return result

    def set_file(self, binary_path: str) -> Dict[str, Any]:
        """Set the file to debug; return raw responses"""
        logger.info(f"Set file: {binary_path}")
        self.gdb.initialize()
        result = self.gdb.set_file(binary_path)
        if result.get("success"):
            self.session.binary_path = binary_path
            self.session.binary_loaded = True
            self.debugger.clear_register_cache()
            self._observe_inferior_transition(result, explicit_reason="binary changed")
        self.session.update_state(result["state"])
        self.session.record_command(
            result.get("command", f"-file-exec-and-symbols {binary_path}"), result
        )
        return result

    def attach(self, pid: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Attach to an existing process; return raw responses"""
        logger.info(f"Attach to pid: {pid}")
        self.gdb.initialize()
        result, context = self.gdb.attach(pid)
        if result.get("success"):
            self._observe_inferior_transition(
                result, explicit_reason="attached to a new inferior"
            )
            self.session.pid = pid
        self.session.update_state(result["state"])
        self.session.record_command(
            result.get("command", f"-target-attach {pid}"), result
        )
        return result, context

    def run(self, args: str = "", start: bool = False) -> Dict[str, Any]:
        """Run the loaded binary; return raw responses"""
        logger.info(f"Run with args: '{args}'")
        self.gdb.initialize()
        if not self.session.binary_loaded:
            return {
                "command": "-exec-run",
                "responses": [],
                "success": False,
                "state": self.gdb.get_state(),
                "error": "No binary loaded. Use set_file first.",
            }
        result = self.gdb.run(args, start=start)
        if result.get("success"):
            self._observe_inferior_transition(
                result, explicit_reason="inferior restarted"
            )
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-run"), result)
        return result

    def finish(self) -> Dict[str, Any]:
        """Run until current function finishes; return raw responses"""
        logger.info("Finish current function")
        self.gdb.initialize()
        result = self.gdb.finish()
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-finish"), result)
        return result

    def jump(self, locspec: str) -> Dict[str, Any]:
        """Jump to a specific location; return raw responses"""
        logger.info(f"Jump to {locspec}")
        self.gdb.initialize()
        result = self.gdb.jump(locspec)
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command(
            result.get("command", f"-exec-jump {locspec}"), result
        )
        return result

    def return_from_function(self) -> Dict[str, Any]:
        """Force return from current function; return raw responses"""
        logger.info("Force return from current function")
        self.gdb.initialize()
        result = self.gdb.return_from_function()
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-return"), result)
        return result

    def until(self, locspec: Optional[str] = None) -> Dict[str, Any]:
        """Run until a location or next source line; return raw responses"""
        logger.info(f"Until {locspec if locspec else '[next line]'}")
        self.gdb.initialize()
        result = self.gdb.until(locspec)
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-until"), result)
        return result

    def step_control(self, command: str) -> Dict[str, Any]:
        """Execute stepping commands (c, n, s, ni, si); return raw responses"""
        logger.info(f"Step control: {command}")
        self.gdb.initialize()
        command_map = {
            "c": "continue",
            "n": "next",
            "s": "step",
            "ni": "nexti",
            "si": "stepi",
        }
        actual = command_map.get(command, command)
        current_state = self.gdb.get_state()
        if current_state != "stopped":
            return {
                "command": actual,
                "responses": [],
                "success": False,
                "state": current_state,
                "error": f"Cannot execute '{command}' in state '{current_state}'",
            }
        if actual == "continue":
            result = self.gdb.continue_execution()
        elif actual == "next":
            result = self.gdb.next()
        elif actual == "step":
            result = self.gdb.step()
        elif actual == "nexti":
            result = self.gdb.nexti()
        elif actual == "stepi":
            result = self.gdb.stepi()
        else:
            return {
                "command": actual,
                "responses": [],
                "success": False,
                "state": current_state,
                "error": f"Unknown step command '{command}'",
            }
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", actual), result)
        return result

    def gdb_poll(self, timeout: float = 0.0) -> Dict[str, Any]:
        """Drain pending async GDB notifications."""
        logger.info("GDB poll")
        self.gdb.initialize()
        result = self.gdb.drain_async(timeout_sec=timeout)
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command("gdb_poll", result)
        return result

    def gdb_interrupt(self, timeout: float = 1.0) -> Dict[str, Any]:
        """Interrupt the inferior and drain async notifications."""
        logger.info("GDB interrupt")
        self.gdb.initialize()
        result = self.gdb.interrupt(timeout_sec=timeout)
        self._observe_inferior_transition(result)
        self.session.update_state(result["state"])
        self.session.record_command("gdb_interrupt", result)
        return result

    def get_context(self, context_type: str = "all") -> Dict[str, Any]:
        """Get debugging context (registers, stack, disassembly, etc.)"""
        logger.info(f"Get context: {context_type}")
        self.gdb.initialize()
        if self.gdb.get_state() != "stopped":
            return {
                "command": f"context {context_type}",
                "responses": [],
                "success": False,
                "state": self.gdb.get_state(),
                "error": f"Cannot get context while inferior is {self.gdb.get_state()}",
            }
        if context_type == "all":
            # Prefer fast MI-based snapshot to avoid slow pwndbg rendering
            aggregated = self.gdb.get_quick_context()
            self.session.record_command("context all (quick)", aggregated)
            return aggregated
        else:
            result = self.gdb.get_context(context_type)
            self.session.record_command(f"context {context_type}", result)
            return result

    def set_breakpoint(
        self,
        location: str,
        condition: Optional[str] = None,
        temporary: bool = False,
    ) -> Dict[str, Any]:
        """Set a breakpoint and return a typed breakpoint record."""
        logger.info(f"Set breakpoint at {location}")
        self.gdb.initialize()
        result = self.debugger.create_breakpoint(
            location=location,
            condition=condition,
            temporary=temporary,
        )
        breakpoint = result.get("parsed", {}).get("breakpoint")
        if result.get("success") and breakpoint:
            number = str(breakpoint.get("number", ""))
            if number.isdigit():
                self.session.add_breakpoint(
                    int(number),
                    breakpoint.get("original_location")
                    or breakpoint.get("address")
                    or location,
                    condition,
                )
        self.session.update_state(result.get("state", self.gdb.get_state()))
        self.session.record_command(result.get("command", "-break-insert"), result)
        return result

    def list_breakpoints(self) -> Dict[str, Any]:
        """List all breakpoints as typed records."""
        logger.info("List breakpoints")
        self.gdb.initialize()
        result = self.debugger.list_breakpoints()
        self.session.record_command(result.get("command", "-break-list"), result)
        return result

    def delete_breakpoint(self, number: int) -> Dict[str, Any]:
        """Delete a breakpoint and synchronize session metadata."""
        logger.info(f"Delete breakpoint #{number}")
        self.gdb.initialize()
        result = self.debugger.delete_breakpoint(number)
        if result.get("success"):
            self.session.remove_breakpoint(number)
        self.session.record_command(
            result.get("command", f"-break-delete {number}"), result
        )
        return result

    def toggle_breakpoint(self, number: int, enable: bool) -> Dict[str, Any]:
        """Enable or disable a breakpoint and synchronize session metadata."""
        action = "enable" if enable else "disable"
        logger.info(f"{action} breakpoint #{number}")
        self.gdb.initialize()
        result = self.debugger.toggle_breakpoint(number, enable)
        if result.get("success") and number in self.session.breakpoints:
            self.session.breakpoints[number].enabled = enable
        self.session.record_command(
            result.get("command", f"-break-{action} {number}"), result
        )
        return result

    def _record_debugger(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize state and record one typed debugger operation."""
        self._observe_inferior_transition(result)
        self.session.update_state(result.get("state", self.gdb.get_state()))
        self.session.record_command(result.get("command", "debugger"), result)
        return result

    def set_watchpoint(self, expression: str, access: str = "write") -> Dict[str, Any]:
        """Set a typed write, read, or access watchpoint."""
        self.gdb.initialize()
        result = self.debugger.set_watchpoint(expression, access)
        watchpoint = result.get("parsed", {}).get("watchpoint")
        if result.get("success") and isinstance(watchpoint, dict):
            number = str(watchpoint.get("number", ""))
            if number.isdigit():
                self.session.add_breakpoint(int(number), expression)
        return self._record_debugger(result)

    def list_watchpoints(self) -> Dict[str, Any]:
        """List only watchpoint breakpoint records."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.list_watchpoints())

    def delete_watchpoint(self, number: int) -> Dict[str, Any]:
        """Delete a verified watchpoint and synchronize session metadata."""
        self.gdb.initialize()
        result = self.debugger.delete_watchpoint(number)
        if result.get("success"):
            self.session.remove_breakpoint(number)
        return self._record_debugger(result)

    def toggle_watchpoint(self, number: int, enable: bool) -> Dict[str, Any]:
        """Enable or disable a verified watchpoint."""
        self.gdb.initialize()
        result = self.debugger.toggle_watchpoint(number, enable)
        if result.get("success") and number in self.session.breakpoints:
            self.session.breakpoints[number].enabled = enable
        return self._record_debugger(result)

    def read_registers(
        self,
        names: Optional[List[str]] = None,
        refresh_names: bool = False,
    ) -> Dict[str, Any]:
        """Read selected or all registers as typed records."""
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.read_registers(names, refresh_names=refresh_names)
        )

    def write_register(self, name: str, value: str) -> Dict[str, Any]:
        """Write a register and return before/after evidence."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.write_register(name, value))

    def evaluate_expression_typed(self, expression: str) -> Dict[str, Any]:
        """Evaluate a validated GDB expression with typed output."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.evaluate_expression(expression))

    def select_thread(self, thread_id: int) -> Dict[str, Any]:
        """Select the active inferior thread."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.select_thread(thread_id))

    def stack_frames(self, low: int = 0, high: Optional[int] = None) -> Dict[str, Any]:
        """Return typed stack frames without an implicit frame limit."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.stack_frames(low=low, high=high))

    def select_frame(self, level: int) -> Dict[str, Any]:
        """Select and inspect one stack frame."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.select_frame(level))

    def list_debugger_threads(self) -> Dict[str, Any]:
        """List typed threads directly from GDB/MI."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.threads())

    def list_inferiors(self, recurse: int = 1) -> Dict[str, Any]:
        """List typed GDB inferiors/thread groups."""
        self.gdb.initialize()
        return self._record_debugger(self.debugger.inferiors(recurse=recurse))

    def read_register_group(
        self, group: str, refresh_names: bool = False
    ) -> Dict[str, Any]:
        """Read a general, vector, floating-point, system, or all register group."""
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.read_register_group(group, refresh_names=refresh_names)
        )

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
        """Search mapped memory through pwndbg with parsed matches."""
        gate = self._backend_gate("search_memory")
        if gate is not None:
            return gate
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.search_memory(
                value=value,
                value_type=value_type,
                mapping=mapping,
                hex_encoded=hex_encoded,
                executable=executable,
                writable=writable,
                limit=limit,
                aligned=aligned,
                step=step,
            )
        )

    @staticmethod
    def _patch_storage_bytes(patch: Dict[str, Any]) -> int:
        explicit = patch.get("storage_bytes")
        if isinstance(explicit, int) and explicit >= 0:
            return explicit
        total = 0
        for field in ("original_hex", "patched_hex"):
            value = patch.get(field)
            if isinstance(value, str):
                total += len(value) // 2
        return total

    def _memory_patch_quota(self) -> Dict[str, Any]:
        retained = list(self._memory_patches.values())
        active = [patch for patch in retained if patch.get("active")]
        return {
            "max_write_bytes": self.debugger.max_memory_write_bytes,
            "max_patch_count": self.max_memory_patches,
            "max_patch_bytes": self.max_memory_patch_bytes,
            "retained_patch_count": len(retained),
            "retained_patch_bytes": sum(
                self._patch_storage_bytes(patch) for patch in retained
            ),
            "active_patch_count": len(active),
            "active_patch_bytes": sum(
                self._patch_storage_bytes(patch) for patch in active
            ),
            "byte_accounting": "original_plus_patched_payload",
        }

    def _plan_memory_patch_capacity(
        self, byte_count: int
    ) -> Tuple[List[str], Dict[str, Any], Optional[str]]:
        """Plan inactive-history retention without mutating it before the write."""
        required_bytes = byte_count * 2
        quota = self._memory_patch_quota()
        quota["required_patch_bytes"] = required_bytes
        quota["retention_candidates"] = []
        if required_bytes > self.max_memory_patch_bytes:
            return (
                [],
                quota,
                "Reversible patch payload exceeds the per-session patch byte limit",
            )
        removable = sorted(
            (
                patch
                for patch in self._memory_patches.values()
                if not patch.get("active")
            ),
            key=lambda patch: int(patch.get("sequence", 0)),
        )
        removed: List[str] = []
        projected_count = int(quota["retained_patch_count"])
        projected_bytes = int(quota["retained_patch_bytes"])
        while removable and (
            projected_count + 1 > self.max_memory_patches
            or projected_bytes + required_bytes > self.max_memory_patch_bytes
        ):
            candidate = removable.pop(0)
            patch_id = str(candidate["patch_id"])
            removed.append(patch_id)
            projected_count -= 1
            projected_bytes -= self._patch_storage_bytes(candidate)

        quota["retention_candidates"] = removed
        quota["projected_patch_count"] = projected_count + 1
        quota["projected_patch_bytes"] = projected_bytes + required_bytes
        if projected_count + 1 > self.max_memory_patches:
            return (
                removed,
                quota,
                "Active memory patches have reached the per-session patch count limit",
            )
        if projected_bytes + required_bytes > self.max_memory_patch_bytes:
            return (
                removed,
                quota,
                "Active memory patches have reached the per-session patch byte limit",
            )
        return removed, quota, None

    def write_memory(
        self,
        address: str,
        data_hex: str,
        verify: bool = True,
        allow_unsafe: bool = False,
    ) -> Dict[str, Any]:
        """Write bytes with mapping validation and session-local patch history."""
        self.gdb.initialize()
        self._sync_inferior_generation("GDB inferior lifecycle notification")
        normalized, byte_count = self.debugger.normalize_memory_write(data_hex)
        retention_candidates, quota, quota_error = self._plan_memory_patch_capacity(
            byte_count
        )
        if quota_error is not None:
            return self._record_debugger(
                {
                    "success": False,
                    "command": "write_memory",
                    "state": self.gdb.get_state(),
                    "parser_version": "1",
                    "parsed": {
                        "address": address,
                        "byte_count": byte_count,
                        "write_applied": False,
                        "patch_quota": quota,
                    },
                    "raw_responses": [],
                    "error": quota_error,
                    "type": "MemoryPatchQuotaError",
                }
            )
        address_result = self.inspection.address_info(address)
        resolved_address = None
        if address_result.get("success"):
            candidate = address_result.get("parsed", {}).get("address")
            if candidate:
                resolved_address = str(candidate)
        effective_address = resolved_address or address
        mapping_result = (
            self.inspection.memory_maps(address_or_module=resolved_address)
            if resolved_address is not None
            else None
        )
        result = self.debugger.write_memory(
            effective_address,
            normalized,
            verify=verify,
            mapping_result=mapping_result,
            allow_unsafe=allow_unsafe,
        )
        parsed = result.get("parsed", {})
        if isinstance(parsed, dict):
            parsed["requested_address"] = address
            parsed["canonical_address"] = resolved_address
            result.setdefault("raw_responses", {})["address_resolution"] = (
                address_result.get("raw_responses", [])
            )
            parsed["retention_pruned"] = []
            parsed["patch_quota"] = quota
        original = parsed.get("original_hex") if isinstance(parsed, dict) else None
        write_applied = (
            bool(parsed.get("write_applied")) if isinstance(parsed, dict) else False
        )
        if write_applied and original is not None:
            for retained_patch_id in retention_candidates:
                self._memory_patches.pop(retained_patch_id, None)
            self._memory_patch_sequence += 1
            patch_id = f"patch-{self._memory_patch_sequence:06d}"
            patch = {
                "patch_id": patch_id,
                "address": effective_address,
                "requested_address": address,
                "canonical_address": resolved_address,
                "byte_count": parsed.get("byte_count"),
                "original_hex": original,
                "patched_hex": parsed.get("data_hex"),
                "verified": parsed.get("verified"),
                "verification_uncertain": parsed.get("verification_uncertain"),
                "allow_unsafe": allow_unsafe,
                "inferior_generation": self._inferior_generation,
                "inferior_pid": self.gdb.get_inferior_pid() or self.session.pid,
                "sequence": self._memory_patch_sequence,
                "storage_bytes": (
                    len(str(original)) // 2 + len(str(parsed.get("data_hex", ""))) // 2
                ),
                "active": True,
                "reverted": False,
            }
            self._memory_patches[patch_id] = patch
            parsed["patch_id"] = patch_id
            parsed["retention_pruned"] = retention_candidates
            parsed["patch_quota"] = self._memory_patch_quota()
        return self._record_debugger(result)

    def list_memory_patches(self, active_only: bool = False) -> Dict[str, Any]:
        """List memory mutations tracked only by this debugger session."""
        self._sync_inferior_generation("GDB inferior lifecycle notification")
        patches = [
            dict(item)
            for item in self._memory_patches.values()
            if not active_only or item.get("active")
        ]
        return {
            "success": True,
            "command": "list_memory_patches",
            "state": self.gdb.get_state(),
            "parser_version": "1",
            "parsed": {
                "count": len(patches),
                "patches": patches,
                "quota": self._memory_patch_quota(),
            },
            "raw_responses": [],
        }

    def revert_memory_patch(
        self, patch_id: str, verify: bool = True, force: bool = False
    ) -> Dict[str, Any]:
        """Restore the original bytes for one active session-local patch."""
        self._sync_inferior_generation("GDB inferior lifecycle notification")
        patch = self._memory_patches.get(patch_id)
        if patch is None:
            raise ValueError(f"Unknown memory patch id: {patch_id}")
        if not patch.get("active"):
            raise ValueError(f"Memory patch is no longer active: {patch_id}")
        current_pid = self.gdb.get_inferior_pid() or self.session.pid
        if (
            patch.get("inferior_generation") != self._inferior_generation
            or patch.get("inferior_pid") != current_pid
        ):
            raise RuntimeError(
                "Memory patch belongs to a different inferior generation"
            )
        canonical_address = patch.get("canonical_address")
        if canonical_address is None and not force:
            raise RuntimeError(
                "Patch address was not canonicalized; pass force=true to revert unsafely"
            )
        start = int(str(patch["address"]), 0) if canonical_address is not None else None
        end = start + int(patch["byte_count"]) if start is not None else None
        overlapping = []
        for candidate in self._memory_patches.values():
            if (
                not candidate.get("active")
                or int(candidate.get("sequence", 0)) <= int(patch["sequence"])
                or candidate.get("canonical_address") is None
            ):
                continue
            candidate_start = int(str(candidate["address"]), 0)
            candidate_end = candidate_start + int(candidate["byte_count"])
            if (
                start is not None
                and end is not None
                and start < candidate_end
                and candidate_start < end
            ):
                overlapping.append(str(candidate["patch_id"]))
        if overlapping and not force:
            raise RuntimeError(
                "A newer active patch overlaps this range: " + ", ".join(overlapping)
            )
        current_result = self.gdb.execute_mi_command(
            "-data-read-memory-bytes "
            f"{self.gdb.quote_mi_argument(str(patch['address']))} "
            f"{patch['byte_count']}",
            timeout_sec=self.debugger.timeout_seconds,
        )
        current_hex = self.debugger._memory_contents(current_result)
        if current_hex != patch.get("patched_hex") and not force:
            return self._record_debugger(
                {
                    "success": False,
                    "command": f"revert_memory_patch {patch_id}",
                    "state": self.gdb.get_state(),
                    "parser_version": "1",
                    "parsed": {
                        "patch_id": patch_id,
                        "expected_hex": patch.get("patched_hex"),
                        "actual_hex": current_hex,
                        "reverted": False,
                    },
                    "raw_responses": current_result.get("responses", []),
                    "error": "Current bytes differ from the recorded patch; pass force=true to override",
                    "type": "PatchConflictError",
                }
            )
        mapping_result = self.inspection.memory_maps(
            address_or_module=str(patch["address"])
        )
        result = self.debugger.write_memory(
            str(patch["address"]),
            str(patch["original_hex"]),
            verify=verify,
            mapping_result=mapping_result,
            allow_unsafe=bool(patch.get("allow_unsafe")),
        )
        if result.get("success"):
            patch["active"] = False
            patch["reverted"] = True
            result.setdefault("parsed", {})["patch_id"] = patch_id
            result["parsed"]["reverted"] = True
            result["parsed"]["forced"] = force
        result["command"] = f"revert_memory_patch {patch_id}"
        return self._record_debugger(result)

    def capture_debugger_snapshot(
        self,
        register_names: Optional[List[str]] = None,
        disassembly_bytes: int = 64,
        include_mappings: bool = True,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture registers, frames, instructions, threads, inferiors, and maps."""
        self.gdb.initialize()
        deadline = self.debugger.snapshot_deadline()
        if cancellation_check is not None:
            cancellation_check()
        mapping_result = self.inspection.memory_maps() if include_mappings else None
        if cancellation_check is not None:
            cancellation_check()
        return self._record_debugger(
            self.debugger.capture_snapshot(
                register_names=register_names,
                disassembly_bytes=disassembly_bytes,
                mapping_result=mapping_result,
                deadline=deadline,
                cancellation_check=cancellation_check,
            )
        )

    def get_last_stop(
        self, fatal_signals: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """Return the latest persisted GDB/MI stop record and classification."""
        self.gdb.initialize()
        return self._record_debugger(self.crash.get_last_stop(fatal_signals))

    def run_until_crash(
        self,
        *,
        arguments: Optional[Sequence[str]] = None,
        mode: str = "auto",
        timeout_seconds: Optional[float] = None,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Run, continue, or wait until a crash, another stop, or normal exit."""
        self.gdb.initialize()
        return self._record_debugger(
            self.crash.run_until_crash(
                arguments=arguments,
                mode=mode,
                timeout_seconds=timeout_seconds,
                fatal_signals=fatal_signals,
                cancellation_check=cancellation_check,
            )
        )

    def capture_crash_snapshot(
        self,
        expected_crash_id: str,
        *,
        stack_bytes: int = 256,
        disassembly_bytes: int = 64,
        include_mappings: bool = True,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Capture a canonical typed snapshot for the exact current crash."""
        self.gdb.initialize()
        return self._record_debugger(
            self.crash.capture_crash_snapshot(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                fatal_signals=fatal_signals,
                cancellation_check=cancellation_check,
            )
        )

    def analyze_crash(
        self,
        expected_crash_id: str,
        *,
        stack_bytes: int = 256,
        disassembly_bytes: int = 64,
        include_mappings: bool = True,
        refresh: bool = False,
        fatal_signals: Optional[Sequence[str]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Dict[str, Any]:
        """Analyze one exact crash id with conservative typed evidence."""
        self.gdb.initialize()
        return self._record_debugger(
            self.crash.analyze_crash(
                expected_crash_id,
                stack_bytes=stack_bytes,
                disassembly_bytes=disassembly_bytes,
                include_mappings=include_mappings,
                refresh=refresh,
                fatal_signals=fatal_signals,
                cancellation_check=cancellation_check,
            )
        )

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
        """Generate a monitored ELF core through the GDB CLI fallback."""
        self.gdb.initialize()
        return self._record_debugger(
            self.crash.generate_core_file(
                expected_crash_id,
                output_path,
                max_size_bytes=max_size_bytes,
                timeout_seconds=timeout_seconds,
                min_free_bytes=min_free_bytes,
                fatal_signals=fatal_signals,
                cancellation_check=cancellation_check,
                progress_callback=progress_callback,
            )
        )

    def diff_debugger_snapshots(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Diff two explicit snapshots without hidden snapshot storage."""
        return self._record_debugger(
            self.debugger.diff_snapshots(before_value=before, after_value=after)
        )

    def disassemble(self, start: str = "$pc", byte_count: int = 64) -> Dict[str, Any]:
        """Return typed MI disassembly for a bounded byte range."""
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.disassemble(start=start, byte_count=byte_count)
        )

    def step_until_event(
        self,
        event: str,
        symbol_regex: Optional[str] = None,
        mnemonic: Optional[str] = None,
        operands: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Step until a typed call, jump, return, syscall, instruction, or address."""
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.step_until_event(
                event=event,
                symbol_regex=symbol_regex,
                mnemonic=mnemonic,
                operands=operands,
                target=target,
            )
        )

    def _get_full_context(self) -> Dict[str, Any]:
        """Get complete debugging context (raw responses per context)"""
        contexts = {}
        for ctx_type in ["regs", "stack", "disasm", "code", "backtrace"]:
            contexts[ctx_type] = self.gdb.get_context(ctx_type)
        return contexts

    def get_memory(
        self, address: str, size: int = 64, format: str = "hex"
    ) -> Dict[str, Any]:
        """Read memory at specified address; return raw responses"""
        logger.info(f"Read memory at {address}, {size} bytes as {format}")
        self.gdb.initialize()
        if format == "hex":
            # Use fast MI bytes read and return raw bytes; caller can format
            result = self.gdb.read_memory_bytes(address, size)
            self.session.record_command(
                f"-data-read-memory-bytes {address} {size}", result
            )
            return result
        elif format == "string":
            # For C-string, classic command is fine (needs pwndbg/pretty print)
            cmd = f"x/s {address}"
            result = self.gdb.execute_command(cmd)
            self.session.record_command(cmd, result)
            return result
        else:
            # Generic bytes read via MI grid as fallback
            word_size = 1
            nr_rows = size
            nr_cols = 1
            result = self.gdb.read_memory_mi(address, "x", word_size, nr_rows, nr_cols)
            self.session.record_command(
                f"-data-read-memory {address} x {word_size} {nr_rows} {nr_cols}", result
            )
            return result

    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information (no GDB sync/parsing)"""
        return {"session": self.session.to_dict(), "gdb_state": self.gdb.get_state()}
