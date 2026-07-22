"""
Pwndbg tools for MCP server

Provides MCP tool implementations for GDB/pwndbg commands.
Each tool returns immediate results suitable for LLM interaction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from paomcp.tools.backends.compatibility import PwndbgCompatibility
from paomcp.tools.backends.debugger import DebuggerOperations
from paomcp.tools.backends.gdb import GdbController
from paomcp.tools.backends.inspection import PwndbgInspection

if TYPE_CHECKING:
    from paomcp.state.session import SessionState

logger = logging.getLogger(__name__)


class PwndbgTools:
    """MCP tools for pwndbg interaction"""

    def __init__(self, gdb_controller: GdbController, session_state: SessionState):
        """
        Initialize pwndbg tools

        Args:
            gdb_controller: GDB controller instance
            session_state: Session state manager
        """
        self.gdb = gdb_controller
        self.session = session_state
        self.compatibility = PwndbgCompatibility(gdb_controller)
        self.inspection = PwndbgInspection(gdb_controller)
        self.debugger = DebuggerOperations(
            gdb_controller, timeout_seconds=self.inspection.timeout_seconds
        )
        # Ensure GDB is initialized once lazily to avoid startup cost unless used
        try:
            self.gdb.initialize()
        except Exception:
            # Defer errors to first actual call to avoid failing construction
            logger.exception(
                "GDB initialize() failed during tool init; will retry on demand"
            )

    def get_debugger_capabilities(self, refresh: bool = False) -> Dict[str, Any]:
        """Return typed pwndbg/GDB runtime compatibility information."""
        self.gdb.initialize()
        return self.compatibility.get_capabilities(refresh=refresh)

    def list_pwndbg_commands(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List installed pwndbg commands with parsed and raw evidence."""
        self.gdb.initialize()
        return self.compatibility.list_commands(category=category, query=query)

    def get_pwndbg_command_help(self, command_name: str) -> Dict[str, Any]:
        """Return raw help for one validated pwndbg command."""
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
        self.gdb.initialize()
        return self._record_inspection(self.inspection.threads(limit=limit))

    def get_process_info(self) -> Dict[str, Any]:
        """Return parsed process metadata from procinfo."""
        self.gdb.initialize()
        return self._record_inspection(self.inspection.process_info())

    def inspect_libc(self, include_link_map: bool = True) -> Dict[str, Any]:
        """Return parsed libc metadata and optional dynamic link map."""
        self.gdb.initialize()
        return self._record_inspection(
            self.inspection.inspect_libc(include_link_map=include_link_map)
        )

    def inspect_address(self, address: str = "$pc") -> Dict[str, Any]:
        """Return parsed mapping, offsets, and sections for an address."""
        self.gdb.initialize()
        return self._record_inspection(self.inspection.address_info(address))

    def execute(self, command: str) -> Dict[str, Any]:
        """Execute arbitrary GDB/pwndbg command and return raw responses"""
        logger.info(f"Execute tool: {command}")
        self.gdb.initialize()
        result = self.gdb.execute_command(command)
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
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-run"), result)
        return result

    def finish(self) -> Dict[str, Any]:
        """Run until current function finishes; return raw responses"""
        logger.info("Finish current function")
        self.gdb.initialize()
        result = self.gdb.finish()
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-finish"), result)
        return result

    def jump(self, locspec: str) -> Dict[str, Any]:
        """Jump to a specific location; return raw responses"""
        logger.info(f"Jump to {locspec}")
        self.gdb.initialize()
        result = self.gdb.jump(locspec)
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
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", "-exec-return"), result)
        return result

    def until(self, locspec: Optional[str] = None) -> Dict[str, Any]:
        """Run until a location or next source line; return raw responses"""
        logger.info(f"Until {locspec if locspec else '[next line]'}")
        self.gdb.initialize()
        result = self.gdb.until(locspec)
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
        self.session.update_state(result["state"])
        self.session.record_command(result.get("command", actual), result)
        return result

    def gdb_poll(self, timeout: float = 0.0) -> Dict[str, Any]:
        """Drain pending async GDB notifications."""
        logger.info("GDB poll")
        self.gdb.initialize()
        result = self.gdb.drain_async(timeout_sec=timeout)
        self.session.update_state(result["state"])
        self.session.record_command("gdb_poll", result)
        return result

    def gdb_interrupt(self, timeout: float = 1.0) -> Dict[str, Any]:
        """Interrupt the inferior and drain async notifications."""
        logger.info("GDB interrupt")
        self.gdb.initialize()
        result = self.gdb.interrupt(timeout_sec=timeout)
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
        self, location: str, condition: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set a breakpoint and return a typed breakpoint record."""
        logger.info(f"Set breakpoint at {location}")
        self.gdb.initialize()
        result = self.debugger.decorate_created_breakpoint(
            self.gdb.set_breakpoint(location, condition)
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
        self.session.update_state(result.get("state", self.gdb.get_state()))
        self.session.record_command(result.get("command", "debugger"), result)
        return result

    def set_watchpoint(self, expression: str, access: str = "write") -> Dict[str, Any]:
        """Set a typed write, read, or access watchpoint."""
        self.gdb.initialize()
        result = self.debugger.set_watchpoint(expression, access)
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

    def write_memory(
        self, address: str, data_hex: str, verify: bool = True
    ) -> Dict[str, Any]:
        """Write bytes to inferior memory with optional read-back verification."""
        self.gdb.initialize()
        return self._record_debugger(
            self.debugger.write_memory(address, data_hex, verify=verify)
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
