"""MCP tools for typed ROP gadget indexing and libc identification (Stage 7)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.backends.chain_confirm import ChainConfirmer
from paomcp.tools.common import (
    catch_errors,
    get_services,
    resolve_debug_session,
    run_session_action,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def find_gadgets(
        binary_path: str,
        filter_pattern: Optional[str] = None,
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        bad_bytes: Optional[List[Any]] = None,
        max_results: int = 100,
        max_instructions: int = 8,
        include_analysis: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List ROP gadgets with side effects (regs read/written, stack delta).

        Args:
            binary_path: Target binary under /workspace.
            filter_pattern: Optional regex on gadget text.
            arch: Override (x86_64/i386/aarch64); auto-detected if omitted.
            image_base: Optional rebase; gadgets reported at base+offset.
            bad_bytes: Drop addresses containing these bytes (0-255/0x..).
            max_results: Cap (max 20000, default 500).
            max_instructions: Drop longer gadgets to keep it cheap.
            include_analysis: False drops per-instruction detail for speed.
        """
        services = get_services(ctx)
        return services.rop_tools.find_gadgets(
            binary_path=binary_path,
            filter_pattern=filter_pattern,
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            max_results=max_results,
            max_instructions=max_instructions,
            include_analysis=include_analysis,
        )

    @mcp.tool()
    @catch_errors()
    async def find_gadget_sequence(
        binary_path: str,
        steps: List[str],
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        bad_bytes: Optional[List[Any]] = None,
        prefer_shortest: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Find one gadget per pattern in ``steps`` for a chain scaffold.

        Each step is an independent case-insensitive regex applied to a
        gadget's instruction text.  With ``prefer_shortest`` (default)
        the shortest matching gadget wins.  The overall ``verdict`` is
        ``complete`` when every step matched; ``partial`` otherwise.
        Bad-byte-tainted gadgets are excluded from all slots.

        Args:
            binary_path: Absolute or workspace-relative path to the
                target binary.
            steps: Ordered list of instruction-text regexes, at most 32
                entries.  Example: ``["pop rdi ; ret", "pop rsi ; ret"]``.
            arch: Optional architecture override.
            image_base: Optional rebase address.
            bad_bytes: Optional forbidden byte values (see
                ``find_gadgets``).
            prefer_shortest: When true, sort candidates by shortest
                instruction count then address before matching.
        """
        services = get_services(ctx)
        return services.rop_tools.find_gadget_sequence(
            binary_path=binary_path,
            steps=steps,
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            prefer_shortest=prefer_shortest,
        )

    @mcp.tool()
    @catch_errors()
    async def inspect_gadget(
        binary_path: str,
        address: int,
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Look up a single gadget by absolute address.

        Returns ``verdict = "found"`` with the typed gadget payload when
        the address maps to a known gadget, or ``verdict = "not_found"``
        when it does not.  Useful for verifying that an address a caller
        already selected is genuinely a walkable gadget in this binary
        under the chosen rebase.

        Args:
            binary_path: Absolute or workspace-relative path to the
                target binary.
            address: Gadget address (respecting ``image_base``).
            arch: Optional architecture override.
            image_base: Optional rebase address.
        """
        services = get_services(ctx)
        return services.rop_tools.inspect_gadget(
            binary_path=binary_path,
            address=address,
            arch=arch,
            image_base=image_base,
        )

    @mcp.tool()
    @catch_errors()
    async def identify_libc(
        leaks: Mapping[str, int],
        candidate_libcs: List[str],
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Score {symbol: addr} leaks vs candidate libcs; verdict confirmed/likely/ambiguous.

        Args:
            leaks: Symbol name to leaked runtime address.
            candidate_libcs: libc.so paths to score.
        """
        services = get_services(ctx)
        return services.rop_tools.identify_libc(
            leaks=dict(leaks),
            candidate_libcs=list(candidate_libcs),
        )

    @mcp.tool()
    @catch_errors()
    async def find_one_gadgets(
        libc_path: str,
        timeout_sec: float = 30.0,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Enumerate one-gadget candidates in a libc via the Ruby tool.

        Shells out to the ``one_gadget`` executable when present on
        PATH.  Returns ``verdict="parsed"`` with per-gadget
        ``{offset, call, constraints[]}`` on success, ``verdict="empty"``
        when the tool ran but returned no gadgets,
        ``verdict="unsupported"`` when the executable is missing (with
        an install hint), or ``verdict="failed"`` with the tool's
        stderr on a non-zero exit.

        Args:
            libc_path: Absolute or workspace-relative path to the libc
                to scan.
            timeout_sec: Wall-clock budget for the subprocess call.
        """
        services = get_services(ctx)
        return services.rop_tools.find_one_gadgets(
            libc_path=libc_path,
            timeout_sec=timeout_sec,
        )

    @mcp.tool()
    @catch_errors()
    async def build_rop_chain(
        binary_path: str,
        kind: str,
        arch: str = "x86_64",
        image_base: Optional[int] = None,
        bad_bytes: Optional[List[Any]] = None,
        target_address: Optional[int] = None,
        target_symbol: Optional[str] = None,
        args: Optional[List[int]] = None,
        align_stack: bool = True,
        syscall_number: Optional[int] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Build ret2libc/ret2syscall chain; returns entries + raw_bytes.

        Args:
            binary_path: Binary to search.
            kind: "ret2libc" or "ret2syscall" (x86_64).
            arch: Only x86_64 supported today.
            image_base: Optional rebase.
            bad_bytes: Forbidden address bytes.
            target_address: ret2libc call target (use identify_libc first).
            target_symbol: Cosmetic label for target.
            args: Loaded into rdi/rsi/rdx in order.
            align_stack: Append ret for 16B alignment (default True).
            syscall_number: ret2syscall number for rax.
        """
        services = get_services(ctx)
        return services.rop_tools.build_rop_chain(
            binary_path=binary_path,
            kind=kind,
            arch=arch,
            image_base=image_base,
            bad_bytes=bad_bytes,
            target_address=target_address,
            target_symbol=target_symbol,
            args=list(args or []),
            align_stack=align_stack,
            syscall_number=syscall_number,
        )

    @mcp.tool()
    @catch_errors()
    async def inspect_sigreturn_frame(
        arch: str = "x86_64",
        registers: Optional[Mapping[str, int]] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Build a typed sigreturn (SROP) frame with per-register offsets.

        Delegates the byte-level layout to pwntools' ``SigreturnFrame``
        so the offsets stay authoritative, but presents the result
        under the typed backend envelope: ``frame_bytes`` (hex-encoded
        raw bytes), ``frame_size``, ``register_offsets`` (name ->
        byte offset within the frame), and ``syscall_number`` for
        rt_sigreturn on the requested arch.  ``x86_64`` and ``i386``
        are supported today; other arches return
        ``verdict="unsupported_arch"``.

        Args:
            arch: Target architecture (``"x86_64"`` or ``"i386"``).
            registers: Optional mapping of register-name to value; any
                register slot present in the arch's frame layout may
                be assigned here (e.g. ``{"rip": 0xdeadbeef, "rsp":
                0xcafe0000, "rdi": 0}``).
        """
        services = get_services(ctx)
        return services.rop_tools.inspect_sigreturn_frame(
            arch=arch,
            registers=dict(registers) if registers else None,
        )

    @mcp.tool()
    @catch_errors()
    async def explain_chain_constraints(
        chain: List[Mapping[str, Any]],
        arch: str = "x86_64",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Advisory: per-slot side effects and risky cross-slot overwrites.

        Walks a chain (as returned by ``build_rop_chain``'s ``entries``
        list) and reports per-slot ``reads`` / ``writes`` / ``clobbers``
        plus a ``risky_overwrites`` list that flags any register a
        later gadget writes over after an earlier slot loaded it.
        Advisory only -- it helps a caller spot a layout that is
        obviously wrong before running it.

        Args:
            chain: Sequence of chain entries; each entry is a dict
                with at minimum a ``kind`` field
                (``gadget`` / ``immediate`` / ``target``).
            arch: Target architecture (``"x86_64"`` recommended).
        """
        services = get_services(ctx)
        return services.rop_tools.explain_chain_constraints(
            chain=list(chain),
            arch=arch,
        )

    @mcp.tool()
    @catch_errors()
    async def validate_rop_chain(
        chain: List[Mapping[str, Any]],
        executable_mappings: Optional[List[Mapping[str, Any]]] = None,
        bad_bytes: Optional[List[Any]] = None,
        arch: str = "x86_64",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Static pass/fail/unknown for a chain (exec map, bad bytes, clobber).

        Args:
            chain: Entries from build_rop_chain.
            executable_mappings: vmmap list; omitted => verdict unknown.
            bad_bytes: Forbidden byte values.
            arch: Target arch.
        """
        services = get_services(ctx)
        return services.rop_tools.validate_rop_chain(
            chain=list(chain),
            executable_mappings=list(executable_mappings or []),
            bad_bytes=bad_bytes,
            arch=arch,
        )

    @mcp.tool()
    @catch_errors()
    async def confirm_chain(
        chain: List[Any],
        session_id: Optional[str] = None,
        stack_address: Optional[Any] = None,
        target_address: Optional[Any] = None,
        max_steps: int = 200,
        arch: str = "x86_64",
        restore: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Dry-run chain in live inferior; verdict reached_target/crashed/exhausted.

        Args:
            chain: Stack-ordered values, chain[0] runs first.
            session_id: Session with stopped inferior.
            stack_address: Scratch; omitted => current $sp.
            target_address: Success addr; default last chain value.
            max_steps: Budget, max 512.
            arch: x86_64/i386.
            restore: Restore $pc/$sp after (default True).
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )

        def _do_confirm() -> Dict[str, Any]:
            return ChainConfirmer(session.gdb).confirm(
                chain=list(chain),
                stack_address=stack_address,
                target_address=target_address,
                max_steps=max_steps,
                arch=arch,
                restore=restore,
            )

        result = await run_session_action(session, _do_confirm)
        result["session_id"] = session.session_id
        return result

    @mcp.tool()
    @catch_errors()
    async def export_chain(
        chain: List[Mapping[str, Any]],
        format: str = "raw",
        arch: str = "x86_64",
        variable_name: str = "chain",
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Serialize a chain into a raw / python / pwntools payload.

        Formats:

        * ``raw``       -- hex-encoded little-endian packed bytes
          (drop straight into a payload buffer).
        * ``python``    -- a self-contained ``struct.pack``-based
          Python 3 snippet (no pwntools dependency).
        * ``pwntools``  -- a ``p64(...)``-based snippet a caller can
          paste into a pwntools exploit template; also sets
          ``context.arch`` for you.

        Every non-``raw`` output annotates each slot with the chain
        entry's ``purpose`` field so the resulting snippet is
        self-documenting.

        Args:
            chain: Chain entries.
            format: One of ``"raw"``, ``"python"``, ``"pwntools"``.
            arch: Target architecture (drives pointer width and
                ``context.arch`` for pwntools output).
            variable_name: Name of the Python variable holding the
                payload in generated snippets.  Ignored for ``raw``.
        """
        services = get_services(ctx)
        return services.rop_tools.export_chain(
            chain=list(chain),
            format=format,
            arch=arch,
            variable_name=variable_name,
        )


__all__ = ["register"]
