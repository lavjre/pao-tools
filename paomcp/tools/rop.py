"""MCP tools for typed ROP gadget indexing and libc identification (Stage 7)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.common import catch_errors, get_services


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def find_gadgets(
        binary_path: str,
        filter_pattern: Optional[str] = None,
        arch: Optional[str] = None,
        image_base: Optional[int] = None,
        bad_bytes: Optional[List[Any]] = None,
        max_results: int = 500,
        max_instructions: int = 8,
        include_analysis: bool = True,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Enumerate ROP gadgets from a binary with typed side-effect analysis.

        Static analysis via ropper: loads ``binary_path``, walks every
        candidate gadget, filters by an optional case-insensitive
        instruction-text regex, drops anything whose absolute address
        (encoded little-endian for the target pointer width) contains a
        listed bad byte, deduplicates identical (bytes, instructions)
        sightings, and returns per-gadget side effects: registers read /
        written / clobbered, stack delta (bytes), and per-instruction
        notes flagging unknown mnemonics.

        Args:
            binary_path: Absolute or workspace-relative path to the
                target binary.
            filter_pattern: Optional case-insensitive regex applied to
                the ``mnemonic op1, op2 ; ...`` text.  Omit for no
                filtering.
            arch: Optional architecture override (``x86_64``, ``i386``,
                ``aarch64``, ``arm``, etc.).  Auto-detected from the
                binary when omitted.
            image_base: Optional rebase address; when set every gadget
                is reported at ``image_base + offset``.
            bad_bytes: Optional list of byte values (ints 0-255 or
                ``"0x0a"`` strings) that any gadget address must not
                contain when encoded little-endian.
            max_results: Cap on returned gadgets (bounded at 20000).
            max_instructions: Drop gadgets longer than this before
                analysis, keeping the response cheap on large binaries.
            include_analysis: When false, drop the heavy
                ``instructions_analyzed`` payload from each gadget for
                small-response searches.
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
        """Score leaked ``{symbol: address}`` pairs against candidate libc files.

        Each candidate is checked with the invariant that every
        ``leak_address - libc_symbol_offset`` must yield the same base.
        A candidate is ``consistent`` when at least two leaks agree on
        a base; ``insufficient`` when only one leak matched;
        ``inconsistent`` when the bases disagree; ``unreadable`` when
        the candidate could not be parsed at all.  The top-level
        verdict aggregates those into ``confirmed`` (single candidate,
        >= 3 matched leaks), ``likely`` (single candidate, 2 matched),
        ``ambiguous`` (multiple candidates consistent), ``insufficient``
        (need more leaks) or ``rejected`` (no candidate matches).

        Args:
            leaks: Mapping of libc symbol name (``"puts"``, ``"read"``,
                ...) to the leaked absolute runtime address.
            candidate_libcs: Ordered list of candidate ``libc.so`` file
                paths to score.  A common corpus is every version in
                ``libc-database`` extracted locally.
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
        """Assemble a typed ROP chain for a real exploit shape.

        Supported ``kind`` values (x86_64 today):

        * ``ret2libc``    -- loads rdi, rsi, rdx via ``pop <reg> ; ret``
          gadgets and transfers control to ``target_address`` (compute
          this via ``identify_libc`` + a libc symbol offset first).  A
          single ``ret`` gadget is inserted before the target call for
          16-byte SysV stack alignment unless ``align_stack=False``.
        * ``ret2syscall`` -- loads rax with ``syscall_number`` then
          sets rdi, rsi, rdx, r10, r8, r9 (Linux SYSCALL calling
          convention; SYSCALL clobbers rcx and r11) before executing a
          ``syscall ; ret`` gadget.

        The returned chain carries the typed entry list, the packed
        little-endian ``raw_bytes`` a caller can splice directly into
        an overflow payload, and a ``missing_gadgets`` list plus
        ``verdict = "missing_gadgets"`` when a required gadget was not
        available.

        Args:
            binary_path: Binary to search for chain gadgets.  Usually
                the target executable itself.
            kind: ``"ret2libc"`` or ``"ret2syscall"``.
            arch: Target architecture (only ``"x86_64"`` is supported
                today; other arches return ``verdict="unsupported_arch"``).
            image_base: Optional rebase for the gadget search.
            bad_bytes: Optional forbidden bytes for gadget addresses.
            target_address: ret2libc: absolute address to call.
            target_symbol: ret2libc: cosmetic label used in the entry
                purposes and ``target.symbol`` field.
            args: Positional arguments (int list) loaded into
                rdi/rsi/rdx/... in order.
            align_stack: ret2libc: prepend a bare ``ret`` gadget before
                the final call so the target sees 16-byte-aligned rsp.
            syscall_number: ret2syscall: syscall number for rax.
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


__all__ = ["register"]
