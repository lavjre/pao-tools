"""MCP tools for typed ROP gadget indexing (Stage 7.1)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


__all__ = ["register"]
