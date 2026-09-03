"""FastMCP tool registration package."""

from typing import Optional

from fastmcp import FastMCP

from paomcp.tools.profiles import resolve_profile


def register_all_tools(mcp: FastMCP, profile: Optional[str] = None) -> None:
    """Register tool modules, restricted to a task profile when named.

    ``profile`` is one of ``full`` (default), ``pwn``, ``rev``, ``bounty``
    (see :mod:`paomcp.tools.profiles`). Modules outside the profile are
    never registered, so ``list_tools`` only costs what the mission needs.
    """
    from . import (
        artifacts,
        attack,
        compatibility,
        crash,
        decompiler,
        debug,
        debugger_memory,
        elf_runtime,
        fuzz,
        heap,
        ida,
        inspect,
        kernel_slub,
        macho,
        operations,
        platform,
        processes,
        paocli,
        python_env,
        repos,
        regions,
        research,
        rop,
        static_analysis,
        targets,
    )

    enabled = resolve_profile(profile)
    modules = {
        "artifacts": artifacts,
        "attack": attack,
        "compatibility": compatibility,
        "crash": crash,
        "decompiler": decompiler,
        "debug": debug,
        "debugger_memory": debugger_memory,
        "elf_runtime": elf_runtime,
        "fuzz": fuzz,
        "heap": heap,
        "ida": ida,
        "inspect": inspect,
        "kernel_slub": kernel_slub,
        "macho": macho,
        "operations": operations,
        "platform": platform,
        "processes": processes,
        "paocli": paocli,
        "python_env": python_env,
        "repos": repos,
        "regions": regions,
        "research": research,
        "rop": rop,
        "static_analysis": static_analysis,
        "targets": targets,
    }
    for name, module in modules.items():
        if enabled is None or name in enabled:
            module.register(mcp)


__all__ = ["register_all_tools"]
