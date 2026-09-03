"""FastMCP tool registration package."""

from fastmcp import FastMCP


def register_all_tools(mcp: FastMCP) -> None:
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

    artifacts.register(mcp)
    attack.register(mcp)
    compatibility.register(mcp)
    crash.register(mcp)
    decompiler.register(mcp)
    debug.register(mcp)
    debugger_memory.register(mcp)
    elf_runtime.register(mcp)
    fuzz.register(mcp)
    heap.register(mcp)
    ida.register(mcp)
    inspect.register(mcp)
    kernel_slub.register(mcp)
    macho.register(mcp)
    operations.register(mcp)
    platform.register(mcp)
    processes.register(mcp)
    repos.register(mcp)
    regions.register(mcp)
    python_env.register(mcp)
    paocli.register(mcp)
    research.register(mcp)
    rop.register(mcp)
    static_analysis.register(mcp)
    targets.register(mcp)


__all__ = ["register_all_tools"]
