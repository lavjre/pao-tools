"""FastMCP tool registration package."""

from fastmcp import FastMCP


def register_all_tools(mcp: FastMCP) -> None:
    from . import (
        artifacts,
        compatibility,
        decompiler,
        debug,
        debugger_memory,
        elf_runtime,
        inspect,
        processes,
        paocli,
        python_env,
        repos,
    )

    artifacts.register(mcp)
    compatibility.register(mcp)
    decompiler.register(mcp)
    debug.register(mcp)
    debugger_memory.register(mcp)
    elf_runtime.register(mcp)
    inspect.register(mcp)
    processes.register(mcp)
    repos.register(mcp)
    python_env.register(mcp)
    paocli.register(mcp)


__all__ = ["register_all_tools"]
