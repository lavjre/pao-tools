"""FastMCP tool registration package."""

from fastmcp import FastMCP


def register_all_tools(mcp: FastMCP) -> None:
    from . import (
        artifacts,
        compatibility,
        debug,
        elf_runtime,
        inspect,
        processes,
        pwncli,
        python_env,
        repos,
        retdec,
    )

    artifacts.register(mcp)
    compatibility.register(mcp)
    debug.register(mcp)
    elf_runtime.register(mcp)
    inspect.register(mcp)
    processes.register(mcp)
    repos.register(mcp)
    python_env.register(mcp)
    pwncli.register(mcp)
    retdec.register(mcp)


__all__ = ["register_all_tools"]
