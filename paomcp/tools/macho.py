"""MCP tools for Mach-O static analysis (Apple binaries).

The repo's static model today is ELF-only (``analyze_static_binary``); this
module adds the Mach-O counterpart for Apple targets -- the file header, the
load-command walk (segments, entry point, loaded dylibs), and the build UUID,
all parsed with the pure-stdlib backend in ``backends/macho.py``.

Workspace-file based like the ELF tools: no debug session, read-only, the
binary resolved from the configured workspace root.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.tools.backends.macho import (
    FatMachOError,
    MachOError,
    load_macho,
    parse_fat_macho,
    parse_macho_slice,
)
from paomcp.tools.common import catch_errors, get_services, run_blocking
from paomcp.utils.paths import resolve_workspace_file


def _macho_envelope(binary_path: str) -> Dict[str, Any]:
    """Parse a Mach-O image into the typed ``analyze_macho`` envelope."""
    try:
        macho = load_macho(binary_path)
    except FatMachOError as exc:
        # A universal (fat) binary IS Mach-O: inventory the slices and
        # parse the first one so callers get segments/dylibs/uuid instead
        # of a bare recognized-but-unparsed verdict.
        try:
            with open(binary_path, "rb") as handle:
                blob = handle.read()
            slices = parse_fat_macho(blob)
            first = parse_macho_slice(blob, 0) if slices else None
        except (FatMachOError, MachOError, OSError) as inner:
            return {
                "success": True,
                "command": "analyze_macho",
                "parsed": {
                    "binary_path": binary_path,
                    "verdict": "fat_macho",
                    "is_macho": True,
                    "reason": f"{exc}; slice inventory failed: {inner}",
                    "slice": None,
                },
            }
        return {
            "success": True,
            "command": "analyze_macho",
            "parsed": {
                "binary_path": binary_path,
                "verdict": "fat_macho",
                "is_macho": True,
                "reason": str(exc),
                "slice_count": len(slices),
                "slices": [
                    {
                        "cpu_name": s.cpu_name,
                        "cputype": s.cputype,
                        "offset": s.offset,
                        "size": s.size,
                    }
                    for s in slices
                ],
                "slice": (
                    {
                        "index": 0,
                        "cpu_name": first.cpu_name,
                        "filetype_name": first.filetype_name,
                        "entry": first.entry,
                        "uuid": first.uuid,
                        "segments": [
                            {
                                "name": segment.name,
                                "vmaddr": segment.vmaddr,
                                "vmsize": segment.vmsize,
                                "initprot": segment.initprot,
                            }
                            for segment in first.segments
                        ],
                        "dylibs": first.dylibs,
                        "links_libsystem": first.links_libsystem,
                        "links_libsystem_malloc": first.links_libsystem_malloc,
                    }
                    if first is not None
                    else None
                ),
            },
        }
    except MachOError as exc:
        return {
            "success": True,
            "command": "analyze_macho",
            "parsed": {
                "binary_path": binary_path,
                "verdict": "not_macho",
                "is_macho": False,
                "reason": str(exc),
            },
        }
    except OSError as exc:
        return {
            "success": True,
            "command": "analyze_macho",
            "parsed": {
                "binary_path": binary_path,
                "verdict": "unreadable",
                "is_macho": False,
                "reason": f"cannot read binary: {exc}",
            },
        }
    return {
        "success": True,
        "command": "analyze_macho",
        "parsed": {
            "binary_path": binary_path,
            "verdict": "parsed",
            "is_macho": True,
            "magic": macho.magic,
            "is_64": macho.is_64,
            "cputype": macho.cputype,
            "cpu_name": macho.cpu_name,
            "cpusubtype": macho.cpusubtype,
            "filetype": macho.filetype,
            "filetype_name": macho.filetype_name,
            "flags": macho.flags,
            "ncmds": macho.ncmds,
            "sizeofcmds": macho.sizeofcmds,
            "entry": macho.entry,
            "entry_stacksize": macho.entry_stacksize,
            "uuid": macho.uuid,
            "segments": [
                {
                    "name": segment.name,
                    "vmaddr": segment.vmaddr,
                    "vmsize": segment.vmsize,
                    "initprot": segment.initprot,
                }
                for segment in macho.segments
            ],
            "dylibs": macho.dylibs,
            "links_libsystem": macho.links_libsystem,
            "links_libsystem_malloc": macho.links_libsystem_malloc,
        },
    }


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def analyze_macho(
        binary_path: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Parse a workspace Mach-O binary into a typed header + load-command model.

        Args:
            binary_path: Mach-O image under the configured workspace; relative
                paths resolve from the workspace root.

        The result reports the CPU/filetype identity, ``LC_MAIN`` entry point
        (a VM/file offset, ``None`` for dylibs), loaded dylibs (including
        whether ``libSystem.B.dylib`` / ``libsystem_malloc`` are linked),
        ``__TEXT``/``__DATA`` segments, and the build UUID.  Non-Mach-O input
        yields an honest ``not_macho`` verdict rather than an exception;
        a universal (fat) image yields ``fat_macho`` with a slice inventory
        plus the parsed first slice.
        """
        services = get_services(ctx)
        resolved = resolve_workspace_file(
            binary_path,
            services.runtime_paths.workspace_root,
            kind="binary_path",
        )

        def analyze_snapshot() -> Dict[str, Any]:
            return _macho_envelope(resolved)

        return await run_blocking(analyze_snapshot)
