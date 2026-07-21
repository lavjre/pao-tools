# pao-tools

`pao-tools` is a stateful MCP server for binary exploitation and low-level
security research. It gives AI agents controlled access to GDB/pwndbg,
interactive exploit processes, Python tooling, repositories, and supporting
analysis workflows.

## Features

- Independent GDB/pwndbg sessions for parallel research
- Runtime-verified GDB/pwndbg versions, commands, categories, and feature groups
- Typed ELF mitigations, sections, GOT/PLT, mappings, threads, libc, and address data
- Typed breakpoint/watchpoint CRUD, registers, expressions, memory search/write,
  disassembly, thread selection, and event-oriented stepping
- Deterministic execution control through GDB/MI
- Register, stack, memory, disassembly, and backtrace inspection
- Breakpoints, stepping, attach, interrupt, and asynchronous event polling
- Binary-safe interactive exploit workflows through `pwncli`
- Managed helper processes with process-tree cleanup
- Shared Python research environment with pwntools
- Workspace-scoped command execution and repository fetching
- Versioned structured responses shared by every MCP tool
- Lossless artifact and chunk delivery for large tool/process output
- HTTP and stdio MCP transports
- Optional RetDec integration

## Response API

Every MCP tool returns the same `api_version: "1.0"` envelope. Payloads are
preserved in full; the server does not truncate stdout, stderr, GDB responses,
events, or decompiled code. Small results stay inline. Large results are stored
as artifacts and retrieved losslessly with `read_artifact` byte ranges.

```json
{
  "api_version": "1.0",
  "success": true,
  "tool": "get_memory",
  "session_id": "chal-a",
  "data": {
    "command": "-data-read-memory-bytes $rsp 64",
    "responses": [],
    "state": "stopped"
  },
  "error": null,
  "warnings": [],
  "meta": {
    "duration_ms": 3
  }
}
```

When `data` exceeds `PAO_INLINE_RESPONSE_BYTES` (1 MiB by default), it becomes
an artifact descriptor:

```json
{
  "delivery": "artifact",
  "artifact_id": "4d3d4d939ae047a1a90a0ffda031bcd2",
  "tool": "get_memory",
  "session_id": "chal-a",
  "content_type": "application/json",
  "encoding": "utf-8",
  "size_bytes": 4194432,
  "sha256": "...",
  "created_at": "2026-07-21T12:00:00+00:00"
}
```

Use the artifact tools to inspect, stream, pin, delete, or clean stored output.
`read_artifact` defaults to 256 KiB per call and returns exact bytes as Base64
plus a decoded text convenience field. By default artifacts expire after 72
hours; the store uses a 10 GiB soft quota and preserves 2 GiB of free disk.

## Quick start

The recommended workflow is Docker. Create a workspace and place a target binary
inside it:

```bash
mkdir -p workspace
cp /path/to/chal workspace/chal
chmod +x workspace/chal
```

Build and run the server:

```bash
docker build -t pao-tools:latest . --platform linux/amd64

docker run --rm -p 5500:5500 \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v "$PWD/workspace:/workspace" \
  pao-tools:latest
```

The default MCP endpoint is:

```text
http://127.0.0.1:5500/mcp
```

For stdio transport:

```bash
python -m pwnomcp --stdio
```

## Local development

Python 3.12 or newer is required. GDB and pwndbg must be available for debugger
workflows.

```bash
uv sync --extra dev
uv run python -m pytest
uv run black --check .
uv run mypy pwnomcp --ignore-missing-imports --disable-error-code=import-untyped
```

Run the HTTP server locally:

```bash
uv run python -m pwnomcp
```

## Runtime layout

- `/workspace`: target binaries, scripts, build output, and cloned repositories
- `/tmp/pwno/sessions`: debugger and exploit-session runtime files
- `/tmp/pwno/processes`: managed process logs
- `/tmp/pwno/python`: shared Python virtual environment
- `/tmp/pwno/artifacts`: complete large-response payloads and metadata

## Pwndbg compatibility

Docker builds pin the official pwndbg portable release to `2026.02.18`. At startup,
the server queries the pwndbg instance inside its own runtime and records a non-fatal
compatibility report. Use `get_debugger_capabilities`, `list_pwndbg_commands`, and
`get_pwndbg_command_help` instead of assuming commands found in another environment
are installed in pao-tools.

## Roadmap

- Crash triage and core-dump analysis
- ROP gadget and libc workflows
- Local, remote, SSH, and QEMU target providers
- Research notes, snapshots, exports, and report generation

## Credits

Based on the original [pwno-mcp](https://github.com/pwno-io/pwno-mcp)
project. Thanks to its authors and contributors for the initial foundation.

## License

This project retains the original CC BY-NC-ND 4.0 license. See
[LICENSE](LICENSE) for details.
