# pao-tools

`pao-tools` is a stateful MCP server for binary exploitation and low-level
security research. It gives AI agents controlled access to GDB/pwndbg,
interactive exploit processes, Python tooling, repositories, and supporting
analysis workflows.

## Features

- Independent GDB/pwndbg sessions for parallel research
- Runtime-verified GDB/pwndbg versions, commands, categories, and feature groups
- Unified platform manifest for debugger, decompiler, helper, architecture,
  allocator, emulator, and provider readiness
- Typed ELF mitigations, sections, GOT/PLT, mappings, threads, libc, and address data
- Complete cached static ELF models, strings, fingerprints, and binary comparison
- Typed breakpoint/watchpoint CRUD, registers, expressions, memory search/write,
  disassembly, thread selection, and event-oriented stepping
- Typed frames/inferiors, register groups, debugger snapshots, and reversible patches
- Typed fatal-signal records, crash snapshots, cyclic offsets, core artifacts, and reports
- Deterministic execution control through GDB/MI
- Register, stack, memory, disassembly, and backtrace inspection
- Breakpoints, stepping, attach, interrupt, and asynchronous event polling
- Binary-safe interactive exploit workflows through `paocli`
- Managed helper processes with process-tree cleanup
- Shared Python research environment with pwntools
- Workspace-scoped command execution and repository fetching
- Versioned structured responses shared by every MCP tool
- Pollable operation IDs, progress, cooperative cancellation, and resource caps
- Session-owned lossless artifacts with chunk delivery and integrity verification
- HTTP and stdio MCP transports
- Offline Ghidra headless decompilation by function name, address, or full binary

## Response API

Every MCP tool returns the same `api_version: "1.0"` envelope. Primary tool
payloads spill to artifacts instead of being silently truncated. Small results stay
inline. Safety-bounded capability probes report original byte counts and explicit
truncation flags. Managed process output has a hard combined stdout+stderr budget;
crossing it kills the process tree and reports exact observed, stored, and dropped
bytes. Artifacts are retrieved losslessly with `read_artifact` byte ranges.

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
    "duration_ms": 3,
    "operation_id": "d93a1a07510149d38702051d2dafd818",
    "idempotency_key": null,
    "idempotency_replayed": false
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

Use the artifact tools to inspect, verify, stream, pin, delete, or clean stored output.
`read_artifact` defaults to 256 KiB per call, allows at most 4 MiB per call, and
returns exact bytes as Base64 plus a decoded text convenience field. By default
artifacts expire after 72 hours; the store uses a 10 GiB soft quota and preserves
2 GiB of free disk. Session-owned artifacts require the exact same `session_id`;
unscoped artifacts can only be accessed with `session_id` omitted.

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

docker run --rm -p 127.0.0.1:5500:5500 \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_ADMIN \
  --security-opt seccomp=unconfined \
  --security-opt apparmor=unconfined \
  -v "$PWD/workspace:/workspace" \
  -v "pao-runtime:/var/lib/pao" \
  pao-tools:latest
```

The default MCP endpoint is:

```text
http://127.0.0.1:5500/mcp
```

For stdio transport:

```bash
python -m paomcp --stdio
```

## Local development

Python 3.12 or newer is required. GDB and pwndbg must be available for debugger
workflows.

```bash
uv sync --extra dev
uv run python -m pytest
uv run black --check .
uv run mypy paomcp
```

Run the HTTP server locally:

```bash
uv run python -m paomcp
```

## Runtime layout

- `/workspace`: target binaries, scripts, build output, and cloned repositories
- `/var/lib/pao/sessions`: debugger and exploit-session runtime files in Docker
- `/var/lib/pao/processes`: managed process logs in Docker
- `/var/lib/pao/python`: mutable Python scripts and runtime state in Docker
- `/var/lib/pao/artifacts`: complete payloads, core files, and metadata in Docker
- `/opt/pao-python/shared_venv`: image-baked Python helper environment

Local runs still default to the platform temporary directory unless
`PAO_RUNTIME_ROOT` is set. Core capture uses the GDB CLI fallback
`generate-core-file` because GDB/MI has no dedicated core-generation command.
On Linux the response records whether the temporary `RLIMIT_FSIZE` cap was applied
and restored. Core capture fails closed when that kernel hard cap cannot be applied;
polling is secondary detection, not a zero-overshoot boundary. Core files can
contain credentials and other process-memory secrets, so restrict access to the
runtime volume and configure retention accordingly. Use only one server process per
artifact volume because store locking is process-local. The staging pathname also
does not isolate an actively hostile same-UID inferior; use a disposable VM or
stronger process/filesystem isolation for those targets.

## Pwndbg compatibility

Docker builds pin the official pwndbg portable release to `2026.02.18`. At startup,
the server queries the pwndbg instance inside its own runtime and records a non-fatal
compatibility report. Use `get_debugger_capabilities`, `list_pwndbg_commands`, and
`get_pwndbg_command_help` instead of assuming commands found in another environment
are installed in pao-tools.

## Roadmap

Stages 1-4 established the response/artifact foundation, compatibility and platform
discovery, typed ELF/static analysis, and the debugger/memory API.

| Stage | Goal | Status |
| --- | --- | --- |
| S5 | Typed crash triage: run, snapshot, analyze, classify, compare, cyclic scan, core evidence, and per-crash report artifact | Code and local unit/static validation complete. Release completion still requires the current candidate to pass the live Linux Docker boot, HTTP MCP handshake, and real crash/core smoke gate. |
| S6 | Typed glibc heap inspection: allocator discovery, arenas, bins/tcache, chunks, safe-linking evidence, and bounded history | Planned |
| S7 | ROP gadget, libc identification, one-gadget, chain construction, and validation workflows | Planned |
| S8 | Local, remote, SSH, Docker, and QEMU target providers behind one target contract | Planned |
| S9 | Research notes, persistent session snapshots, exports, and final multi-artifact reporting | Planned |

The S5 `generate_crash_report` tool is an evidence-preserving report for one exact
crash. It does not replace the broader cross-session research records and final
reporting planned for S9.

This roadmap assumes a private, single-user research tool. Authentication, RBAC,
multi-user tenancy, and public-service audit infrastructure are intentionally out of
scope. Resource caps, cleanup, artifact integrity, loopback defaults, and debugger
isolation remain operational safeguards for the owner's machine rather than a
separate product stage.

## License

This project retains the original CC BY-NC-ND 4.0 license. See
[LICENSE](LICENSE) for details.

Credit: based on the original [pwno-mcp](https://github.com/pwno-io/pwno-mcp).
