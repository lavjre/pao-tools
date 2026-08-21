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
- Typed allocator discovery plus glibc chunks, arenas, bins, corruption checks,
  free validation, and bounded heap snapshot diffs
- Static ROP gadget and side-effect analysis, libc candidate scoring, optional
  one-gadget integration, typed chain construction, validation, and export
- One target lifecycle contract for local processes, PID attach, gdbserver,
  GDB-remote TCP, SSH, Docker, QEMU user, and QEMU system providers
- Bounded in-memory research workspaces with notes, hypotheses, evidence,
  snapshots, timelines, export/import bundles, and final reports
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

The current server contract exposes 150 MCP tools. Every tool returns the same
`api_version: "1.0"` envelope. Primary tool
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

## Install as a Claude Code plugin

The repository ships an in-repo Claude Code plugin (`.claude-plugin/` +
`.mcp.json`), so it installs the way [ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp)
does — no manual MCP config.

Requirements on the host: `uv`, GDB with `pwndbg` on `PATH` (gdb backend), and
optionally `pwndbg-lldb` for the lldb backend. The server is launched over stdio
by `uv` from the plugin root.

```bash
# Point Claude Code at this repository as a marketplace. From a local checkout:
claude plugin marketplace add ./pao-tools

# Or from GitHub (clone/install goes through your existing gh/ssh/https auth,
# so a private repository works for you and any collaborator with access):
claude plugin marketplace add git@github.com:lavjre/pao-tools.git

claude plugin install pao-tools@pao-tools
```

After install, the `pao-tools` MCP server appears as a stdio server and its 150+
tools (debugger sessions, heap analysis, ROP tooling, research sessions) are
available to Claude Code. Runtime state is written under `<plugin-root>/runtime`
(`PAO_RUNTIME_ROOT`) — a `runtime/` directory is created at install time and is
git-ignored.

Backend selection is environment-controlled and per-session:

- `PAO_DEBUGGER_BACKEND=gdb|lldb|auto` — default `auto` (prefers gdb; falls back
  to lldb when gdb is unavailable or the target is Mach-O).
- Each `create_debug_session` / `create_target` accepts `backend=` to override.
- All heap/ELF/pwndbg typed tooling requires the gdb/pwndbg backend; on an lldb
  session those tools return a typed `requires the gdb/pwndbg backend` verdict.

Validate the packaging locally with:

```bash
claude plugin validate .
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
discovery, typed ELF/static analysis, and the debugger/memory API. Stages 5-9 now
have implemented baselines; the status column records their intentionally bounded
scope and the remaining extension points.

| Stage | Goal | Status |
| --- | --- | --- |
| S5 | Typed crash triage: run, snapshot, analyze, classify, compare, cyclic scan, core evidence, and per-crash report artifact | Implemented. The CI definitions gate image publication on Linux container boot, `/healthz`, an HTTP MCP registration check, and a real crash/core smoke test; consult the workflow run for the current candidate rather than assuming that live gate passed. |
| S6 | Typed heap analysis: allocator discovery, glibc chunks, arenas, fastbins/tcache/regular bins, safe-linking evidence, corruption/free validation, and bounded snapshot history | Implemented for glibc, including reconciled typed-capability metadata. musl mallocng, jemalloc, and Scudo are detected through the allocator/plugin contract but do not yet have typed internal parsers. Allocation-event tracking is not included. |
| S7 | ROP gadget analysis, libc identification, one-gadget lookup, chain construction, static validation, and export | Implemented baseline. Gadget analysis is multi-architecture, but chain construction is currently x86-64 `ret2libc`/`ret2syscall`; SROP frames support x86-64/i386 and validation is static. The standard Docker image pins `one_gadget` 1.10.0; it remains optional for non-Docker installs. |
| S8 | Local process, PID attach, gdbserver, GDB-remote TCP, SSH, Docker, QEMU user, and QEMU system providers behind one target contract | Implemented provider matrix with reconciled aggregate metadata, strict SSH host-key verification, and registry-wide shutdown cleanup. Runtime availability still depends on the corresponding host executables, credentials, known-host entries, images, kernels, and root filesystems. SSH and Docker targets use a separate gdbserver/GDB-remote target for debugger attachment. |
| S9 | Research notes, hypotheses, evidence, named snapshots, timeline, export/import bundles, and final reporting | Implemented baseline. ZIP verification/import applies compressed, uncompressed, entry, control-document, expansion-ratio, integrity, canonical-path, and per-session record quotas before rehydration. The live registry remains bounded and in-memory, so explicit ZIP export/import is the durability path. |

The S5 `generate_crash_report` tool is an evidence-preserving report for one exact
crash. The S9 `generate_final_report` tool instead summarizes the evidence,
hypotheses, snapshots, and timeline that a caller linked into a research session.

The Linux image build and publish definitions now include a live S6-S9 gate that
checks tool registration plus representative glibc inspection, gadget/one-gadget
analysis, local target lifecycle, and research export/verify/import workflows.
This describes the configured CI gate, not a claimed local Docker pass; consult
the workflow run for the exact image candidate.

### Potential builds

- Add allocation-event tracking and typed musl, jemalloc, and Scudo internals.
- Extend ROP chain construction to more architectures and add debugger-backed
  dynamic chain validation.
- Expand live provider coverage beyond the current local-target smoke with
  repeatable SSH, Docker-provider, gdbserver, and QEMU integration fixtures.
- Add research autosave, direct artifact re-ingestion on import, and optionally
  signed or encrypted research bundles. Stream the bounded compressed ZIP
  container directly from disk if avoiding its current in-memory copy becomes
  important for very large private investigations.

This roadmap assumes a private, single-user research tool. Authentication, RBAC,
multi-user tenancy, and public-service audit infrastructure are intentionally out of
scope. Resource caps, cleanup, artifact integrity, loopback defaults, and debugger
isolation remain operational safeguards for the owner's machine rather than a
separate product stage. There is intentionally no S10 in this roadmap.

## License

This project retains the original CC BY-NC-ND 4.0 license. See
[LICENSE](LICENSE) for details.

Credit: based on the original [pwno-mcp](https://github.com/pwno-io/pwno-mcp).
