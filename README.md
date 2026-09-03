# pao-tools

`pao-tools` is a stateful MCP server for binary exploitation and low-level
security research. It gives AI agents controlled access to a live debugging
environment — GDB/pwndbg, interactive exploit processes, Python tooling,
repositories, and the analysis workflows built on top of them — and returns
typed, versioned evidence instead of raw ANSI-formatted terminal output.

Every conclusion the server produces (allocator state, heap corruption,
crash attribution, JIT memory surface, ROP chain validity, research findings)
carries the runtime evidence it was derived from, so an agent can act on the
answer and trust the record.

## What it does

At the core, `pao-tools` runs one debugger session per target and exposes
typed tooling over it:

- **Debugger control** — sessions, run/attach/stop, registers, memory, threads,
  breakpoints/watchpoints, expression evaluation, disassembly, and
  event-oriented stepping.
- **Target lifecycle** — one contract for local processes, PID attach,
  gdbserver, GDB-remote TCP, SSH, Docker, QEMU user, and QEMU system targets.
- **Static and runtime analysis** — ELF mitigations and structure, Mach-O
  parsing, libc identification, decompilation, and crash triage.
- **Heap analysis** — allocator detection and typed parsers for the families
  it supports, plus allocator-independent memory-region helpers.
- **Exploit engineering** — ROP gadget indexing, chain construction and
  validation, and targeted attack helpers.
- **Research workspace** — notes, hypotheses, evidence, snapshots, timelines,
  and exportable bundles for a private investigation.

Responses are structured and bounded: big payloads spill to integrity-checked
artifacts, long-running operations expose progress and cancellation, and every
tool answers in the same envelope.

## Features

### Debugger and process control

- Independent GDB/pwndbg sessions for parallel research, with runtime-verified
  versions, commands, categories, and feature groups
- Deterministic execution control through GDB/MI: run/attach/stop, stepping,
  interrupts, and asynchronous event polling
- Typed registers, register groups, frames/inferiors, memory search and write,
  disassembly, thread/frame selection, and reversible memory patches
- Typed breakpoint/watchpoint CRUD and debugger snapshots
- A unified platform manifest for debugger, decompiler, helper, architecture,
  allocator, emulator, and provider readiness

### Target management

- One target lifecycle contract across local processes, PID attach, gdbserver,
  GDB-remote TCP, SSH, Docker, QEMU user, and QEMU system providers
- Binary-safe interactive exploit workflows through `paocli`
- Managed helper processes with process-tree cleanup and a shared Python
  research environment with pwntools

### Static and crash analysis

- Typed ELF mitigations, sections, GOT/PLT, mappings, threads, libc, and address
  data, backed by cached static ELF models, strings, fingerprints, and binary
  comparison
- Mach-O parsing for macOS/iOS targets (`analyze_macho`)
- Typed fatal-signal records, crash snapshots, cyclic offsets, core artifacts,
  and evidence-preserving crash reports
- Offline Ghidra headless decompilation by function name, address, or full
  binary

### Heap analysis

- Allocator detection and family classification for glibc, musl mallocng,
  jemalloc, Scudo, and Apple libmalloc
- Typed glibc parsers: chunks, arenas, fastbins/tcache/regular bins with
  safe-linking evidence, corruption checks, free validation, and bounded heap
  snapshot diffs
- Apple libmalloc zone introspection: zone discovery, per-zone function and
  introspection tables, size-class probes, and pointer-ownership classification
- Kernel SLUB slab analysis: `struct kmem_cache` geometry and hardened
  freelist decoding from BTF
- Allocator-independent memory-region tracking: anonymous-executable (V8 JIT
  heap) candidate scanning and per-address region classification over vmmap
- Targeted attack helpers such as `poison_tcache_link` for write-what-where
  planning

### Exploit engineering

- Static ROP gadget and side-effect analysis, libc candidate scoring, optional
  one-gadget integration, typed chain construction, validation, and export
- SROP frame construction and static chain-constraint checks

### Research workspace

- Bounded in-memory research sessions with notes, hypotheses, evidence,
  snapshots, timelines, and typed final reports
- Research bundle export/import with SHA-256 integrity verification and
  rehydration quotas

### Operational

- Session-owned lossless artifacts with chunk delivery, SHA-256 verification,
  and retention control
- Versioned structured responses shared by every MCP tool, with pollable
  operation IDs, progress, cooperative cancellation, and resource caps
- HTTP and stdio MCP transports

## Response API

The current server contract exposes 188 MCP tools. Every tool returns the same
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

When `data` exceeds `PAO_INLINE_RESPONSE_BYTES` (32 KiB by default), it becomes
an artifact descriptor. Inline copies stay token-lean: verbose
`raw_responses` are summarized and very long lists are capped with exact
totals, while the full payload remains lossless in the artifact:

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
defaults to `format=auto` (decoded text for text artifacts, Base64 for binary;
`both` restores the legacy dual-field shape). By default
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

After install, the `pao-tools` MCP server appears as a stdio server and its 188
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

## IDA Pro MCP bridge

`pao-tools` speaks to an already running
[mrexodia/ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp) server instead of
reimplementing IDA analysis. Configure endpoints (defaults target a local IDA):

```bash
PAO_IDA_MCP_URL=http://127.0.0.1:13337/mcp    # GUI / legacy SSE
PAO_IDALIB_MCP_URL=http://127.0.0.1:8745/mcp  # headless idalib supervisor
```

Flow: `ida_idb_open(path)` returns a session id, pass it as `database` to
`ida_decompile` / `ida_disasm` / `ida_list_funcs` / `ida_xrefs_to` /
`ida_py_eval` / `ida_rename`, then `ida_idb_close`. Debugger tools
(`ida_dbg_start` / `continue` / `run_to` / `step` / `bps` / `add_bp` /
`regs` / `stacktrace` / `read` / `write`) need the IDA side started with
`?ext=dbg` and auto-append it. When IDA is not running,
`get_ida_status` reports `ready: false` and every bridge tool degrades to a
typed verdict instead of failing startup. Bridge defaults are token-lean
(limit 100, max 1000) and large payloads spill to artifacts like the rest.

## Scope and limits

`pao-tools` is designed for a private, single-user research workflow. The typed
heap surface covers glibc deeply; musl mallocng, jemalloc, and Scudo are detected
and represented through the allocator contract, but do not yet have typed internal
parsers, and allocation-event tracking is not included. The PartitionAlloc family
is deliberately not walked — its freelists are per-PartitionPage and encoded
`((slot ^ random) << 16)`, its metadata changes across Chrome versions, and
release builds are stripped — so Chrome work uses the allocator-independent
JIT/memory-region helpers instead of a parser.

Authentication, RBAC, multi-user tenancy, and public-service audit infrastructure
are intentionally out of scope. Resource caps, cleanup, artifact integrity,
loopback defaults, and debugger isolation remain operational safeguards for the
owner's machine.

## License

This project retains the original CC BY-NC-ND 4.0 license. See
[LICENSE](LICENSE) for details.

Credit: based on the original [pwno-mcp](https://github.com/pwno-io/pwno-mcp).
