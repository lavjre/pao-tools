---
name: reverse-engineering
description: Reverse engineer a binary statically and dynamically with pao-tools. Use when asked to analyze what a binary does,map its attack surface, decompile functions, trace execution, or document findings.
---

# Reverse engineering (static + dynamic reading)

Pairs with `PAO_PROFILE=rev` (static analysis, Ghidra + IDA, dynamic reading — no heap/ROP/crash/attack surface). Nothing here writes to the target.

## Workflow

1. **Map the binary.**
   - `analyze_elf` (sections, segments, entry), `checksec`, `inspect_got_plt`
   - `analyze_macho` for macOS/iOS targets
   - `fetch_repo` when source or a reference build exists upstream
2. **Decompile in layers.**
   - `decompile_binary` (Ghidra headless, whole binary or one function) for structure
   - `ida_idb_open` → `ida_decompile` / `ida_disasm` / `ida_list_funcs` / `ida_lookup` for precision, `ida_xrefs_to` / `ida_find_bytes` to chase references
   - `ida_publish_note` to write conclusions back into the IDB as comments
3. **Confirm dynamically (read-only).**
   - `create_debug_session` → `set_file` → breakpoints → `run`
   - `read_registers`, `get_memory`, `disassemble`, `search_memory` to verify what the static view claims
   - `scan_jit_regions` / `classify_memory_region` for JIT or packed code
4. **Record it.**
   - research session per target: notes for findings, hypotheses with states for open questions, evidence links to decomp artifacts
   - `export_research_session` to hand the case to someone (or to a future you)

## Rules

- Static first, dynamic to confirm — never assert behavior from bytes alone when a breakpoint is cheap.
- Name things in the IDB (`ida_rename`, `ida_set_comments`) as you learn them; a marked-up database is the deliverable.
- `execute` (raw GDB) is the escape hatch, not the workflow; prefer typed tools so evidence stays structured.
