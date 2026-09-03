---
name: firmware-reversing
description: Reverse engineer firmware images statically with pao-tools. Use when asked to analyze a firmware dump, map its binaries and services, decompile target functions, or document findings.
---

# Firmware reversing (static-only)

Pairs with `PAO_PROFILE=firmware` (static analysis + decompilers, no live target at all). There is deliberately no debugger here: a blob on disk gets no inferior.

## Workflow

1. **Inventory the image.**
   - `analyze_elf` per extracted binary (sections, entry,arch), `checksec` for mitigations that survive in firmware builds
   - Filesystem extraction (binwalk/unblob-style carving) is not yet a tool — do it on the host, then point pao-tools at the extracted tree under `/workspace`
2. **Decompile the interesting parts.**
   - `decompile_binary` (Ghidra headless) for breadth, `ida_idb_open` → `ida_decompile` / `ida_disasm` for depth
   - `ida_find_bytes` for magic constants, version strings, crypto material; `ida_xrefs_to` from interesting imports (system, memcpy, socket handlers)
   - `ida_publish_note` / `ida_set_comments` to mark findings in the IDB as you go
3. **Trace data flow on paper.**
   - Network-facing parsers first (HTTP, UPnP, custom binary protocols): entry → parse → sink
   - `search_memory` is unavailable (no live target); reason from disassembly instead
4. **Record it.**
   - research session per image: notes per binary/service, hypotheses for suspected vulns with states, evidence links to decomp artifacts
   - `export_research_session` to hand the case over

## Rules

- No debugger means no confirmation: mark every claim `proposed` until a second static path corroborates it.
- Version strings and build dates go in session metadata first — firmware bugs are versioned bugs.
- When emulation becomes possible (QEMU user for the target arch), switch to the `rev` profile and confirm dynamically.
