---
name: ctf-pwn
description: Solve a CTF pwn challenge fast with pao-tools. Use when asked to pwn a CTF binary, get a quick crash triage, build an exploit under time pressure, or find a flag.
---

# CTF pwn (time-boxed)

Pairs with `PAO_PROFILE=ctf` (alias of `pwn`). Same tools, different discipline: speed over completeness, flag over report.

## Workflow

1. **60-second recon.**
   - `checksec` → mitigations dictate the shape (no PIE/canary = ret2win/ROP fast path)
   - `analyze_elf` + `decompile_binary` on `main` and the obviously-named vuln function; skip everything else until needed
2. **Crash first, theory second.**
   - `create_debug_session` → `set_file` → `run` with cyclic input → `find_cyclic_offset` on the crash value
   - `run_until_crash` → `capture_crash_snapshot` → `analyze_crash`: signal + fault address is usually all you need
3. **Shortest path to control.**
   - ret2win symbol present? `build_rop_chain` ret2libc to it and skip the heap entirely
   - Need libc? `identify_libc` from one leak, `find_one_gadgets` before hand-rolling chains
   - Heap challenge? `classify_heap_corruption` + `analyze_heap_technique` for the named primitive, then `poison_tcache_link`-style planning; snapshots only around the winning primitive
4. **Confirm and flag.**
   - `confirm_chain` dry-run, then `paocli` or `export_chain` (pwntools snippet) for the real run
   - Flag goes in the final message, exploit stays reproducible from the session

## Rules

- Time-box each step: recon 5 min, crash 10 min, primitive 20 min, then simplify the plan instead of deepening it.
- `generate_bounty_report` is for real targets; CTF findings go in chat, not reports.
- Remote target? `create_target` (remote/tcp) + `attach_debugger` mirrors the local flow; mind the libcs differing between local and remote.
