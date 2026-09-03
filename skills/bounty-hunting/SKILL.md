---
name: bounty-hunting
description: Hunt vulnerabilities and produce triager-ready findings with pao-tools. Use when asked to find bugs in a target, triage a crash for security impact, fuzz for variant crashes, or write a bounty report.
---

# Bug bounty hunting (find → triage → report)

Pairs with `PAO_PROFILE=bounty` (everything except write primitives and interactive drivers). The mission ends in a report, so every step links evidence.

## Workflow

1. **Scope the target.**
   - research session per target; record binary/endpoint/version in session metadata (reports read it as the target spec)
   - `analyze_elf` + `checksec` + `decompile_binary` for attack surface; `fetch_repo` for version history and patch diffing (1-days live in diffs)
2. **Find crashes.**
   - `generate_fuzz_inputs` (deterministic seeds) → `run_fuzz_batch` under GDB; novel crashes dedup by signature and link straight into the research session
   - `analyze_kernel_oops` for kernel log crashes (fault kind, RIP, stable fingerprint)
3. **Triage for impact.**
   - `run_until_crash` → `capture_crash_snapshot` → `analyze_crash` → `classify_crash`
   - Heap targets: `detect_allocator`, `classify_heap_corruption`, `detect_uaf`, `validate_free` — write down which primitive the crash gives (read/write, control of what, how constrained)
   - `analyze_heap_technique` only if you are chasing exploitability, not for every crash
4. **File it.**
   - Hypotheses carry the argument: `supported` with rationale for root cause, `proposed` for open leads
   - `generate_bounty_report` (title, severity, affected target, remediation): summary, PoC from the reproduction guide, root cause from supported hypotheses only, explicit gaps
   - Link crash/heap/artifact evidence so a triager can replay without asking questions

## Rules

- One finding per report: split distinct bugs, never bundle.
- Severity needs a primitive, not a crash: a SIGSEGV with no controlled impact is low until proven otherwise.
- Every claim links evidence; anything unlinked goes under gaps, not into root cause.
