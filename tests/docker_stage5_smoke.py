"""Manual crash/core smoke for the project's pinned Docker runtime."""

import asyncio
import base64
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
from typing import Any, Dict, Optional, Tuple

from fastmcp import Client

from paomcp.artifacts import (
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    ArtifactStore,
    WriterNotQuiescentError,
)
from paomcp.server import create_mcp
from paomcp.services import AppServices, close_services, create_services
from paomcp.state.registry import DebugSessionRegistry
from paomcp.tools.backends.crash import CoreDumpGenerationError

# The only error types that legitimately signal an enforced/aborted oversized
# core.  Anything else (a crash in our own code, a GDB launch failure, etc.)
# must NOT be silently accepted as "the cap worked".
EXPECTED_CAP_ERRORS = (
    ArtifactSizeLimitError,
    ArtifactStorageFullError,
    CoreDumpGenerationError,
    WriterNotQuiescentError,
)


def require_success(name, result):
    if not result.get("success"):
        raise RuntimeError(f"{name} failed: {result}")
    return result


async def _exercise_stage5_via_mcp(
    services: AppServices,
    *,
    session_id: str,
    cap: int,
) -> Dict[str, Dict[str, Any]]:
    """Exercise every public crash-triage API through the MCP router and envelope."""
    mcp = create_mcp(services)
    async with Client(mcp) as client:
        published = {tool.name for tool in await client.list_tools()}
        required = {
            "run_until_crash",
            "capture_crash_snapshot",
            "analyze_crash",
            "classify_crash",
            "compare_crashes",
            "find_cyclic_offset",
            "capture_core_dump",
            "generate_crash_report",
        }
        missing = sorted(required - published)
        if missing:
            raise RuntimeError(f"crash-triage MCP tools are not published: {missing}")

        async def call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            result = await client.call_tool(name, arguments)
            data = result.data
            if not isinstance(data, dict):
                raise RuntimeError(f"{name} returned no typed data: {result}")
            return require_success(f"{name} MCP", data)

        crashed = await call(
            "run_until_crash",
            {
                "session_id": session_id,
                "timeout_seconds": 20,
            },
        )
        crash_parsed = crashed["data"]["parsed"]
        if not crash_parsed.get("crashed"):
            raise RuntimeError(f"Expected a fatal crash: {crashed}")
        crash_id = crash_parsed["crash_id"]

        snapshot = await call(
            "capture_crash_snapshot",
            {
                "expected_crash_id": crash_id,
                "session_id": session_id,
            },
        )
        analyzed = await call(
            "analyze_crash",
            {
                "expected_crash_id": crash_id,
                "session_id": session_id,
            },
        )
        classified = await call(
            "classify_crash",
            {
                "expected_crash_id": crash_id,
                "session_id": session_id,
            },
        )
        compared = await call(
            "compare_crashes",
            {
                "left_crash_id": crash_id,
                "right_crash_id": crash_id,
                "session_id": session_id,
            },
        )
        cyclic = await call(
            "find_cyclic_offset",
            {
                "value": "0x61616162",
                "expected_crash_id": crash_id,
                "session_id": session_id,
                "n": 4,
                "endian": "little",
            },
        )
        captured = await call(
            "capture_core_dump",
            {
                "expected_crash_id": crash_id,
                "session_id": session_id,
                "max_size_bytes": cap,
                "timeout_seconds": 30,
            },
        )
        artifact_id = captured["data"]["parsed"]["artifact"]["artifact_id"]
        report = await call(
            "generate_crash_report",
            {
                "expected_crash_id": crash_id,
                "session_id": session_id,
                "core_artifact_id": artifact_id,
                "report_format": "json",
            },
        )
        return {
            "run_until_crash": crashed,
            "capture_crash_snapshot": snapshot,
            "analyze_crash": analyzed,
            "classify_crash": classified,
            "compare_crashes": compared,
            "find_cyclic_offset": cyclic,
            "capture_core_dump": captured,
            "generate_crash_report": report,
        }


def _deleted_but_open_core_fds(pid: int) -> list[str]:
    """Return any deleted regular files the gdb pid still holds open (Linux only).

    A deleted-but-open descriptor is the exact writer-quiescence leak we guard
    against: if gdb still holds the unlinked core open, /proc/<pid>/fd shows a
    symlink target ending in " (deleted)".
    """
    fd_dir = f"/proc/{pid}/fd"
    if not os.path.isdir(fd_dir):
        return []
    leaked = []
    for entry in os.listdir(fd_dir):
        try:
            target = os.readlink(os.path.join(fd_dir, entry))
        except OSError:
            continue
        if target.endswith("(deleted)") and "artifact" in target.lower():
            leaked.append(target)
    return leaked


def _read_fsize_limit(pid: Optional[int]) -> Optional[Tuple[int, int]]:
    """Read the live gdb RLIMIT_FSIZE pair, or return None after process exit."""
    if pid is None:
        return None
    try:
        import resource

        return resource.prlimit(pid, resource.RLIMIT_FSIZE)
    except (AttributeError, OSError, ProcessLookupError):
        return None


def _gdb_return_code(gdb: Any) -> Optional[int]:
    """Return the original gdb subprocess return code when it is observable."""
    controller = getattr(gdb, "controller", None)
    process = getattr(controller, "gdb_process", None) if controller else None
    if process is None or not hasattr(process, "poll"):
        return None
    return process.poll()


def _allocated_bytes(root: str) -> int:
    """Return allocated bytes for regular files below root (Linux smoke helper)."""
    total = 0
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            path = os.path.join(directory, filename)
            try:
                stat_result = os.stat(path, follow_symlinks=False)
            except (FileNotFoundError, OSError):
                continue
            blocks = getattr(stat_result, "st_blocks", None)
            total += (
                int(blocks) * 512 if blocks is not None else int(stat_result.st_size)
            )
    return total


def _observe_fsize_cycle(gdb: Any) -> Dict[str, list[Dict[str, Any]]]:
    """Record the exact limit calls made internally by generate_core_file."""
    events: Dict[str, list[Dict[str, Any]]] = {"apply": [], "restore": []}
    original_apply = gdb.apply_fsize_limit
    original_restore = gdb.restore_fsize_limit

    def observed_apply(cap_bytes: int) -> Dict[str, Any]:
        pid = gdb.get_gdb_process_pid()
        before = _read_fsize_limit(pid)
        state = original_apply(cap_bytes)
        after = _read_fsize_limit(pid)
        events["apply"].append(
            {
                "requested_cap": cap_bytes,
                "pid": pid,
                "before": before,
                "after": after,
                "state": dict(state),
            }
        )
        return state

    def observed_restore(previous: Dict[str, Any]) -> bool:
        pid = gdb.get_gdb_process_pid()
        before = _read_fsize_limit(pid)
        restored = original_restore(previous)
        after = _read_fsize_limit(pid)
        events["restore"].append(
            {
                "pid": pid,
                "before": before,
                "after": after,
                "restored": restored,
            }
        )
        return restored

    # generate_core_file resolves these methods from the controller instance.
    # Wrapping them here observes the production path without a separate probe.
    gdb.apply_fsize_limit = observed_apply
    gdb.restore_fsize_limit = observed_restore
    return events


def _assert_successful_fsize_cycle(
    events: Dict[str, list[Dict[str, Any]]], requested_cap: int
) -> None:
    """Require one real apply/restore cycle around successful core generation."""
    if len(events["apply"]) != 1 or len(events["restore"]) != 1:
        raise RuntimeError(
            "Successful generate_core_file did not perform exactly one RLIMIT "
            f"apply/restore cycle: {events}"
        )
    applied = events["apply"][0]
    restored = events["restore"][0]
    if not applied["state"].get("applied"):
        raise RuntimeError(
            f"Successful core generation did not apply RLIMIT_FSIZE: {applied}"
        )
    if applied["after"] is None or applied["after"][0] > requested_cap:
        raise RuntimeError(
            f"Successful core generation did not lower RLIMIT_FSIZE: {applied}"
        )
    if restored["restored"] is not True or restored["after"] != applied["before"]:
        raise RuntimeError(
            "Successful core generation did not restore the original "
            f"RLIMIT_FSIZE pair: apply={applied}, restore={restored}"
        )


def main() -> None:
    runtime_dir = tempfile.mkdtemp(prefix="stage5-smoke-", dir="/var/lib/pao")
    source_path = os.path.join(runtime_dir, "crash.c")
    binary_path = os.path.join(runtime_dir, "crash")
    with open(source_path, "w", encoding="utf-8") as source:
        source.write(
            "int main(void) { volatile int *pointer = (int *)0; "
            "*pointer = 7; return 0; }\n"
        )
    subprocess.run(
        ["gcc", "-g", "-O0", "-fno-omit-frame-pointer", source_path, "-o", binary_path],
        check=True,
    )

    services = create_services(
        workspace_root=runtime_dir,
        runtime_root=os.path.join(runtime_dir, "runtime"),
    )
    registry = services.session_registry
    session = registry.ensure_session()
    gdb = session.gdb
    store = services.artifact_store
    tools = session.tools
    summary = {}
    try:
        require_success("set_file", tools.set_file(binary_path))
        cap = 128 * 1024 * 1024
        fsize_events = _observe_fsize_cycle(gdb)
        results = asyncio.run(
            _exercise_stage5_via_mcp(
                services,
                session_id=session.session_id,
                cap=cap,
            )
        )
        crashed = results["run_until_crash"]
        parsed = crashed["data"]["parsed"]
        if not parsed.get("crashed"):
            raise RuntimeError(f"Expected a fatal crash: {crashed}")
        if (parsed.get("stop", {}).get("signal") or {}).get("name") != "SIGSEGV":
            raise RuntimeError(f"Expected SIGSEGV: {crashed}")
        crash_id = parsed["crash_id"]
        summary["crash_id"] = crash_id
        summary["signal"] = parsed["stop"]["signal"]["name"]
        summary["snapshot_complete"] = parsed.get("snapshot_complete")

        snapshot = results["capture_crash_snapshot"]["data"]["parsed"]
        if snapshot.get("crash_snapshot", {}).get("crash_id") != crash_id:
            raise RuntimeError(f"Snapshot is not bound to the crash: {snapshot}")

        analyzed = results["analyze_crash"]["data"]["parsed"]
        summary["category"] = analyzed["analysis"]["category"]

        classified = results["classify_crash"]["data"]["parsed"]
        signature = classified.get("signature") or {}
        if signature.get("schema") != "pao.crash-fingerprint.v1":
            raise RuntimeError(f"Missing typed crash signature: {classified}")
        summary["classification"] = {
            "primary": (classified.get("classification") or {}).get("primary"),
            "signature": signature.get("digest"),
        }

        compared = results["compare_crashes"]["data"]["parsed"]
        if compared.get("relationship") not in {"same", "unknown"}:
            raise RuntimeError(f"A crash differs from itself: {compared}")
        summary["self_comparison"] = {
            "relationship": compared.get("relationship"),
            "confidence": compared.get("confidence"),
        }

        cyclic = results["find_cyclic_offset"]["data"]["parsed"]
        if not any(match.get("offset") == 4 for match in cyclic.get("matches", [])):
            raise RuntimeError(f"Unexpected cyclic scan result: {cyclic}")
        if cyclic.get("crash_scan", {}).get("crash_id") != crash_id:
            raise RuntimeError(
                f"Cyclic scan is not bound to the current crash: {cyclic}"
            )
        summary["cyclic_offset"] = next(
            match["offset"] for match in cyclic["matches"] if match.get("offset") == 4
        )

        estimate = tools.crash.estimate_core_size()
        summary["core_estimate"] = {
            "success": estimate.get("success"),
            "estimated_upper_bound_bytes": estimate.get("estimated_upper_bound_bytes"),
            "mapping_count": estimate.get("mapping_count"),
        }
        captured = results["capture_core_dump"]
        report_response = results["generate_crash_report"]
        capture_parsed = captured["data"]["parsed"]
        artifact = store.get(capture_parsed["artifact"]["artifact_id"])
        generation = capture_parsed["generation"]
        enforcement = generation.get("limit_enforcement", {})
        if enforcement.get("kernel_rlimit_applied") is not True:
            raise RuntimeError(
                "Successful core generation did not report kernel RLIMIT "
                f"enforcement: {enforcement}"
            )
        if enforcement.get("kernel_rlimit_restored") is not True:
            raise RuntimeError(
                "Successful core generation did not report RLIMIT restoration: "
                f"{enforcement}"
            )
        _assert_successful_fsize_cycle(fsize_events, cap)
        chunk = store.read_for_session(
            artifact.artifact_id, session.session_id, offset=0, length=64
        )
        if "content" in chunk:
            raise RuntimeError("Binary artifact unexpectedly exposed decoded text")
        header = base64.b64decode(chunk["content_base64"])
        if header[:4] != b"\x7fELF":
            raise RuntimeError("Core artifact is not ELF")
        verified = store.verify_for_session(artifact.artifact_id, session.session_id)
        if not verified.get("valid"):
            raise RuntimeError(f"Core artifact verification failed: {verified}")
        summary["core"] = {
            "size_bytes": artifact.size_bytes,
            "sha256": artifact.sha256,
            "elf_type": generation["validation"]["elf_type"],
            "source": generation["source"],
            "limit_enforcement": enforcement,
        }
        report_artifact_id = report_response["data"]["parsed"]["artifact"][
            "artifact_id"
        ]
        report_verification = store.verify_for_session(
            report_artifact_id, session.session_id
        )
        if not report_verification.get("valid"):
            raise RuntimeError(
                f"Crash report artifact verification failed: {report_verification}"
            )
        report_metadata = store.get_for_session(report_artifact_id, session.session_id)
        report_payload = store.read_for_session(
            report_artifact_id,
            session.session_id,
            offset=0,
            length=report_metadata.size_bytes,
        )
        report_document = json.loads(report_payload["content"])
        report_core = report_document.get("core_artifact") or {}
        if (
            report_core.get("artifact_id") != artifact.artifact_id
            or report_core.get("integrity_verified") is not True
        ):
            raise RuntimeError(
                f"Crash report did not bind the verified core artifact: {report_core}"
            )
        summary["report"] = {
            "artifact_id": report_artifact_id,
            "sha256": report_metadata.sha256,
            "core_integrity_verified": True,
        }

        help_result = require_success(
            "generate-core-file help", gdb.execute_command("help generate-core-file")
        )
        summary["gdb_core_cli"] = bool(help_result.get("responses"))

        # Blocker B/C proof: force an undersized cap and confirm the write is
        # aborted, no staging survives, and free space is restored.
        summary["undersized_cap"] = _verify_undersized_cap(
            tools,
            store,
            crash_id,
            gdb,
            binary_path,
            registry,
            session,
            fsize_events,
        )

        print(json.dumps({"success": True, "summary": summary}, sort_keys=True))
    finally:
        close_services(services)
        shutil.rmtree(runtime_dir, ignore_errors=True)


def _verify_undersized_cap(
    tools: Any,
    store: ArtifactStore,
    crash_id: str,
    gdb: Any,
    binary_path: str,
    registry: DebugSessionRegistry,
    original_session: Any,
    fsize_events: Dict[str, list[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Force a cap below the real core size; prove abort + cleanup + no leak.

    The core-dump path applies a transient RLIMIT_FSIZE to gdb's own pid for the
    duration of ``generate-core-file`` only, so on Linux the kernel raises
    SIGXFSZ at the cap (zero overshoot) without touching the inferior's limits.
    The artifact layer must then leave no staging file behind, the gdb pid must
    hold no deleted-but-open core descriptor, and disk use must not run away.
    """
    staging_dir = store.root_dir
    free_before = shutil.disk_usage(staging_dir).free
    allocated_before = _allocated_bytes(staging_dir)
    gdb_pid = gdb.get_gdb_process_pid()
    tiny_cap = 4096  # far below any real core
    fsize_events["apply"].clear()
    fsize_events["restore"].clear()

    error_type = None
    error_message = None
    peak_staging_bytes = 0

    def _track_peak(path: str) -> Dict[str, Any]:
        nonlocal peak_staging_bytes
        sampler_done = threading.Event()
        sampler_errors = []

        def sample_once() -> None:
            nonlocal peak_staging_bytes
            try:
                peak_staging_bytes = max(peak_staging_bytes, os.path.getsize(path))
            except FileNotFoundError:
                pass
            except OSError as exc:
                sampler_errors.append(exc)

        def sample_until_done() -> None:
            while not sampler_done.wait(0.002):
                sample_once()

        sampler = threading.Thread(
            target=sample_until_done,
            name="stage5-peak-sampler",
            daemon=True,
        )
        sampler.start()
        try:
            return tools.generate_core_file(
                crash_id,
                path,
                max_size_bytes=tiny_cap,
                timeout_seconds=30,
                min_free_bytes=0,
            )
        finally:
            # This runs before save_generated_file handles the exception and
            # removes/quarantines staging, so the final sample stays meaningful.
            sample_once()
            sampler_done.set()
            sampler.join(timeout=2)
            if sampler.is_alive():
                raise RuntimeError("Peak staging sampler did not terminate")
            if sampler_errors:
                raise RuntimeError(f"Peak staging sampler failed: {sampler_errors[0]}")

    try:
        store.save_generated_file(
            _track_peak,
            reserve_bytes=tiny_cap,
            max_size_bytes=tiny_cap,
            tool="capture_core_dump",
            session_id=original_session.session_id,
            content_type="application/x-coredump",
            encoding="binary",
        )
    except EXPECTED_CAP_ERRORS as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
    except Exception as exc:  # noqa: BLE001 - surface an unexpected failure loudly
        raise RuntimeError(
            f"Undersized cap raised an unexpected error type: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        raise RuntimeError("Undersized cap did not abort core generation")

    apply_events = fsize_events["apply"]
    restore_events = fsize_events["restore"]
    if len(apply_events) != 1:
        raise RuntimeError(
            "generate_core_file did not invoke apply_fsize_limit exactly once: "
            f"{apply_events}"
        )
    applied = apply_events[0]
    applied_state = applied["state"]
    if not applied_state.get("applied"):
        raise RuntimeError(
            "generate_core_file could not apply RLIMIT_FSIZE: " f"{applied_state}"
        )
    if applied["after"] is None or applied["after"][0] > tiny_cap:
        raise RuntimeError(
            "RLIMIT_FSIZE was not lowered by generate_core_file: " f"{applied}"
        )
    if len(restore_events) != 1:
        raise RuntimeError(
            "generate_core_file did not attempt restore_fsize_limit exactly once: "
            f"{restore_events}"
        )
    restored = bool(restore_events[0]["restored"])
    if restored and restore_events[0]["after"] != applied["before"]:
        raise RuntimeError(
            "generate_core_file restored a different RLIMIT_FSIZE pair: "
            f"apply={applied}, restore={restore_events[0]}"
        )

    # No .staging file may survive in the artifact root (quarantine files are a
    # separate, explicit signal and are also disallowed here since the writer
    # should have been confirmed dead on this platform).
    leftovers = [
        name
        for name in os.listdir(staging_dir)
        if name.endswith(".staging")
        or name.endswith(".quarantine")
        or name.endswith(".quarantine-marker")
    ]
    if leftovers:
        raise RuntimeError(f"Staging files survived an aborted core write: {leftovers}")

    # The gdb process must not still hold the unlinked core open.
    leaked_fds = _deleted_but_open_core_fds(gdb_pid) if gdb_pid is not None else []
    if leaked_fds:
        raise RuntimeError(f"gdb still holds a deleted core open: {leaked_fds}")

    free_after = shutil.disk_usage(staging_dir).free
    consumed = max(0, free_before - free_after)
    # Keep the filesystem-wide delta as diagnostic evidence only: unrelated
    # runner activity can move it. The scoped artifact allocation check below
    # is the deterministic leak gate.

    # Peak staging size must not exceed the cap (kernel enforcement guarantee).
    if peak_staging_bytes > tiny_cap:
        raise RuntimeError(
            f"Staging file grew to {peak_staging_bytes} bytes, exceeding "
            f"the {tiny_cap}-byte cap (RLIMIT_FSIZE not enforced)"
        )
    if peak_staging_bytes <= 0:
        raise RuntimeError(
            "Peak staging sampler observed no core bytes; cap enforcement was "
            "not proven"
        )

    gdb_return_code = _gdb_return_code(gdb)
    terminated_by_sigxfsz = gdb_return_code == -int(signal.SIGXFSZ)
    reached_exact_cap = peak_staging_bytes == tiny_cap
    if not reached_exact_cap and not terminated_by_sigxfsz:
        raise RuntimeError(
            "Undersized core raised an accepted exception without hard-cap "
            "evidence. An unrelated CoreDumpGenerationError must not pass the "
            f"smoke gate: error={error_type}: {error_message!r}, "
            f"peak={peak_staging_bytes}, cap={tiny_cap}, "
            f"gdb_return_code={gdb_return_code}"
        )

    allocated_after = _allocated_bytes(staging_dir)
    allocated_growth = max(0, allocated_after - allocated_before)
    if allocated_growth > tiny_cap:
        raise RuntimeError(
            "Aborted core write left allocated artifact data: "
            f"growth={allocated_growth} bytes, cap={tiny_cap}"
        )

    # Production either retains a clean controller after a successful restore,
    # or marks a dead/limited controller destroyed and rebuilds it through the
    # registry. Prove exactly one of those supported recovery paths.
    controller_destroyed = gdb.is_destroyed()
    recovered_session = registry.ensure_session(original_session.session_id)
    if controller_destroyed:
        if restored:
            raise RuntimeError(
                "Destroyed controller unexpectedly reported a successful limit restore"
            )
        if recovered_session is original_session:
            raise RuntimeError("Registry did not replace the destroyed controller")
        recovery_mode = "registry_rebuilt"
    else:
        if not restored:
            raise RuntimeError(
                "Live controller retained a lowered limit after restore failure"
            )
        if recovered_session is not original_session:
            raise RuntimeError("Registry replaced a healthy controller unexpectedly")
        recovery_mode = "same_controller"

    recovered_tools = recovered_session.tools
    require_success("set_file (recovery)", recovered_tools.set_file(binary_path))
    rerun = recovered_tools.run_until_crash(timeout_seconds=20)
    if not rerun.get("parsed", {}).get("crashed"):
        raise RuntimeError(f"GDB recovery failed after tiny-cap abort: {rerun}")

    return {
        "aborted_with": error_type,
        "error_message": error_message,
        "hard_cap_evidence": {
            "reached_exact_cap": reached_exact_cap,
            "terminated_by_sigxfsz": terminated_by_sigxfsz,
            "gdb_return_code": gdb_return_code,
        },
        "staging_leftovers": leftovers,
        "deleted_open_fds": leaked_fds,
        "disk_consumed_bytes": consumed,
        "artifact_allocated_growth_bytes": allocated_growth,
        "peak_staging_bytes": peak_staging_bytes,
        "gdb_pid_checked": gdb_pid,
        "fsize_apply": applied,
        "fsize_restore": restore_events[0],
        "recovery_mode": recovery_mode,
        "gdb_reusable": recovery_mode == "same_controller",
        "registry_rebuilt": recovery_mode == "registry_rebuilt",
    }


if __name__ == "__main__":
    main()
