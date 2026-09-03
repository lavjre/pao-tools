"""Live heap/ROP/targets/research smoke test for the project's pinned Docker runtime."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict

from fastmcp import Client

from paomcp.server import create_mcp
from paomcp.services import AppServices, close_services, create_services

STAGE6_TOOLS = {
    "detect_allocator",
    "get_allocator_capabilities",
    "inspect_chunk",
    "validate_chunk",
    "list_arenas",
    "inspect_arena",
    "inspect_fastbins",
    "inspect_tcache",
    "inspect_bins",
    "classify_heap_corruption",
    "validate_free",
    "capture_heap_snapshot",
    "list_heap_snapshots",
    "get_heap_snapshot",
    "diff_heap_snapshots",
    "link_snapshot_to_crash",
}
STAGE7_TOOLS = {
    "find_gadgets",
    "find_gadget_sequence",
    "inspect_gadget",
    "identify_libc",
    "find_one_gadgets",
    "build_rop_chain",
    "inspect_sigreturn_frame",
    "explain_chain_constraints",
    "validate_rop_chain",
    "export_chain",
}
STAGE8_TOOLS = {
    "create_target",
    "list_targets",
    "get_target_status",
    "get_target_capabilities",
    "start_target",
    "stop_target",
    "connect_target",
    "disconnect_target",
    "reconnect_target",
    "sync_target_files",
    "fetch_runtime_libraries",
    "attach_debugger",
}
STAGE9_TOOLS = {
    "create_research_session",
    "get_research_session",
    "list_research_sessions",
    "add_research_note",
    "link_evidence",
    "add_hypothesis",
    "update_hypothesis_state",
    "create_named_snapshot",
    "list_snapshots",
    "compare_snapshots",
    "append_timeline_event",
    "get_research_timeline",
    "export_research_session",
    "import_research_session",
    "verify_research_bundle",
    "generate_final_report",
}
REQUIRED_TOOLS = STAGE6_TOOLS | STAGE7_TOOLS | STAGE8_TOOLS | STAGE9_TOOLS


def _require_backend_success(name: str, response: Dict[str, Any]) -> Dict[str, Any]:
    if response.get("success") is not True:
        raise RuntimeError(f"{name} outer response failed: {response}")
    backend = response.get("data")
    if not isinstance(backend, dict):
        raise RuntimeError(f"{name} returned no backend data: {response}")
    parsed = backend.get("parsed")
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} returned no parsed payload: {backend}")
    return parsed


async def _exercise_mcp(
    services: AppServices,
    *,
    session_id: str,
    binary_path: str,
    bundle_path: str,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    async with Client(create_mcp(services)) as client:
        published = {tool.name for tool in await client.list_tools()}
        missing = sorted(REQUIRED_TOOLS - published)
        if missing:
            raise RuntimeError(
                f"heap/ROP/targets/research MCP tools are not published: {missing}"
            )
        summary["registered_tools"] = len(REQUIRED_TOOLS)

        async def call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            result = await client.call_tool(name, arguments)
            if not isinstance(result.data, dict):
                raise RuntimeError(f"{name} returned no MCP envelope: {result}")
            return _require_backend_success(name, result.data)

        allocator = await call("detect_allocator", {"session_id": session_id})
        if allocator.get("family") != "glibc":
            raise RuntimeError(f"Expected live glibc detection: {allocator}")
        allocator_caps = await call(
            "get_allocator_capabilities", {"session_id": session_id}
        )
        glibc_caps = (allocator_caps.get("allocators") or {}).get("glibc") or {}
        if (glibc_caps.get("typed_inspection_api") or {}).get("available") is not True:
            raise RuntimeError(f"Typed glibc API is not ready: {allocator_caps}")
        summary["stage6"] = {
            "family": allocator["family"],
            "version": allocator.get("version"),
            "typed_inspection": True,
        }

        gadgets = await call(
            "find_gadgets",
            {
                "binary_path": binary_path,
                "filter_pattern": "pop rdi.*ret",
                "max_results": 16,
            },
        )
        if not gadgets.get("gadgets"):
            raise RuntimeError(f"Ropper found no injected smoke gadget: {gadgets}")
        libc_path = "/lib/x86_64-linux-gnu/libc.so.6"
        one_gadgets = await call(
            "find_one_gadgets",
            {"libc_path": libc_path, "timeout_sec": 60.0},
        )
        if one_gadgets.get("verdict") != "parsed" or not one_gadgets.get("gadgets"):
            raise RuntimeError(f"Pinned one_gadget did not parse libc: {one_gadgets}")
        summary["stage7"] = {
            "gadget_count": len(gadgets["gadgets"]),
            "one_gadget_count": len(one_gadgets["gadgets"]),
        }

        created_target = await call(
            "create_target",
            {
                "kind": "local_process",
                "session_id": session_id,
                "config": {
                    "binary_path": "/bin/sleep",
                    "argv": ["30"],
                    "name": "stage8-live-smoke",
                },
            },
        )
        target_id = created_target["target"]["target_id"]
        started_target = await call("start_target", {"target_id": target_id})
        if (
            started_target.get("success") is not True
            or (started_target.get("target") or {}).get("state") != "running"
        ):
            raise RuntimeError(
                f"Local target did not enter running state: {started_target}"
            )
        status = await call("get_target_status", {"target_id": target_id})
        if (status.get("target") or {}).get("state") != "running":
            raise RuntimeError(f"Target status did not observe liveness: {status}")
        stopped_target = await call(
            "stop_target",
            {"target_id": target_id, "timeout_sec": 5.0, "remove": True},
        )
        if (
            stopped_target.get("success") is not True
            or (stopped_target.get("target") or {}).get("state") != "stopped"
            or stopped_target.get("removed") is not True
        ):
            raise RuntimeError(f"Local target did not stop cleanly: {stopped_target}")
        summary["stage8"] = {
            "kind": "local_process",
            "started": True,
            "stopped": True,
        }

        research = await call(
            "create_research_session",
            {
                "name": "stage9-live-smoke",
                "owning_debug_session": session_id,
                "metadata": {"binary_path": binary_path},
            },
        )
        research_id = research["session"]["session_id"]
        hypothesis = await call(
            "add_hypothesis",
            {
                "session_id": research_id,
                "statement": "The heap/ROP/targets/research Docker runtime is internally consistent",
            },
        )
        hypothesis_id = hypothesis["hypothesis"]["hypothesis_id"]
        await call(
            "add_research_note",
            {
                "session_id": research_id,
                "body": "Live Docker smoke completed representative heap/ROP/targets operations.",
                "tags": ["smoke"],
                "hypothesis_id": hypothesis_id,
            },
        )
        await call(
            "create_named_snapshot",
            {
                "session_id": research_id,
                "name": "live-runtime",
                "slots": {
                    "metadata": {
                        "allocator": allocator.get("family"),
                        "target_provider": "local_process",
                    }
                },
            },
        )
        await call(
            "update_hypothesis_state",
            {
                "session_id": research_id,
                "hypothesis_id": hypothesis_id,
                "state": "supported",
                "rationale": "Representative live operations and export round-trip passed.",
            },
        )
        report = await call(
            "generate_final_report",
            {"session_id": research_id, "formats": ["json", "markdown"]},
        )
        if not {"json", "markdown"} <= set((report.get("rendered") or {}).keys()):
            raise RuntimeError(f"Final report formats are incomplete: {report}")
        exported = await call(
            "export_research_session",
            {"session_id": research_id, "destination": bundle_path},
        )
        if not os.path.isfile(bundle_path):
            raise RuntimeError(
                f"Research export did not create {bundle_path}: {exported}"
            )
        verified = await call("verify_research_bundle", {"source_path": bundle_path})
        if verified.get("verdict") != "valid":
            raise RuntimeError(f"Research bundle failed verification: {verified}")
        imported = await call("import_research_session", {"source_path": bundle_path})
        imported_session_id = imported.get("session_id")
        if (
            not isinstance(imported_session_id, str)
            or not imported_session_id
            or imported_session_id == research_id
        ):
            raise RuntimeError(
                f"Research import returned an invalid or reused id: {imported}"
            )
        summary["stage9"] = {
            "bundle_valid": True,
            "imported_as_new_session": True,
            "report_formats": sorted((report.get("rendered") or {}).keys()),
        }
    return summary


def main() -> None:
    runtime_dir = tempfile.mkdtemp(prefix="stage6-9-smoke-", dir="/var/lib/pao")
    services: AppServices | None = None
    try:
        workspace_dir = os.path.join(runtime_dir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)
        source_path = os.path.join(workspace_dir, "stage6_9_smoke.c")
        binary_path = os.path.join(workspace_dir, "stage6_9_smoke")
        bundle_path = os.path.join(workspace_dir, "stage9-smoke.pao.zip")
        with open(source_path, "w", encoding="utf-8") as source:
            source.write(
                "#include <stdlib.h>\n"
                "__attribute__((used, naked)) void smoke_pop_rdi(void) "
                '{ __asm__("pop %rdi; ret"); }\n'
                "__attribute__((noinline)) void heap_marker(void *pointer) "
                '{ __asm__ volatile("" : : "r"(pointer) : "memory"); }\n'
                "int main(void) { void *pointer = malloc(0x80); "
                "if (!pointer) return 1; heap_marker(pointer); "
                "free(pointer); return 0; }\n"
            )
        subprocess.run(
            [
                "gcc",
                "-g",
                "-O0",
                "-fno-pie",
                "-no-pie",
                source_path,
                "-o",
                binary_path,
            ],
            check=True,
        )

        services = create_services(
            workspace_root=workspace_dir,
            runtime_root=os.path.join(runtime_dir, "runtime"),
        )
        session = services.session_registry.ensure_session()
        loaded = session.tools.set_file(binary_path)
        if loaded.get("success") is not True:
            raise RuntimeError(f"Could not load smoke binary: {loaded}")
        breakpoint = session.tools.set_breakpoint("heap_marker")
        if breakpoint.get("success") is not True:
            raise RuntimeError(f"Could not set heap marker: {breakpoint}")
        stopped = session.tools.run()
        if stopped.get("success") is not True or stopped.get("state") != "stopped":
            raise RuntimeError(f"Smoke inferior did not stop at heap marker: {stopped}")

        summary = asyncio.run(
            _exercise_mcp(
                services,
                session_id=session.session_id,
                binary_path=binary_path,
                bundle_path=bundle_path,
            )
        )
        print(json.dumps({"success": True, "summary": summary}, sort_keys=True))
    finally:
        try:
            if services is not None:
                close_services(services)
        finally:
            shutil.rmtree(runtime_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
