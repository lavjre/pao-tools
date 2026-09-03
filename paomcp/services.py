import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationRegistry
from paomcp.paopipe import PaoPipe
from paomcp.state import DebugSession, DebugSessionRegistry
from paomcp.tools.backends import (
    GhidraDecompiler,
    GitTools,
    IdaProMCP,
    PythonTools,
    ResearchBundle,
    ResearchRegistry,
    ResearchReport,
    RopTools,
    SubprocessTools,
    TargetRegistry,
)
from paomcp.utils.paths import DEFAULT_WORKSPACE, RuntimePaths, build_runtime_paths

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    runtime_paths: RuntimePaths
    session_registry: DebugSessionRegistry
    default_session_id: Optional[str]
    subprocess_tools: SubprocessTools
    git_tools: GitTools
    python_tools: PythonTools
    decompiler: GhidraDecompiler
    ida: IdaProMCP
    artifact_store: ArtifactStore
    operation_registry: OperationRegistry
    debugger_compatibility: Dict[str, Any]
    rop_tools: RopTools
    target_registry: TargetRegistry
    research_registry: ResearchRegistry
    research_bundle: ResearchBundle
    research_report: ResearchReport
    paopipe_sessions: Dict[str, PaoPipe] = field(default_factory=dict)
    paopipe_lock: threading.Lock = field(default_factory=threading.Lock)


def create_services(
    workspace_root: str = DEFAULT_WORKSPACE,
    runtime_root: Optional[str] = None,
) -> AppServices:
    if not os.path.exists(workspace_root):
        try:
            os.makedirs(workspace_root, exist_ok=True)
            logger.info("Created default workspace directory: %s", workspace_root)
        except OSError as exc:
            logger.warning(
                "Could not create workspace directory %s: %s", workspace_root, exc
            )
            logger.info("Continuing without default workspace directory")

    runtime_paths = (
        build_runtime_paths(workspace_root)
        if runtime_root is None
        else build_runtime_paths(workspace_root, runtime_root)
    )
    session_registry = DebugSessionRegistry(runtime_paths)
    default_session: Optional[DebugSession] = None
    try:
        # Creating the default session resolves the debugger executable and
        # initializes it; a missing/unusable debugger must degrade the server
        # (non-debugger tools keep working) instead of failing startup.
        default_session = session_registry.create_session("default")
    except Exception as exc:
        logger.exception("Failed to create the default debug session")
        logger.warning("Continuing without a default debug session: %s", exc)

    if default_session is not None:
        init_result = default_session.gdb.initialize()
        logger.info("GDB initialization: %s", init_result.get("status"))

    if default_session is not None:
        try:
            debugger_compatibility = default_session.tools.get_debugger_capabilities(
                refresh=True
            )
        except Exception as exc:
            logger.exception("Pwndbg compatibility probe failed")
            debugger_compatibility = {
                "success": False,
                "ready": False,
                "error": str(exc),
                "type": type(exc).__name__,
            }
    else:
        debugger_compatibility = {
            "success": False,
            "ready": False,
            "error": "no default debug session (debugger spawn failed at startup)",
            "type": "DebugSessionUnavailable",
        }
    if debugger_compatibility.get("ready"):
        logger.info(
            "Pwndbg compatibility ready: version=%s commands=%s",
            debugger_compatibility.get("versions", {}).get("pwndbg"),
            debugger_compatibility.get("command_count"),
        )
    else:
        logger.warning(
            "Pwndbg compatibility is not ready: %s",
            debugger_compatibility.get("warnings")
            or debugger_compatibility.get("error"),
        )

    artifact_store = ArtifactStore(runtime_paths.artifacts_dir)
    operation_registry = OperationRegistry()
    decompiler = GhidraDecompiler(
        runtime_dir=runtime_paths.decompiler_dir,
        artifact_store=artifact_store,
        operation_registry=operation_registry,
    )
    logger.info("Local Ghidra decompiler ready=%s", decompiler.status()["ready"])
    ida = IdaProMCP()
    try:
        logger.info("IDA Pro MCP ready=%s", ida.status()["ready"])
    except Exception as exc:
        logger.warning("IDA status probe failed (degraded, not fatal): %s", exc)

    services = AppServices(
        runtime_paths=runtime_paths,
        session_registry=session_registry,
        default_session_id=(
            default_session.session_id if default_session is not None else None
        ),
        subprocess_tools=SubprocessTools(
            process_root=runtime_paths.processes_dir,
            workspace_root=runtime_paths.workspace_root,
            artifact_store=artifact_store,
            operation_registry=operation_registry,
        ),
        # Git repos land in the caller-visible workspace (docs promise
        # /workspace/<name>), and fetch_repo enforces containment against
        # this same root.
        git_tools=GitTools(workspace_dir=runtime_paths.workspace_root),
        python_tools=PythonTools(workspace_dir=runtime_paths.python_dir),
        decompiler=decompiler,
        ida=ida,
        artifact_store=artifact_store,
        operation_registry=operation_registry,
        debugger_compatibility=debugger_compatibility,
        rop_tools=RopTools(),
        target_registry=TargetRegistry(),
        research_registry=(research_registry := ResearchRegistry()),
        research_bundle=ResearchBundle(
            research_registry, artifact_store=artifact_store
        ),
        research_report=ResearchReport(research_registry),
    )
    artifact_store.start_cleanup_worker()
    return services


def close_services(services: AppServices) -> None:
    logger.info("Shutting down Pao Tools server components...")

    with services.paopipe_lock:
        for pipe in services.paopipe_sessions.values():
            try:
                pipe.kill()
            except Exception:
                # A wedged pipe must not abort the rest of shutdown:
                # target / subprocess / session / artifact cleanup still
                # needs to run below.
                logger.exception("Failed to stop a PaO pipe during shutdown")
        services.paopipe_sessions.clear()

    target_cleanup = services.target_registry.close_all()
    if target_cleanup["errors"]:
        logger.warning(
            "Target cleanup completed with %d error(s): %s",
            len(target_cleanup["errors"]),
            target_cleanup["errors"],
        )

    services.subprocess_tools.close_all()
    services.session_registry.close_all()
    services.artifact_store.close()
