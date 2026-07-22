import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from paomcp.artifacts import ArtifactStore
from paomcp.operations import OperationRegistry
from paomcp.paopipe import PaoPipe
from paomcp.state import DebugSessionRegistry
from paomcp.tools.backends import (
    GhidraDecompiler,
    GitTools,
    PythonTools,
    SubprocessTools,
)
from paomcp.utils.paths import DEFAULT_WORKSPACE, RuntimePaths, build_runtime_paths

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    runtime_paths: RuntimePaths
    session_registry: DebugSessionRegistry
    default_session_id: str
    subprocess_tools: SubprocessTools
    git_tools: GitTools
    python_tools: PythonTools
    decompiler: GhidraDecompiler
    artifact_store: ArtifactStore
    operation_registry: OperationRegistry
    debugger_compatibility: Dict[str, Any]
    paopipe_sessions: Dict[str, PaoPipe] = field(default_factory=dict)
    paopipe_lock: threading.Lock = field(default_factory=threading.Lock)


def create_services(workspace_root: str = DEFAULT_WORKSPACE) -> AppServices:
    if not os.path.exists(workspace_root):
        try:
            os.makedirs(workspace_root, exist_ok=True)
            logger.info("Created default workspace directory: %s", workspace_root)
        except OSError as exc:
            logger.warning(
                "Could not create workspace directory %s: %s", workspace_root, exc
            )
            logger.info("Continuing without default workspace directory")

    runtime_paths = build_runtime_paths(workspace_root)
    session_registry = DebugSessionRegistry(runtime_paths)
    default_session = session_registry.create_session("default")

    init_result = default_session.gdb.initialize()
    logger.info("GDB initialization: %s", init_result.get("status"))
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

    services = AppServices(
        runtime_paths=runtime_paths,
        session_registry=session_registry,
        default_session_id=default_session.session_id,
        subprocess_tools=SubprocessTools(
            process_root=runtime_paths.processes_dir,
            workspace_root=runtime_paths.workspace_root,
            artifact_store=artifact_store,
            operation_registry=operation_registry,
        ),
        git_tools=GitTools(workspace_dir=runtime_paths.repos_dir),
        python_tools=PythonTools(workspace_dir=runtime_paths.python_dir),
        decompiler=decompiler,
        artifact_store=artifact_store,
        operation_registry=operation_registry,
        debugger_compatibility=debugger_compatibility,
    )
    artifact_store.start_cleanup_worker()
    return services


def close_services(services: AppServices) -> None:
    logger.info("Shutting down Pao Tools server components...")

    with services.paopipe_lock:
        for pipe in services.paopipe_sessions.values():
            pipe.kill()
        services.paopipe_sessions.clear()

    services.subprocess_tools.close_all()
    services.session_registry.close_all()
    services.artifact_store.close()
