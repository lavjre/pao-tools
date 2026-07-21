import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict

from pwnomcp.artifacts import ArtifactStore
from pwnomcp.pwnpipe import PwnPipe
from pwnomcp.state import DebugSessionRegistry
from pwnomcp.tools.backends import (
    GitTools,
    PythonTools,
    RetDecAnalyzer,
    SubprocessTools,
)
from pwnomcp.utils.paths import DEFAULT_WORKSPACE, RuntimePaths, build_runtime_paths

logger = logging.getLogger(__name__)


@dataclass
class AppServices:
    runtime_paths: RuntimePaths
    session_registry: DebugSessionRegistry
    default_session_id: str
    subprocess_tools: SubprocessTools
    git_tools: GitTools
    python_tools: PythonTools
    retdec_analyzer: RetDecAnalyzer
    artifact_store: ArtifactStore
    debugger_compatibility: Dict[str, Any]
    pwnpipe_sessions: Dict[str, PwnPipe] = field(default_factory=dict)
    pwnpipe_lock: threading.Lock = field(default_factory=threading.Lock)


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

    retdec_analyzer = RetDecAnalyzer()
    logger.info("RetDec analyzer created (lazy initialization)")
    artifact_store = ArtifactStore(runtime_paths.artifacts_dir)

    services = AppServices(
        runtime_paths=runtime_paths,
        session_registry=session_registry,
        default_session_id=default_session.session_id,
        subprocess_tools=SubprocessTools(
            process_root=runtime_paths.processes_dir,
            workspace_root=runtime_paths.workspace_root,
            artifact_store=artifact_store,
        ),
        git_tools=GitTools(workspace_dir=runtime_paths.repos_dir),
        python_tools=PythonTools(workspace_dir=runtime_paths.python_dir),
        retdec_analyzer=retdec_analyzer,
        artifact_store=artifact_store,
        debugger_compatibility=debugger_compatibility,
    )
    artifact_store.start_cleanup_worker()
    return services


def close_services(services: AppServices) -> None:
    logger.info("Shutting down Pwno MCP server components...")

    with services.pwnpipe_lock:
        for pipe in services.pwnpipe_sessions.values():
            pipe.kill()
        services.pwnpipe_sessions.clear()

    services.subprocess_tools.close_all()
    services.session_registry.close_all()
    services.artifact_store.close()
