import asyncio
import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI

from paomcp.http.models import AttachRequest, AttachResponse
from paomcp.services import AppServices
from paomcp.tools.backends.foundation import validate_attach_pid
from paomcp.tools.common import resolve_debug_session, run_session_locked
from paomcp.utils.paths import resolve_workspace_path

logger = logging.getLogger(__name__)

# GDB escapes that run host shell commands. pre/after are debugger-command
# automation by design (loopback sidecar), but host-shell execution is never
# part of that contract and is rejected per command instead of executed.
_BLOCKED_ATTACH_COMMAND_PREFIXES = ("shell", "!")


def _attach_command_blocked(command: str) -> bool:
    text = (command or "").strip().lower()
    if not text:
        return False
    if text == "shell" or text.startswith("shell ") or text.startswith("shell\t"):
        return True
    return text.startswith("!")


def _resolve_session(body: AttachRequest, services: AppServices):
    return resolve_debug_session(
        services, session_id=body.session_id, create_if_missing=False
    )


async def run_attach_request(
    body: AttachRequest, services: AppServices
) -> AttachResponse:
    try:
        session = _resolve_session(body, services)
    except Exception as exc:
        logger.exception("Failed to resolve debug session for attach request")
        return AttachResponse(
            successful=False,
            attach={
                "success": False,
                "error": str(exc),
                "pid": body.pid,
                "session_id": body.session_id,
            },
            result={},
        )

    try:
        validated_pid = validate_attach_pid(body.pid)
    except ValueError as exc:
        return AttachResponse(
            successful=False,
            attach={
                "success": False,
                "error": str(exc),
                "pid": body.pid,
                "session_id": body.session_id,
            },
            result={},
        )

    tools = session.tools
    command_results: Dict[str, Any] = {}

    def _guarded_execute(cmd: str) -> Dict[str, Any]:
        if _attach_command_blocked(cmd):
            return {
                "success": False,
                "error": (
                    f"refusing host-shell escape in attach command {cmd!r}: "
                    "pre/after run debugger commands only"
                ),
                "type": "AttachCommandBlockedError",
            }
        return tools.execute(cmd)

    def _run_attach_sequence() -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        if body.where:
            try:
                resolved_binary = resolve_workspace_path(
                    body.where,
                    workspace_root=services.runtime_paths.workspace_root,
                    require_exists=False,
                    kind="where",
                )
                logger.info("[attach] set-file: %s", resolved_binary)
                set_res = tools.set_file(resolved_binary)
                command_results["set-file"] = set_res
            except Exception as e:
                logger.exception("Error setting file: %s", body.where)
                command_results["set-file"] = {"success": False, "error": str(e)}

        for cmd in body.pre or []:
            try:
                logger.info("[attach] pre: %s", cmd)
                res = _guarded_execute(cmd)
                logger.info("[attach] pre command result: %s", res)
                command_results[cmd] = res
            except Exception as e:
                logger.exception("Error executing pre command: %s", cmd)
                command_results[cmd] = {"success": False, "error": str(e)}

        attach_info: Optional[Dict[str, Any]] = None
        try:
            logger.info("[attach] attaching to pid=%s", validated_pid)
            attach_result, _ = tools.attach(validated_pid)
            logger.info("[attach] attach result: %s", attach_result)
            attach_success = bool(attach_result.get("success"))
            if attach_success:
                session.state.pid = validated_pid
            attach_info = {
                "command": attach_result.get("command"),
                "success": attach_success,
                "state": attach_result.get("state"),
                "pid": attach_result.get("pid"),
                "session_id": session.session_id,
            }
        except Exception:
            logger.exception("Error attaching to pid %s", validated_pid)
            attach_success = False
            attach_info = {
                "success": False,
                "error": f"failed to attach to pid {validated_pid}",
            }

        if attach_success:
            for cmd in body.after or []:
                try:
                    logger.info("[attach] after: %s", cmd)
                    res = _guarded_execute(cmd)
                    logger.info("[attach] after command result: %s", res)
                    command_results[cmd] = res
                except Exception as e:
                    logger.exception("Error executing after command: %s", cmd)
                    command_results[cmd] = {"success": False, "error": str(e)}

        return attach_success, attach_info, command_results

    def _run_with_session_lock() -> (
        Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]
    ):
        # run_session_locked adds the closing/closed/destroyed guards a raw
        # lock skips: a concurrent close can no longer leave this driving a
        # dead gdb pipe.
        return run_session_locked(session, _run_attach_sequence, sync_pid=False)

    attach_success, attach_info, command_results = await asyncio.to_thread(
        _run_with_session_lock
    )

    return AttachResponse(
        successful=attach_success,
        attach=attach_info,
        result=command_results,
    )


def create_attach_app(services: AppServices) -> FastAPI:
    app = FastAPI(title="pao-tools attach", version="0.2.0")

    @app.get("/")
    async def root():
        return {"message": "Pao Tools Attach"}

    @app.post("/attach", response_model=AttachResponse)
    async def attach_endpoint(body: AttachRequest) -> AttachResponse:
        return await run_attach_request(body, services)

    return app
