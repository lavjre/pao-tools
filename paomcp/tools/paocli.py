import os
import shlex
import time
from typing import Any, Dict, List, Optional, Tuple

from fastmcp import Context, FastMCP
from fastmcp.dependencies import CurrentContext

from paomcp.paopipe import PaoPipe
from paomcp.tools.common import (
    catch_errors,
    get_paopipe,
    get_services,
    resolve_binary_path,
    resolve_debug_session,
    resolve_pipe_session_id,
    require_session_registry,
    run_blocking,
)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @catch_errors()
    async def paocli(
        file: str,
        session_id: str,
        argument: str = "",
        wait_timeout: float = 3.0,
        binary_path: Optional[str] = None,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Run paocli exploit script; persistent pipe per session.

        Exposes paocli_attach_result marker as attachment.result. Only create
        persistent /workspace scripts when the user explicitly requests it.

        Args:
            file: Script contents.
            argument: Args after "debug <binary>".
            wait_timeout: Max wait secs for attach/output.
            binary_path: Target under /workspace.
            session_id: Debug session id.
        """
        services = get_services(ctx)
        session = resolve_debug_session(
            services, session_id=session_id, create_if_missing=False
        )
        resolved_binary = resolve_binary_path(
            binary_path,
            session,
            require_exists=True,
            workspace_root=services.runtime_paths.workspace_root,
        )

        runtime_dir = session.runtime_dir
        os.makedirs(runtime_dir, exist_ok=True)
        script_path = os.path.join(runtime_dir, f"exp_{int(time.time() * 1000)}.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(file)

        replaced = False
        driver_pid: Optional[int] = None
        old_pipe: Optional[PaoPipe] = None
        with services.paopipe_lock:
            old_pipe = services.paopipe_sessions.get(session.session_id)
            if old_pipe and old_pipe.is_alive():
                replaced = True
            services.paopipe_sessions.pop(session.session_id, None)

            cmd = ["uv", "run", script_path, "debug", resolved_binary]
            if argument:
                cmd.extend(shlex.split(argument))
            pipe = PaoPipe(
                command=cmd,
                cwd=os.path.dirname(resolved_binary),
                env={"PYTHONUNBUFFERED": "1"},
            )
            services.paopipe_sessions[session.session_id] = pipe
            driver_pid = pipe.get_pid()
            session.driver_pid = driver_pid

        if old_pipe and old_pipe.is_alive():
            old_pipe.kill()

        def _collect_startup() -> Tuple[Dict[str, Any], str, Any]:
            startup_result = pipe.wait_ready(timeout=wait_timeout)
            output_result = pipe.release()
            attach_result_value = pipe.get_attach_result()
            return startup_result, output_result, attach_result_value

        startup, output, attach_result = await run_blocking(_collect_startup)

        return {
            "session_id": session.session_id,
            "runtime_dir": runtime_dir,
            "binary_path": resolved_binary,
            "io": {
                "pipeinput": True,
                "pipeoutput": True,
                "current_output": output,
            },
            "attachment": {
                "result": attach_result,
            },
            "startup": {
                "ready": startup.get("ready"),
                "reason": startup.get("reason"),
                "wait_ms": startup.get("wait_ms"),
                "alive": pipe.is_alive(),
                "pid": driver_pid,
                "replaced": replaced,
            },
        }

    @mcp.tool()
    @catch_errors()
    async def sendinput(
        data: str,
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Send raw input to a session-scoped paocli process stdin.

        Important:
            This call does not append a newline. If the target expects a line, include
            "\n" yourself.

        Args:
            data: Raw text to write to stdin.

        Returns:
            { "success": bool } indicating whether the input was written successfully.
        """
        services = get_services(ctx)
        resolved_session_id, pipe = get_paopipe(services, session_id=session_id)
        with services.paopipe_lock:
            if not pipe.is_alive():
                return {
                    "success": False,
                    "session_id": resolved_session_id,
                    "error": "No active PaoPipe",
                }
            ok = pipe.send(data)
        return {"success": bool(ok), "session_id": resolved_session_id}

    @mcp.tool()
    @catch_errors()
    async def checkoutput(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Release and return accumulated output from a session PaoPipe.

        Returns:
            { "success": True, "output": str } on success, or a failure object when no
            pipe exists. The internal buffer is cleared by this call (subsequent calls only
            return new output).
        """
        services = get_services(ctx)
        resolved_session_id, pipe = get_paopipe(services, session_id=session_id)
        with services.paopipe_lock:
            out = pipe.release()
        return {"success": True, "session_id": resolved_session_id, "output": out}

    @mcp.tool()
    @catch_errors()
    async def checkevents(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Release and return structured events from a session PaoPipe.

        Returns:
            {"success": True, "events": [...], "alive": bool, "exit_code": int|None}
        """
        services = get_services(ctx)
        resolved_session_id, pipe = get_paopipe(services, session_id=session_id)
        with services.paopipe_lock:
            events = pipe.release_events()
            alive = pipe.is_alive()
            exit_code = pipe.get_exit_code()
            driver_pid = pipe.get_pid()
        return {
            "success": True,
            "session_id": resolved_session_id,
            "driver_pid": driver_pid,
            "events": events,
            "alive": alive,
            "exit_code": exit_code,
        }

    @mcp.tool()
    @catch_errors()
    async def paocli_stop(
        session_id: str,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Stop a paocli driver session and clear its session pipe."""
        services = get_services(ctx)
        resolved_session_id = resolve_pipe_session_id(services, session_id=session_id)
        with services.paopipe_lock:
            pipe = services.paopipe_sessions.pop(resolved_session_id, None)
            if not pipe:
                return {
                    "success": False,
                    "session_id": resolved_session_id,
                    "error": "No active PaoPipe",
                }
            was_alive = pipe.is_alive()
            exit_code = pipe.get_exit_code()
            pipe.kill()
        session = require_session_registry(services).get_session(resolved_session_id)
        if session:
            session.driver_pid = None
        return {
            "success": True,
            "session_id": resolved_session_id,
            "was_alive": was_alive,
            "exit_code": exit_code,
        }

    @mcp.tool()
    @catch_errors()
    async def paocli_link_evidence(
        session_id: str,
        research_session_id: str,
        label: Optional[str] = None,
        max_chars: int = 8000,
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """Drain a paocli pipe and link its output as research evidence.

        Releases accumulated stdout + structured events without killing the
        driver, stores them as a research note (capped), and links a
        ``tool_result`` evidence entry pointing at the pipe session.

        Args:
            session_id: Pipe session id owning the paocli driver.
            research_session_id: Research session receiving the evidence.
            label: Human tag for the note/evidence description.
            max_chars: Cap output chars stored in the note.
        """
        services = get_services(ctx)
        resolved_session_id, pipe = get_paopipe(services, session_id=session_id)
        with services.paopipe_lock:
            output = pipe.release()
            events = pipe.release_events()
            alive = pipe.is_alive()
            exit_code = pipe.get_exit_code()
            driver_pid = pipe.get_pid()
        body = (output or "")[-max(512, max_chars) :]
        tag = (label or "paocli").strip() or "paocli"

        def _link() -> Dict[str, Any]:
            note = services.research_registry.add_note(
                session_id=research_session_id,
                body=f"[{tag}] paocli {resolved_session_id} output:\n{body}",
                tags=["paocli", tag],
            )
            evidence = services.research_registry.link_evidence(
                session_id=research_session_id,
                kind="tool_result",
                ref=f"paocli:{resolved_session_id}",
                description=f"{tag}: {len(output or '')} chars, {len(events)} events",
                note_id=note.note_id,
                metadata={"driver_pid": driver_pid, "alive": alive},
            )
            return {
                "note_id": note.note_id,
                "evidence_id": evidence.evidence_id,
            }

        linked = await run_blocking(_link)
        return {
            "success": True,
            "session_id": resolved_session_id,
            "research_session_id": research_session_id,
            "alive": alive,
            "exit_code": exit_code,
            "output_chars": len(output or ""),
            "event_count": len(events),
            **linked,
        }

    @mcp.tool()
    @catch_errors()
    async def list_paocli_sessions(
        ctx: Context = CurrentContext(),
    ) -> Dict[str, Any]:
        """List all active paocli driver sessions."""
        services = get_services(ctx)
        sessions: List[Dict[str, Any]] = []
        with services.paopipe_lock:
            for sid, pipe in services.paopipe_sessions.items():
                sessions.append(
                    {
                        "session_id": sid,
                        "driver_pid": pipe.get_pid(),
                        "alive": pipe.is_alive(),
                        "exit_code": pipe.get_exit_code(),
                    }
                )
        return {"success": True, "count": len(sessions), "sessions": sessions}
