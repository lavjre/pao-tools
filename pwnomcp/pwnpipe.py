import threading
import subprocess
import queue
import json
import os
import shlex
import signal
import time
import codecs
from typing import Optional, List, Dict, Any, Sequence, Union


class PwnPipe:
    """
    Simple subprocess I/O pipeline with queued output buffer.

    - Accumulates stdout/stderr into an internal queue
    - release() returns accumulated output and clears the queue
    - send(data) writes raw data to stdin (no newline automatically)
    - Detects attach marker lines: 'PWNCLI_ATTACH_RESULT:<json>'
    - Provides a structured event queue for output/state markers
    """

    def __init__(
        self,
        command: Union[str, Sequence[str]],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ):
        self.command = list(command) if not isinstance(command, str) else command
        self.cwd = cwd
        self.env = env or {}
        env_full = os.environ.copy()
        env_full.update(self.env)

        argv = shlex.split(command) if isinstance(command, str) else list(command)
        group_options: Dict[str, Any]
        if os.name == "nt":
            group_options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        else:
            group_options = {"start_new_session": True}

        self.proc = subprocess.Popen(
            argv,
            cwd=self.cwd,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env_full,
            **group_options,
        )
        self._q: "queue.Queue[str]" = queue.Queue()
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._alive = True
        self._lock = threading.Lock()
        self._attach_result = None
        self._exit_code: Optional[int] = None

        self._attach_event = threading.Event()
        self._output_event = threading.Event()
        self._exit_event = threading.Event()
        self._activity_event = threading.Event()

        self._t_out = threading.Thread(
            target=self._reader, args=(self.proc.stdout, "out")
        )
        self._t_err = threading.Thread(
            target=self._reader, args=(self.proc.stderr, "err")
        )
        self._t_out.daemon = True
        self._t_err.daemon = True
        self._t_out.start()
        self._t_err.start()

        self._t_wait = threading.Thread(target=self._waiter)
        self._t_wait.daemon = True
        self._t_wait.start()

    def _handle_control_line(self, line: bytes) -> bool:
        decoded = line.decode("utf-8", errors="replace")
        if decoded.startswith("PWNCLI_ATTACH_RESULT:"):
            payload = decoded.split(":", 1)[1].strip()
            try:
                with self._lock:
                    self._attach_result = json.loads(payload)
                self._attach_event.set()
                self._activity_event.set()
                self._events.put(
                    {
                        "type": "attached",
                        "ok": bool(
                            self._attach_result.get("successful")
                            if isinstance(self._attach_result, dict)
                            else False
                        ),
                        "result": self._attach_result,
                    }
                )
            except Exception:
                pass
            return True

        if decoded.startswith("PWNO_IPC:"):
            payload = decoded.split(":", 1)[1].strip()
            try:
                event = json.loads(payload)
                if isinstance(event, dict):
                    self._events.put(event)
                    self._output_event.set()
                    self._activity_event.set()
                    return True
            except Exception:
                pass
        return False

    def _reader(self, pipe, stream: str):
        prefixes = (b"PWNCLI_ATTACH_RESULT:", b"PWNO_IPC:")
        pending_control = b""
        at_line_start = True
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def emit(data: bytes, final: bool = False) -> None:
            text = decoder.decode(data, final=final)
            if not text:
                return
            self._q.put(text)
            self._events.put({"type": stream, "data": text})
            self._output_event.set()
            self._activity_event.set()

        while True:
            chunk = os.read(pipe.fileno(), 4096)
            if not chunk:
                break

            while chunk:
                if pending_control:
                    newline = chunk.find(b"\n")
                    if newline == -1:
                        pending_control += chunk
                        if not any(
                            prefix.startswith(pending_control)
                            or pending_control.startswith(prefix)
                            for prefix in prefixes
                        ):
                            emit(pending_control)
                            pending_control = b""
                            at_line_start = False
                        chunk = b""
                        continue
                    pending_control += chunk[: newline + 1]
                    chunk = chunk[newline + 1 :]
                    if not self._handle_control_line(pending_control):
                        emit(pending_control)
                    pending_control = b""
                    at_line_start = True
                    continue

                if not at_line_start:
                    newline = chunk.find(b"\n")
                    if newline == -1:
                        emit(chunk)
                        chunk = b""
                    else:
                        emit(chunk[: newline + 1])
                        chunk = chunk[newline + 1 :]
                        at_line_start = True
                    continue

                newline = chunk.find(b"\n")
                segment = chunk if newline == -1 else chunk[: newline + 1]
                chunk = b"" if newline == -1 else chunk[newline + 1 :]
                if any(
                    segment.startswith(prefix) or prefix.startswith(segment)
                    for prefix in prefixes
                ):
                    if segment.endswith(b"\n"):
                        if not self._handle_control_line(segment):
                            emit(segment)
                    else:
                        pending_control = segment
                    at_line_start = segment.endswith(b"\n")
                    continue

                emit(segment)
                at_line_start = segment.endswith(b"\n")

        if pending_control and not self._handle_control_line(pending_control):
            emit(pending_control)
        emit(b"", final=True)
        pipe.close()

    def _waiter(self):
        self.proc.wait()
        with self._lock:
            self._alive = False
            self._exit_code = self.proc.returncode
        self._exit_event.set()
        self._activity_event.set()
        self._events.put({"type": "exit", "code": self._exit_code})

    def is_alive(self) -> bool:
        with self._lock:
            return self._alive and (self.proc.poll() is None)

    def send(self, data: str) -> bool:
        if not self.is_alive():
            return False
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(data.encode("utf-8"))
            self.proc.stdin.flush()
            return True
        except Exception:
            return False

    def release(self) -> str:
        chunks = []
        try:
            while True:
                chunks.append(self._q.get_nowait())
        except queue.Empty:
            pass
        return "".join(chunks)

    def release_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        try:
            while True:
                events.append(self._events.get_nowait())
        except queue.Empty:
            pass
        return events

    def get_attach_result(self):
        with self._lock:
            return self._attach_result

    def get_exit_code(self) -> Optional[int]:
        with self._lock:
            return self._exit_code

    def get_pid(self) -> Optional[int]:
        try:
            return self.proc.pid
        except Exception:
            return None

    def wait_ready(self, timeout: float = 3.0) -> Dict[str, Any]:
        start = time.monotonic()
        if self._attach_event.is_set():
            reason = "attached"
        elif self._output_event.is_set():
            reason = "output"
        elif self._exit_event.is_set():
            reason = "exited"
        else:
            signaled = self._activity_event.wait(timeout)
            if not signaled:
                reason = "timeout"
            elif self._attach_event.is_set():
                reason = "attached"
            elif self._output_event.is_set():
                reason = "output"
            elif self._exit_event.is_set():
                reason = "exited"
            else:
                reason = "activity"
        wait_ms = int((time.monotonic() - start) * 1000)
        return {"ready": reason != "timeout", "reason": reason, "wait_ms": wait_ms}

    def kill(self):
        try:
            if self.proc and self.proc.poll() is None:
                if os.name == "nt":
                    self.proc.kill()
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=5)
        finally:
            with self._lock:
                self._alive = False
