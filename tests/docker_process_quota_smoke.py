"""Manual Linux smoke test for bounded managed-process output."""

import json
from pathlib import Path
import shlex
import sys
import tempfile
import time

from paomcp.tools.backends.subproc import SubprocessTools


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pao-process-smoke-") as temp_dir:
        root = Path(temp_dir)
        emitter = root / "emit.py"
        emitter.write_text(
            "import sys, time\n"
            "time.sleep(0.2)\n"
            "sys.stdout.buffer.write(b'O' * 800)\n"
            "sys.stdout.buffer.flush()\n"
            "sys.stderr.buffer.write(b'E' * 800)\n"
            "sys.stderr.buffer.flush()\n"
            "time.sleep(10)\n",
            encoding="utf-8",
        )
        tools = SubprocessTools(
            process_root=str(root / "processes"),
            workspace_root=str(root),
            max_output_bytes=1024,
        )
        try:
            spawned = tools.spawn_process(
                shlex.join([sys.executable, str(emitter)]),
                cwd=str(root),
            )
            if not spawned.get("success"):
                raise RuntimeError(f"Could not spawn quota smoke process: {spawned}")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                listed = tools.list_processes()
                if listed["completed_count"]:
                    break
                time.sleep(0.02)
            result = tools.get_process(spawned["pid"])
            evidence = result.get("output_limit", {})
            if result.get("code") != "RESOURCE_EXHAUSTED":
                raise RuntimeError(f"Output quota was not enforced: {result}")
            if evidence.get("total_stored_bytes") != 1024:
                raise RuntimeError(f"Stored-byte accounting is invalid: {evidence}")
            if not evidence.get("measurement_complete"):
                raise RuntimeError(f"Pipe readers did not finish: {evidence}")
            print(
                json.dumps(
                    {
                        "success": True,
                        "process_tree_terminated": True,
                        "stored_bytes": evidence["total_stored_bytes"],
                        "dropped_bytes": evidence["total_dropped_bytes"],
                    },
                    sort_keys=True,
                )
            )
        finally:
            tools.close_all()


if __name__ == "__main__":
    main()
