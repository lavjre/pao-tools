import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

from paomcp.tools.backends.subproc import SubprocessTools


def _script_command(path: Path) -> str:
    argv = [sys.executable, str(path)]
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _wait_for_output_limit(entry: dict, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if entry["output_limiter"].snapshot()["exceeded"]:
            break
        time.sleep(0.01)
    assert entry["output_limiter"].snapshot()["exceeded"] is True

    process = entry["process"]
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.01)
    assert process.poll() is not None


def _assert_exact_quota_evidence(result: dict, expected_observed: int) -> None:
    output_limit = result["output_limit"]
    assert result["success"] is False
    assert result["code"] == "RESOURCE_EXHAUSTED"
    assert result["type"] == "ProcessOutputLimitError"
    assert output_limit["exceeded"] is True
    assert output_limit["total_observed_bytes"] == expected_observed
    assert output_limit["total_stored_bytes"] == output_limit["limit_bytes"]
    assert output_limit["total_dropped_bytes"] == (
        expected_observed - output_limit["limit_bytes"]
    )
    assert output_limit["measurement_complete"] is True
    assert output_limit["drain_complete"] is True
    assert output_limit["reader_threads_alive"] == []
    assert (
        output_limit["stdout"]["stored_bytes"] + output_limit["stderr"]["stored_bytes"]
        == output_limit["limit_bytes"]
    )


def test_run_command_enforces_exact_combined_output_quota(tmp_path) -> None:
    script = tmp_path / "emit_both.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'O' * 700)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'E' * 700)\n"
        "sys.stderr.buffer.flush()\n",
        encoding="utf-8",
    )
    process_root = tmp_path / "processes"
    tools = SubprocessTools(
        process_root=str(process_root),
        workspace_root=str(tmp_path),
        max_output_bytes=1024,
    )

    result = tools.run_command(_script_command(script), cwd=str(tmp_path))

    _assert_exact_quota_evidence(result, expected_observed=1400)
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) == 1024
    assert list(process_root.iterdir()) == []
    assert not any(
        thread.name.startswith("pao-process-") for thread in threading.enumerate()
    )


def test_get_process_preserves_limit_evidence_across_reap_and_owner_check(
    tmp_path,
) -> None:
    script = tmp_path / "emit_later.py"
    script.write_text(
        "import sys, time\n"
        "time.sleep(0.3)\n"
        "sys.stdout.buffer.write(b'O' * 800)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'E' * 800)\n"
        "sys.stderr.buffer.flush()\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    tools = SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_processes=1,
        max_output_bytes=1024,
    )
    spawned = tools.spawn_process(
        _script_command(script), cwd=str(tmp_path), session_id="alpha"
    )
    assert spawned["success"] is True
    entry = tools.background_processes[spawned["pid"]]
    limiter = entry["output_limiter"]
    process_dir = Path(spawned["process_dir"])

    _wait_for_output_limit(entry)
    listed = tools.list_processes("alpha")
    assert listed["completed_count"] == 1
    assert listed["completed_processes"][0]["output_limit"]["exceeded"] is True
    denied = tools.get_process(spawned["pid"], session_id="beta")
    assert denied["success"] is False
    assert denied["type"] == "PermissionError"

    result = tools.get_process(spawned["pid"], session_id="alpha")

    _assert_exact_quota_evidence(result, expected_observed=1600)
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) == 1024
    assert process_dir.exists() is False
    assert all(not thread.is_alive() for thread in limiter._threads)
    assert spawned["pid"] not in tools.background_processes
    assert spawned["pid"] not in tools._completed_processes

    sleeper = tmp_path / "sleeper.py"
    sleeper.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    replacement = tools.spawn_process(_script_command(sleeper), cwd=str(tmp_path))
    assert replacement["success"] is True
    killed = tools.kill_process(replacement["pid"])
    assert killed["success"] is True
    assert Path(replacement["process_dir"]).exists() is False


def test_kill_process_reports_output_limit_and_cleans_reader_threads(tmp_path) -> None:
    script = tmp_path / "emit_later.py"
    script.write_text(
        "import sys, time\n"
        "time.sleep(0.3)\n"
        "sys.stdout.buffer.write(b'O' * 900)\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'E' * 900)\n"
        "sys.stderr.buffer.flush()\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    tools = SubprocessTools(
        process_root=str(tmp_path / "processes"),
        workspace_root=str(tmp_path),
        max_output_bytes=1200,
    )
    spawned = tools.spawn_process(_script_command(script), cwd=str(tmp_path))
    assert spawned["success"] is True
    entry = tools.background_processes[spawned["pid"]]
    limiter = entry["output_limiter"]

    _wait_for_output_limit(entry)
    result = tools.kill_process(spawned["pid"])

    _assert_exact_quota_evidence(result, expected_observed=1800)
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) == 1200
    assert Path(spawned["process_dir"]).exists() is False
    assert all(not thread.is_alive() for thread in limiter._threads)


def test_process_output_quota_must_be_positive(tmp_path) -> None:
    try:
        SubprocessTools(
            process_root=str(tmp_path / "processes"),
            workspace_root=str(tmp_path),
            max_output_bytes=0,
        )
    except ValueError as exc:
        assert str(exc) == "max_output_bytes must be positive"
    else:
        raise AssertionError("zero process-output quota must be rejected")
