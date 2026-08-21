import sys
import json

from paomcp.paopipe import PaoPipe


def test_paopipe_parses_paocli_attach_marker():
    expected = {"successful": True, "attach": {"pid": 31337}}
    payload = json.dumps(expected)
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            f"print('PAOCLI_ATTACH_RESULT:' + {payload!r})",
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        pipe.proc.wait(timeout=2)

        assert ready["ready"] is True
        assert pipe.get_attach_result() == expected
        assert "PAOCLI_ATTACH_RESULT" not in pipe.release()
    finally:
        pipe.kill()


def test_paopipe_releases_prompt_without_newline():
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            "import sys,time; sys.stdout.write('prompt: '); sys.stdout.flush(); time.sleep(5)",
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        output = pipe.release()

        assert ready["ready"] is True
        assert "prompt: " in output
    finally:
        pipe.kill()


def test_paopipe_releases_fragmented_prompt_without_newline():
    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            (
                "import sys,time; sys.stdout.write('P'); sys.stdout.flush(); "
                "time.sleep(.2); sys.stdout.write('rompt: '); sys.stdout.flush(); "
                "time.sleep(5)"
            ),
        ]
    )
    try:
        ready = pipe.wait_ready(timeout=2)
        output = pipe.release()

        assert ready["ready"] is True
        assert "Prompt: " in output
    finally:
        pipe.kill()


def test_paopipe_replaces_invalid_utf8():
    pipe = PaoPipe([sys.executable, "-c", "import os; os.write(1, b'bad: \\xff\\n')"])
    try:
        pipe.wait_ready(timeout=2)
        pipe.proc.wait(timeout=2)
        output = pipe.release()

        assert "bad: " in output
        assert "\ufffd" in output
    finally:
        pipe.kill()


def test_paopipe_kill_after_process_exited_does_not_raise():
    pipe = PaoPipe([sys.executable, "-c", "pass"])
    pipe.proc.wait(timeout=5)
    assert pipe.is_alive() is False
    # kill() after exit exercises the ProcessLookupError path (group already
    # gone) and must swallow it rather than aborting shutdown callers.
    pipe.kill()
    assert pipe.get_exit_code() == 0


def test_paopipe_kill_terminates_live_process_tree():
    import subprocess
    import time

    pipe = PaoPipe(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )
    try:
        pipe.wait_ready(timeout=2)
        assert pipe.is_alive() is True
        pipe.kill()
        assert pipe.is_alive() is False
        assert pipe.proc.poll() is not None
    finally:
        try:
            pipe.kill()
        except Exception:
            pass
