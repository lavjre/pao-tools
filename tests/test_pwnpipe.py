import sys

from pwnomcp.pwnpipe import PwnPipe


def test_pwnpipe_releases_prompt_without_newline():
    pipe = PwnPipe(
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


def test_pwnpipe_releases_fragmented_prompt_without_newline():
    pipe = PwnPipe(
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


def test_pwnpipe_replaces_invalid_utf8():
    pipe = PwnPipe([sys.executable, "-c", "import os; os.write(1, b'bad: \\xff\\n')"])
    try:
        pipe.wait_ready(timeout=2)
        pipe.proc.wait(timeout=2)
        output = pipe.release()

        assert "bad: " in output
        assert "\ufffd" in output
    finally:
        pipe.kill()
