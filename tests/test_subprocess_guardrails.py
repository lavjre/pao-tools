import logging

import pytest

from paomcp.tools.backends.subproc import SubprocessTools

_SUBPROC_LOGGER = "paomcp.tools.backends.subproc"


def test_run_command_blocks_inline_python_c():
    tools = SubprocessTools()

    result = tools.run_command('python3 -c "print(1)"')

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "execute_python_code"
    assert "python -c" in result["error"].lower()


def test_run_command_blocks_python_c_after_cd():
    # M1: guardrails must inspect every shell segment, not just cmd_parts[0].
    tools = SubprocessTools()

    result = tools.run_command('cd /tmp && python3 -c "print(1)"')

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "execute_python_code"


def test_run_command_blocks_env_wrapped_python_c():
    tools = SubprocessTools()

    result = tools.run_command('env python3 -c "print(1)"')

    assert result["success"] is False
    assert result["recommended_tool"] == "execute_python_code"


def test_guardrails_allow_python_without_dash_c():
    tools = SubprocessTools()

    result = tools._check_command_guardrails(
        command="python3 -V",
        cmd_parts=["python3", "-V"],
        cwd=None,
        operation="run_command",
    )
    assert result is None


def test_guardrails_allow_grep_dash_c():
    # ``grep -c`` counts matches; the head is not python so it must pass.
    tools = SubprocessTools()

    result = tools._check_command_guardrails(
        command="grep -c foo bar",
        cmd_parts=["grep", "-c", "foo", "bar"],
        cwd=None,
        operation="run_command",
    )
    assert result is None


def test_run_command_blocks_shell_wrapped_inline_python_c():
    tools = SubprocessTools()

    result = tools.run_command("sh -lc 'python3 -c \"print(1)\"'")

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "execute_python_code"


def test_run_command_blocks_workspace_elf_execution(tmp_path):
    target = tmp_path / "chal"
    target.write_bytes(b"\x7fELF" + b"\x00" * 16)
    target.chmod(0o755)
    tools = SubprocessTools(workspace_root=str(tmp_path))

    result = tools.run_command(str(target), cwd=str(tmp_path))

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "set_file+run"
    assert str(tmp_path).lower() in result["error"].lower()


def test_run_command_blocks_workspace_elf_after_cd(tmp_path):
    # M1: a workspace ELF run after ``cd ... &&`` must be blocked too.
    target = tmp_path / "chal"
    target.write_bytes(b"\x7fELF" + b"\x00" * 16)
    target.chmod(0o755)
    tools = SubprocessTools(workspace_root=str(tmp_path))

    result = tools.run_command(f"cd /tmp && {target}", cwd=str(tmp_path))

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "set_file+run"


def test_spawn_process_blocks_workspace_elf_execution(tmp_path):
    target = tmp_path / "chal"
    target.write_bytes(b"\x7fELF" + b"\x00" * 16)
    target.chmod(0o755)
    tools = SubprocessTools(workspace_root=str(tmp_path))

    result = tools.spawn_process(str(target), cwd=str(tmp_path))

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "set_file+run"


def test_process_operations_reject_unmanaged_pid():
    tools = SubprocessTools()

    status = tools.get_process(123456)
    killed = tools.kill_process(123456)

    assert status["success"] is False
    assert status["type"] == "UnmanagedProcessError"
    assert killed["success"] is False
    assert killed["type"] == "UnmanagedProcessError"


@pytest.mark.parametrize(
    "command",
    [
        'python3 -Oc "print(1)"',
        'python3 -Ic "print(1)"',
        'python3 -uc "print(1)"',
        "python3 -cprint(1)",
        'python3.13.1 -c "print(1)"',
        'python3 -X utf8 -c "print(1)"',
        'env python3.13.1 -Oc "print(1)"',
    ],
)
def test_run_command_blocks_python_c_variants(command):
    # E2.3: short-option clusters (``-Oc``), version suffixes
    # (``python3.13.1``) and operand-taking options (``-X utf8``) all still
    # execute an inline script.
    tools = SubprocessTools()

    result = tools.run_command(command)

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "execute_python_code"


def test_guardrails_allow_python_module_arguments_after_dash_m():
    # ``-m`` hands the remaining tokens to the module; ``python -m pytest
    # -c pytest.ini`` is not an inline script and must stay allowed.
    tools = SubprocessTools()

    result = tools._check_command_guardrails(
        command="python3 -m pytest -c pytest.ini",
        cmd_parts=["python3", "-m", "pytest", "-c", "pytest.ini"],
        cwd=None,
        operation="run_command",
    )
    assert result is None


def test_run_command_blocks_shell_wrapped_python_c_cluster():
    tools = SubprocessTools()

    result = tools.run_command("sh -lc 'python3.13.1 -Oc \"print(1)\"'")

    assert result["success"] is False
    assert result["type"] == "ToolUsageError"
    assert result["recommended_tool"] == "execute_python_code"


def test_command_summary_never_contains_argument_values():
    summary = SubprocessTools._command_summary(
        "curl -H 'Authorization: Bearer super-secret' https://example.invalid"
    )

    assert "curl" in summary
    assert "super-secret" not in summary
    assert "super-secret" not in SubprocessTools._command_summary(
        "TOKEN=super-secret ./run"
    )


def test_run_command_logs_summary_at_info_and_full_argv_at_debug(tmp_path, caplog):
    tools = SubprocessTools(
        process_root=str(tmp_path / "processes"), workspace_root=str(tmp_path)
    )

    with caplog.at_level(logging.DEBUG, logger=_SUBPROC_LOGGER):
        result = tools.run_command("echo super-secret-token", cwd=str(tmp_path))

    assert result["success"] is True
    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == _SUBPROC_LOGGER and record.levelno == logging.INFO
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == _SUBPROC_LOGGER and record.levelno == logging.DEBUG
    ]
    assert any("echo" in message for message in info_messages)
    assert all("super-secret-token" not in message for message in info_messages)
    assert any("super-secret-token" in message for message in debug_messages)


def test_spawn_process_logs_summary_at_info_and_full_argv_at_debug(tmp_path, caplog):
    tools = SubprocessTools(
        process_root=str(tmp_path / "processes"), workspace_root=str(tmp_path)
    )

    try:
        with caplog.at_level(logging.DEBUG, logger=_SUBPROC_LOGGER):
            tools.spawn_process("echo super-secret-token", cwd=str(tmp_path))
    finally:
        tools.close_all()

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == _SUBPROC_LOGGER and record.levelno == logging.INFO
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == _SUBPROC_LOGGER and record.levelno == logging.DEBUG
    ]
    assert any("echo" in message for message in info_messages)
    assert all("super-secret-token" not in message for message in info_messages)
    assert any("super-secret-token" in message for message in debug_messages)
