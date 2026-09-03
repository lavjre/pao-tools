import os

from paomcp.tools.backends.python import PythonTools


def test_python_tools_uses_configured_baked_venv(tmp_path, monkeypatch):
    workspace = tmp_path / "runtime-python"
    baked_venv = tmp_path / "image" / "shared_venv"
    baked_venv.mkdir(parents=True)
    monkeypatch.setenv("PAO_PYTHON_VENV", str(baked_venv))

    tools = PythonTools(workspace_dir=str(workspace))

    assert tools.venv_path == os.path.abspath(baked_venv)
    assert tools.scripts_dir == str(workspace / "scripts")
    assert not tools.venv_path.startswith(os.path.abspath(workspace) + os.sep)


def test_explicit_venv_path_overrides_environment(tmp_path, monkeypatch):
    workspace = tmp_path / "runtime-python"
    environment_venv = tmp_path / "environment-venv"
    explicit_venv = tmp_path / "explicit-venv"
    environment_venv.mkdir()
    explicit_venv.mkdir()
    monkeypatch.setenv("PAO_PYTHON_VENV", str(environment_venv))

    tools = PythonTools(
        workspace_dir=str(workspace),
        venv_path=str(explicit_venv),
    )

    assert tools.venv_path == os.path.abspath(explicit_venv)


def test_python_tools_defaults_to_workspace_venv(tmp_path, monkeypatch):
    workspace = tmp_path / "runtime-python"
    default_venv = workspace / "shared_venv"
    default_venv.mkdir(parents=True)
    monkeypatch.delenv("PAO_PYTHON_VENV", raising=False)

    tools = PythonTools(workspace_dir=str(workspace))

    assert tools.venv_path == os.path.abspath(default_venv)


def _tools_with_venv(tmp_path, monkeypatch):
    workspace = tmp_path / "runtime-python"
    venv = tmp_path / "venv"
    venv.mkdir(parents=True)
    monkeypatch.setenv("PAO_PYTHON_VENV", str(venv))
    return PythonTools(workspace_dir=str(workspace))


def test_install_packages_gated_behind_env_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("PAO_ALLOW_PIP_INSTALL", raising=False)
    tools = _tools_with_venv(tmp_path, monkeypatch)
    result = tools.install_packages(["requests"])
    assert result["success"] is False
    assert "PAO_ALLOW_PIP_INSTALL" in result["error"]


def test_execute_code_rejects_empty_and_oversize(tmp_path, monkeypatch):
    tools = _tools_with_venv(tmp_path, monkeypatch)
    assert tools.execute_code("   ")["success"] is False
    big = "x = 1\n" * (PythonTools.MAX_CODE_BYTES // 6 + 1)
    result = tools.execute_code(big)
    assert result["success"] is False
    assert "cap" in result["error"]
