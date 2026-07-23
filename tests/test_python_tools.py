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
