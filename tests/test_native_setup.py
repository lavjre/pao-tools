"""Tests for native (no-Docker) setup: doctor checks, script markers, docs nav."""

import json
from pathlib import Path

from paocli.utils.doctor import check_one, detect_os, run_doctor

ROOT = Path(__file__).resolve().parent.parent


def _which(mapping):
    def _fn(name):
        return mapping.get(name)

    return _fn


def _env(**overrides):
    base = {"PATH": "/usr/bin:/bin"}
    base.update(overrides)
    return base


class TestDetectOs:
    def test_darwin_is_macos(self):
        assert detect_os("darwin") == "macos"

    def test_linux_platforms(self):
        assert detect_os("linux") == "linux"
        assert detect_os("win32") == "linux"


class TestRunDoctor:
    def _present(self):
        return {
            "uv": "/home/u/.local/bin/uv",
            "pwndbg": "/home/u/.local/bin/pwndbg",
            "gdb": "/usr/bin/gdb",
            "lldb": "/usr/bin/lldb",
            "git": "/usr/bin/git",
            "java": "/usr/bin/java",
            "analyzeHeadless": "/opt/ghidra/support/analyzeHeadless",
            "qemu-system-x86_64": "/usr/bin/qemu-system-x86_64",
            "afl-fuzz": "/usr/bin/afl-fuzz",
            "unblob": "/home/u/.local/bin/unblob",
            "pwninit": "/home/u/.local/bin/pwninit",
            "one_gadget": "/usr/local/bin/one_gadget",
        }

    def test_linux_all_present(self):
        report = run_doctor(
            platform="linux", which_fn=_which(self._present()), env=_env()
        )
        assert report["os"] == "linux"
        assert report["ok_required"] is True
        assert report["missing_required"] == []

    def test_linux_missing_pwndbg_is_required(self):
        mapping = self._present()
        del mapping["pwndbg"]
        report = run_doctor(platform="linux", which_fn=_which(mapping), env=_env())
        assert report["ok_required"] is False
        assert any("pwndbg" in name for name in report["missing_required"])

    def test_linux_missing_afl_is_optional(self):
        mapping = self._present()
        del mapping["afl-fuzz"]
        report = run_doctor(platform="linux", which_fn=_which(mapping), env=_env())
        assert report["ok_required"] is True
        afl = next(c for c in report["checks"] if c["name"] == "afl-fuzz")
        assert afl["ok"] is False
        assert afl["required"] is False

    def test_macos_gdb_not_required_lldb_is(self):
        mapping = self._present()
        del mapping["gdb"]
        del mapping["pwndbg"]
        report = run_doctor(platform="darwin", which_fn=_which(mapping), env=_env())
        assert report["os"] == "macos"
        assert report["ok_required"] is True
        del mapping["lldb"]
        report = run_doctor(platform="darwin", which_fn=_which(mapping), env=_env())
        assert report["ok_required"] is False

    def test_ghidra_env_key_counts(self):
        report = run_doctor(
            platform="linux",
            which_fn=_which({}),
            env=_env(PAO_GHIDRA_HEADLESS="/tmp/fake/analyzeHeadless"),
        )
        assert report["os"] == "linux"

    def test_check_one_python_reports_version(self):
        spec = {
            "name": "python3 (>=3.12)",
            "kind": "python",
            "required_linux": True,
            "required_macos": True,
            "hint_linux": "x",
            "hint_macos": "y",
        }
        result = check_one(spec, os_name="linux", which_fn=_which({}), env=_env())
        assert result["required"] is True
        assert result["ok"] is True  # test env runs 3.12+


class TestNativeArtifacts:
    def test_setup_script_markers(self):
        text = (ROOT / "scripts" / "setup-native.sh").read_text(encoding="utf-8")
        for marker in (
            "--with-ghidra",
            "b62e81a0390618466c019c60d8c2f796ced2509c4c1aea4a37644a77272cf99d",
            "pipx install pwndbg",
            "PAO_DEBUGGER_BACKEND",
            ".env.native",
            "paocli doctor",
            "brew install",
        ):
            assert marker in text, marker

    def test_doctor_command_registered(self):
        from click.testing import CliRunner

        from paocli.cli import cli as root

        runner = CliRunner()
        result = runner.invoke(root, ["doctor"])
        assert result.exit_code == 0
        assert "paocli doctor" in result.output

    def test_docs_page_in_nav(self):
        assert (ROOT / "docs" / "guides" / "native-setup.mdx").is_file()
        nav = json.loads((ROOT / "docs" / "docs.json").read_text(encoding="utf-8"))
        pages: list = []
        for tab in nav["navigation"]["tabs"]:
            pages.extend(tab.get("pages", []))
            for group in tab.get("groups", []):
                pages.extend(group.get("pages", []))
        assert "guides/native-setup" in pages
