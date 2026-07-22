import pytest

from paomcp.tools.backends.compatibility import PwndbgCompatibility


class FakeGdb:
    def __init__(self):
        self.calls = []

    def execute_command(self, command):
        self.calls.append(command)
        outputs = {
            "pwndbg": (
                "pwndbg\n"
                "Context Commands [Aliases]    Description\n"
                "----------------------------  -------------------------\n"
                "context [ctx]                 Print the current context.\n"
                "regs                          Print registers.\n"
                "Memory Commands [Aliases]     Description\n"
                "----------------------------  -------------------------\n"
                "telescope                     Recursively dereference pointers.\n"
                "vmmap                         Print virtual memory mappings.\n"
                "checksec -- Check binary security settings\n"
            ),
            "pwndbg --list-categories": (
                "pwndbg --list-categories\nMemory\nLinux/libc/ELF\n"
            ),
            "version": "Gdb: 16.2\nPwndbg: 2026.02.18\nPython: 3.12\n",
            "help checksec": "Check binary security settings.\n",
        }
        return {
            "success": command in outputs,
            "responses": [
                {
                    "type": "console",
                    "payload": outputs.get(command, "Undefined command"),
                }
            ],
        }


def test_list_commands_returns_parsed_and_raw_evidence():
    adapter = PwndbgCompatibility(FakeGdb())

    result = adapter.list_commands()

    assert result["success"] is True
    assert result["parsed"]["count"] == 5
    assert result["parsed"]["commands"][0]["name"] == "checksec"
    assert "Context Commands" in result["raw_output"]
    assert result["raw_responses"]


def test_list_categories_ignores_echoed_command():
    adapter = PwndbgCompatibility(FakeGdb())

    result = adapter.list_categories()

    assert result["parsed"]["categories"] == ["Linux/libc/ELF", "Memory"]


def test_capabilities_validate_pinned_version_and_core_commands():
    adapter = PwndbgCompatibility(FakeGdb(), expected_version="2026.02.18")

    result = adapter.get_capabilities()

    assert result["ready"] is True
    assert result["version_matches"] is True
    assert result["versions"]["pwndbg"] == "2026.02.18"
    assert result["command_groups"]["core"]["missing"] == []
    assert result["command_count"] == 5


def test_capabilities_report_version_mismatch():
    adapter = PwndbgCompatibility(FakeGdb(), expected_version="2025.10.20")

    result = adapter.get_capabilities()

    assert result["ready"] is False
    assert result["version_matches"] is False
    assert "Expected pwndbg" in result["warnings"][0]


def test_command_help_rejects_gdb_command_injection():
    adapter = PwndbgCompatibility(FakeGdb())

    with pytest.raises(ValueError):
        adapter.get_help("checksec\nquit")


def test_command_help_returns_runtime_evidence():
    adapter = PwndbgCompatibility(FakeGdb())

    result = adapter.get_help("checksec")

    assert result["success"] is True
    assert result["command"] == "help checksec"
    assert "security settings" in result["raw_output"]
