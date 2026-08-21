"""Tests for the debugger backend decision layer (Feature B)."""

from typing import Optional

import pytest

from paomcp.tools.backends.backend import (
    BACKEND_CHOICES,
    BackendUnavailableError,
    UnknownBackendError,
    available_backends,
    resolve_backend,
    validate_backend_name,
)


def _which(present: set) -> callable:
    def _which(name: str) -> Optional[str]:
        return f"/usr/bin/{name}" if name in present else None

    return _which


def _probe(result: bool) -> callable:
    return lambda: result


def _gdb_only_which(_name: str) -> Optional[str]:
    return "/usr/bin/pwndbg"


def _lldb_only_which(_name: str) -> Optional[str]:
    return "/usr/bin/pwndbg-lldb"


class TestValidateBackendName:
    def test_known_values_pass(self) -> None:
        assert validate_backend_name("gdb") == "gdb"
        assert validate_backend_name("lldb") == "lldb"
        assert validate_backend_name("auto") == "auto"

    def test_none_passes(self) -> None:
        assert validate_backend_name(None) is None

    def test_unknown_raises(self) -> None:
        with pytest.raises(UnknownBackendError):
            validate_backend_name("vanilla-gdb")


class TestResolveBackend:
    def test_no_request_defaults_to_gdb_even_without_pwndbg(self) -> None:
        spec = resolve_backend(which=_which(set()), probe_lldb=_probe(False))
        assert spec.name == "gdb"
        assert spec.protocol == "gdb-mi3"
        assert spec.reason == "default"
        assert spec.available is False  # binary absent, but default still gdb

    def test_no_request_preserves_gdb_default_when_present(self) -> None:
        spec = resolve_backend(which=_which({"pwndbg"}), probe_lldb=_probe(True))
        assert spec.name == "gdb"
        assert spec.reason == "default"

    def test_explicit_gdb(self) -> None:
        spec = resolve_backend(
            session_backend="gdb", which=_which({"pwndbg"}), probe_lldb=_probe(False)
        )
        assert spec.name == "gdb"
        assert spec.reason == "explicit"

    def test_explicit_lldb(self) -> None:
        spec = resolve_backend(
            session_backend="lldb",
            which=_which({"pwndbg-lldb"}),
            probe_lldb=_probe(True),
        )
        assert spec.name == "lldb"
        assert spec.protocol == "lldb-cli"
        assert spec.executable == "pwndbg-lldb"
        assert spec.reason == "explicit"

    def test_explicit_unavailable_gdb_raises(self) -> None:
        with pytest.raises(BackendUnavailableError):
            resolve_backend(
                session_backend="gdb", which=_which(set()), probe_lldb=_probe(True)
            )

    def test_explicit_unavailable_lldb_raises_even_with_gdb(self) -> None:
        with pytest.raises(BackendUnavailableError):
            resolve_backend(
                session_backend="lldb",
                which=_which({"pwndbg"}),
                probe_lldb=_probe(False),
            )

    def test_unknown_value_raises(self) -> None:
        with pytest.raises(UnknownBackendError):
            resolve_backend(
                session_backend="qemu", which=_which(set()), probe_lldb=_probe(False)
            )

    def test_auto_prefers_gdb_on_linux(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_which({"pwndbg", "pwndbg-lldb"}),
            probe_lldb=_probe(True),
            target_os="linux",
        )
        assert spec.name == "gdb"
        assert spec.reason == "auto-availability"

    def test_auto_prefers_lldb_on_darwin(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_which({"pwndbg", "pwndbg-lldb"}),
            probe_lldb=_probe(True),
            target_os="darwin",
        )
        assert spec.name == "lldb"
        assert spec.reason == "target-os"
        assert spec.target_hint == "darwin"

    def test_auto_prefers_lldb_on_mach_o_platform(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_which({"pwndbg", "pwndbg-lldb"}),
            probe_lldb=_probe(True),
            target_platform="mach-o",
        )
        assert spec.name == "lldb"

    def test_auto_falls_back_to_lldb_when_gdb_missing(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_which({"pwndbg-lldb"}),
            probe_lldb=_probe(True),
        )
        assert spec.name == "lldb"

    def test_auto_falls_back_to_gdb_on_darwin_when_lldb_missing(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_which({"pwndbg"}),
            probe_lldb=_probe(False),
            target_os="darwin",
        )
        assert spec.name == "gdb"
        assert spec.reason == "auto-fallback"

    def test_auto_none_available_raises(self) -> None:
        with pytest.raises(BackendUnavailableError):
            resolve_backend(
                session_backend="auto", which=_which(set()), probe_lldb=_probe(False)
            )

    def test_env_backend_is_honored_before_default(self) -> None:
        spec = resolve_backend(
            env_backend="lldb",
            which=_which({"pwndbg-lldb"}),
            probe_lldb=_probe(True),
        )
        assert spec.name == "lldb"

    def test_session_backend_wins_over_env(self) -> None:
        spec = resolve_backend(
            env_backend="lldb",
            session_backend="gdb",
            which=_which({"pwndbg"}),
            probe_lldb=_probe(False),
        )
        assert spec.name == "gdb"


class TestAvailableBackends:
    def test_shape_and_availability(self) -> None:
        report = available_backends(
            which=_which({"pwndbg", "pwndbg-lldb"}), probe_lldb=_probe(True)
        )
        by_name = {entry["name"]: entry for entry in report}
        assert set(by_name) == {"gdb", "lldb"}
        assert by_name["gdb"]["protocol"] == "gdb-mi3"
        assert by_name["gdb"]["available"] is True
        assert by_name["lldb"]["protocol"] == "lldb-cli"
        assert by_name["lldb"]["available"] is True
        assert by_name["lldb"]["path"] is not None

    def test_lldb_unavailable_when_probe_fails(self) -> None:
        report = available_backends(
            which=_which({"pwndbg", "pwndbg-lldb"}), probe_lldb=_probe(False)
        )
        by_name = {entry["name"]: entry for entry in report}
        assert by_name["lldb"]["available"] is False

    def test_lldb_unavailable_when_binary_missing(self) -> None:
        report = available_backends(which=_which({"pwndbg"}), probe_lldb=_probe(True))
        by_name = {entry["name"]: entry for entry in report}
        assert by_name["lldb"]["available"] is False
        assert by_name["lldb"]["path"] is None


class TestSpecDict:
    def test_to_dict_omits_target_hint_when_unset(self) -> None:
        spec = resolve_backend(session_backend="gdb", which=_gdb_only_which)
        data = spec.to_dict()
        assert data["name"] == "gdb"
        assert "target_hint" not in data

    def test_to_dict_includes_target_hint_when_set(self) -> None:
        spec = resolve_backend(
            session_backend="auto",
            which=_lldb_only_which,
            probe_lldb=_probe(True),
            target_os="darwin",
        )
        data = spec.to_dict()
        assert data["target_hint"] == "darwin"


def test_backend_choices_exported() -> None:
    assert BACKEND_CHOICES == ("gdb", "lldb", "auto")
