"""Tests for Stage 8.3 SshProvider + file sync + fetch_runtime_libraries."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import pytest

from paomcp.tools.backends.targets import (
    DEFAULT_LIBC_FETCH_PATHS,
    TARGET_FETCH_LIBS_SCHEMA,
    TARGET_FILE_SYNC_SCHEMA,
    SshProvider,
    TargetError,
    TargetKind,
    TargetRegistry,
    TargetSpec,
    TargetState,
)


# ---------------------------------------------------------------------------
# Fake paramiko client + SFTP for hermetic tests
# ---------------------------------------------------------------------------
class _FakeTransport:
    def __init__(self, active: bool = True) -> None:
        self._active = active

    def is_active(self) -> bool:
        return self._active


class _FakeSftp:
    """Backed by an in-memory {remote_path: local-source-path} map."""

    def __init__(
        self,
        *,
        remote_files: Dict[str, bytes],
        remote_missing: Optional[List[str]] = None,
    ) -> None:
        # Share the dict by reference so uploads made through this SFTP
        # are observable from the enclosing _FakeSshClient.
        self._files = remote_files
        self._missing = set(remote_missing or [])
        self.closed = False

    def stat(self, remote_path: str) -> Any:
        if remote_path in self._missing or remote_path not in self._files:
            raise FileNotFoundError(remote_path)
        return {"size": len(self._files[remote_path])}

    def get(self, remote_path: str, local_path: str) -> None:
        if remote_path not in self._files or remote_path in self._missing:
            raise FileNotFoundError(remote_path)
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        with open(local_path, "wb") as handle:
            handle.write(self._files[remote_path])

    def put(self, local_path: str, remote_path: str) -> None:
        with open(local_path, "rb") as handle:
            self._files[remote_path] = handle.read()

    def close(self) -> None:
        self.closed = True


class _FakeSshClient:
    def __init__(
        self,
        *,
        remote_files: Optional[Dict[str, bytes]] = None,
        active: bool = True,
    ) -> None:
        self._remote_files: Dict[str, bytes] = remote_files or {
            "/lib/x86_64-linux-gnu/libc.so.6": b"\x7fELF\x02\x01\x01\x00" + b"X" * 32,
            "/lib64/ld-linux-x86-64.so.2": b"\x7fELF" + b"L" * 24,
        }
        self._active = active
        self.connect_calls: List[Dict[str, Any]] = []
        self.closed = False
        self._sftp: Optional[_FakeSftp] = None

    def connect(self, **kwargs: Any) -> None:
        self.connect_calls.append(kwargs)

    def open_sftp(self) -> _FakeSftp:
        self._sftp = _FakeSftp(remote_files=self._remote_files)
        return self._sftp

    def get_transport(self) -> _FakeTransport:
        return _FakeTransport(active=self._active)

    def close(self) -> None:
        self.closed = True


def _build_registry(fake_client: _FakeSshClient) -> TargetRegistry:
    registry = TargetRegistry()
    registry._providers[TargetKind.SSH] = SshProvider(
        client_factory=lambda: fake_client,
    )
    return registry


def _ssh_spec() -> TargetSpec:
    return TargetSpec.from_config(
        "ssh",
        {
            "host": "ctf.remote",
            "port": 22,
            "ssh_user": "player",
            "ssh_key_path": "/tmp/id_rsa",
        },
        session_id="sess",
    )


# ---------------------------------------------------------------------------
# Spec validation
# ---------------------------------------------------------------------------
class TestSshSpecValidation:
    def test_requires_host(self) -> None:
        with pytest.raises(TargetError, match="'host'"):
            TargetSpec.from_config("ssh", {"ssh_user": "player", "ssh_key_path": "k"})

    def test_requires_user(self) -> None:
        with pytest.raises(TargetError, match="'ssh_user'"):
            TargetSpec.from_config("ssh", {"host": "h", "ssh_key_path": "k"})

    def test_requires_key_or_password(self) -> None:
        with pytest.raises(TargetError, match="ssh_key_path.*ssh_password"):
            TargetSpec.from_config("ssh", {"host": "h", "ssh_user": "u"})

    def test_accepts_password(self) -> None:
        spec = TargetSpec.from_config(
            "ssh",
            {"host": "h", "ssh_user": "u", "ssh_password": "p"},
        )
        assert spec.kind == TargetKind.SSH
        assert spec.ssh_password == "p"


# ---------------------------------------------------------------------------
# Lifecycle over the fake client
# ---------------------------------------------------------------------------
class TestSshLifecycle:
    def test_start_connects_and_probes_transport(self) -> None:
        fake = _FakeSshClient()
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        response = registry.connect_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is True
        assert handle.state == TargetState.CONNECTED
        assert fake.connect_calls
        kwargs = fake.connect_calls[0]
        assert kwargs["hostname"] == "ctf.remote"
        assert kwargs["username"] == "player"
        assert kwargs["key_filename"] == "/tmp/id_rsa"

    def test_start_fails_when_transport_reports_inactive(self) -> None:
        fake = _FakeSshClient(active=False)
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        response = registry.start_target_response(target_id=handle.target_id)
        assert response["parsed"]["success"] is False
        assert handle.state == TargetState.ERROR

    def test_stop_closes_client_and_detaches(self) -> None:
        fake = _FakeSshClient()
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        registry.disconnect_target_response(target_id=handle.target_id)
        assert handle.state == TargetState.DETACHED
        assert fake.closed is True

    def test_attach_debugger_is_explicitly_unsupported(self) -> None:
        fake = _FakeSshClient()
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)

        class _AnyGdb:
            def attach(self, pid: int) -> Any:
                raise AssertionError("must not be called for ssh")

            def execute_mi_command(self, *a: Any, **kw: Any) -> Any:
                raise AssertionError("must not be called for ssh")

        response = registry.attach_debugger_response(
            target_id=handle.target_id, gdb=_AnyGdb()  # type: ignore[arg-type]
        )
        parsed = response["parsed"]
        assert parsed["success"] is False
        assert "ssh provider" in parsed["error"]


# ---------------------------------------------------------------------------
# File sync + fetch_runtime_libraries
# ---------------------------------------------------------------------------
class TestSshFileSync:
    def test_download_writes_bytes_to_destination(self, tmp_path) -> None:
        fake = _FakeSshClient(
            remote_files={"/remote/binary": b"HELLO PAO"},
        )
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        local = str(tmp_path / "downloaded.bin")
        response = registry.sync_target_files_response(
            target_id=handle.target_id,
            source="/remote/binary",
            destination=local,
            direction="download",
        )
        parsed = response["parsed"]
        assert parsed["schema"] == TARGET_FILE_SYNC_SCHEMA
        assert parsed["result"]["direction"] == "download"
        with open(local, "rb") as f:
            assert f.read() == b"HELLO PAO"

    def test_upload_puts_local_bytes_to_remote(self, tmp_path) -> None:
        fake = _FakeSshClient(remote_files={})
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        local = tmp_path / "payload.bin"
        local.write_bytes(b"PAYLOAD")
        registry.sync_target_files_response(
            target_id=handle.target_id,
            source=str(local),
            destination="/remote/payload.bin",
            direction="upload",
        )
        # Verify the fake got the bytes.
        assert fake._remote_files["/remote/payload.bin"] == b"PAYLOAD"

    def test_bad_direction_raises(self, tmp_path) -> None:
        fake = _FakeSshClient()
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        with pytest.raises(TargetError, match="direction must be"):
            registry.sync_target_files_response(
                target_id=handle.target_id,
                source="/x",
                destination=str(tmp_path / "y"),
                direction="sideways",
            )

    def test_sync_on_non_ssh_target_raises(self, tmp_path) -> None:
        registry = TargetRegistry()
        handle = registry.create(
            TargetSpec.from_config(
                "attach_pid", {"pid": os.getpid()}, session_id="sess"
            )
        )
        with pytest.raises(TargetError, match="does not support file sync"):
            registry.sync_target_files_response(
                target_id=handle.target_id,
                source="/x",
                destination=str(tmp_path / "y"),
            )


class TestFetchRuntimeLibraries:
    def test_fetches_present_paths_and_reports_missing(self, tmp_path) -> None:
        libc_body = b"\x7fELF" + b"L" * 128
        loader_body = b"\x7fELF" + b"D" * 128
        fake = _FakeSshClient(
            remote_files={
                "/lib/x86_64-linux-gnu/libc.so.6": libc_body,
                "/lib64/ld-linux-x86-64.so.2": loader_body,
            }
        )
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        response = registry.fetch_runtime_libraries_response(
            target_id=handle.target_id,
            destination_dir=str(tmp_path),
        )
        parsed = response["parsed"]
        assert parsed["schema"] == TARGET_FETCH_LIBS_SCHEMA
        result = parsed["result"]
        fetched_names = sorted(
            os.path.basename(entry["local_path"]) for entry in result["fetched"]
        )
        assert "libc.so.6" in fetched_names
        assert "ld-linux-x86-64.so.2" in fetched_names
        assert result["missing"]  # musl / i386 paths not present in the fake

    def test_custom_candidate_paths_are_honored(self, tmp_path) -> None:
        fake = _FakeSshClient(
            remote_files={"/custom/libc.so.42": b"\x7fELF" + b"C" * 32},
        )
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        registry.connect_target_response(target_id=handle.target_id)
        response = registry.fetch_runtime_libraries_response(
            target_id=handle.target_id,
            destination_dir=str(tmp_path),
            candidate_paths=["/custom/libc.so.42", "/does/not/exist"],
        )
        result = response["parsed"]["result"]
        assert len(result["fetched"]) == 1
        assert result["missing"] == ["/does/not/exist"]

    def test_capabilities_flag_file_sync_true_for_ssh(self) -> None:
        fake = _FakeSshClient()
        registry = _build_registry(fake)
        handle = registry.create(_ssh_spec())
        caps = registry.get_capabilities_response(target_id=handle.target_id)["parsed"][
            "capabilities"
        ]
        assert caps["file_sync"] is True
        assert caps["attach_debugger"] is False


def test_default_libc_fetch_paths_are_documented() -> None:
    assert "/lib/x86_64-linux-gnu/libc.so.6" in DEFAULT_LIBC_FETCH_PATHS
    assert any("musl" in p for p in DEFAULT_LIBC_FETCH_PATHS)
