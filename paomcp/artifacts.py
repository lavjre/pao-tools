"""Persistent storage for large MCP tool payloads."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from pydantic import BaseModel

DEFAULT_INLINE_RESPONSE_BYTES = int(
    os.environ.get("PAO_INLINE_RESPONSE_BYTES", str(1024 * 1024))
)
DEFAULT_ARTIFACT_CHUNK_BYTES = int(
    os.environ.get("PAO_ARTIFACT_CHUNK_BYTES", str(256 * 1024))
)
DEFAULT_ARTIFACT_MAX_READ_BYTES = int(
    os.environ.get("PAO_ARTIFACT_MAX_READ_BYTES", str(4 * 1024 * 1024))
)
DEFAULT_ARTIFACT_TTL_SECONDS = int(
    os.environ.get("PAO_ARTIFACT_TTL_SECONDS", str(72 * 60 * 60))
)
DEFAULT_ARTIFACT_QUOTA_BYTES = int(
    os.environ.get("PAO_ARTIFACT_QUOTA_BYTES", str(10 * 1024**3))
)
DEFAULT_ARTIFACT_CLEANUP_TARGET_RATIO = float(
    os.environ.get("PAO_ARTIFACT_CLEANUP_TARGET_RATIO", "0.8")
)
DEFAULT_ARTIFACT_CLEANUP_INTERVAL_SECONDS = float(
    os.environ.get("PAO_ARTIFACT_CLEANUP_INTERVAL_SECONDS", "900")
)
DEFAULT_ARTIFACT_MIN_FREE_BYTES = int(
    os.environ.get("PAO_ARTIFACT_MIN_FREE_BYTES", str(2 * 1024**3))
)
DEFAULT_ARTIFACT_ORPHAN_GRACE_SECONDS = float(
    os.environ.get("PAO_ARTIFACT_ORPHAN_GRACE_SECONDS", "300")
)
ARTIFACT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
BINARY_ARTIFACT_ENCODING = "binary"
QUARANTINE_SUFFIX = ".quarantine"
QUARANTINE_MARKER_SUFFIX = ".quarantine-marker"
logger = logging.getLogger(__name__)

GeneratorResult = TypeVar("GeneratorResult")


class ArtifactStorageFullError(RuntimeError):
    """Raised when an artifact cannot be stored without exhausting the disk."""


class ArtifactSizeLimitError(RuntimeError):
    """Raised when a generated artifact exceeds its caller-supplied hard cap."""


class WriterNotQuiescentError(RuntimeError):
    """Raised when a generator's external writer cannot be proven to have stopped.

    When this escapes an artifact generator, the staging file MUST NOT be
    unlinked: a still-open descriptor on a deleted inode would keep consuming
    disk with no monitor.  ``save_generated_file`` quarantines the file instead.
    """


class ArtifactPinnedError(RuntimeError):
    """Raised when deletion is attempted on a pinned artifact."""


class ArtifactOwnershipError(PermissionError):
    """Raised when an artifact belongs to another debugger session."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata persisted next to an artifact payload."""

    artifact_id: str
    tool: str
    session_id: Optional[str]
    content_type: str
    encoding: str
    size_bytes: int
    sha256: str
    created_at: str
    expires_at: Optional[str]
    pinned: bool = False
    provenance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return metadata with an explicit delivery mode."""
        return {"delivery": "artifact", **asdict(self)}


class ArtifactStore:
    """Store and retrieve complete large tool responses on disk."""

    def __init__(
        self,
        root_dir: str,
        inline_response_bytes: int = DEFAULT_INLINE_RESPONSE_BYTES,
        default_chunk_bytes: int = DEFAULT_ARTIFACT_CHUNK_BYTES,
        max_read_bytes: int = DEFAULT_ARTIFACT_MAX_READ_BYTES,
        ttl_seconds: int = DEFAULT_ARTIFACT_TTL_SECONDS,
        quota_bytes: int = DEFAULT_ARTIFACT_QUOTA_BYTES,
        cleanup_target_ratio: float = DEFAULT_ARTIFACT_CLEANUP_TARGET_RATIO,
        cleanup_interval_seconds: float = DEFAULT_ARTIFACT_CLEANUP_INTERVAL_SECONDS,
        min_free_bytes: int = DEFAULT_ARTIFACT_MIN_FREE_BYTES,
        orphan_grace_seconds: float = DEFAULT_ARTIFACT_ORPHAN_GRACE_SECONDS,
        dir_quota_enforcer: Optional[Callable[[str, int], None]] = None,
    ):
        if inline_response_bytes < 0:
            raise ValueError("inline_response_bytes cannot be negative")
        if default_chunk_bytes <= 0:
            raise ValueError("default_chunk_bytes must be positive")
        if max_read_bytes <= 0 or default_chunk_bytes > max_read_bytes:
            raise ValueError(
                "max_read_bytes must be positive and at least default_chunk_bytes"
            )
        if ttl_seconds < 0 or quota_bytes < 0 or min_free_bytes < 0:
            raise ValueError("artifact retention sizes and TTL cannot be negative")
        if not 0 < cleanup_target_ratio <= 1:
            raise ValueError(
                "cleanup_target_ratio must be greater than 0 and at most 1"
            )
        if cleanup_interval_seconds < 0:
            raise ValueError("cleanup_interval_seconds cannot be negative")
        if orphan_grace_seconds < 0:
            raise ValueError("orphan_grace_seconds cannot be negative")
        self.root_dir = os.path.abspath(root_dir)
        self.inline_response_bytes = inline_response_bytes
        self.default_chunk_bytes = default_chunk_bytes
        self.max_read_bytes = max_read_bytes
        self.ttl_seconds = ttl_seconds
        self.quota_bytes = quota_bytes
        self.cleanup_target_ratio = cleanup_target_ratio
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.min_free_bytes = min_free_bytes
        self.orphan_grace_seconds = orphan_grace_seconds
        # Optional absolute-enforcement seam.  A deployment can supply a callback
        # ``(staging_dir, max_size_bytes) -> None`` that provisions a per-operation
        # loopback filesystem, project quota, or cgroup storage limit under the
        # staging directory.  This complements the in-process RLIMIT_FSIZE cap
        # (kernel-enforced per gdb child) with an infrastructure-enforced ceiling.
        # Default is a no-op; the RLIMIT path remains the primary guarantee.
        self.dir_quota_enforcer = dir_quota_enforcer
        self._lock = threading.RLock()
        # Reentrant because save operations reserve capacity through cleanup while
        # holding the same lock.  Reconciliation, retention cleanup, and every
        # payload mutation share this lock, so a generated staging file cannot be
        # mistaken for an orphan while its trusted generator is still running.
        self._write_lock = threading.RLock()
        # In-memory protection complements the durable marker written when an
        # external writer cannot be proven quiescent.  The reservation map keeps
        # quota accounting fail-closed even when the marker write itself fails
        # during this process lifetime.
        self._protected_staging_names: Set[str] = set()
        self._protected_staging_reservations: Dict[str, int] = {}
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._last_reconciliation: Optional[Dict[str, Any]] = None
        os.makedirs(self.root_dir, exist_ok=True)

    def _new_metadata(
        self,
        artifact_id: str,
        *,
        tool: str,
        session_id: Optional[str],
        content_type: str,
        encoding: str,
        size_bytes: int,
        sha256: str,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ArtifactMetadata:
        created = datetime.now(timezone.utc)
        expires = (
            created + timedelta(seconds=self.ttl_seconds)
            if self.ttl_seconds > 0
            else None
        )
        return ArtifactMetadata(
            artifact_id=artifact_id,
            tool=tool,
            session_id=session_id,
            content_type=content_type,
            encoding=encoding,
            size_bytes=size_bytes,
            sha256=sha256,
            created_at=created.isoformat(),
            expires_at=expires.isoformat() if expires else None,
            provenance=dict(provenance) if provenance is not None else None,
        )

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    def serialize_json(self, data: Any) -> bytes:
        """Serialize a response payload exactly once for sizing and storage."""
        return json.dumps(
            data,
            ensure_ascii=True,
            separators=(",", ":"),
            default=self._json_default,
        ).encode("utf-8")

    def should_store(self, payload: bytes) -> bool:
        """Return whether a serialized payload should use artifact delivery."""
        return len(payload) > self.inline_response_bytes

    def _validate_artifact_id(self, artifact_id: str) -> str:
        if not ARTIFACT_ID_PATTERN.fullmatch(artifact_id):
            raise ValueError("artifact_id must be a 32-character lowercase hex value")
        return artifact_id

    def _data_path(self, artifact_id: str) -> str:
        validated_id = self._validate_artifact_id(artifact_id)
        return os.path.join(self.root_dir, f"{validated_id}.data")

    def _metadata_path(self, artifact_id: str) -> str:
        validated_id = self._validate_artifact_id(artifact_id)
        return os.path.join(self.root_dir, f"{validated_id}.meta.json")

    def _write_atomic(self, target_path: str, payload: bytes) -> None:
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".artifact_", dir=self.root_dir
        )
        try:
            with os.fdopen(descriptor, "wb") as temporary_file:
                temporary_file.write(payload)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, target_path)
        except Exception:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    def save_bytes(
        self,
        payload: bytes,
        *,
        tool: str,
        session_id: Optional[str] = None,
        content_type: str = "application/octet-stream",
        encoding: str = "utf-8",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ArtifactMetadata:
        """Persist an in-memory payload without truncation."""
        artifact_id = uuid.uuid4().hex
        metadata = self._new_metadata(
            artifact_id,
            tool=tool,
            session_id=session_id,
            content_type=content_type,
            encoding=encoding,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            provenance=provenance,
        )
        metadata_payload = json.dumps(
            asdict(metadata), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

        with self._write_lock:
            self.ensure_capacity(len(payload))
            with self._lock:
                self._write_atomic(self._data_path(artifact_id), payload)
                try:
                    self._write_atomic(
                        self._metadata_path(artifact_id), metadata_payload
                    )
                except Exception:
                    try:
                        os.unlink(self._data_path(artifact_id))
                    except OSError:
                        pass
                    raise
        return metadata

    def save_json_bytes(
        self,
        payload: bytes,
        *,
        tool: str,
        session_id: Optional[str] = None,
    ) -> ArtifactMetadata:
        """Persist an already serialized JSON payload without truncation."""
        return self.save_bytes(
            payload,
            tool=tool,
            session_id=session_id,
            content_type="application/json",
            encoding="utf-8",
        )

    def save_file(
        self,
        source_path: str,
        *,
        tool: str,
        session_id: Optional[str] = None,
        content_type: str = "application/octet-stream",
        encoding: str = "utf-8",
        provenance: Optional[Dict[str, Any]] = None,
    ) -> ArtifactMetadata:
        """Copy a file into the store incrementally so RAM usage stays bounded."""
        source_size = os.path.getsize(source_path)
        artifact_id = uuid.uuid4().hex
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".artifact_", dir=self.root_dir
        )
        digest = hashlib.sha256()
        size_bytes = 0
        descriptor_open = True

        try:
            with self._write_lock:
                self.ensure_capacity(source_size)
                destination = os.fdopen(descriptor, "wb")
                descriptor_open = False
                with (
                    destination,
                    open(source_path, "rb") as source,
                ):
                    remaining = source_size
                    while remaining > 0:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise OSError(
                                "Source file changed or was truncated while copying"
                            )
                        destination.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
                        remaining -= len(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())

                metadata = self._new_metadata(
                    artifact_id,
                    tool=tool,
                    session_id=session_id,
                    content_type=content_type,
                    encoding=encoding,
                    size_bytes=size_bytes,
                    sha256=digest.hexdigest(),
                    provenance=provenance,
                )
                metadata_payload = json.dumps(
                    asdict(metadata), ensure_ascii=True, separators=(",", ":")
                ).encode("utf-8")

                with self._lock:
                    os.replace(temporary_path, self._data_path(artifact_id))
                    try:
                        self._write_atomic(
                            self._metadata_path(artifact_id), metadata_payload
                        )
                    except Exception:
                        try:
                            os.unlink(self._data_path(artifact_id))
                        except OSError:
                            pass
                        raise
            return metadata
        except Exception:
            if descriptor_open:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
            raise

    @staticmethod
    def _remove_staging_path(staging_path: str) -> None:
        """Remove one private staging entry without following replacement links."""
        try:
            staging_stat = os.lstat(staging_path)
        except FileNotFoundError:
            return
        if stat.S_ISDIR(staging_stat.st_mode):
            shutil.rmtree(staging_path)
        else:
            os.unlink(staging_path)

    def _mark_staging_path_protected(
        self, staging_path: str, reserved_bytes: int
    ) -> Optional[str]:
        """Persistently protect a staging name and its worst-case reservation."""
        staging_name = os.path.basename(staging_path)
        self._protected_staging_names.add(staging_name)
        self._protected_staging_reservations[staging_name] = max(
            reserved_bytes,
            self._protected_staging_reservations.get(staging_name, 0),
        )
        marker_path = f"{staging_path}{QUARANTINE_MARKER_SUFFIX}"
        marker_payload = (
            "protected_at="
            f"{datetime.now(timezone.utc).isoformat()}\n"
            f"staging_name={staging_name}\n"
            f"reserved_bytes={self._protected_staging_reservations[staging_name]}\n"
        ).encode("utf-8")
        try:
            self._write_atomic(marker_path, marker_payload)
        except OSError:
            # The in-memory set still protects the path for this process.  Do
            # not unlink even when the durable marker cannot be written.
            logger.exception(
                "Failed to persist a quarantine marker for staging file '%s'; "
                "the path remains protected in this artifact-store process",
                staging_path,
            )
            return None
        return marker_path

    def _quarantine_staging_path(self, staging_path: str, reserved_bytes: int) -> str:
        """Keep a possibly active staging inode reachable and protected.

        Used when a generator's external writer could not be proven quiescent:
        unlinking a file a writer still holds open would leak disk via the
        deleted inode.  Renaming keeps the (single) inode reachable under a
        ``.quarantine`` name.  A durable marker always protects the original
        staging name as well, because an external writer can recreate that name
        after the rename.  If the rename fails (for example, because Windows
        denies renaming an open file), the original name remains protected.
        """
        self._mark_staging_path_protected(staging_path, reserved_bytes)
        try:
            os.lstat(staging_path)
        except FileNotFoundError:
            # The external writer may create the path after reporting failure,
            # so protect the expected name even when it is not visible yet.
            return staging_path
        quarantine_path = f"{staging_path}{QUARANTINE_SUFFIX}"
        try:
            os.replace(staging_path, quarantine_path)
        except OSError:
            logger.exception(
                "Failed to rename staging file '%s' for quarantine; leaving the "
                "protected path reachable",
                staging_path,
            )
            return staging_path
        logger.error(
            "Quarantined a core staging file whose writer could not be confirmed "
            "stopped: %s",
            quarantine_path,
        )
        return quarantine_path

    def _marker_reserved_bytes(self, marker_name: str) -> int:
        """Read a bounded reservation from one marker without trusting its payload."""
        marker_path = os.path.join(self.root_dir, marker_name)
        unknown_reservation = self.quota_bytes if self.quota_bytes > 0 else 0
        try:
            marker_stat = os.lstat(marker_path)
            if not stat.S_ISREG(marker_stat.st_mode) or marker_stat.st_size > 4096:
                raise ValueError("marker is not a small regular file")
            with open(marker_path, "r", encoding="utf-8") as marker_file:
                for line in marker_file:
                    key, separator, value = line.partition("=")
                    if separator and key == "reserved_bytes":
                        reservation = int(value.strip())
                        if reservation < 0:
                            raise ValueError("marker reservation is negative")
                        return reservation
            raise ValueError("marker has no reservation")
        except (OSError, UnicodeError, ValueError):
            logger.warning(
                "Could not read quarantine reservation from marker '%s'; "
                "the target remains protected and capacity accounting fails closed",
                marker_name,
            )
        return unknown_reservation

    def _protected_staging_reservation_map(self, names: Set[str]) -> Dict[str, int]:
        """Resolve protected staging names and reservations across restarts."""
        protected = dict(self._protected_staging_reservations)
        for staging_name in self._protected_staging_names:
            protected.setdefault(staging_name, 0)
        for name in names:
            if name.startswith(".artifact_") and name.endswith(
                QUARANTINE_MARKER_SUFFIX
            ):
                target = name.removesuffix(QUARANTINE_MARKER_SUFFIX)
                if target.endswith(".staging"):
                    protected[target] = max(
                        protected.get(target, 0),
                        self._marker_reserved_bytes(name),
                    )
        return protected

    def _protected_staging_targets(self, names: Set[str]) -> Set[str]:
        """Resolve in-memory and durable-marker protected staging names."""
        return set(self._protected_staging_reservation_map(names))

    @staticmethod
    def _entry_size(root_dir: str, name: str) -> int:
        """Return one directory entry's allocated logical size without following it."""
        try:
            return os.lstat(os.path.join(root_dir, name)).st_size
        except OSError:
            return 0

    def _operational_debt(self) -> Dict[str, int]:
        """Measure protected/quarantined files that normal cleanup cannot reclaim."""
        with self._lock:
            try:
                names = set(os.listdir(self.root_dir))
            except OSError:
                names = set()
            protected_reservations = self._protected_staging_reservation_map(names)
            protected_targets = set(protected_reservations)
            quarantine_names = {
                name
                for name in names
                if name.startswith(".artifact_") and name.endswith(QUARANTINE_SUFFIX)
            }
            protected_payloads = protected_targets.intersection(names)
            paired_quarantine_names = {
                f"{target}{QUARANTINE_SUFFIX}"
                for target in protected_targets
                if f"{target}{QUARANTINE_SUFFIX}" in names
            }
            unmarked_quarantine_names = quarantine_names - paired_quarantine_names
            incident_bytes = sum(
                max(
                    reserved_bytes,
                    self._entry_size(self.root_dir, target)
                    + self._entry_size(self.root_dir, f"{target}{QUARANTINE_SUFFIX}"),
                )
                for target, reserved_bytes in protected_reservations.items()
            )
            quarantine_bytes = incident_bytes + sum(
                max(
                    self._entry_size(self.root_dir, name),
                    self.quota_bytes if self.quota_bytes > 0 else 0,
                )
                for name in unmarked_quarantine_names
            )
            return {
                "quarantine_count": len(protected_targets)
                + len(unmarked_quarantine_names),
                "quarantine_bytes": quarantine_bytes,
                "quarantine_rename_failure_count": sum(
                    target in names and f"{target}{QUARANTINE_SUFFIX}" not in names
                    for target in protected_targets
                ),
                "protected_staging_count": len(protected_payloads),
                "protected_staging_bytes": sum(
                    self._entry_size(self.root_dir, name) for name in protected_payloads
                ),
            }

    @staticmethod
    def _raise_for_capacity(result: Dict[str, Any]) -> None:
        """Raise when cleanup could not satisfy quota or disk-reserve policy."""
        if not result["quota_satisfied"]:
            raise ArtifactStorageFullError(
                "Not enough artifact quota after cleanup; pinned artifacts prevent "
                "reserving the requested capacity"
            )
        if not result["disk_reserve_satisfied"]:
            raise ArtifactStorageFullError(
                "Not enough artifact storage after cleanup; pinned artifacts or "
                "filesystem usage prevent preserving the configured free-space reserve"
            )

    def save_generated_file(
        self,
        generator: Callable[[str], GeneratorResult],
        *,
        reserve_bytes: int,
        max_size_bytes: int,
        tool: str,
        session_id: Optional[str] = None,
        content_type: str = "application/octet-stream",
        encoding: str = BINARY_ARTIFACT_ENCODING,
        provenance: Optional[Dict[str, Any]] = None,
        cancellation_check: Optional[Callable[[], None]] = None,
    ) -> Tuple[ArtifactMetadata, GeneratorResult]:
        """Generate and atomically adopt a file without making a second copy.

        The trusted generator receives a private path on the artifact filesystem.
        ``reserve_bytes`` is used for policy preflight before generation, while
        ``max_size_bytes`` is a hard acceptance limit on the completed file.  The
        write lock remains held through generation, hashing, and finalization so
        cleanup cannot reclaim the active staging path.
        """
        if reserve_bytes < 0:
            raise ValueError("reserve_bytes cannot be negative")
        if max_size_bytes < 0:
            raise ValueError("max_size_bytes cannot be negative")

        artifact_id = uuid.uuid4().hex
        # Reserve the unpredictable name under the same lock used by orphan
        # reconciliation.  Some external generators require a nonexistent target,
        # so only the name (rather than an open placeholder) survives this block.
        with self._write_lock:
            descriptor, staging_path = tempfile.mkstemp(
                prefix=".artifact_", suffix=".staging", dir=self.root_dir
            )
            os.close(descriptor)
            os.unlink(staging_path)

        preserve_staging = False
        try:
            with self._write_lock:
                preflight = self.cleanup(incoming_size=reserve_bytes)
                self._raise_for_capacity(preflight)
                if self.dir_quota_enforcer is not None:
                    # Give a deployment-supplied enforcer a chance to bound the
                    # staging directory (loopback fs / project quota / cgroup)
                    # before the trusted generator starts writing.
                    self.dir_quota_enforcer(self.root_dir, max_size_bytes)
                if cancellation_check is not None:
                    cancellation_check()

                try:
                    generator_result = generator(staging_path)
                except WriterNotQuiescentError:
                    # Preserve/protect while still holding the same lock used by
                    # reconciliation, closing the delete-before-quarantine race.
                    preserve_staging = True
                    try:
                        self._quarantine_staging_path(staging_path, max_size_bytes)
                    except OSError:
                        # Preservation is fail-closed: keep the original path and
                        # propagate the writer-state failure, never unlink here.
                        logger.exception(
                            "Could not record quarantine state for staging file '%s'; "
                            "leaving the path reachable",
                            staging_path,
                        )
                    raise
                if cancellation_check is not None:
                    cancellation_check()

                try:
                    generated_stat = os.lstat(staging_path)
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        "Artifact generator did not create its staging file"
                    ) from exc
                if not stat.S_ISREG(generated_stat.st_mode):
                    raise OSError("Artifact generator must create a regular file")

                size_bytes = generated_stat.st_size
                if size_bytes > max_size_bytes:
                    raise ArtifactSizeLimitError(
                        f"Generated artifact is {size_bytes} bytes, exceeding the "
                        f"{max_size_bytes}-byte limit"
                    )

                # The staging allocation already consumes disk space, so only the
                # artifact quota still treats it as incoming during this check.
                actual_capacity = self.cleanup(
                    incoming_size=size_bytes,
                    incoming_already_allocated=True,
                )
                self._raise_for_capacity(actual_capacity)

                digest = hashlib.sha256()
                observed_size = 0
                read_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
                read_flags |= getattr(os, "O_NOFOLLOW", 0)
                generated_descriptor = os.open(staging_path, read_flags)
                with os.fdopen(generated_descriptor, "rb") as generated_file:
                    opened_stat = os.fstat(generated_file.fileno())
                    if (
                        not stat.S_ISREG(opened_stat.st_mode)
                        or (opened_stat.st_dev, opened_stat.st_ino)
                        != (generated_stat.st_dev, generated_stat.st_ino)
                        or opened_stat.st_size != size_bytes
                    ):
                        raise OSError(
                            "Generated artifact path changed before finalization"
                        )

                    while True:
                        if cancellation_check is not None:
                            cancellation_check()
                        chunk = generated_file.read(1024 * 1024)
                        if not chunk:
                            break
                        observed_size += len(chunk)
                        if observed_size > max_size_bytes:
                            raise ArtifactSizeLimitError(
                                "Generated artifact grew beyond its hard size limit "
                                "while it was being hashed"
                            )
                        digest.update(chunk)

                    final_descriptor_stat = os.fstat(generated_file.fileno())
                    try:
                        final_path_stat = os.lstat(staging_path)
                    except FileNotFoundError as exc:
                        raise OSError(
                            "Generated artifact path disappeared during finalization"
                        ) from exc
                    expected_identity = (opened_stat.st_dev, opened_stat.st_ino)
                    if (
                        observed_size != size_bytes
                        or final_descriptor_stat.st_size != size_bytes
                        or (
                            final_descriptor_stat.st_dev,
                            final_descriptor_stat.st_ino,
                        )
                        != expected_identity
                        or (final_path_stat.st_dev, final_path_stat.st_ino)
                        != expected_identity
                        or final_path_stat.st_size != size_bytes
                    ):
                        raise OSError(
                            "Generated artifact changed while it was being finalized"
                        )

                if cancellation_check is not None:
                    cancellation_check()

                metadata = self._new_metadata(
                    artifact_id,
                    tool=tool,
                    session_id=session_id,
                    content_type=content_type,
                    encoding=encoding,
                    size_bytes=size_bytes,
                    sha256=digest.hexdigest(),
                    provenance=provenance,
                )
                metadata_payload = json.dumps(
                    asdict(metadata), ensure_ascii=True, separators=(",", ":")
                ).encode("utf-8")

                finalized_path = self._data_path(artifact_id)
                with self._lock:
                    pre_adoption_stat = os.lstat(staging_path)
                    if (
                        (pre_adoption_stat.st_dev, pre_adoption_stat.st_ino)
                        != expected_identity
                        or pre_adoption_stat.st_size != size_bytes
                    ):
                        raise OSError(
                            "Generated artifact identity changed before adoption"
                        )
                    os.replace(staging_path, finalized_path)
                    adopted_stat = os.lstat(finalized_path)
                    if (
                        adopted_stat.st_dev,
                        adopted_stat.st_ino,
                    ) != expected_identity or adopted_stat.st_size != size_bytes:
                        try:
                            os.unlink(finalized_path)
                        except OSError:
                            pass
                        raise OSError(
                            "Generated artifact identity changed during adoption"
                        )
                    try:
                        self._write_atomic(
                            self._metadata_path(artifact_id), metadata_payload
                        )
                    except Exception:
                        try:
                            os.unlink(finalized_path)
                        except OSError:
                            pass
                        raise
                return metadata, generator_result
        finally:
            # The rename makes staging_path disappear on success.  Every other
            # failure — generator errors, cancellation during hashing — lands
            # here and unlinks the private staging file.
            if staging_path and not preserve_staging:
                try:
                    self._remove_staging_path(staging_path)
                except FileNotFoundError:
                    pass

    def get(self, artifact_id: str) -> ArtifactMetadata:
        """Load metadata for an existing artifact."""
        metadata_path = self._metadata_path(artifact_id)
        data_path = self._data_path(artifact_id)
        with self._lock:
            if not os.path.isfile(metadata_path) or not os.path.isfile(data_path):
                raise FileNotFoundError(f"Artifact '{artifact_id}' not found")
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata_data = json.load(metadata_file)
        metadata_data.setdefault("pinned", False)
        metadata_data.setdefault("provenance", None)
        if "expires_at" not in metadata_data:
            created = datetime.fromisoformat(metadata_data["created_at"])
            expires = (
                created + timedelta(seconds=self.ttl_seconds)
                if self.ttl_seconds > 0
                else None
            )
            metadata_data["expires_at"] = expires.isoformat() if expires else None
        return ArtifactMetadata(**metadata_data)

    def list(self, session_id: Optional[str] = None) -> List[ArtifactMetadata]:
        """List artifacts, optionally filtered by debugger session."""
        artifacts: List[ArtifactMetadata] = []
        with self._lock:
            metadata_names = [
                name
                for name in os.listdir(self.root_dir)
                if name.endswith(".meta.json")
            ]
        for metadata_name in metadata_names:
            artifact_id = metadata_name.removesuffix(".meta.json")
            try:
                metadata = self.get(artifact_id)
            except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if session_id is None or metadata.session_id == session_id:
                artifacts.append(metadata)
        return sorted(artifacts, key=lambda item: item.created_at, reverse=True)

    def list_owned(self, session_id: Optional[str]) -> List[ArtifactMetadata]:
        """List only artifacts owned by the exact session (or unscoped owner)."""
        return [item for item in self.list() if item.session_id == session_id]

    def reconcile_orphans(
        self, *, grace_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Remove stale temporary files and incomplete data/metadata pairs."""
        effective_grace = (
            self.orphan_grace_seconds if grace_seconds is None else float(grace_seconds)
        )
        if effective_grace < 0:
            raise ValueError("grace_seconds cannot be negative")
        cutoff = datetime.now(timezone.utc).timestamp() - effective_grace
        reclaimed_bytes = 0
        removed = {
            "temporary": 0,
            "data_without_metadata": 0,
            "metadata_without_data": 0,
        }
        failures: List[Dict[str, str]] = []

        with self._write_lock:
            with self._lock:
                names = set(os.listdir(self.root_dir))
                protected_targets = self._protected_staging_targets(names)
                candidates: List[tuple[str, str]] = []
                for name in names:
                    if name in protected_targets or name.endswith(
                        QUARANTINE_MARKER_SUFFIX
                    ):
                        continue
                    if name.startswith(".artifact_") and not name.endswith(
                        QUARANTINE_SUFFIX
                    ):
                        candidates.append((name, "temporary"))
                        continue
                    if name.endswith(".data"):
                        artifact_id = name.removesuffix(".data")
                        if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) and (
                            f"{artifact_id}.meta.json" not in names
                        ):
                            candidates.append((name, "data_without_metadata"))
                        continue
                    if name.endswith(".meta.json"):
                        artifact_id = name.removesuffix(".meta.json")
                        if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) and (
                            f"{artifact_id}.data" not in names
                        ):
                            candidates.append((name, "metadata_without_data"))

                for name, reason in candidates:
                    path = os.path.join(self.root_dir, name)
                    try:
                        candidate_stat = os.lstat(path)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        failures.append(
                            {
                                "name": name,
                                "reason": reason,
                                "error": type(exc).__name__,
                            }
                        )
                        continue
                    if effective_grace > 0 and candidate_stat.st_mtime > cutoff:
                        continue
                    if stat.S_ISDIR(candidate_stat.st_mode):
                        failures.append(
                            {
                                "name": name,
                                "reason": reason,
                                "error": "IsADirectoryError",
                            }
                        )
                        continue
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        failures.append(
                            {
                                "name": name,
                                "reason": reason,
                                "error": type(exc).__name__,
                            }
                        )
                        continue
                    reclaimed_bytes += candidate_stat.st_size
                    removed[reason] += 1

        result = {
            "grace_seconds": effective_grace,
            "removed_count": sum(removed.values()),
            "reclaimed_bytes": reclaimed_bytes,
            "removed": removed,
            "failed_count": len(failures),
            "failures": failures,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        stored_result = {
            **result,
            "removed": dict(removed),
            "failures": [dict(item) for item in failures],
        }
        with self._lock:
            self._last_reconciliation = stored_result
        return {
            **stored_result,
            "removed": dict(removed),
            "failures": [dict(item) for item in failures],
        }

    def assert_owner(
        self, artifact_id: str, session_id: Optional[str]
    ) -> ArtifactMetadata:
        """Validate session ownership and return the artifact metadata."""
        metadata = self.get(artifact_id)
        if metadata.session_id != session_id:
            raise ArtifactOwnershipError(
                "The artifact belongs to a different debugger session"
            )
        return metadata

    def get_for_session(
        self, artifact_id: str, session_id: Optional[str]
    ) -> ArtifactMetadata:
        """Load metadata after enforcing debugger-session ownership."""
        return self.assert_owner(artifact_id, session_id)

    def status(self) -> Dict[str, Any]:
        """Return retention policy and current artifact storage usage."""
        artifacts = self.list()
        with self._lock:
            last_reconciliation = (
                {
                    **self._last_reconciliation,
                    "removed": dict(self._last_reconciliation["removed"]),
                    "failures": [
                        dict(item)
                        for item in self._last_reconciliation.get("failures", [])
                    ],
                }
                if self._last_reconciliation is not None
                else None
            )
        now = datetime.now(timezone.utc)
        usage = sum(item.size_bytes for item in artifacts)
        operational_debt = self._operational_debt()
        return {
            "artifact_count": len(artifacts),
            "pinned_count": sum(item.pinned for item in artifacts),
            "expired_count": sum(
                item.expires_at is not None
                and datetime.fromisoformat(item.expires_at) <= now
                for item in artifacts
            ),
            "usage_bytes": usage,
            "total_usage_bytes": usage + operational_debt["quarantine_bytes"],
            **operational_debt,
            "free_bytes": shutil.disk_usage(self.root_dir).free,
            "last_reconciliation": last_reconciliation,
            "worker_running": bool(
                self._cleanup_thread and self._cleanup_thread.is_alive()
            ),
            "policy": {
                "ttl_seconds": self.ttl_seconds,
                "quota_bytes": self.quota_bytes,
                "cleanup_target_ratio": self.cleanup_target_ratio,
                "cleanup_target_bytes": int(
                    self.quota_bytes * self.cleanup_target_ratio
                ),
                "cleanup_interval_seconds": self.cleanup_interval_seconds,
                "min_free_bytes": self.min_free_bytes,
                "max_read_bytes": self.max_read_bytes,
                "orphan_grace_seconds": self.orphan_grace_seconds,
            },
        }

    def set_pinned(self, artifact_id: str, pinned: bool = True) -> ArtifactMetadata:
        """Pin or unpin an artifact against automatic cleanup."""
        with self._write_lock:
            with self._lock:
                metadata = self.get(artifact_id)
                updated = replace(metadata, pinned=pinned)
                payload = json.dumps(
                    asdict(updated), ensure_ascii=True, separators=(",", ":")
                ).encode("utf-8")
                self._write_atomic(self._metadata_path(artifact_id), payload)
        return updated

    def set_pinned_for_session(
        self,
        artifact_id: str,
        session_id: Optional[str],
        pinned: bool = True,
    ) -> ArtifactMetadata:
        """Pin an artifact after enforcing debugger-session ownership."""
        self.assert_owner(artifact_id, session_id)
        return self.set_pinned(artifact_id, pinned)

    def delete(self, artifact_id: str, *, force: bool = False) -> ArtifactMetadata:
        """Delete one artifact, rejecting pinned entries unless forced."""
        with self._write_lock:
            with self._lock:
                metadata = self.get(artifact_id)
                if metadata.pinned and not force:
                    raise ArtifactPinnedError(
                        f"Artifact '{artifact_id}' is pinned; use force=true to delete it"
                    )
                for path in (
                    self._data_path(artifact_id),
                    self._metadata_path(artifact_id),
                ):
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass
        return metadata

    def delete_for_session(
        self,
        artifact_id: str,
        session_id: Optional[str],
        *,
        force: bool = False,
    ) -> ArtifactMetadata:
        """Delete an artifact after enforcing debugger-session ownership."""
        self.assert_owner(artifact_id, session_id)
        return self.delete(artifact_id, force=force)

    def cleanup(
        self,
        *,
        dry_run: bool = False,
        incoming_size: int = 0,
        incoming_already_allocated: bool = False,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Apply TTL, quota, and minimum-free-space retention rules."""
        with self._write_lock:
            return self._cleanup_locked(
                dry_run=dry_run,
                incoming_size=incoming_size,
                incoming_already_allocated=incoming_already_allocated,
                now=now,
            )

    def _cleanup_locked(
        self,
        *,
        dry_run: bool,
        incoming_size: int,
        incoming_already_allocated: bool,
        now: Optional[datetime],
    ) -> Dict[str, Any]:
        """Cleanup implementation called while the reentrant write lock is held."""
        if incoming_size < 0:
            raise ValueError("incoming_size cannot be negative")
        now = now or datetime.now(timezone.utc)
        artifacts = self.list()
        operational_debt = self._operational_debt()
        total_before = (
            sum(item.size_bytes for item in artifacts)
            + operational_debt["quarantine_bytes"]
        )
        free_before = shutil.disk_usage(self.root_dir).free
        selected: List[Dict[str, Any]] = []
        selected_ids = set()

        def select(item: ArtifactMetadata, reason: str) -> None:
            if item.artifact_id not in selected_ids:
                selected_ids.add(item.artifact_id)
                selected.append(
                    {
                        "artifact_id": item.artifact_id,
                        "size_bytes": item.size_bytes,
                        "reason": reason,
                    }
                )

        for item in artifacts:
            if (
                not item.pinned
                and item.expires_at
                and datetime.fromisoformat(item.expires_at) <= now
            ):
                select(item, "ttl")

        deleted_bytes = sum(item["size_bytes"] for item in selected)
        simulated_total = total_before - deleted_bytes
        simulated_free = free_before + deleted_bytes
        target_bytes = int(self.quota_bytes * self.cleanup_target_ratio)
        quota_pressure = bool(
            self.quota_bytes and simulated_total + incoming_size > self.quota_bytes
        )
        target_total = (
            max(0, target_bytes - incoming_size) if quota_pressure else simulated_total
        )
        required_free = self.min_free_bytes + (
            0 if incoming_already_allocated else incoming_size
        )

        survivors = [
            item
            for item in sorted(artifacts, key=lambda value: value.created_at)
            if not item.pinned and item.artifact_id not in selected_ids
        ]
        if incoming_size == 0 and survivors:
            survivors = survivors[:-1]
        for item in survivors:
            needs_quota = quota_pressure and simulated_total > target_total
            needs_disk = simulated_free < required_free
            if not needs_quota and not needs_disk:
                break
            reason = (
                "quota+disk"
                if needs_quota and needs_disk
                else ("quota" if needs_quota else "disk")
            )
            select(item, reason)
            simulated_total -= item.size_bytes
            simulated_free += item.size_bytes

        if not dry_run:
            for candidate in selected:
                try:
                    self.delete(candidate["artifact_id"], force=True)
                except FileNotFoundError:
                    pass

        free_after = (
            shutil.disk_usage(self.root_dir).free if not dry_run else simulated_free
        )
        return {
            "dry_run": dry_run,
            "incoming_size": incoming_size,
            "quarantine_bytes": operational_debt["quarantine_bytes"],
            "total_before_bytes": total_before,
            "total_after_bytes": simulated_total,
            "free_before_bytes": free_before,
            "free_after_bytes": free_after,
            "deleted_count": len(selected),
            "deleted_bytes": sum(item["size_bytes"] for item in selected),
            "deletions": selected,
            "skipped_pinned_count": sum(item.pinned for item in artifacts),
            "quota_satisfied": not self.quota_bytes
            or simulated_total + incoming_size <= self.quota_bytes,
            "disk_reserve_satisfied": free_after >= required_free,
        }

    def ensure_capacity(self, incoming_size: int) -> None:
        """Evict eligible old artifacts and enforce the physical disk reserve."""
        result = self.cleanup(incoming_size=incoming_size)
        self._raise_for_capacity(result)

    def start_cleanup_worker(self) -> None:
        """Run startup cleanup and start the periodic daemon worker."""
        with self._lock:
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                return
            self._stop_event.clear()
        # No tool can be using this newly constructed store yet, so startup can
        # safely reclaim even freshly timestamped leftovers from a prior crash.
        self.reconcile_orphans(grace_seconds=0)
        self.cleanup()
        if self.cleanup_interval_seconds == 0:
            return

        def worker() -> None:
            while not self._stop_event.wait(self.cleanup_interval_seconds):
                try:
                    self.reconcile_orphans()
                    self.cleanup()
                except Exception:
                    logger.exception("Periodic artifact cleanup failed")

        self._cleanup_thread = threading.Thread(
            target=worker,
            name="artifact-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def close(self) -> None:
        """Stop the periodic cleanup worker."""
        self._stop_event.set()
        thread = self._cleanup_thread
        if thread and thread.is_alive():
            thread.join(timeout=2)
        self._cleanup_thread = None

    def read(
        self,
        artifact_id: str,
        *,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read a byte range, or the complete artifact when length is None."""
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if length is not None and length < 0:
            raise ValueError("length cannot be negative")
        if length is not None and length > self.max_read_bytes:
            raise ValueError(
                f"length cannot exceed {self.max_read_bytes} bytes; use next_offset"
            )

        metadata = self.get(artifact_id)
        requested_length = length
        if requested_length is None:
            requested_length = min(
                self.max_read_bytes, max(0, metadata.size_bytes - offset)
            )

        with self._lock:
            with open(self._data_path(artifact_id), "rb") as artifact_file:
                artifact_file.seek(offset)
                payload = artifact_file.read(requested_length)

        next_offset = offset + len(payload)
        result = {
            "artifact": metadata.to_dict(),
            "offset": offset,
            "length": len(payload),
            "next_offset": next_offset,
            "eof": next_offset >= metadata.size_bytes,
            "content_base64": base64.b64encode(payload).decode("ascii"),
        }
        if metadata.encoding != BINARY_ARTIFACT_ENCODING:
            result["content"] = payload.decode(metadata.encoding, errors="replace")
        return result

    def read_for_session(
        self,
        artifact_id: str,
        session_id: Optional[str],
        *,
        offset: int = 0,
        length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Read an artifact range after enforcing debugger-session ownership."""
        self.assert_owner(artifact_id, session_id)
        return self.read(artifact_id, offset=offset, length=length)

    def verify(
        self, artifact_id: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Recompute size and SHA-256 to detect incomplete or corrupt payloads."""
        metadata = self.assert_owner(artifact_id, session_id)
        digest = hashlib.sha256()
        observed_size = 0
        with self._lock:
            with open(self._data_path(artifact_id), "rb") as artifact_file:
                while chunk := artifact_file.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
        observed_sha256 = digest.hexdigest()
        valid = (
            observed_size == metadata.size_bytes and observed_sha256 == metadata.sha256
        )
        return {
            "artifact": metadata.to_dict(),
            "valid": valid,
            "expected_size_bytes": metadata.size_bytes,
            "observed_size_bytes": observed_size,
            "expected_sha256": metadata.sha256,
            "observed_sha256": observed_sha256,
        }
