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
import tempfile
import threading
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

DEFAULT_INLINE_RESPONSE_BYTES = int(
    os.environ.get("PAO_INLINE_RESPONSE_BYTES", str(1024 * 1024))
)
DEFAULT_ARTIFACT_CHUNK_BYTES = int(
    os.environ.get("PAO_ARTIFACT_CHUNK_BYTES", str(256 * 1024))
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
ARTIFACT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
logger = logging.getLogger(__name__)


class ArtifactStorageFullError(RuntimeError):
    """Raised when an artifact cannot be stored without exhausting the disk."""


class ArtifactPinnedError(RuntimeError):
    """Raised when deletion is attempted on a pinned artifact."""


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
        ttl_seconds: int = DEFAULT_ARTIFACT_TTL_SECONDS,
        quota_bytes: int = DEFAULT_ARTIFACT_QUOTA_BYTES,
        cleanup_target_ratio: float = DEFAULT_ARTIFACT_CLEANUP_TARGET_RATIO,
        cleanup_interval_seconds: float = DEFAULT_ARTIFACT_CLEANUP_INTERVAL_SECONDS,
        min_free_bytes: int = DEFAULT_ARTIFACT_MIN_FREE_BYTES,
    ):
        if inline_response_bytes < 0:
            raise ValueError("inline_response_bytes cannot be negative")
        if default_chunk_bytes <= 0:
            raise ValueError("default_chunk_bytes must be positive")
        if ttl_seconds < 0 or quota_bytes < 0 or min_free_bytes < 0:
            raise ValueError("artifact retention sizes and TTL cannot be negative")
        if not 0 < cleanup_target_ratio <= 1:
            raise ValueError(
                "cleanup_target_ratio must be greater than 0 and at most 1"
            )
        if cleanup_interval_seconds < 0:
            raise ValueError("cleanup_interval_seconds cannot be negative")
        self.root_dir = os.path.abspath(root_dir)
        self.inline_response_bytes = inline_response_bytes
        self.default_chunk_bytes = default_chunk_bytes
        self.ttl_seconds = ttl_seconds
        self.quota_bytes = quota_bytes
        self.cleanup_target_ratio = cleanup_target_ratio
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.min_free_bytes = min_free_bytes
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
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
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
                        digest.update(chunk)
                        size_bytes += len(chunk)
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

    def status(self) -> Dict[str, Any]:
        """Return retention policy and current artifact storage usage."""
        artifacts = self.list()
        now = datetime.now(timezone.utc)
        usage = sum(item.size_bytes for item in artifacts)
        return {
            "artifact_count": len(artifacts),
            "pinned_count": sum(item.pinned for item in artifacts),
            "expired_count": sum(
                item.expires_at is not None
                and datetime.fromisoformat(item.expires_at) <= now
                for item in artifacts
            ),
            "usage_bytes": usage,
            "free_bytes": shutil.disk_usage(self.root_dir).free,
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
            },
        }

    def set_pinned(self, artifact_id: str, pinned: bool = True) -> ArtifactMetadata:
        """Pin or unpin an artifact against automatic cleanup."""
        with self._lock:
            metadata = self.get(artifact_id)
            updated = replace(metadata, pinned=pinned)
            payload = json.dumps(
                asdict(updated), ensure_ascii=True, separators=(",", ":")
            ).encode("utf-8")
            self._write_atomic(self._metadata_path(artifact_id), payload)
        return updated

    def delete(self, artifact_id: str, *, force: bool = False) -> ArtifactMetadata:
        """Delete one artifact, rejecting pinned entries unless forced."""
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

    def cleanup(
        self,
        *,
        dry_run: bool = False,
        incoming_size: int = 0,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Apply TTL, quota, and minimum-free-space retention rules."""
        if incoming_size < 0:
            raise ValueError("incoming_size cannot be negative")
        now = now or datetime.now(timezone.utc)
        artifacts = self.list()
        total_before = sum(item.size_bytes for item in artifacts)
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
        required_free = self.min_free_bytes + incoming_size

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
        if not result["disk_reserve_satisfied"]:
            raise ArtifactStorageFullError(
                "Not enough artifact storage after cleanup; pinned artifacts or "
                "filesystem usage prevent preserving the configured free-space reserve"
            )

    def start_cleanup_worker(self) -> None:
        """Run startup cleanup and start the periodic daemon worker."""
        with self._lock:
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                return
            self._stop_event.clear()
        self.cleanup()
        if self.cleanup_interval_seconds == 0:
            return

        def worker() -> None:
            while not self._stop_event.wait(self.cleanup_interval_seconds):
                try:
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

        metadata = self.get(artifact_id)
        requested_length = length
        if requested_length is None:
            requested_length = max(0, metadata.size_bytes - offset)

        with self._lock:
            with open(self._data_path(artifact_id), "rb") as artifact_file:
                artifact_file.seek(offset)
                payload = artifact_file.read(requested_length)

        next_offset = offset + len(payload)
        return {
            "artifact": metadata.to_dict(),
            "offset": offset,
            "length": len(payload),
            "next_offset": next_offset,
            "eof": next_offset >= metadata.size_bytes,
            "content": payload.decode(metadata.encoding, errors="replace"),
            "content_base64": base64.b64encode(payload).decode("ascii"),
        }
