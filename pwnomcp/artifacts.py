"""Persistent storage for large MCP tool payloads."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
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
ARTIFACT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


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
    ):
        if inline_response_bytes < 0:
            raise ValueError("inline_response_bytes cannot be negative")
        if default_chunk_bytes <= 0:
            raise ValueError("default_chunk_bytes must be positive")
        self.root_dir = os.path.abspath(root_dir)
        self.inline_response_bytes = inline_response_bytes
        self.default_chunk_bytes = default_chunk_bytes
        self._lock = threading.RLock()
        os.makedirs(self.root_dir, exist_ok=True)

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
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            tool=tool,
            session_id=session_id,
            content_type=content_type,
            encoding=encoding,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        metadata_payload = json.dumps(
            asdict(metadata), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

        with self._lock:
            self._write_atomic(self._data_path(artifact_id), payload)
            try:
                self._write_atomic(self._metadata_path(artifact_id), metadata_payload)
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
        artifact_id = uuid.uuid4().hex
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".artifact_", dir=self.root_dir
        )
        digest = hashlib.sha256()
        size_bytes = 0

        try:
            with (
                os.fdopen(descriptor, "wb") as destination,
                open(source_path, "rb") as source,
            ):
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                destination.flush()
                os.fsync(destination.fileno())

            metadata = ArtifactMetadata(
                artifact_id=artifact_id,
                tool=tool,
                session_id=session_id,
                content_type=content_type,
                encoding=encoding,
                size_bytes=size_bytes,
                sha256=digest.hexdigest(),
                created_at=datetime.now(timezone.utc).isoformat(),
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
