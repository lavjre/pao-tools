"""Typed research bundle export / import / verify.

A research bundle is a single ``.zip`` file that lets a caller close a
research session and reopen it later (or share it) without losing
provenance.  The layout is intentionally boring so an out-of-band
tool can read it too:

    manifest.json      -- one JSON document with the full research
                          session (session metadata, notes,
                          hypotheses, evidence pointers, snapshots,
                          timeline) plus a ``bundle`` header
                          (schema version, tool/parser versions,
                          created-at, artifact index).
    integrity.json     -- SHA-256 over every payload in ``artifacts/``
                          (keyed by ``arcname``) plus the digest of
                          ``manifest.json`` itself, so verify covers the
                          session narrative and not just the blobs.
                          Signed only by the hash function; a caller who
                          needs a real signature layers that on top of
                          the bundle.
    artifacts/<sha256>  -- one file per referenced artifact,
                          content-addressed by SHA-256 so identical
                          payloads are stored once even across
                          different sessions.

``export_research_session`` follows every ``kind == "artifact"``
evidence entry and copies the underlying blob out of
:class:`ArtifactStore` into the bundle.  Crash / heap / tool_result /
external / note references stay as opaque strings in the manifest --
they name state that lives in stores the recipient may or may not
have; the bundle records the reference so the eventual final report
can still cite them, but does not duplicate their bytes here.

``verify_research_bundle`` checks:

  1. The zip opens cleanly and carries both ``manifest.json`` and
     ``integrity.json``.
  2. Every artifact listed in the manifest / integrity file exists
     in ``artifacts/`` and its SHA-256 matches the recorded value.
  3. ``manifest.json`` has a recorded SHA-256 in ``integrity.json``
     and its bytes match it, so the session record itself is covered.
  4. No files outside the documented layout are present (defensive
     against accidental leakage of unrelated files).

``import_research_session`` verifies first, then rehydrates a new
:class:`ResearchSession` (with a fresh session id so an imported
bundle never clobbers an existing session).  Every original id
inside the manifest is remapped through a translation table so
cross-references between notes / hypotheses / evidence / snapshots
/ timeline events remain internally consistent after import.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import struct
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from paomcp.artifacts import ArtifactStore
from paomcp.tools.backends.foundation import build_backend_response
from paomcp.tools.backends.research import (
    ResearchError,
    ResearchEvidence,
    ResearchHypothesis,
    ResearchNote,
    ResearchRegistry,
    ResearchSession,
    ResearchSnapshot,
    EVIDENCE_KINDS,
    HYPOTHESIS_STATES,
    SNAPSHOT_SLOTS,
    TIMELINE_EVENT_KINDS,
    TimelineEvent,
)

PARSER_VERSION = "1"
RESEARCH_EXPORT_SCHEMA = "pao.research-export.v1"
RESEARCH_IMPORT_SCHEMA = "pao.research-import.v1"
RESEARCH_VERIFY_SCHEMA = "pao.research-verify.v1"

BUNDLE_SCHEMA_VERSION = "pao.research-bundle.v1"
MANIFEST_ARCNAME = "manifest.json"
INTEGRITY_ARCNAME = "integrity.json"
ARTIFACTS_DIRNAME = "artifacts"
DEFAULT_MAX_BUNDLE_BYTES = 512 * 1024 * 1024  # 512 MiB
DEFAULT_MAX_BUNDLE_ENTRIES = 4096
DEFAULT_MAX_MANIFEST_BYTES = 32 * 1024 * 1024  # 32 MiB
DEFAULT_MAX_INTEGRITY_BYTES = 4 * 1024 * 1024  # 4 MiB
DEFAULT_MAX_COMPRESSION_RATIO = 200.0
DEFAULT_MAX_ENTRY_NAME_BYTES = 255
DEFAULT_MAX_ZIP_METADATA_FIELD_BYTES = 4096
_STREAM_CHUNK_BYTES = 1024 * 1024
_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_EOCD_STRUCT = struct.Struct("<4s4H2LH")
_CENTRAL_DIRECTORY_HEADER_BYTES = 46
_CANONICAL_ARTIFACT_RE = re.compile(r"^artifacts/[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BundleError(RuntimeError):
    """Raised when a bundle cannot be built, verified or imported."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class ResearchBundle:
    """Export / import / verify typed research bundles.

    The class holds no state beyond its dependencies -- a research
    registry and an artifact store -- so a caller can share one
    instance across every research session in the process.
    """

    def __init__(
        self,
        registry: ResearchRegistry,
        artifact_store: Optional[ArtifactStore] = None,
        *,
        max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES,
        max_bundle_entries: int = DEFAULT_MAX_BUNDLE_ENTRIES,
        max_uncompressed_bytes: Optional[int] = None,
        max_manifest_bytes: int = DEFAULT_MAX_MANIFEST_BYTES,
        max_integrity_bytes: int = DEFAULT_MAX_INTEGRITY_BYTES,
        max_compression_ratio: float = DEFAULT_MAX_COMPRESSION_RATIO,
        max_entry_name_bytes: int = DEFAULT_MAX_ENTRY_NAME_BYTES,
    ) -> None:
        if max_bundle_bytes <= 0:
            raise ValueError("max_bundle_bytes must be positive")
        if max_bundle_entries <= 0:
            raise ValueError("max_bundle_entries must be positive")
        if max_uncompressed_bytes is not None and max_uncompressed_bytes <= 0:
            raise ValueError("max_uncompressed_bytes must be positive")
        if max_manifest_bytes <= 0:
            raise ValueError("max_manifest_bytes must be positive")
        if max_integrity_bytes <= 0:
            raise ValueError("max_integrity_bytes must be positive")
        if max_compression_ratio < 1:
            raise ValueError("max_compression_ratio must be at least 1")
        if max_entry_name_bytes <= 0:
            raise ValueError("max_entry_name_bytes must be positive")
        self._registry = registry
        self._artifact_store = artifact_store
        self._max_bundle_bytes = max_bundle_bytes
        self._max_bundle_entries = max_bundle_entries
        self._max_uncompressed_bytes = (
            max_bundle_bytes
            if max_uncompressed_bytes is None
            else max_uncompressed_bytes
        )
        self._max_manifest_bytes = min(max_manifest_bytes, self._max_uncompressed_bytes)
        self._max_integrity_bytes = min(
            max_integrity_bytes, self._max_uncompressed_bytes
        )
        self._max_compression_ratio = float(max_compression_ratio)
        self._max_entry_name_bytes = max_entry_name_bytes

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _collect_artifact_payloads(
        self, session: ResearchSession
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bytes], List[str]]:
        """Read every ``kind=artifact`` evidence blob out of the store.

        Returns ``(index_entries, blob_by_sha, warnings)``.  Missing
        artifacts do not fail the export -- they are dropped from the
        blob map, still recorded in the manifest, and surfaced in
        ``warnings`` so the caller knows the eventual report will be
        one short of provenance.
        """
        index_entries: List[Dict[str, Any]] = []
        blob_by_sha: Dict[str, bytes] = {}
        warnings: List[str] = []
        if self._artifact_store is None:
            # No artifact store wired -- record the artifact refs but
            # do not try to read them.
            for evidence in session.evidence.values():
                if evidence.kind != "artifact":
                    continue
                index_entries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.ref,
                        "resolved": False,
                        "reason": "artifact_store not wired to bundle",
                    }
                )
                warnings.append(
                    f"artifact '{evidence.ref}' not included; no store wired"
                )
            return index_entries, blob_by_sha, warnings
        for evidence in session.evidence.values():
            if evidence.kind != "artifact":
                continue
            try:
                metadata = self._artifact_store.get(evidence.ref)
            except FileNotFoundError:
                warnings.append(
                    f"artifact '{evidence.ref}' referenced by evidence "
                    f"'{evidence.evidence_id}' is missing from the store"
                )
                index_entries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.ref,
                        "resolved": False,
                        "reason": "not_found",
                    }
                )
                continue
            except ValueError:
                warnings.append(
                    f"artifact '{evidence.ref}' referenced by evidence "
                    f"'{evidence.evidence_id}' is not a valid artifact id"
                )
                index_entries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.ref,
                        "resolved": False,
                        "reason": "invalid_ref",
                    }
                )
                continue
            data_path = self._artifact_store._data_path(evidence.ref)
            try:
                with open(data_path, "rb") as handle:
                    payload = handle.read()
            except OSError:
                warnings.append(
                    f"artifact '{evidence.ref}' referenced by evidence "
                    f"'{evidence.evidence_id}' could not be read"
                )
                index_entries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.ref,
                        "resolved": False,
                        "reason": "read_failed",
                    }
                )
                continue
            observed_sha = _sha256_bytes(payload)
            if observed_sha != metadata.sha256:
                warnings.append(
                    f"artifact '{evidence.ref}' sha256 mismatch "
                    f"(recorded={metadata.sha256[:12]}, "
                    f"observed={observed_sha[:12]}); dropped from bundle"
                )
                index_entries.append(
                    {
                        "evidence_id": evidence.evidence_id,
                        "artifact_id": evidence.ref,
                        "resolved": False,
                        "reason": "sha_mismatch",
                    }
                )
                continue
            blob_by_sha.setdefault(observed_sha, payload)
            index_entries.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "artifact_id": evidence.ref,
                    "resolved": True,
                    "sha256": observed_sha,
                    "size_bytes": metadata.size_bytes,
                    "content_type": metadata.content_type,
                    "arcname": f"{ARTIFACTS_DIRNAME}/{observed_sha}",
                }
            )
        return index_entries, blob_by_sha, warnings

    def _build_manifest(
        self,
        session: ResearchSession,
        *,
        artifact_index: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        # ``session.to_dict`` includes children summaries but omits
        # timeline events + snapshot slot payloads; for the bundle we
        # want the full payload so a reopened bundle has everything.
        session_payload = session.to_dict(include_children=True)
        # Replace the summary-only snapshots list with the full-slot version.
        session_payload["snapshots"] = [
            snap.to_dict(include_slots=True) for snap in session.snapshots.values()
        ]
        session_payload["timeline"] = [event.to_dict() for event in session.timeline]
        return {
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parser_version": PARSER_VERSION,
            "artifact_index": artifact_index,
            "session": session_payload,
        }

    def export_to_bytes(self, *, session_id: str) -> Tuple[bytes, Dict[str, Any]]:
        """Return ``(bundle_bytes, summary)`` for a research session."""
        session = self._registry.get_session(session_id)
        with session._lock:
            index_entries, blob_by_sha, warnings = self._collect_artifact_payloads(
                session
            )
            manifest = self._build_manifest(session, artifact_index=index_entries)
        manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        # The manifest carries the session narrative (notes, hypotheses,
        # evidence, snapshots, timeline).  Hash it into the integrity index so
        # verify detects tampering of the record itself, not just artifacts/.
        manifest_sha = _sha256_bytes(manifest_bytes)
        integrity: Dict[str, str] = {
            f"{ARTIFACTS_DIRNAME}/{sha}": sha for sha in blob_by_sha
        }
        integrity[MANIFEST_ARCNAME] = manifest_sha
        integrity_bytes = json.dumps(integrity, sort_keys=True, indent=2).encode(
            "utf-8"
        )
        if len(manifest_bytes) > self._max_manifest_bytes:
            raise BundleError(
                "research bundle manifest exceeds the byte cap "
                f"({len(manifest_bytes)} > {self._max_manifest_bytes})"
            )
        if len(integrity_bytes) > self._max_integrity_bytes:
            raise BundleError(
                "research bundle integrity index exceeds the byte cap "
                f"({len(integrity_bytes)} > {self._max_integrity_bytes})"
            )
        # Enforce budget before we actually zip.  Byte sum is
        # approximate (zip DEFLATE usually shrinks) but the pre-check
        # protects the caller from OOMing on export.
        total_bytes = (
            len(manifest_bytes)
            + len(integrity_bytes)
            + sum(len(payload) for payload in blob_by_sha.values())
        )
        if total_bytes > self._max_bundle_bytes:
            raise BundleError(
                f"research bundle exceeds the byte cap "
                f"({total_bytes} > {self._max_bundle_bytes})"
            )
        if total_bytes > self._max_uncompressed_bytes:
            raise BundleError(
                "research bundle exceeds the uncompressed byte cap "
                f"({total_bytes} > {self._max_uncompressed_bytes})"
            )
        total_entries = 2 + len(blob_by_sha)
        if total_entries > self._max_bundle_entries:
            raise BundleError(
                f"research bundle exceeds the entry cap "
                f"({total_entries} > {self._max_bundle_entries})"
            )
        export_entries: List[Tuple[str, bytes]] = [
            (MANIFEST_ARCNAME, manifest_bytes),
            (INTEGRITY_ARCNAME, integrity_bytes),
            *[
                (f"{ARTIFACTS_DIRNAME}/{sha}", payload)
                for sha, payload in blob_by_sha.items()
            ],
        ]

        def build_archive(stored_entries: set[str]) -> bytes:
            buffer = io.BytesIO()
            with zipfile.ZipFile(
                buffer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
            ) as zf:
                for arcname, payload in export_entries:
                    zf.writestr(
                        arcname,
                        payload,
                        compress_type=(
                            zipfile.ZIP_STORED
                            if arcname in stored_entries
                            else zipfile.ZIP_DEFLATED
                        ),
                    )
            return buffer.getvalue()

        bundle_bytes = build_archive(set())
        # A legitimate export must pass the same expansion-ratio rule as an
        # imported bundle.  Store only unusually compressible members rather
        # than emitting an archive that our own verifier would reject.
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as generated_zip:
            ratio_offenders = {
                info.filename
                for info in generated_zip.infolist()
                if info.file_size > 0
                and (
                    info.compress_size <= 0
                    or info.file_size / info.compress_size > self._max_compression_ratio
                )
            }
        if ratio_offenders:
            bundle_bytes = build_archive(ratio_offenders)
        if len(bundle_bytes) > self._max_bundle_bytes:
            raise BundleError(
                "research bundle exceeds the compressed byte cap "
                f"({len(bundle_bytes)} > {self._max_bundle_bytes})"
            )
        summary: Dict[str, Any] = {
            "schema": RESEARCH_EXPORT_SCHEMA,
            "session_id": session_id,
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "bundle_bytes": len(bundle_bytes),
            "sha256": _sha256_bytes(bundle_bytes),
            "manifest_sha256": manifest_sha,
            "artifact_count": len(blob_by_sha),
            "note_count": len(session.notes),
            "hypothesis_count": len(session.hypotheses),
            "evidence_count": len(session.evidence),
            "snapshot_count": len(session.snapshots),
            "timeline_length": len(session.timeline),
            "warnings": warnings,
        }
        return bundle_bytes, summary

    def export_to_file(self, *, session_id: str, destination: str) -> Dict[str, Any]:
        bundle_bytes, summary = self.export_to_bytes(session_id=session_id)
        parent = os.path.dirname(destination)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Write to a temp file first so a partial write does not leave
        # a truncated bundle on disk.
        tmp_path = f"{destination}.tmp"
        with open(tmp_path, "wb") as handle:
            handle.write(bundle_bytes)
        os.replace(tmp_path, destination)
        summary["destination"] = os.path.abspath(destination)
        return summary

    # ------------------------------------------------------------------
    # Verify (no state changes)
    # ------------------------------------------------------------------
    _KNOWN_TOPLEVEL = {MANIFEST_ARCNAME, INTEGRITY_ARCNAME}

    def _verification_result(
        self,
        *,
        bundle_bytes: int,
        problems: List[str],
        manifest: Optional[Mapping[str, Any]] = None,
        artifact_shas: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        session_meta = None
        if manifest is not None and isinstance(manifest.get("session"), dict):
            session = manifest["session"]
            session_meta = {
                "name": session.get("name"),
                "session_id": session.get("session_id"),
                "note_count": session.get("note_count"),
                "hypothesis_count": session.get("hypothesis_count"),
                "evidence_count": session.get("evidence_count"),
                "snapshot_count": session.get("snapshot_count"),
                "timeline_length": session.get("timeline_length"),
            }
        return {
            "schema": RESEARCH_VERIFY_SCHEMA,
            "verdict": "valid" if not problems else "invalid",
            "bundle_bytes": bundle_bytes,
            "bundle_schema": (
                manifest.get("bundle_schema") if manifest is not None else None
            ),
            "artifact_count": len(artifact_shas or []),
            "problems": problems,
            "session": session_meta,
        }

    def _central_directory_problems(self, bundle_bytes: bytes) -> List[str]:
        """Inspect the central-directory envelope before ``ZipFile`` allocates it.

        ``zipfile.ZipFile`` materializes every central-directory record on
        open.  Reading and checking the EOCD plus fixed-size central headers
        first keeps a forged entry count from turning that convenience into an
        allocation primitive.
        """
        if len(bundle_bytes) > self._max_bundle_bytes:
            return [
                "bundle exceeds the compressed byte cap "
                f"({len(bundle_bytes)} > {self._max_bundle_bytes})"
            ]
        minimum_eocd_bytes = _EOCD_STRUCT.size
        if len(bundle_bytes) < minimum_eocd_bytes:
            return ["bundle is not a valid zip: end-of-central-directory not found"]

        search_start = max(0, len(bundle_bytes) - (65535 + minimum_eocd_bytes))
        search_end = len(bundle_bytes)
        eocd_offset = -1
        eocd_fields: Optional[Tuple[Any, ...]] = None
        while search_end > search_start:
            candidate = bundle_bytes.rfind(_EOCD_SIGNATURE, search_start, search_end)
            if candidate < 0 or candidate + minimum_eocd_bytes > len(bundle_bytes):
                break
            fields = _EOCD_STRUCT.unpack_from(bundle_bytes, candidate)
            comment_length = int(fields[-1])
            if candidate + minimum_eocd_bytes + comment_length == len(bundle_bytes):
                eocd_offset = candidate
                eocd_fields = fields
                break
            search_end = candidate
        if eocd_fields is None:
            return ["bundle is not a valid zip: end-of-central-directory not found"]

        (
            _signature,
            disk_number,
            central_directory_disk,
            entries_on_disk,
            entry_count,
            central_directory_bytes,
            central_directory_offset,
            _comment_length,
        ) = eocd_fields
        if (
            entry_count == 0xFFFF
            or entries_on_disk == 0xFFFF
            or central_directory_bytes == 0xFFFFFFFF
            or central_directory_offset == 0xFFFFFFFF
        ):
            return ["ZIP64 bundles are not supported by the configured bundle caps"]
        if disk_number != 0 or central_directory_disk != 0:
            return ["multi-disk ZIP bundles are not supported"]
        if entries_on_disk != entry_count:
            return ["central-directory entry counts do not match"]
        if entry_count > self._max_bundle_entries:
            return [
                "bundle exceeds the entry cap "
                f"({entry_count} > {self._max_bundle_entries})"
            ]
        if central_directory_bytes > self._max_bundle_bytes:
            return [
                "central directory exceeds the compressed byte cap "
                f"({central_directory_bytes} > {self._max_bundle_bytes})"
            ]

        # The archives produced here have no central-directory digital
        # signature.  Calculating the physical start from the EOCD also
        # permits a harmless prepended self-extracting stub without trusting
        # its attacker-controlled logical offset.
        central_start = eocd_offset - int(central_directory_bytes)
        if central_start < 0:
            return ["central directory points outside the bundle"]
        cursor = central_start
        for _ in range(int(entry_count)):
            fixed_end = cursor + _CENTRAL_DIRECTORY_HEADER_BYTES
            if (
                fixed_end > eocd_offset
                or bundle_bytes[cursor : cursor + 4] != _CENTRAL_DIRECTORY_SIGNATURE
            ):
                return ["central directory is malformed"]
            name_length, extra_length, comment_length = struct.unpack_from(
                "<HHH", bundle_bytes, cursor + 28
            )
            if name_length > self._max_entry_name_bytes:
                return [
                    "ZIP entry name exceeds the byte cap "
                    f"({name_length} > {self._max_entry_name_bytes})"
                ]
            if extra_length > DEFAULT_MAX_ZIP_METADATA_FIELD_BYTES:
                return [
                    "ZIP entry extra field exceeds the byte cap "
                    f"({extra_length} > {DEFAULT_MAX_ZIP_METADATA_FIELD_BYTES})"
                ]
            if comment_length > DEFAULT_MAX_ZIP_METADATA_FIELD_BYTES:
                return [
                    "ZIP entry comment exceeds the byte cap "
                    f"({comment_length} > {DEFAULT_MAX_ZIP_METADATA_FIELD_BYTES})"
                ]
            cursor = fixed_end + name_length + extra_length + comment_length
            if cursor > eocd_offset:
                return ["central directory is malformed"]
        if cursor != eocd_offset:
            return ["central-directory size or entry count is inconsistent"]
        return []

    def _archive_metadata_problems(
        self, zf: zipfile.ZipFile
    ) -> Tuple[List[str], Dict[str, zipfile.ZipInfo]]:
        problems: List[str] = []
        infos = zf.infolist()
        if len(infos) > self._max_bundle_entries:
            problems.append(
                "bundle exceeds the entry cap "
                f"({len(infos)} > {self._max_bundle_entries})"
            )
            return problems, {}

        info_by_name: Dict[str, zipfile.ZipInfo] = {}
        compressed_bytes = 0
        uncompressed_bytes = 0
        for info in infos:
            name = info.filename
            if name in info_by_name:
                problems.append(f"duplicate bundle entry '{name}'")
                continue
            info_by_name[name] = info
            encoded_name_bytes = len(name.encode("utf-8", errors="surrogatepass"))
            if encoded_name_bytes > self._max_entry_name_bytes:
                problems.append(
                    f"bundle entry name '{name[:80]}' exceeds the byte cap "
                    f"({encoded_name_bytes} > {self._max_entry_name_bytes})"
                )
            if info.flag_bits & 0x1:
                problems.append(f"encrypted bundle entry '{name}' is not supported")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                problems.append(
                    f"bundle entry '{name}' uses unsupported compression "
                    f"type {info.compress_type}"
                )
            compressed_bytes += info.compress_size
            uncompressed_bytes += info.file_size
            if info.file_size > 0:
                if info.compress_size <= 0:
                    problems.append(
                        f"bundle entry '{name}' has an invalid compressed size"
                    )
                elif info.file_size / info.compress_size > self._max_compression_ratio:
                    problems.append(
                        f"bundle entry '{name}' exceeds the compression-ratio cap "
                        f"({info.file_size / info.compress_size:.1f} > "
                        f"{self._max_compression_ratio:.1f})"
                    )

            if name in self._KNOWN_TOPLEVEL:
                pass
            elif _CANONICAL_ARTIFACT_RE.fullmatch(name):
                pass
            elif name.startswith(f"{ARTIFACTS_DIRNAME}/"):
                problems.append(
                    f"artifact entry '{name}' is not a canonical SHA-256 path"
                )
            else:
                problems.append(f"unexpected bundle entry '{name}'")

        if compressed_bytes > self._max_bundle_bytes:
            problems.append(
                "bundle entries exceed the compressed byte cap "
                f"({compressed_bytes} > {self._max_bundle_bytes})"
            )
        if uncompressed_bytes > self._max_uncompressed_bytes:
            problems.append(
                "bundle exceeds the uncompressed byte cap "
                f"({uncompressed_bytes} > {self._max_uncompressed_bytes})"
            )
        manifest_info = info_by_name.get(MANIFEST_ARCNAME)
        integrity_info = info_by_name.get(INTEGRITY_ARCNAME)
        if manifest_info is None:
            problems.append("missing manifest.json")
        elif manifest_info.file_size > self._max_manifest_bytes:
            problems.append(
                "manifest.json exceeds the byte cap "
                f"({manifest_info.file_size} > {self._max_manifest_bytes})"
            )
        if integrity_info is None:
            problems.append("missing integrity.json")
        elif integrity_info.file_size > self._max_integrity_bytes:
            problems.append(
                "integrity.json exceeds the byte cap "
                f"({integrity_info.file_size} > {self._max_integrity_bytes})"
            )
        return problems, info_by_name

    @staticmethod
    def _read_member_bytes(
        zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, limit: int
    ) -> bytes:
        chunks: List[bytes] = []
        total = 0
        with zf.open(info, "r") as source:
            while True:
                remaining = limit - total
                chunk = source.read(min(_STREAM_CHUNK_BYTES, remaining + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise BundleError(
                        f"bundle entry '{info.filename}' exceeded its "
                        f"decompression cap ({total} > {limit})"
                    )
                chunks.append(chunk)
        if total != info.file_size:
            raise BundleError(
                f"bundle entry '{info.filename}' size changed while decompressing "
                f"({total} != {info.file_size})"
            )
        return b"".join(chunks)

    @staticmethod
    def _hash_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, *, limit: int) -> str:
        digest = hashlib.sha256()
        total = 0
        with zf.open(info, "r") as source:
            while True:
                remaining = limit - total
                chunk = source.read(min(_STREAM_CHUNK_BYTES, remaining + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise BundleError(
                        f"bundle entry '{info.filename}' exceeded its "
                        f"decompression cap ({total} > {limit})"
                    )
                digest.update(chunk)
        if total != info.file_size:
            raise BundleError(
                f"bundle entry '{info.filename}' size changed while decompressing "
                f"({total} != {info.file_size})"
            )
        return digest.hexdigest()

    @staticmethod
    def _optional_string_id(value: Any) -> bool:
        return value is None or isinstance(value, str)

    def _control_document_problems(
        self,
        *,
        manifest: Any,
        integrity: Any,
        info_by_name: Mapping[str, zipfile.ZipInfo],
    ) -> List[str]:
        problems: List[str] = []
        if not isinstance(manifest, dict):
            problems.append("manifest.json must contain a JSON object")
            return problems
        if manifest.get("bundle_schema") != BUNDLE_SCHEMA_VERSION:
            problems.append(
                "manifest.json has unsupported bundle_schema "
                f"{manifest.get('bundle_schema')!r}"
            )

        session = manifest.get("session")
        if not isinstance(session, dict):
            problems.append("manifest.session is not a dict")
        else:
            metadata = session.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                problems.append("manifest.session.metadata must be an object")
            child_specs = (
                ("hypotheses", "_max_hypotheses"),
                ("notes", "_max_notes"),
                ("evidence", "_max_evidence"),
                ("snapshots", "_max_snapshots"),
                ("timeline", "_max_timeline"),
            )
            for child_name, cap_attribute in child_specs:
                children = session.get(child_name, [])
                if children is None:
                    children = []
                if not isinstance(children, list):
                    problems.append(f"manifest.session.{child_name} must be a list")
                    continue
                cap = int(getattr(self._registry, cap_attribute))
                if len(children) > cap:
                    problems.append(
                        f"manifest.session.{child_name} exceeds the registry cap "
                        f"({len(children)} > {cap})"
                    )
                if any(not isinstance(child, dict) for child in children):
                    problems.append(
                        f"manifest.session.{child_name} entries must be objects"
                    )

            hypotheses = session.get("hypotheses", [])
            if hypotheses is None:
                hypotheses = []
            if isinstance(hypotheses, list):
                for index, hypothesis in enumerate(hypotheses):
                    if not isinstance(hypothesis, dict):
                        continue
                    if not self._optional_string_id(hypothesis.get("hypothesis_id")):
                        problems.append(
                            f"manifest hypothesis {index} has an invalid id"
                        )
                    statement = hypothesis.get("statement")
                    if not isinstance(statement, str) or not statement.strip():
                        problems.append(
                            f"manifest hypothesis {index} has an empty statement"
                        )
                    if hypothesis.get("state", "proposed") not in HYPOTHESIS_STATES:
                        problems.append(
                            f"manifest hypothesis {index} has an invalid state"
                        )

            notes = session.get("notes", [])
            if notes is None:
                notes = []
            if isinstance(notes, list):
                for index, note in enumerate(notes):
                    if not isinstance(note, dict):
                        continue
                    if not self._optional_string_id(note.get("note_id")):
                        problems.append(f"manifest note {index} has an invalid id")
                    if not self._optional_string_id(note.get("hypothesis_id")):
                        problems.append(
                            f"manifest note {index} has an invalid hypothesis id"
                        )
                    body = note.get("body")
                    if not isinstance(body, str) or not body.strip():
                        problems.append(f"manifest note {index} has an empty body")
                    tags = note.get("tags", [])
                    if tags is None:
                        tags = []
                    if not isinstance(tags, list):
                        problems.append(f"manifest note {index} tags must be a list")

            evidence_items = session.get("evidence", [])
            if evidence_items is None:
                evidence_items = []
            if isinstance(evidence_items, list):
                for index, evidence in enumerate(evidence_items):
                    if not isinstance(evidence, dict):
                        continue
                    for field_name in ("evidence_id", "hypothesis_id", "note_id"):
                        if not self._optional_string_id(evidence.get(field_name)):
                            problems.append(
                                f"manifest evidence {index} has an invalid "
                                f"{field_name}"
                            )
                    if evidence.get("kind", "external") not in EVIDENCE_KINDS:
                        problems.append(
                            f"manifest evidence {index} has an invalid kind"
                        )
                    ref = evidence.get("ref")
                    if not isinstance(ref, str) or not ref.strip():
                        problems.append(f"manifest evidence {index} has an empty ref")
                    evidence_metadata = evidence.get("metadata", {})
                    if evidence_metadata is None:
                        evidence_metadata = {}
                    if not isinstance(evidence_metadata, dict):
                        problems.append(
                            f"manifest evidence {index} metadata must be an object"
                        )

            snapshots = session.get("snapshots", [])
            if snapshots is None:
                snapshots = []
            if isinstance(snapshots, list):
                for index, snapshot in enumerate(snapshots):
                    if not isinstance(snapshot, dict):
                        continue
                    if not self._optional_string_id(snapshot.get("snapshot_id")):
                        problems.append(f"manifest snapshot {index} has an invalid id")
                    name = snapshot.get("name")
                    if not isinstance(name, str) or not name.strip():
                        problems.append(f"manifest snapshot {index} has an empty name")
                    slots = snapshot.get("slots", {})
                    if slots is None:
                        slots = {}
                    if not isinstance(slots, dict):
                        problems.append(
                            f"manifest snapshot {index} slots must be an object"
                        )
                    else:
                        unknown_slots = sorted(set(slots) - set(SNAPSHOT_SLOTS))
                        if unknown_slots:
                            problems.append(
                                f"manifest snapshot {index} has unknown slots "
                                f"{unknown_slots}"
                            )

            timeline = session.get("timeline", [])
            if timeline is None:
                timeline = []
            if isinstance(timeline, list):
                for index, event in enumerate(timeline):
                    if not isinstance(event, dict):
                        continue
                    for field_name in (
                        "event_id",
                        "hypothesis_id",
                        "note_id",
                        "snapshot_id",
                    ):
                        if not self._optional_string_id(event.get(field_name)):
                            problems.append(
                                f"manifest timeline event {index} has an invalid "
                                f"{field_name}"
                            )
                    if event.get("kind", "other") not in TIMELINE_EVENT_KINDS:
                        problems.append(
                            f"manifest timeline event {index} has an invalid kind"
                        )
                    summary = event.get("summary")
                    if not isinstance(summary, str) or not summary.strip():
                        problems.append(
                            f"manifest timeline event {index} has an empty summary"
                        )
                    event_metadata = event.get("metadata", {})
                    if event_metadata is None:
                        event_metadata = {}
                    if not isinstance(event_metadata, dict):
                        problems.append(
                            f"manifest timeline event {index} metadata "
                            "must be an object"
                        )

        if not isinstance(integrity, dict):
            problems.append("integrity.json must contain a JSON object")
            return problems

        listed_artifacts = set()
        declared_manifest_sha = None
        for arcname, expected_sha in integrity.items():
            if arcname == MANIFEST_ARCNAME:
                if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(
                    expected_sha
                ):
                    problems.append("integrity manifest.json has an invalid SHA-256")
                    continue
                declared_manifest_sha = expected_sha
                if MANIFEST_ARCNAME not in info_by_name:
                    problems.append(
                        "integrity references missing entry 'manifest.json'"
                    )
                continue
            if not isinstance(arcname, str) or not _CANONICAL_ARTIFACT_RE.fullmatch(
                arcname
            ):
                problems.append(
                    f"integrity contains non-canonical artifact path {arcname!r}"
                )
                continue
            if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(
                expected_sha
            ):
                problems.append(f"integrity entry '{arcname}' has an invalid SHA-256")
                continue
            if os.path.basename(arcname) != expected_sha:
                problems.append(
                    f"integrity entry '{arcname}' does not match its content address"
                )
            if arcname not in info_by_name:
                problems.append(f"integrity references missing entry '{arcname}'")
            listed_artifacts.add(arcname)

        present_artifacts = {
            name for name in info_by_name if _CANONICAL_ARTIFACT_RE.fullmatch(name)
        }
        for name in sorted(present_artifacts - listed_artifacts):
            problems.append(f"artifact '{name}' present but not in integrity.json")

        artifact_index = manifest.get("artifact_index")
        if not isinstance(artifact_index, list):
            problems.append("manifest.artifact_index must be a list")
        else:
            for index, entry in enumerate(artifact_index):
                if not isinstance(entry, dict):
                    problems.append(
                        f"manifest artifact index {index} must be an object"
                    )
                    continue
                resolved = entry.get("resolved")
                if not isinstance(resolved, bool):
                    problems.append(
                        f"manifest artifact index {index} resolved must be boolean"
                    )
                    continue
                if not resolved:
                    continue
                arcname = entry.get("arcname")
                sha256 = entry.get("sha256")
                if not isinstance(arcname, str) or not _CANONICAL_ARTIFACT_RE.fullmatch(
                    arcname
                ):
                    problems.append(
                        f"manifest artifact index {index} has a non-canonical path"
                    )
                    continue
                if (
                    not isinstance(sha256, str)
                    or not _SHA256_RE.fullmatch(sha256)
                    or os.path.basename(arcname) != sha256
                ):
                    problems.append(
                        f"manifest artifact index {index} has an invalid SHA-256"
                    )
                if arcname not in listed_artifacts:
                    problems.append(
                        f"manifest artifact index {index} is absent from integrity.json"
                    )
                info = info_by_name.get(arcname)
                recorded_size = entry.get("size_bytes")
                if info is not None and (
                    not isinstance(recorded_size, int)
                    or isinstance(recorded_size, bool)
                    or recorded_size != info.file_size
                ):
                    problems.append(
                        f"manifest artifact index {index} size does not match "
                        "the ZIP entry"
                    )
        if declared_manifest_sha is None:
            problems.append(
                "integrity.json does not cover manifest.json; "
                "the session narrative is unprotected"
            )
        return problems

    def verify_bytes(self, bundle_bytes: bytes) -> Dict[str, Any]:
        """Verify a bundle without importing it."""
        problems: List[str] = []
        manifest: Any = None
        integrity: Any = None
        artifact_shas: List[str] = []
        problems.extend(self._central_directory_problems(bundle_bytes))
        if problems:
            return self._verification_result(
                bundle_bytes=len(bundle_bytes), problems=problems
            )
        try:
            with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
                metadata_problems, info_by_name = self._archive_metadata_problems(zf)
                problems.extend(metadata_problems)
                if not problems:
                    try:
                        manifest_payload = self._read_member_bytes(
                            zf,
                            info_by_name[MANIFEST_ARCNAME],
                            limit=self._max_manifest_bytes,
                        )
                        manifest = json.loads(manifest_payload)
                    except (ValueError, UnicodeDecodeError) as exc:
                        problems.append(f"manifest.json is not valid JSON: {exc}")
                    try:
                        integrity_payload = self._read_member_bytes(
                            zf,
                            info_by_name[INTEGRITY_ARCNAME],
                            limit=self._max_integrity_bytes,
                        )
                        integrity = json.loads(integrity_payload)
                    except (ValueError, UnicodeDecodeError) as exc:
                        problems.append(f"integrity.json is not valid JSON: {exc}")
                if not problems:
                    problems.extend(
                        self._control_document_problems(
                            manifest=manifest,
                            integrity=integrity,
                            info_by_name=info_by_name,
                        )
                    )
                if isinstance(integrity, dict):
                    for arcname, expected_sha in integrity.items():
                        if arcname == MANIFEST_ARCNAME:
                            if (
                                isinstance(expected_sha, str)
                                and _SHA256_RE.fullmatch(expected_sha)
                                and MANIFEST_ARCNAME in info_by_name
                            ):
                                info = info_by_name[MANIFEST_ARCNAME]
                                observed_sha = self._hash_member(
                                    zf, info, limit=self._max_manifest_bytes
                                )
                                if observed_sha != expected_sha:
                                    problems.append(
                                        f"manifest.json sha256 mismatch "
                                        f"(expected={expected_sha[:12]}, "
                                        f"observed={observed_sha[:12]})"
                                    )
                            continue
                        if (
                            not isinstance(arcname, str)
                            or not _CANONICAL_ARTIFACT_RE.fullmatch(arcname)
                            or not isinstance(expected_sha, str)
                            or not _SHA256_RE.fullmatch(expected_sha)
                            or arcname not in info_by_name
                        ):
                            continue
                        info = info_by_name[arcname]
                        observed_sha = self._hash_member(zf, info, limit=info.file_size)
                        if observed_sha != expected_sha:
                            problems.append(
                                f"{arcname} sha256 mismatch "
                                f"(expected={expected_sha[:12]}, "
                                f"observed={observed_sha[:12]})"
                            )
                        else:
                            artifact_shas.append(observed_sha)
        except (
            BundleError,
            EOFError,
            NotImplementedError,
            OSError,
            RuntimeError,
            zipfile.BadZipFile,
        ) as exc:
            problems.append(f"bundle is not a valid safe zip: {exc}")
        manifest_mapping = manifest if isinstance(manifest, dict) else None
        return self._verification_result(
            bundle_bytes=len(bundle_bytes),
            problems=problems,
            manifest=manifest_mapping,
            artifact_shas=artifact_shas,
        )

    def verify_file(self, path: str) -> Dict[str, Any]:
        absolute_path = os.path.abspath(path)
        file_size = os.path.getsize(path)
        if file_size > self._max_bundle_bytes:
            result = self._verification_result(
                bundle_bytes=file_size,
                problems=[
                    "bundle exceeds the compressed byte cap "
                    f"({file_size} > {self._max_bundle_bytes})"
                ],
            )
            result["source_path"] = absolute_path
            return result
        with open(path, "rb") as handle:
            opened_size = os.fstat(handle.fileno()).st_size
            if opened_size > self._max_bundle_bytes:
                result = self._verification_result(
                    bundle_bytes=opened_size,
                    problems=[
                        "bundle exceeds the compressed byte cap "
                        f"({opened_size} > {self._max_bundle_bytes})"
                    ],
                )
                result["source_path"] = absolute_path
                return result
            payload = handle.read(self._max_bundle_bytes + 1)
        if len(payload) > self._max_bundle_bytes:
            result = self._verification_result(
                bundle_bytes=len(payload),
                problems=[
                    "bundle exceeded the compressed byte cap while reading "
                    f"({len(payload)} > {self._max_bundle_bytes})"
                ],
            )
            result["source_path"] = absolute_path
            return result
        result = self.verify_bytes(payload)
        result["source_path"] = absolute_path
        return result

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    def _rehydrate_session(
        self, manifest: Mapping[str, Any]
    ) -> Tuple[ResearchSession, Dict[str, Dict[str, str]]]:
        session_meta = manifest.get("session") or {}
        if not isinstance(session_meta, dict):
            raise BundleError("manifest.session is not a dict")
        # Create a fresh session so the imported bundle never clobbers
        # an existing session id.  Record the origin in metadata.
        origin_id = session_meta.get("session_id")
        name = session_meta.get("name") or f"imported-{origin_id or 'session'}"
        origin_metadata = dict(session_meta.get("metadata") or {})
        origin_metadata["imported_from"] = origin_id
        new_session = self._registry.create_session(
            name=str(name),
            description=session_meta.get("description"),
            owning_debug_session=session_meta.get("owning_debug_session"),
            metadata=origin_metadata,
        )
        id_map: Dict[str, Dict[str, str]] = {
            "hypothesis": {},
            "note": {},
            "evidence": {},
            "snapshot": {},
            "session": {str(origin_id): new_session.session_id} if origin_id else {},
        }
        with new_session._lock:
            # Hypotheses first -- notes / evidence may reference them.
            for hyp_data in session_meta.get("hypotheses") or []:
                new_hyp = ResearchHypothesis(
                    hypothesis_id=uuid.uuid4().hex[:12],
                    statement=str(hyp_data.get("statement") or ""),
                    state=str(hyp_data.get("state") or "proposed"),
                    rationale=hyp_data.get("rationale"),
                    created_at=str(hyp_data.get("created_at") or ""),
                    updated_at=str(hyp_data.get("updated_at") or ""),
                )
                new_session.hypotheses[new_hyp.hypothesis_id] = new_hyp
                if hyp_data.get("hypothesis_id"):
                    id_map["hypothesis"][
                        hyp_data["hypothesis_id"]
                    ] = new_hyp.hypothesis_id
            for note_data in session_meta.get("notes") or []:
                new_note = ResearchNote(
                    note_id=uuid.uuid4().hex[:12],
                    body=str(note_data.get("body") or ""),
                    tags=[str(t) for t in note_data.get("tags") or []],
                    hypothesis_id=id_map["hypothesis"].get(
                        note_data.get("hypothesis_id") or ""
                    )
                    or None,
                    created_at=str(note_data.get("created_at") or ""),
                )
                new_session.notes[new_note.note_id] = new_note
                if note_data.get("note_id"):
                    id_map["note"][note_data["note_id"]] = new_note.note_id
            for evidence_data in session_meta.get("evidence") or []:
                new_ev = ResearchEvidence(
                    evidence_id=uuid.uuid4().hex[:12],
                    kind=str(evidence_data.get("kind") or "external"),
                    ref=str(evidence_data.get("ref") or ""),
                    description=evidence_data.get("description"),
                    hypothesis_id=id_map["hypothesis"].get(
                        evidence_data.get("hypothesis_id") or ""
                    )
                    or None,
                    note_id=id_map["note"].get(evidence_data.get("note_id") or "")
                    or None,
                    created_at=str(evidence_data.get("created_at") or ""),
                    metadata=dict(evidence_data.get("metadata") or {}),
                )
                new_session.evidence[new_ev.evidence_id] = new_ev
                if evidence_data.get("evidence_id"):
                    id_map["evidence"][
                        evidence_data["evidence_id"]
                    ] = new_ev.evidence_id
            for snap_data in session_meta.get("snapshots") or []:
                new_snap = ResearchSnapshot(
                    snapshot_id=uuid.uuid4().hex[:12],
                    name=str(snap_data.get("name") or "unnamed"),
                    slots=dict(snap_data.get("slots") or {}),
                    description=snap_data.get("description"),
                    created_at=str(snap_data.get("created_at") or ""),
                )
                new_session.snapshots[new_snap.snapshot_id] = new_snap
                if snap_data.get("snapshot_id"):
                    id_map["snapshot"][snap_data["snapshot_id"]] = new_snap.snapshot_id
            for event_data in session_meta.get("timeline") or []:
                new_event = TimelineEvent(
                    event_id=uuid.uuid4().hex[:12],
                    kind=str(event_data.get("kind") or "other"),
                    summary=str(event_data.get("summary") or "(imported)"),
                    created_at=str(event_data.get("created_at") or ""),
                    ref=event_data.get("ref"),
                    hypothesis_id=id_map["hypothesis"].get(
                        event_data.get("hypothesis_id") or ""
                    )
                    or None,
                    note_id=id_map["note"].get(event_data.get("note_id") or "") or None,
                    snapshot_id=id_map["snapshot"].get(
                        event_data.get("snapshot_id") or ""
                    )
                    or None,
                    metadata=dict(event_data.get("metadata") or {}),
                )
                new_session.timeline.append(new_event)
            new_session._touch()
        return new_session, id_map

    @staticmethod
    def _extract_member(
        zf: zipfile.ZipFile,
        info: zipfile.ZipInfo,
        *,
        destination: str,
    ) -> None:
        """Stream one already-verified member to an atomic destination."""
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=".pao-bundle-", suffix=".tmp", dir=parent
        )
        total = 0
        try:
            with os.fdopen(fd, "wb") as output, zf.open(info, "r") as source:
                while True:
                    remaining = info.file_size - total
                    chunk = source.read(min(_STREAM_CHUNK_BYTES, remaining + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > info.file_size:
                        raise BundleError(
                            f"bundle entry '{info.filename}' exceeded its "
                            f"decompression cap ({total} > {info.file_size})"
                        )
                    output.write(chunk)
            if total != info.file_size:
                raise BundleError(
                    f"bundle entry '{info.filename}' size changed while extracting "
                    f"({total} != {info.file_size})"
                )
            os.replace(temporary_path, destination)
        except Exception:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
            raise

    def import_bytes(
        self,
        bundle_bytes: bytes,
        *,
        artifact_destination_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        verify_result = self.verify_bytes(bundle_bytes)
        if verify_result["verdict"] != "valid":
            raise BundleError(
                "bundle failed verification: " + "; ".join(verify_result["problems"])
            )
        with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
            info_by_name = {info.filename: info for info in zf.infolist()}
            manifest = json.loads(
                self._read_member_bytes(
                    zf,
                    info_by_name[MANIFEST_ARCNAME],
                    limit=self._max_manifest_bytes,
                )
            )
            extracted_artifacts: List[Dict[str, Any]] = []
            if artifact_destination_dir:
                os.makedirs(artifact_destination_dir, exist_ok=True)
                artifact_names = sorted(
                    name
                    for name in info_by_name
                    if _CANONICAL_ARTIFACT_RE.fullmatch(name)
                )
                for arcname in artifact_names:
                    local_name = os.path.basename(arcname)
                    local_path = os.path.join(artifact_destination_dir, local_name)
                    self._extract_member(
                        zf, info_by_name[arcname], destination=local_path
                    )
                    extracted_artifacts.append(
                        {
                            "arcname": arcname,
                            "local_path": os.path.abspath(local_path),
                            "sha256": local_name,
                        }
                    )
        new_session, id_map = self._rehydrate_session(manifest)
        return {
            "schema": RESEARCH_IMPORT_SCHEMA,
            "session_id": new_session.session_id,
            "bundle_schema": manifest.get("bundle_schema"),
            "note_count": len(new_session.notes),
            "hypothesis_count": len(new_session.hypotheses),
            "evidence_count": len(new_session.evidence),
            "snapshot_count": len(new_session.snapshots),
            "timeline_length": len(new_session.timeline),
            "extracted_artifacts": extracted_artifacts,
            "id_map": id_map,
            "verify": verify_result,
        }

    def import_file(
        self, path: str, *, artifact_destination_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        file_size = os.path.getsize(path)
        if file_size > self._max_bundle_bytes:
            raise BundleError(
                "bundle exceeds the compressed byte cap "
                f"({file_size} > {self._max_bundle_bytes})"
            )
        with open(path, "rb") as handle:
            opened_size = os.fstat(handle.fileno()).st_size
            if opened_size > self._max_bundle_bytes:
                raise BundleError(
                    "bundle exceeds the compressed byte cap "
                    f"({opened_size} > {self._max_bundle_bytes})"
                )
            payload = handle.read(self._max_bundle_bytes + 1)
        if len(payload) > self._max_bundle_bytes:
            raise BundleError(
                "bundle exceeded the compressed byte cap while reading "
                f"({len(payload)} > {self._max_bundle_bytes})"
            )
        result = self.import_bytes(
            payload, artifact_destination_dir=artifact_destination_dir
        )
        result["source_path"] = os.path.abspath(path)
        return result

    # ------------------------------------------------------------------
    # High-level response wrappers used by the tool layer
    # ------------------------------------------------------------------
    def export_response(
        self,
        *,
        session_id: str,
        destination: str,
    ) -> Dict[str, Any]:
        summary = self.export_to_file(session_id=session_id, destination=destination)
        return build_backend_response(
            "export_research_session",
            parsed=summary,
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=summary.get("warnings") or None,
        )

    def import_response(
        self,
        *,
        source_path: str,
        artifact_destination_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.import_file(
            source_path, artifact_destination_dir=artifact_destination_dir
        )
        return build_backend_response(
            "import_research_session",
            parsed=result,
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )

    def verify_response(self, *, source_path: str) -> Dict[str, Any]:
        result = self.verify_file(source_path)
        return build_backend_response(
            "verify_research_bundle",
            parsed=result,
            raw_responses=[],
            state="MANAGED",
            parser_version=PARSER_VERSION,
            warnings=None,
        )


__all__ = [
    "BundleError",
    "ResearchBundle",
    "BUNDLE_SCHEMA_VERSION",
    "MANIFEST_ARCNAME",
    "INTEGRITY_ARCNAME",
    "ARTIFACTS_DIRNAME",
    "DEFAULT_MAX_BUNDLE_BYTES",
    "DEFAULT_MAX_BUNDLE_ENTRIES",
    "DEFAULT_MAX_MANIFEST_BYTES",
    "DEFAULT_MAX_INTEGRITY_BYTES",
    "DEFAULT_MAX_COMPRESSION_RATIO",
    "DEFAULT_MAX_ENTRY_NAME_BYTES",
    "RESEARCH_EXPORT_SCHEMA",
    "RESEARCH_IMPORT_SCHEMA",
    "RESEARCH_VERIFY_SCHEMA",
]
