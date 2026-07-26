"""Typed research bundle export / import / verify (Stage 9.4).

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
    integrity.json     -- SHA-256 over every payload in ``artifacts/``,
                          keyed by ``arcname``.  Signed only by the
                          hash function; a caller who needs a real
                          signature layers that on top of the bundle.
    artifacts/<sha256>  -- one file per referenced artifact,
                          content-addressed by SHA-256 so identical
                          payloads are stored once even across
                          different sessions.

``export_research_session`` follows every ``kind == "artifact"``
evidence entry and copies the underlying blob out of
:class:`ArtifactStore` into the bundle.  Crash / heap / tool_result /
external / note references stay as opaque strings in the manifest --
they name state that lives in stores the recipient may or may not
have; the bundle records the reference so the eventual S9.5 report
can still cite them, but does not duplicate their bytes here.

``verify_research_bundle`` checks:

  1. The zip opens cleanly and carries both ``manifest.json`` and
     ``integrity.json``.
  2. Every artifact listed in the manifest / integrity file exists
     in ``artifacts/`` and its SHA-256 matches the recorded value.
  3. No files outside the documented layout are present (defensive
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
    ) -> None:
        if max_bundle_bytes <= 0:
            raise ValueError("max_bundle_bytes must be positive")
        if max_bundle_entries <= 0:
            raise ValueError("max_bundle_entries must be positive")
        self._registry = registry
        self._artifact_store = artifact_store
        self._max_bundle_bytes = max_bundle_bytes
        self._max_bundle_entries = max_bundle_entries

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
            data_path = self._artifact_store._data_path(evidence.ref)
            with open(data_path, "rb") as handle:
                payload = handle.read()
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
        integrity: Dict[str, str] = {
            f"{ARTIFACTS_DIRNAME}/{sha}": sha for sha in blob_by_sha
        }
        integrity_bytes = json.dumps(integrity, sort_keys=True, indent=2).encode(
            "utf-8"
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
        total_entries = 2 + len(blob_by_sha)
        if total_entries > self._max_bundle_entries:
            raise BundleError(
                f"research bundle exceeds the entry cap "
                f"({total_entries} > {self._max_bundle_entries})"
            )
        buffer = io.BytesIO()
        with zipfile.ZipFile(
            buffer, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as zf:
            zf.writestr(MANIFEST_ARCNAME, manifest_bytes)
            zf.writestr(INTEGRITY_ARCNAME, integrity_bytes)
            for sha, payload in blob_by_sha.items():
                zf.writestr(f"{ARTIFACTS_DIRNAME}/{sha}", payload)
        bundle_bytes = buffer.getvalue()
        summary: Dict[str, Any] = {
            "schema": RESEARCH_EXPORT_SCHEMA,
            "session_id": session_id,
            "bundle_schema": BUNDLE_SCHEMA_VERSION,
            "bundle_bytes": len(bundle_bytes),
            "sha256": _sha256_bytes(bundle_bytes),
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

    def verify_bytes(self, bundle_bytes: bytes) -> Dict[str, Any]:
        """Verify a bundle without importing it."""
        problems: List[str] = []
        manifest: Optional[Dict[str, Any]] = None
        integrity: Optional[Dict[str, str]] = None
        artifact_shas: List[str] = []
        try:
            with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zf:
                names = set(zf.namelist())
                if MANIFEST_ARCNAME not in names:
                    problems.append("missing manifest.json")
                if INTEGRITY_ARCNAME not in names:
                    problems.append("missing integrity.json")
                # Detect unexpected top-level entries.  ``artifacts/``
                # sub-entries are allowed; every other stray file is
                # not.  This is defensive against accidental leaks.
                for name in names:
                    if name in self._KNOWN_TOPLEVEL:
                        continue
                    if name.startswith(f"{ARTIFACTS_DIRNAME}/"):
                        continue
                    problems.append(f"unexpected bundle entry '{name}'")
                if MANIFEST_ARCNAME in names:
                    try:
                        manifest = json.loads(zf.read(MANIFEST_ARCNAME))
                    except json.JSONDecodeError as exc:
                        problems.append(f"manifest.json is not valid JSON: {exc}")
                if INTEGRITY_ARCNAME in names:
                    try:
                        integrity = json.loads(zf.read(INTEGRITY_ARCNAME))
                    except json.JSONDecodeError as exc:
                        problems.append(f"integrity.json is not valid JSON: {exc}")
                if integrity is not None and isinstance(integrity, dict):
                    for arcname, expected_sha in integrity.items():
                        if arcname not in names:
                            problems.append(
                                f"integrity references missing entry '{arcname}'"
                            )
                            continue
                        payload = zf.read(arcname)
                        observed_sha = _sha256_bytes(payload)
                        if observed_sha != expected_sha:
                            problems.append(
                                f"{arcname} sha256 mismatch "
                                f"(expected={expected_sha[:12]}, "
                                f"observed={observed_sha[:12]})"
                            )
                        else:
                            artifact_shas.append(observed_sha)
                    # Every artifact/* entry must be listed in integrity
                    for name in names:
                        if not name.startswith(f"{ARTIFACTS_DIRNAME}/"):
                            continue
                        if name not in integrity:
                            problems.append(
                                f"artifact '{name}' present but not in integrity.json"
                            )
        except zipfile.BadZipFile as exc:
            problems.append(f"bundle is not a valid zip: {exc}")
        session_meta = None
        if manifest is not None and isinstance(manifest.get("session"), dict):
            session_meta = {
                "name": manifest["session"].get("name"),
                "session_id": manifest["session"].get("session_id"),
                "note_count": manifest["session"].get("note_count"),
                "hypothesis_count": manifest["session"].get("hypothesis_count"),
                "evidence_count": manifest["session"].get("evidence_count"),
                "snapshot_count": manifest["session"].get("snapshot_count"),
                "timeline_length": manifest["session"].get("timeline_length"),
            }
        verdict = "valid" if not problems else "invalid"
        return {
            "schema": RESEARCH_VERIFY_SCHEMA,
            "verdict": verdict,
            "bundle_bytes": len(bundle_bytes),
            "bundle_schema": (manifest.get("bundle_schema") if manifest else None),
            "artifact_count": len(artifact_shas),
            "problems": problems,
            "session": session_meta,
        }

    def verify_file(self, path: str) -> Dict[str, Any]:
        with open(path, "rb") as handle:
            payload = handle.read()
        result = self.verify_bytes(payload)
        result["source_path"] = os.path.abspath(path)
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
            manifest = json.loads(zf.read(MANIFEST_ARCNAME))
            extracted_artifacts: List[Dict[str, Any]] = []
            if artifact_destination_dir:
                os.makedirs(artifact_destination_dir, exist_ok=True)
                for arcname in zf.namelist():
                    if not arcname.startswith(f"{ARTIFACTS_DIRNAME}/"):
                        continue
                    local_name = os.path.basename(arcname)
                    local_path = os.path.join(artifact_destination_dir, local_name)
                    with open(local_path, "wb") as handle:
                        handle.write(zf.read(arcname))
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
        with open(path, "rb") as handle:
            payload = handle.read()
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
    "RESEARCH_EXPORT_SCHEMA",
    "RESEARCH_IMPORT_SCHEMA",
    "RESEARCH_VERIFY_SCHEMA",
]
