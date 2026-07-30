"""Tests for Stage 9.4 research bundle export / import / verify."""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

from paomcp.tools.backends.research import ResearchRegistry
from paomcp.tools.backends.research_bundle import (
    ARTIFACTS_DIRNAME,
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    INTEGRITY_ARCNAME,
    MANIFEST_ARCNAME,
    RESEARCH_EXPORT_SCHEMA,
    RESEARCH_IMPORT_SCHEMA,
    RESEARCH_VERIFY_SCHEMA,
    ResearchBundle,
)


class _FakeMetadata:
    def __init__(self, size: int, sha: str) -> None:
        self.size_bytes = size
        self.sha256 = sha
        self.content_type = "application/octet-stream"


class _FakeArtifactStore:
    """In-memory artifact store mirroring the tiny subset ResearchBundle uses.

    Only implements ``get(artifact_id)`` and the private ``_data_path``
    helper; anything else in :class:`ArtifactStore` is out of scope
    here.
    """

    def __init__(self, tmp_root: Path) -> None:
        self._root = tmp_root
        self._payloads: dict = {}
        self._sha: dict = {}

    def add(self, artifact_id: str, payload: bytes) -> None:
        import hashlib

        path = self._root / f"{artifact_id}.dat"
        path.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()
        self._payloads[artifact_id] = payload
        self._sha[artifact_id] = sha

    def get(self, artifact_id: str) -> _FakeMetadata:
        if artifact_id not in self._sha:
            raise FileNotFoundError(artifact_id)
        return _FakeMetadata(
            size=len(self._payloads[artifact_id]),
            sha=self._sha[artifact_id],
        )

    def _data_path(self, artifact_id: str) -> str:
        return str(self._root / f"{artifact_id}.dat")


def _seeded_session(registry: ResearchRegistry) -> str:
    sid = registry.create_session_response(name="test-bundle")["parsed"]["session"][
        "session_id"
    ]
    hyp = registry.add_hypothesis_response(session_id=sid, statement="stack overflow")[
        "parsed"
    ]["hypothesis"]
    registry.add_note_response(
        session_id=sid,
        body="leaked puts",
        hypothesis_id=hyp["hypothesis_id"],
    )
    registry.create_snapshot_response(
        session_id=sid,
        name="pre",
        slots={"registers": {"rip": 0x400100}},
    )
    registry.append_timeline_event_response(
        session_id=sid, kind="reproduction", summary="repro ok"
    )
    return sid


def _minimal_manifest() -> dict[str, Any]:
    return {
        "bundle_schema": BUNDLE_SCHEMA_VERSION,
        "parser_version": "1",
        "artifact_index": [],
        "session": {
            "session_id": "source-session",
            "name": "import-control",
            "description": None,
            "owning_debug_session": None,
            "metadata": {},
            "hypotheses": [],
            "notes": [],
            "evidence": [],
            "snapshots": [],
            "timeline": [],
        },
    }


def _build_bundle(
    manifest: dict[str, Any],
    *,
    integrity: Any = None,
    integrity_payload: bytes | None = None,
    artifacts: dict[str, bytes] | None = None,
    extra_entries: dict[str, bytes] | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_ARCNAME, json.dumps(manifest))
        if integrity_payload is None:
            zf.writestr(
                INTEGRITY_ARCNAME,
                json.dumps({} if integrity is None else integrity),
            )
        else:
            zf.writestr(INTEGRITY_ARCNAME, integrity_payload)
        for name, payload in (artifacts or {}).items():
            zf.writestr(name, payload)
        for name, payload in (extra_entries or {}).items():
            zf.writestr(name, payload)
    return buffer.getvalue()


class TestExport:
    def test_export_writes_zip_with_manifest_and_integrity(
        self, tmp_path: Path
    ) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry)
        destination = tmp_path / "bundle.zip"
        summary = bundle.export_to_file(session_id=sid, destination=str(destination))
        assert summary["schema"] == RESEARCH_EXPORT_SCHEMA
        assert destination.exists()
        with zipfile.ZipFile(destination) as zf:
            names = zf.namelist()
            assert MANIFEST_ARCNAME in names
            assert INTEGRITY_ARCNAME in names
            manifest = json.loads(zf.read(MANIFEST_ARCNAME))
            assert manifest["bundle_schema"] == BUNDLE_SCHEMA_VERSION
            assert manifest["session"]["name"] == "test-bundle"
            # snapshots must carry full slot payloads in the bundle
            snap = manifest["session"]["snapshots"][0]
            assert "slots" in snap
            assert snap["slots"]["registers"]["rip"] == 0x400100
            # timeline is preserved
            assert manifest["session"]["timeline"]

    def test_export_bytes_returns_matching_sha(self) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry)
        data, summary = bundle.export_to_bytes(session_id=sid)
        import hashlib

        assert summary["sha256"] == hashlib.sha256(data).hexdigest()

    def test_export_includes_artifact_payload(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        store = _FakeArtifactStore(tmp_path / "store")
        os.makedirs(tmp_path / "store", exist_ok=True)
        store.add("art-1", b"HELLO PAO EXPLOIT")
        sid = _seeded_session(registry)
        registry.link_evidence_response(
            session_id=sid,
            kind="artifact",
            ref="art-1",
            description="capture",
        )
        bundle = ResearchBundle(registry, artifact_store=store)
        destination = tmp_path / "bundle.zip"
        summary = bundle.export_to_file(session_id=sid, destination=str(destination))
        assert summary["artifact_count"] == 1
        with zipfile.ZipFile(destination) as zf:
            names = zf.namelist()
            artifact_entries = [
                n for n in names if n.startswith(f"{ARTIFACTS_DIRNAME}/")
            ]
            assert len(artifact_entries) == 1
            # content-addressed by sha256 of the payload
            import hashlib

            expected_sha = hashlib.sha256(b"HELLO PAO EXPLOIT").hexdigest()
            assert artifact_entries[0].endswith(expected_sha)

    def test_export_flags_missing_artifact_as_warning(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        store = _FakeArtifactStore(tmp_path / "store")
        os.makedirs(tmp_path / "store", exist_ok=True)
        sid = _seeded_session(registry)
        registry.link_evidence_response(
            session_id=sid, kind="artifact", ref="does-not-exist"
        )
        bundle = ResearchBundle(registry, artifact_store=store)
        _, summary = bundle.export_to_bytes(session_id=sid)
        assert summary["artifact_count"] == 0
        assert any("does-not-exist" in w for w in summary["warnings"])

    def test_export_without_store_records_unresolved_artifact(self) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        registry.link_evidence_response(session_id=sid, kind="artifact", ref="orphan")
        bundle = ResearchBundle(registry)  # no store wired
        _, summary = bundle.export_to_bytes(session_id=sid)
        assert summary["artifact_count"] == 0
        assert any("orphan" in w for w in summary["warnings"])

    def test_repetitive_legitimate_export_remains_verifiable(self) -> None:
        registry = ResearchRegistry()
        sid = registry.create_session_response(name="compressible")["parsed"][
            "session"
        ]["session_id"]
        registry.add_note_response(session_id=sid, body="A" * 100_000)
        bundle = ResearchBundle(registry)
        data, _ = bundle.export_to_bytes(session_id=sid)
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "valid"

    def test_byte_cap_enforced(self) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry, max_bundle_bytes=32)
        with pytest.raises(BundleError, match="byte cap"):
            bundle.export_to_bytes(session_id=sid)


class TestVerify:
    def test_valid_bundle_is_valid(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry)
        data, _ = bundle.export_to_bytes(session_id=sid)
        result = bundle.verify_bytes(data)
        assert result["schema"] == RESEARCH_VERIFY_SCHEMA
        assert result["verdict"] == "valid"
        assert result["problems"] == []
        assert result["bundle_schema"] == BUNDLE_SCHEMA_VERSION
        assert result["session"]["name"] == "test-bundle"

    def test_bad_zip_reports_invalid(self) -> None:
        registry = ResearchRegistry()
        bundle = ResearchBundle(registry)
        result = bundle.verify_bytes(b"not a zip file")
        assert result["verdict"] == "invalid"
        assert any("not a valid zip" in p for p in result["problems"])

    def test_missing_manifest_is_invalid(self) -> None:
        registry = ResearchRegistry()
        bundle = ResearchBundle(registry)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(INTEGRITY_ARCNAME, "{}")
        result = bundle.verify_bytes(buf.getvalue())
        assert result["verdict"] == "invalid"
        assert any("missing manifest.json" in p for p in result["problems"])

    def test_stray_entry_reported(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry)
        data, _ = bundle.export_to_bytes(session_id=sid)
        # Rebuild the zip with a stray file to prove the verifier catches it.
        modified = io.BytesIO()
        with (
            zipfile.ZipFile(io.BytesIO(data)) as src,
            zipfile.ZipFile(modified, "w") as dst,
        ):
            for name in src.namelist():
                dst.writestr(name, src.read(name))
            dst.writestr("uninvited/leak.txt", "should not be here")
        result = bundle.verify_bytes(modified.getvalue())
        assert result["verdict"] == "invalid"
        assert any("uninvited/leak.txt" in p for p in result["problems"])

    def test_sha_mismatch_flagged(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        store = _FakeArtifactStore(tmp_path / "store")
        os.makedirs(tmp_path / "store", exist_ok=True)
        store.add("art-1", b"legit")
        sid = _seeded_session(registry)
        registry.link_evidence_response(session_id=sid, kind="artifact", ref="art-1")
        bundle = ResearchBundle(registry, artifact_store=store)
        data, _ = bundle.export_to_bytes(session_id=sid)
        # Rewrite the artifact payload but keep the integrity file
        # unchanged so the observed sha diverges from the expected sha.
        buf = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(buf, "w") as dst:
            for name in src.namelist():
                payload = src.read(name)
                if name.startswith(f"{ARTIFACTS_DIRNAME}/"):
                    payload = b"TAMPERED"
                dst.writestr(name, payload)
        result = bundle.verify_bytes(buf.getvalue())
        assert result["verdict"] == "invalid"
        assert any("sha256 mismatch" in p for p in result["problems"])

    def test_uncompressed_zip_bomb_is_rejected_before_json_parse(self) -> None:
        manifest = _minimal_manifest()
        manifest["session"]["description"] = "A" * 20_000
        data = _build_bundle(manifest)
        assert len(data) < 1024
        bundle = ResearchBundle(
            ResearchRegistry(),
            max_bundle_bytes=1024,
            max_compression_ratio=10_000,
        )
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert any("uncompressed byte cap" in problem for problem in result["problems"])

    def test_high_compression_ratio_is_rejected(self) -> None:
        manifest = _minimal_manifest()
        manifest["session"]["description"] = "A" * 20_000
        data = _build_bundle(manifest)
        bundle = ResearchBundle(
            ResearchRegistry(),
            max_bundle_bytes=100_000,
            max_uncompressed_bytes=100_000,
            max_compression_ratio=5,
        )
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert any("compression-ratio cap" in problem for problem in result["problems"])

    def test_entry_count_cap_is_checked_from_central_directory(self) -> None:
        data = _build_bundle(
            _minimal_manifest(),
            extra_entries={"unexpected.txt": b""},
        )
        bundle = ResearchBundle(ResearchRegistry(), max_bundle_entries=2)
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert result["problems"] == ["bundle exceeds the entry cap (3 > 2)"]

    def test_oversized_manifest_is_rejected_before_decompression(self) -> None:
        data = _build_bundle(_minimal_manifest())
        bundle = ResearchBundle(ResearchRegistry(), max_manifest_bytes=64)
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert any("manifest.json exceeds" in problem for problem in result["problems"])

    def test_oversized_integrity_index_is_rejected_before_decompression(self) -> None:
        data = _build_bundle(
            _minimal_manifest(),
            integrity_payload=b"{}" + (b" " * 128),
        )
        bundle = ResearchBundle(ResearchRegistry(), max_integrity_bytes=64)
        result = bundle.verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert any(
            "integrity.json exceeds" in problem for problem in result["problems"]
        )

    def test_json_numeric_limit_error_is_returned_as_typed_invalid(self) -> None:
        manifest_payload = json.dumps(_minimal_manifest()).replace(
            '"metadata": {}',
            f'"metadata": {{"huge": {"9" * 5000}}}',
            1,
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_ARCNAME, manifest_payload)
            zf.writestr(INTEGRITY_ARCNAME, "{}")
        result = ResearchBundle(
            ResearchRegistry(), max_compression_ratio=10_000
        ).verify_bytes(buffer.getvalue())
        assert result["verdict"] == "invalid"
        assert any("manifest.json is not valid JSON" in p for p in result["problems"])

    def test_noncanonical_artifact_path_is_rejected(self) -> None:
        payload = b"safe bytes"
        sha256 = hashlib.sha256(payload).hexdigest()
        arcname = "artifacts/../not-content-addressed"
        manifest = _minimal_manifest()
        manifest["artifact_index"] = [
            {
                "resolved": True,
                "arcname": arcname,
                "sha256": sha256,
                "size_bytes": len(payload),
            }
        ]
        data = _build_bundle(
            manifest,
            integrity={arcname: sha256},
            artifacts={arcname: payload},
        )
        result = ResearchBundle(ResearchRegistry()).verify_bytes(data)
        assert result["verdict"] == "invalid"
        assert any("canonical" in problem for problem in result["problems"])

    def test_file_size_cap_prevents_verify_and_import_read(
        self, tmp_path: Path
    ) -> None:
        source = tmp_path / "oversized.zip"
        source.write_bytes(b"x" * 128)
        bundle = ResearchBundle(ResearchRegistry(), max_bundle_bytes=64)
        result = bundle.verify_file(str(source))
        assert result["verdict"] == "invalid"
        assert result["bundle_bytes"] == 128
        assert any("compressed byte cap" in problem for problem in result["problems"])
        with pytest.raises(BundleError, match="compressed byte cap"):
            bundle.import_file(str(source))


class TestImport:
    def test_import_creates_new_session_with_remapped_ids(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        bundle = ResearchBundle(registry)
        data, _ = bundle.export_to_bytes(session_id=sid)
        result = bundle.import_bytes(data)
        assert result["schema"] == RESEARCH_IMPORT_SCHEMA
        assert result["session_id"] != sid
        imported = registry.get_session(result["session_id"])
        # id remap: the imported note.hypothesis_id must point at the
        # imported hypothesis, not the original one
        note = next(iter(imported.notes.values()))
        hyp = next(iter(imported.hypotheses.values()))
        assert note.hypothesis_id == hyp.hypothesis_id
        assert note.hypothesis_id != _first_hypothesis_id(registry.get_session(sid))
        assert imported.metadata.get("imported_from") == sid

    def test_import_extracts_artifacts_when_dir_provided(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        store = _FakeArtifactStore(tmp_path / "store")
        os.makedirs(tmp_path / "store", exist_ok=True)
        store.add("art-1", b"blob-1")
        sid = _seeded_session(registry)
        registry.link_evidence_response(session_id=sid, kind="artifact", ref="art-1")
        bundle = ResearchBundle(registry, artifact_store=store)
        data, _ = bundle.export_to_bytes(session_id=sid)
        destination = tmp_path / "extracted"
        result = bundle.import_bytes(data, artifact_destination_dir=str(destination))
        extracted = result["extracted_artifacts"]
        assert len(extracted) == 1
        local_path = Path(extracted[0]["local_path"])
        assert local_path.exists()
        assert local_path.read_bytes() == b"blob-1"

    def test_invalid_bundle_refuses_import(self) -> None:
        registry = ResearchRegistry()
        bundle = ResearchBundle(registry)
        with pytest.raises(BundleError, match="failed verification"):
            bundle.import_bytes(b"not a zip")

    def test_round_trip_preserves_timeline_and_snapshots(self, tmp_path: Path) -> None:
        registry = ResearchRegistry()
        sid = _seeded_session(registry)
        original = registry.get_session(sid)
        bundle = ResearchBundle(registry)
        data, _ = bundle.export_to_bytes(session_id=sid)
        result = bundle.import_bytes(data)
        imported = registry.get_session(result["session_id"])
        assert len(imported.timeline) == len(original.timeline)
        assert len(imported.snapshots) == len(original.snapshots)
        # slot payload survives the round trip
        snap = next(iter(imported.snapshots.values()))
        assert snap.slots["registers"]["rip"] == 0x400100

    @pytest.mark.parametrize(
        ("child_name", "registry_limit", "child"),
        [
            (
                "notes",
                {"max_notes_per_session": 1},
                {"note_id": "note", "body": "bounded", "tags": []},
            ),
            (
                "hypotheses",
                {"max_hypotheses_per_session": 1},
                {
                    "hypothesis_id": "hypothesis",
                    "statement": "bounded",
                    "state": "proposed",
                },
            ),
            (
                "evidence",
                {"max_evidence_per_session": 1},
                {
                    "evidence_id": "evidence",
                    "kind": "external",
                    "ref": "bounded",
                    "metadata": {},
                },
            ),
            (
                "snapshots",
                {"max_snapshots_per_session": 1},
                {"snapshot_id": "snapshot", "name": "bounded", "slots": {}},
            ),
            (
                "timeline",
                {"max_timeline_events_per_session": 1},
                {
                    "event_id": "event",
                    "kind": "other",
                    "summary": "bounded",
                    "metadata": {},
                },
            ),
        ],
    )
    def test_import_cannot_bypass_child_record_caps(
        self,
        child_name: str,
        registry_limit: dict[str, int],
        child: dict[str, Any],
    ) -> None:
        manifest = _minimal_manifest()
        manifest["session"][child_name] = [dict(child), dict(child)]
        data = _build_bundle(manifest)
        registry = ResearchRegistry(**registry_limit)
        bundle = ResearchBundle(registry)

        verify = bundle.verify_bytes(data)
        assert verify["verdict"] == "invalid"
        assert any(
            f"manifest.session.{child_name} exceeds the registry cap" in problem
            for problem in verify["problems"]
        )
        with pytest.raises(BundleError, match="registry cap"):
            bundle.import_bytes(data)
        assert registry.list_sessions() == []


def _first_hypothesis_id(session) -> str:
    return next(iter(session.hypotheses.values())).hypothesis_id
