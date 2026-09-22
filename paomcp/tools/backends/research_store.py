"""SQLite-backed persistence for research cases.

``ResearchRegistry`` is in-memory by design (bounded, lock-protected);
:class:`CaseStore` gives it durability across restarts: every mutation
persists its session in one transaction, and :meth:`load_all` rehydrates
the registry at startup. Research payloads are small next to artifacts,
so whole-session delete+insert per mutation is cheap and always
consistent. Failures never break the calling operation -- persistence is
best-effort and loudly logged.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cases (
    session_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    owning_debug_session TEXT,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    notes_json TEXT NOT NULL DEFAULT '[]',
    evidence_json TEXT NOT NULL DEFAULT '[]',
    hypotheses_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS case_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS case_timeline (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_case_snapshots_session ON case_snapshots(session_id);
CREATE INDEX IF NOT EXISTS idx_case_timeline_session ON case_timeline(session_id, seq);
"""


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


class CaseStore:
    """Durable store for research sessions under one sqlite file."""

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(path)
        self._lock = threading.RLock()
        self._conn = _connect(self.path)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            elif row[0] != str(_SCHEMA_VERSION):
                raise RuntimeError(
                    f"Unsupported case-store schema version {row[0]!r} "
                    f"(expected {_SCHEMA_VERSION})"
                )
            self._conn.commit()

    def close(self) -> None:
        """Close the sqlite handle (idempotent, never raises)."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001 - shutdown path
                logger.debug("CaseStore close failed", exc_info=True)

    def save_session(self, session: Any) -> None:
        """Persist one live session atomically (children included)."""
        notes = [n.to_dict() for n in session.notes.values()]
        evidence = [e.to_dict() for e in session.evidence.values()]
        hypotheses = [h.to_dict() for h in session.hypotheses.values()]
        snapshots = {
            sid: snap.to_dict(include_slots=True)
            for sid, snap in session.snapshots.items()
        }
        timeline = [event.to_dict() for event in session.timeline]
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO cases("
                "session_id, name, owning_debug_session, description, "
                "created_at, updated_at, metadata_json, notes_json, "
                "evidence_json, hypotheses_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    session.session_id,
                    session.name,
                    session.owning_debug_session,
                    session.description,
                    session.created_at,
                    session.updated_at,
                    json.dumps(dict(session.metadata)),
                    json.dumps(notes),
                    json.dumps(evidence),
                    json.dumps(hypotheses),
                ),
            )
            cur.execute(
                "DELETE FROM case_snapshots WHERE session_id=?", (session.session_id,)
            )
            cur.executemany(
                "INSERT INTO case_snapshots(snapshot_id, session_id, payload_json)"
                " VALUES(?,?,?)",
                [
                    (sid, session.session_id, json.dumps(payload))
                    for sid, payload in snapshots.items()
                ],
            )
            cur.execute(
                "DELETE FROM case_timeline WHERE session_id=?", (session.session_id,)
            )
            cur.executemany(
                "INSERT INTO case_timeline(event_id, session_id, seq, payload_json)"
                " VALUES(?,?,?,?)",
                [
                    (event["event_id"], session.session_id, seq, json.dumps(event))
                    for seq, event in enumerate(timeline)
                ],
            )
            self._conn.commit()

    def delete_session(self, session_id: str) -> None:
        """Remove one session and its children (missing ids are a no-op)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM case_timeline WHERE session_id=?", (session_id,))
            cur.execute("DELETE FROM case_snapshots WHERE session_id=?", (session_id,))
            cur.execute("DELETE FROM cases WHERE session_id=?", (session_id,))
            self._conn.commit()

    def load_all(self) -> List[Dict[str, Any]]:
        """Load every stored session as rehydration dicts (no backend types).

        Rows come back newest-first (``updated_at`` descending, with
        ``session_id`` as the deterministic tie-breaker) so a registry
        capped below the stored count rehydrates the most recent
        investigations and drops the oldest.
        """
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT session_id, name, owning_debug_session, description,"
                " created_at, updated_at, metadata_json, notes_json,"
                " evidence_json, hypotheses_json FROM cases"
                " ORDER BY updated_at DESC, session_id DESC"
            ).fetchall()
            result = []
            for (
                session_id,
                name,
                owning_debug_session,
                description,
                created_at,
                updated_at,
                metadata_json,
                notes_json,
                evidence_json,
                hypotheses_json,
            ) in rows:
                snapshots = {
                    snapshot_id: json.loads(payload_json)
                    for snapshot_id, payload_json in cur.execute(
                        "SELECT snapshot_id, payload_json FROM case_snapshots"
                        " WHERE session_id=?",
                        (session_id,),
                    ).fetchall()
                }
                timeline = [
                    json.loads(payload_json)
                    for (payload_json,) in cur.execute(
                        "SELECT payload_json FROM case_timeline"
                        " WHERE session_id=? ORDER BY seq",
                        (session_id,),
                    ).fetchall()
                ]
                result.append(
                    {
                        "session_id": session_id,
                        "name": name,
                        "owning_debug_session": owning_debug_session,
                        "description": description,
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "metadata": json.loads(metadata_json),
                        "notes": json.loads(notes_json),
                        "evidence": json.loads(evidence_json),
                        "hypotheses": json.loads(hypotheses_json),
                        "snapshots": snapshots,
                        "timeline": timeline,
                    }
                )
            return result

    def session_ids(self) -> List[str]:
        """List stored session ids (cheap probe for tests/ops)."""
        with self._lock:
            return [
                row[0]
                for row in self._conn.execute(
                    "SELECT session_id FROM cases"
                    " ORDER BY updated_at DESC, session_id DESC"
                ).fetchall()
            ]


__all__ = ["CaseStore"]
