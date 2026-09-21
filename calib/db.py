"""SQLite persistence: versions, visibilities, flags and run snapshots."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    version TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    data_hash TEXT NOT NULL,
    meta_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS antennas (
    version TEXT NOT NULL,
    name TEXT NOT NULL,
    label TEXT NOT NULL,
    PRIMARY KEY (version, name)
);

CREATE TABLE IF NOT EXISTS visibilities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    antenna_a TEXT NOT NULL,
    antenna_b TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    channel INTEGER NOT NULL,
    re REAL NOT NULL,
    im REAL NOT NULL,
    model_re REAL NOT NULL,
    model_im REAL NOT NULL,
    quality INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_vis_lookup
    ON visibilities (version, antenna_a, antenna_b, t_idx, channel);

CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('manual', 'auto')),
    status TEXT NOT NULL DEFAULT 'applied'
        CHECK (status IN ('applied', 'suggested')),
    reason TEXT NOT NULL,
    antenna_a TEXT,
    antenna_b TEXT,
    t_start INTEGER,
    t_end INTEGER,
    ch_start INTEGER,
    ch_end INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    ref_antenna TEXT NOT NULL,
    t_start INTEGER NOT NULL,
    t_end INTEGER NOT NULL,
    group_size INTEGER NOT NULL,
    params_hash TEXT NOT NULL,
    flagset_hash TEXT NOT NULL,
    data_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_flags (
    run_id INTEGER NOT NULL,
    flag_id INTEGER NOT NULL,
    source TEXT NOT NULL,
    reason TEXT NOT NULL,
    antenna_a TEXT,
    antenna_b TEXT,
    t_start INTEGER,
    t_end INTEGER,
    ch_start INTEGER,
    ch_end INTEGER,
    PRIMARY KEY (run_id, flag_id)
);

CREATE TABLE IF NOT EXISTS run_gains (
    run_id INTEGER NOT NULL,
    group_index INTEGER NOT NULL,
    antenna TEXT NOT NULL,
    gain_re REAL,
    gain_im REAL,
    solvable INTEGER NOT NULL,
    PRIMARY KEY (run_id, group_index, antenna)
);

CREATE TABLE IF NOT EXISTS run_residuals (
    run_id INTEGER NOT NULL,
    t_idx INTEGER NOT NULL,
    channel INTEGER NOT NULL,
    antenna_a TEXT NOT NULL,
    antenna_b TEXT NOT NULL,
    flagged INTEGER NOT NULL,
    residual_re REAL,
    residual_im REAL,
    phase_residual REAL,
    amp_residual REAL,
    PRIMARY KEY (run_id, antenna_a, antenna_b, t_idx, channel)
);

CREATE TABLE IF NOT EXISTS run_closures (
    run_id INTEGER NOT NULL,
    triplet TEXT NOT NULL,
    t_idx INTEGER NOT NULL,
    group_index INTEGER NOT NULL,
    closure_raw REAL,
    PRIMARY KEY (run_id, triplet, t_idx, group_index)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Content hash of the canonical fixture payload (visibilities + meta)."""
    body = {
        "meta": payload["meta"],
        "antennas": payload["antennas"],
        "visibilities": sorted(
            payload["visibilities"],
            key=lambda r: (r["antenna_a"], r["antenna_b"], r["t_idx"], r["channel"]),
        ),
    }
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def flags_hash(flags: list) -> str:
    """Hash of the flag set a solve depends on (applied flags only)."""
    tuples = []
    for row in flags:
        def get(key: str, _row=row):
            return _row[key]
        tuples.append(
            (
                get("source"),
                get("antenna_a"),
                get("antenna_b"),
                get("t_start"),
                get("t_end"),
                get("ch_start"),
                get("ch_end"),
            )
        )
    encoded = json.dumps(
        sorted(tuples, key=lambda x: json.dumps(x, ensure_ascii=False)),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def import_payload(conn: sqlite3.Connection, payload: dict[str, Any]) -> str:
    """Import a fixture payload; returns the version string.

    Idempotent: the same version is re-used and rows are not duplicated.
    """
    version = payload["meta"]["version"]
    digest = canonical_payload_hash(payload)
    init_db(conn)
    existing = conn.execute(
        "SELECT data_hash FROM versions WHERE version = ?", (version,)
    ).fetchone()
    if existing is not None:
        return version

    conn.execute(
        "INSERT INTO versions (version, data_hash, meta_json) VALUES (?, ?, ?)",
        (version, digest, json.dumps(payload["meta"], ensure_ascii=False)),
    )
    conn.executemany(
        "INSERT INTO antennas (version, name, label) VALUES (?, ?, ?)",
        [
            (version, ant["name"], ant["label"])
            for ant in sorted(payload["antennas"], key=lambda a: a["name"])
        ],
    )
    rows = sorted(
        payload["visibilities"],
        key=lambda r: (r["antenna_a"], r["antenna_b"], r["t_idx"], r["channel"]),
    )
    conn.executemany(
        """
        INSERT INTO visibilities
            (version, antenna_a, antenna_b, t_idx, channel,
             re, im, model_re, model_im, quality)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                version,
                r["antenna_a"],
                r["antenna_b"],
                r["t_idx"],
                r["channel"],
                r["re"],
                r["im"],
                r["model_re"],
                r["model_im"],
                r["quality"],
            )
            for r in rows
        ],
    )
    for flag in payload.get("flags", []):
        conn.execute(
            """
            INSERT INTO flags
                (version, source, status, reason, antenna_a, antenna_b,
                 t_start, t_end, ch_start, ch_end)
            VALUES (?, ?, 'applied', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                flag["source"],
                flag["reason"],
                flag.get("antenna_a"),
                flag.get("antenna_b"),
                flag.get("t_start"),
                flag.get("t_end"),
                flag.get("ch_start"),
                flag.get("ch_end"),
            ),
        )
    conn.commit()
    return version


def list_versions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT version, imported_at, data_hash FROM versions ORDER BY version"
        )
    ]


def latest_version(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(version) AS v FROM versions").fetchone()
    return row["v"] if row and row["v"] else None


def load_version(conn: sqlite3.Connection, version: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM versions WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        return None
    return dict(row)


def get_meta(conn: sqlite3.Connection, version: str) -> dict[str, Any]:
    row = load_version(conn, version)
    if row is None:
        raise KeyError(f"unknown version: {version}")
    return json.loads(row["meta_json"])


def get_visibilities(conn: sqlite3.Connection, version: str) -> list[sqlite3.Row]:
    """Return visibilities in canonical order.

    Canonical ordering makes the solver independent of import/insertion order.
    """
    return list(
        conn.execute(
            """
            SELECT * FROM visibilities
            WHERE version = ?
            ORDER BY antenna_a, antenna_b, t_idx, channel
            """,
            (version,),
        )
    )


def get_flags(
    conn: sqlite3.Connection, version: str, status: str | None = None
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM flags WHERE version = ?"
    params: list[Any] = [version]
    if status is not None:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id"
    return list(conn.execute(sql, params))


def add_flag(
    conn: sqlite3.Connection,
    version: str,
    source: str,
    reason: str,
    antenna_a: str | None,
    antenna_b: str | None = None,
    t_start: int | None = None,
    t_end: int | None = None,
    ch_start: int | None = None,
    ch_end: int | None = None,
    status: str = "applied",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO flags
            (version, source, status, reason, antenna_a, antenna_b,
             t_start, t_end, ch_start, ch_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            source,
            status,
            reason,
            antenna_a,
            antenna_b,
            t_start,
            t_end,
            ch_start,
            ch_end,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def set_flag_status(
    conn: sqlite3.Connection, version: str, flag_id: int, status: str
) -> bool:
    cur = conn.execute(
        "UPDATE flags SET status = ? WHERE id = ? AND version = ?",
        (status, flag_id, version),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_flag(conn: sqlite3.Connection, version: str, flag_id: int) -> bool:
    cur = conn.execute(
        "DELETE FROM flags WHERE id = ? AND version = ?", (flag_id, version)
    )
    conn.commit()
    return cur.rowcount > 0


def clear_auto_flags(conn: sqlite3.Connection, version: str) -> int:
    cur = conn.execute(
        "DELETE FROM flags WHERE version = ? AND source = 'auto'", (version,)
    )
    conn.commit()
    return cur.rowcount


def reset_database(conn: sqlite3.Connection) -> None:
    """Wipe every table so the fixture can be re-imported from scratch."""
    conn.executescript(
        """
        DELETE FROM run_closures;
        DELETE FROM run_residuals;
        DELETE FROM run_gains;
        DELETE FROM run_flags;
        DELETE FROM runs;
        DELETE FROM flags;
        DELETE FROM visibilities;
        DELETE FROM antennas;
        DELETE FROM versions;
        """
    )
    conn.commit()


def export_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any]:
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)
    snap = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM run_flags WHERE run_id = ? ORDER BY flag_id", (run_id,)
        )
    ]
    gains = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM run_gains WHERE run_id = ? "
            "ORDER BY group_index, antenna",
            (run_id,),
        )
    ]
    residuals = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM run_residuals WHERE run_id = ? "
            "ORDER BY antenna_a, antenna_b, t_idx, channel",
            (run_id,),
        )
    ]
    closures = [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM run_closures WHERE run_id = ? "
            "ORDER BY triplet, t_idx, group_index",
            (run_id,),
        )
    ]
    return {
        "run": dict(run),
        "flag_snapshot": snap,
        "gains": gains,
        "residuals": residuals,
        "closures": closures,
    }


def list_runs(conn: sqlite3.Connection, version: str | None = None) -> list[dict]:
    sql = "SELECT * FROM runs"
    params: list[Any] = []
    if version:
        sql += " WHERE version = ?"
        params.append(version)
    sql += " ORDER BY id"
    return [dict(row) for row in conn.execute(sql, params)]
