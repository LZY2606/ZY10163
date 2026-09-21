"""Solve-run persistence: immutable parameter/flag snapshots and replay."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from . import db as dbmod
from .flags import Flag
from .solver import SolveResult, solve


REPLAY_TOLERANCE = 1e-9


def params_hash(
    ref_antenna: str, t_start: int, t_end: int, group_size: int
) -> str:
    body = json.dumps(
        {
            "ref_antenna": ref_antenna,
            "t_start": t_start,
            "t_end": t_end,
            "group_size": group_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _persist_result(
    conn: sqlite3.Connection,
    version: str,
    ref_antenna: str,
    t_start: int,
    t_end: int,
    group_size: int,
    applied_flags: list,
    result: SolveResult,
) -> int:
    version_row = dbmod.load_version(conn, version)
    assert version_row is not None
    ph = params_hash(ref_antenna, t_start, t_end, group_size)
    fh = dbmod.flags_hash(applied_flags)
    cur = conn.execute(
        """
        INSERT INTO runs
            (version, ref_antenna, t_start, t_end, group_size,
             params_hash, flagset_hash, data_hash, status, diagnostics_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            version,
            ref_antenna,
            t_start,
            t_end,
            group_size,
            ph,
            fh,
            version_row["data_hash"],
            result.status,
            json.dumps(result.diagnostics, ensure_ascii=False),
        ),
    )
    run_id = int(cur.lastrowid)
    conn.executemany(
        """
        INSERT INTO run_flags
            (run_id, flag_id, source, reason, antenna_a, antenna_b,
             t_start, t_end, ch_start, ch_end)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                row["id"],
                row["source"],
                row["reason"],
                row["antenna_a"],
                row["antenna_b"],
                row["t_start"],
                row["t_end"],
                row["ch_start"],
                row["ch_end"],
            )
            for row in applied_flags
        ],
    )
    for group in result.groups:
        g_index = group["group_index"]
        for antenna, gain in group["gains"].items():
            if gain is None:
                conn.execute(
                    "INSERT INTO run_gains (run_id, group_index, antenna, "
                    "gain_re, gain_im, solvable) VALUES (?, ?, ?, NULL, NULL, 0)",
                    (run_id, g_index, antenna),
                )
            else:
                conn.execute(
                    "INSERT INTO run_gains (run_id, group_index, antenna, "
                    "gain_re, gain_im, solvable) VALUES (?, ?, ?, ?, ?, 1)",
                    (run_id, g_index, antenna, gain["re"], gain["im"]),
                )
    conn.executemany(
        """
        INSERT INTO run_residuals
            (run_id, t_idx, channel, antenna_a, antenna_b, flagged,
             residual_re, residual_im, phase_residual, amp_residual)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                r["t_idx"],
                r["channel"],
                r["antenna_a"],
                r["antenna_b"],
                1 if r["flagged"] else 0,
                r["residual_re"],
                r["residual_im"],
                r["phase_residual"],
                r["amp_residual"],
            )
            for r in result.residuals
        ],
    )
    conn.executemany(
        """
        INSERT INTO run_closures
            (run_id, triplet, t_idx, group_index, closure_raw)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                run_id,
                r["triplet"],
                r["t_idx"],
                r["group_index"],
                r["closure_raw"],
            )
            for r in result.closures
        ],
    )
    conn.commit()
    return run_id


def create_run(
    conn: sqlite3.Connection,
    version: str,
    ref_antenna: str,
    t_start: int,
    t_end: int,
    group_size: int,
) -> dict[str, Any]:
    """Solve against the *current* applied flag set and freeze the snapshot."""
    meta = dbmod.get_meta(conn, version)
    antennas = [row["name"] for row in conn.execute(
        "SELECT name FROM antennas WHERE version = ? ORDER BY name", (version,)
    )]
    rows = dbmod.get_visibilities(conn, version)
    applied = dbmod.get_flags(conn, version, status="applied")
    flag_objs = [Flag.from_row(row) for row in applied]
    result = solve(
        antennas=antennas,
        rows=rows,
        flags=flag_objs,
        ntime=meta["ntime"],
        nchan=meta["nchan"],
        ref_antenna=ref_antenna,
        t_start=t_start,
        t_end=t_end,
        group_size=group_size,
    )
    run_id = _persist_result(
        conn, version, ref_antenna, t_start, t_end, group_size, applied, result
    )
    return {"run_id": run_id, "result": _serialize_result(result)}


def _serialize_result(result: SolveResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "groups": result.groups,
        "diagnostics": result.diagnostics,
        "residuals": result.residuals,
        "closures": result.closures,
    }


def replay_run(
    conn: sqlite3.Connection, run_id: int, tolerance: float = REPLAY_TOLERANCE
) -> dict[str, Any]:
    """Re-solve using the frozen snapshot flags; current flags are ignored.

    Verifies data/params/flag hashes and compares recomputed gains against the
    stored values, so a later-added flag cannot silently change an old run.
    """
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise KeyError(run_id)
    version = run["version"]
    version_row = dbmod.load_version(conn, version)
    if version_row is None:
        raise RuntimeError("run version is missing from the database")
    meta = dbmod.get_meta(conn, version)
    antennas = [row["name"] for row in conn.execute(
        "SELECT name FROM antennas WHERE version = ? ORDER BY name", (version,)
    )]

    snapshot_rows = conn.execute(
        "SELECT * FROM run_flags WHERE run_id = ? ORDER BY flag_id", (run_id,)
    ).fetchall()
    snapshot_flags = [
        Flag(
            id=row["flag_id"],
            source=row["source"],
            status="applied",
            reason=row["reason"],
            antenna_a=row["antenna_a"],
            antenna_b=row["antenna_b"],
            t_start=row["t_start"],
            t_end=row["t_end"],
            ch_start=row["ch_start"],
            ch_end=row["ch_end"],
        )
        for row in snapshot_rows
    ]
    snapshot_for_hash = [
        {
            "source": f.source,
            "antenna_a": f.antenna_a,
            "antenna_b": f.antenna_b,
            "t_start": f.t_start,
            "t_end": f.t_end,
            "ch_start": f.ch_start,
            "ch_end": f.ch_end,
        }
        for f in snapshot_flags
    ]

    rows = dbmod.get_visibilities(conn, version)
    result = solve(
        antennas=antennas,
        rows=rows,
        flags=snapshot_flags,
        ntime=meta["ntime"],
        nchan=meta["nchan"],
        ref_antenna=run["ref_antenna"],
        t_start=run["t_start"],
        t_end=run["t_end"],
        group_size=run["group_size"],
    )

    checks = {
        "data_hash_match": version_row["data_hash"] == run["data_hash"],
        "params_hash_match": params_hash(
            run["ref_antenna"], run["t_start"], run["t_end"], run["group_size"]
        )
        == run["params_hash"],
        "flagset_hash_match": dbmod.flags_hash(snapshot_for_hash)
        == run["flagset_hash"],
        "status_match": result.status == run["status"],
        "tolerance": tolerance,
    }

    stored_gains = {
        (row["group_index"], row["antenna"]): (
            None if not row["solvable"] else complex(row["gain_re"], row["gain_im"])
        )
        for row in conn.execute(
            "SELECT * FROM run_gains WHERE run_id = ?", (run_id,)
        )
    }
    max_diff = 0.0
    mismatches: list[dict[str, Any]] = []
    for group in result.groups:
        for antenna, gain in group["gains"].items():
            key = (group["group_index"], antenna)
            stored = stored_gains.get(key)
            recomputed = None if gain is None else complex(gain["re"], gain["im"])
            if stored is None or recomputed is None:
                if stored is not recomputed and not (stored is None and recomputed is None):
                    mismatches.append(
                        {"group": key[0], "antenna": antenna, "kind": "solvability"}
                    )
                continue
            diff = abs(stored - recomputed)
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                mismatches.append(
                    {
                        "group": key[0],
                        "antenna": antenna,
                        "difference": diff,
                    }
                )
    checks["max_gain_difference"] = max_diff
    checks["replay_matches"] = (
        all(checks[k] for k in (
            "data_hash_match",
            "params_hash_match",
            "flagset_hash_match",
            "status_match",
        ))
        and not mismatches
        and max_diff <= tolerance
    )
    checks["mismatches"] = mismatches
    return {
        "run_id": run_id,
        "checks": checks,
        "result": _serialize_result(result),
        "snapshot_flags": [dict(row) for row in snapshot_rows],
    }
