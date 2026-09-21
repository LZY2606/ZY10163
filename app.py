"""视界标定室 - FastAPI application."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from calib import db as dbmod
from calib import runs as runsmod
from calib.fixture import load_fixture
from calib.flags import Flag, suggest_flags

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CALIB_DB", BASE_DIR / "data" / "chamber.db"))


app = FastAPI(title="视界标定室", version="1.0.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = dbmod.connect(DB_PATH)
    dbmod.init_db(conn)
    if dbmod.latest_version(conn) is None:
        dbmod.import_payload(conn, load_fixture())
    return conn


class FlagIn(BaseModel):
    source: str = "manual"
    reason: str = "手工旗标"
    antenna_a: str | None = None
    antenna_b: str | None = None
    t_start: int | None = None
    t_end: int | None = None
    ch_start: int | None = None
    ch_end: int | None = None
    status: str = "applied"


class SolveIn(BaseModel):
    ref_antenna: str
    t_start: int = 0
    t_end: int = 5
    group_size: int = Field(default=2, ge=1)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"service": "视界标定室", "status": "ok"}


@app.get("/api/state")
def state() -> dict[str, Any]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        if version is None:
            raise HTTPException(404, "no data imported")
        meta = dbmod.get_meta(conn, version)
        version_row = dbmod.load_version(conn, version)
        antennas = [
            dict(row)
            for row in conn.execute(
                "SELECT name, label FROM antennas WHERE version = ? ORDER BY name",
                (version,),
            )
        ]
        flags = [dict(row) for row in dbmod.get_flags(conn, version)]
        run_rows = dbmod.list_runs(conn, version)
        return {
            "version": version,
            "data_hash": version_row["data_hash"],
            "meta": meta,
            "antennas": antennas,
            "flags": flags,
            "runs": [
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "ref_antenna": r["ref_antenna"],
                    "t_start": r["t_start"],
                    "t_end": r["t_end"],
                    "group_size": r["group_size"],
                    "status": r["status"],
                    "params_hash": r["params_hash"],
                    "flagset_hash": r["flagset_hash"],
                    "data_hash": r["data_hash"],
                }
                for r in run_rows
            ],
        }
    finally:
        conn.close()


@app.get("/api/data")
def data() -> dict[str, Any]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        if version is None:
            raise HTTPException(404, "no data imported")
        vis = [dict(row) for row in dbmod.get_visibilities(conn, version)]
        flags = [dict(row) for row in dbmod.get_flags(conn, version)]
        return {"version": version, "visibilities": vis, "flags": flags}
    finally:
        conn.close()


@app.post("/api/flags")
def create_flag(flag: FlagIn) -> dict[str, Any]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        if version is None:
            raise HTTPException(404, "no data imported")
        if flag.source not in ("manual", "auto"):
            raise HTTPException(400, "source must be manual or auto")
        if flag.status not in ("applied", "suggested"):
            raise HTTPException(400, "status must be applied or suggested")
        flag_id = dbmod.add_flag(
            conn,
            version,
            flag.source,
            flag.reason,
            flag.antenna_a,
            flag.antenna_b,
            flag.t_start,
            flag.t_end,
            flag.ch_start,
            flag.ch_end,
            flag.status,
        )
        return {"id": flag_id}
    finally:
        conn.close()


@app.post("/api/flags/{flag_id}/apply")
def apply_flag(flag_id: int) -> dict[str, bool]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        ok = dbmod.set_flag_status(conn, version, flag_id, "applied")
        if not ok:
            raise HTTPException(404, "flag not found")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/flags/{flag_id}/reject")
def reject_flag(flag_id: int) -> dict[str, bool]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        # rejecting a suggestion removes the auto row; manual rows untouched
        row = conn.execute(
            "SELECT source FROM flags WHERE id = ? AND version = ?",
            (flag_id, version),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "flag not found")
        ok = dbmod.delete_flag(conn, version, flag_id)
        return {"ok": ok}
    finally:
        conn.close()


@app.delete("/api/flags/{flag_id}")
def remove_flag(flag_id: int) -> dict[str, bool]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        ok = dbmod.delete_flag(conn, version, flag_id)
        if not ok:
            raise HTTPException(404, "flag not found")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/suggest")
def suggest() -> dict[str, Any]:
    """Regenerate auto-suggestions (idempotent: clears prior auto rows)."""
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        if version is None:
            raise HTTPException(404, "no data imported")
        meta = dbmod.get_meta(conn, version)
        dbmod.clear_auto_flags(conn, version)
        rows = dbmod.get_visibilities(conn, version)
        flag_objs = [Flag.from_row(r) for r in dbmod.get_flags(conn, version)]
        suggestions = suggest_flags(
            rows, flag_objs, meta["ntime"], meta["nchan"]
        )
        created = []
        for item in suggestions:
            flag_id = dbmod.add_flag(
                conn,
                version,
                item["source"],
                item["reason"],
                item["antenna_a"],
                item["antenna_b"],
                item["t_start"],
                item["t_end"],
                item["ch_start"],
                item["ch_end"],
                status="suggested",
            )
            created.append({"id": flag_id, **item})
        return {"suggestions": created}
    finally:
        conn.close()


@app.post("/api/solve")
def solve_endpoint(body: SolveIn) -> dict[str, Any]:
    conn = get_conn()
    try:
        version = dbmod.latest_version(conn)
        if version is None:
            raise HTTPException(404, "no data imported")
        meta = dbmod.get_meta(conn, version)
        known = [
            row["name"]
            for row in conn.execute(
                "SELECT name FROM antennas WHERE version = ?", (version,)
            )
        ]
        if body.ref_antenna not in known:
            raise HTTPException(400, "unknown reference antenna")
        if not (0 <= body.t_start <= body.t_end < meta["ntime"]):
            raise HTTPException(400, "invalid time window")
        if body.group_size < 1 or body.group_size > meta["nchan"]:
            raise HTTPException(400, "invalid group size")
        out = runsmod.create_run(
            conn,
            version,
            body.ref_antenna,
            body.t_start,
            body.t_end,
            body.group_size,
        )
        return out
    finally:
        conn.close()


@app.get("/api/runs")
def list_runs() -> dict[str, Any]:
    conn = get_conn()
    try:
        return {"runs": dbmod.list_runs(conn)}
    finally:
        conn.close()


@app.get("/api/runs/{run_id}")
def get_run(run_id: int) -> dict[str, Any]:
    conn = get_conn()
    try:
        try:
            return dbmod.export_run(conn, run_id)
        except KeyError:
            raise HTTPException(404, "run not found")
    finally:
        conn.close()


@app.post("/api/runs/{run_id}/replay")
def replay(run_id: int) -> dict[str, Any]:
    conn = get_conn()
    try:
        try:
            return runsmod.replay_run(conn, run_id)
        except KeyError:
            raise HTTPException(404, "run not found")
    finally:
        conn.close()


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: int) -> JSONResponse:
    conn = get_conn()
    try:
        try:
            payload = dbmod.export_run(conn, run_id)
        except KeyError:
            raise HTTPException(404, "run not found")
        return JSONResponse(payload)
    finally:
        conn.close()


@app.post("/api/reimport")
def reimport() -> dict[str, Any]:
    """Clear the database and re-import the fixed fixture from scratch."""
    conn = get_conn()
    try:
        dbmod.reset_database(conn)
        version = dbmod.import_payload(conn, load_fixture())
        return {"ok": True, "version": version}
    finally:
        conn.close()


@app.exception_handler(404)
def _not_found(request, exc):  # pragma: no cover - thin wrapper
    return JSONResponse({"detail": str(exc.detail)}, status_code=404)
