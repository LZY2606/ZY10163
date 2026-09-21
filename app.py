"""视界标定室: FastAPI 服务入口。

运行: uvicorn app:app --host 127.0.0.1 --port 5503
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import db
import engine
import fixture
from solver import unwrap_phases

DEFAULT_PARAMS = {"ref_ant": "A1", "t0": 0.0, "t1": 3.0,
                  "time_bin": 1.0, "chan_group": 8, "c0": 0, "c1": 7}
STATIC = Path(__file__).parent / "static" / "index.html"

class FlagIn(BaseModel):
    dataset_id: int
    reason: str
    scope: str = "antenna"          # antenna | baseline
    ant1: str
    ant2: str | None = None
    t0: float
    t1: float
    c0: int
    c1: int


class RunIn(BaseModel):
    dataset_id: int
    ref_ant: str = "A1"
    t0: float = 0.0
    t1: float = 3.0
    time_bin: float = 1.0
    chan_group: int = 8
    c0: int = 0
    c1: int = 7
    flag_ids: list[int] | None = None  # 缺省 = 当前全部旗标的快照




def create_app(db_path=None):
    app = FastAPI(title="视界标定室")
    app.state.db_path = db_path or os.environ.get("CALIB_DB", "calib.db")

    def conn():
        return db.connect(app.state.db_path)

    @app.get("/")
    def index():
        return FileResponse(STATIC)

    # ---------- 数据集 ----------
    @app.post("/api/datasets/import-fixture")
    def import_fixture(order_seed: int | None = None):
        samples = fixture.make_samples(order_seed=order_seed)
        with conn() as c:
            ds_id, h, created = db.import_dataset(c, "fixture-4ant", samples)
            nflags = 0
            if created:
                for fl in engine.ingest_flags(samples):
                    db.add_flag(c, ds_id, "ingest", fl["reason"], fl["scope"],
                                fl["ant1"], fl["ant2"], fl["t0"], fl["t1"],
                                fl["c0"], fl["c1"])
                    nflags += 1
        return {"dataset_id": ds_id, "hash": h, "created": created,
                "samples": len(samples), "ingest_flags": nflags}

    @app.get("/api/datasets")
    def list_datasets():
        with conn() as c:
            rows = c.execute(
                "SELECT d.id, d.name, d.hash, d.created_at, "
                " (SELECT COUNT(*) FROM visibilities v WHERE v.dataset_id=d.id) AS samples,"
                " (SELECT COUNT(*) FROM flags f WHERE f.dataset_id=d.id) AS flags"
                " FROM datasets d ORDER BY d.id").fetchall()
        return [dict(r) for r in rows]

    @app.get("/api/datasets/{dataset_id}/visibilities")
    def visibilities(dataset_id: int):
        with conn() as c:
            samples = db.get_samples(c, dataset_id)
            flags = db.get_flags(c, dataset_id)
        hits = db.flagged_mask(samples, flags)
        src_by_id = {f["id"]: f["source"] for f in flags}
        for s, h in zip(samples, hits):
            s["flag_ids"] = h
            s["flag_sources"] = sorted({src_by_id[fid] for fid in h})
        return {"samples": samples, "flags": flags,
                "antennas": sorted({s["ant1"] for s in samples}
                                   | {s["ant2"] for s in samples})}

    @app.post("/api/datasets/{dataset_id}/unwrap")
    def unwrap(dataset_id: int, ant1: str, ant2: str, time: float = 0.0):
        """演示/校验用: 返回某基线某时刻沿频道的原始与解缠相位。"""
        import cmath
        with conn() as c:
            samples = db.get_samples(c, dataset_id)
        pair = [s for s in samples if {s["ant1"], s["ant2"]} == {ant1, ant2}
                and s["time"] == time]
        pair.sort(key=lambda s: s["channel"])
        conj = pair and pair[0]["ant1"] == ant2
        phases = [cmath.phase(complex(s["real"], -s["imag"] if conj else s["imag"]))
                  for s in pair]
        return {"channels": [s["channel"] for s in pair], "raw_phase": phases,
                "unwrapped_phase": unwrap_phases(phases)}

    # ---------- 旗标 ----------
    @app.post("/api/flags")
    def add_flag(body: FlagIn):
        with conn() as c:
            fid = db.add_flag(c, body.dataset_id, "manual", body.reason,
                              body.scope, body.ant1, body.ant2,
                              body.t0, body.t1, body.c0, body.c1)
        return {"flag_id": fid, "source": "manual"}

    @app.get("/api/flags")
    def list_flags(dataset_id: int):
        with conn() as c:
            return db.get_flags(c, dataset_id)

    @app.post("/api/flags/suggest")
    def suggest(dataset_id: int, apply: bool = False, sigma: float = 5.0):
        with conn() as c:
            samples = db.get_samples(c, dataset_id)
            flags = db.get_flags(c, dataset_id)
            sugs = engine.suggest_flags(samples, flags, sigma=sigma)
            ids = []
            if apply:
                for s in sugs:
                    ids.append(db.add_flag(c, dataset_id, "auto", s["reason"],
                                           s["scope"], s["ant1"], s["ant2"],
                                           s["t0"], s["t1"], s["c0"], s["c1"]))
        return {"suggestions": sugs, "applied_flag_ids": ids}

    # ---------- 求解运行 ----------
    def _execute_run(c, dataset_id, params, flag_ids):
        samples = db.get_samples(c, dataset_id)
        flags = db.get_flags(c, dataset_id, flag_ids=flag_ids)
        result = engine.solve_run(samples, flags, params)
        run_id, rh = db.create_run(c, dataset_id, params,
                                   flag_ids, result["status"], result)
        return run_id, rh, result

    @app.post("/api/runs")
    def create_run(body: RunIn):
        params = body.model_dump(exclude={"dataset_id", "flag_ids"})
        with conn() as c:
            if body.flag_ids is None:
                flag_ids = [f["id"] for f in db.get_flags(c, body.dataset_id)]
            else:
                flag_ids = sorted(body.flag_ids)
            run_id, rh, result = _execute_run(c, body.dataset_id, params, flag_ids)
        return {"run_id": run_id, "result_hash": rh, "status": result["status"]}

    @app.get("/api/runs")
    def runs(dataset_id: int | None = None):
        with conn() as c:
            return db.list_runs(c, dataset_id)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: int):
        with conn() as c:
            r = db.get_run(c, run_id)
        if not r:
            raise HTTPException(404, "run not found")
        import json
        return {"id": r["id"], "dataset_id": r["dataset_id"],
                "params": json.loads(r["params"]),
                "flag_ids": json.loads(r["flag_ids"]),
                "status": r["status"], "result_hash": r["result_hash"],
                "result": json.loads(r["result"])}

    @app.post("/api/runs/{run_id}/replay")
    def replay(run_id: int):
        """重放旧运行: 严格使用其固定的数据版本、参数与旗标集合。"""
        import json
        with conn() as c:
            r = db.get_run(c, run_id)
            if not r:
                raise HTTPException(404, "run not found")
            run_id2, rh, _ = _execute_run(c, r["dataset_id"],
                                          json.loads(r["params"]),
                                          json.loads(r["flag_ids"]))
        return {"run_id": run_id2, "result_hash": rh,
                "matches_original": rh == r["result_hash"]}

    # ---------- 导出 / 导入 ----------
    @app.get("/api/runs/{run_id}/export")
    def export_run(run_id: int):
        import json
        with conn() as c:
            r = db.get_run(c, run_id)
            if not r:
                raise HTTPException(404, "run not found")
            samples = db.get_samples(c, r["dataset_id"])
            flag_ids = json.loads(r["flag_ids"])
            flags = db.get_flags(c, r["dataset_id"], flag_ids=flag_ids)
            ds = c.execute("SELECT * FROM datasets WHERE id=?",
                           (r["dataset_id"],)).fetchone()
        return {"format": "calib-lab-run@v1",
                "dataset": {"name": ds["name"], "hash": ds["hash"],
                            "samples": samples},
                "flags": flags,  # 顺序即 run.flag_ids 的下标顺序
                "run": {"params": json.loads(r["params"]),
                        "flag_ids": flag_ids,
                        "result_hash": r["result_hash"]}}

    @app.post("/api/runs/import")
    def import_run(record: dict):
        """在 (可为清空的) 数据库中重放导出的运行记录并校验结果哈希。"""
        if record.get("format") != "calib-lab-run@v1":
            raise HTTPException(400, "unsupported format")
        with conn() as c:
            ds_id, _, _ = db.import_dataset(
                c, record["dataset"]["name"], record["dataset"]["samples"])
            id_map = {}
            for fl in record["flags"]:
                new_id = db.add_flag(c, ds_id, fl["source"], fl["reason"],
                                     fl["scope"], fl["ant1"], fl["ant2"],
                                     fl["t0"], fl["t1"], fl["c0"], fl["c1"])
                id_map[fl["id"]] = new_id
            flag_ids = sorted(id_map[fid] for fid in record["run"]["flag_ids"])
            run_id, rh, _ = _execute_run(c, ds_id, record["run"]["params"], flag_ids)
        return {"run_id": run_id, "result_hash": rh,
                "verified": rh == record["run"]["result_hash"]}

    @app.post("/api/dev/wipe")
    def wipe():
        """清空数据库 (供清库重放复核)。"""
        with conn() as c:
            for t in ("runs", "flags", "visibilities", "datasets"):
                c.execute(f"DELETE FROM {t}")
            c.commit()
        return {"wiped": True}

    return app


app = create_app()
