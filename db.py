"""SQLite 存储层: 数据集版本、可见度、旗标区间、求解运行记录。

要点:
- 数据集按规范化内容哈希去重, 与基线导入顺序无关。
- 旗标是只增不改的区间记录, 携带来源 (ingest/manual/auto) 与理由。
- 每次求解运行固定 dataset_id + 参数 + 旗标 id 集合, 之后新增旗标不会改写旧结果。
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  hash TEXT NOT NULL UNIQUE,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS visibilities(
  dataset_id INTEGER NOT NULL,
  ant1 TEXT NOT NULL, ant2 TEXT NOT NULL,
  time REAL NOT NULL, channel INTEGER NOT NULL,
  real REAL NOT NULL, imag REAL NOT NULL,
  quality INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS flags(
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL,
  source TEXT NOT NULL,          -- ingest | manual | auto
  reason TEXT NOT NULL,
  scope TEXT NOT NULL,           -- antenna | baseline
  ant1 TEXT NOT NULL, ant2 TEXT,
  t0 REAL NOT NULL, t1 REAL NOT NULL,
  c0 INTEGER NOT NULL, c1 INTEGER NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  dataset_id INTEGER NOT NULL,
  params TEXT NOT NULL,          -- JSON: ref_ant/t0/t1/time_bin/chan_group
  flag_ids TEXT NOT NULL,        -- JSON: 固定的旗标 id 集合
  status TEXT NOT NULL,
  result TEXT NOT NULL,          -- JSON: 增益/诊断/残差/闭相位
  result_hash TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def canonical_samples(samples):
    """规范化样本: 天线排序 (交换时取共轭), 按 (ant1,ant2,time,channel) 排序。"""
    norm = []
    for s in samples:
        a1, a2 = s["ant1"], s["ant2"]
        re, im = float(s["real"]), float(s["imag"])
        if a2 < a1:
            a1, a2, im = a2, a1, -im
        norm.append((a1, a2, float(s["time"]), int(s["channel"]), re, im,
                     int(s.get("quality", 0))))
    norm.sort()
    return norm


def dataset_hash(samples):
    norm = canonical_samples(samples)
    payload = json.dumps(norm, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def result_hash(result):
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def import_dataset(conn, name, samples):
    """导入数据集; 内容相同 (与顺序无关) 时返回已有 id。"""
    h = dataset_hash(samples)
    row = conn.execute("SELECT id FROM datasets WHERE hash=?", (h,)).fetchone()
    if row:
        return row["id"], h, False
    cur = conn.execute(
        "INSERT INTO datasets(name, hash, created_at) VALUES(?,?,?)",
        (name, h, time.time()))
    ds_id = cur.lastrowid
    conn.executemany(
        "INSERT INTO visibilities VALUES(?,?,?,?,?,?,?,?)",
        [(ds_id, a1, a2, t, c, re, im, q)
         for a1, a2, t, c, re, im, q in canonical_samples(samples)])
    conn.commit()
    return ds_id, h, True


def get_samples(conn, dataset_id):
    rows = conn.execute(
        "SELECT ant1,ant2,time,channel,real,imag,quality FROM visibilities "
        "WHERE dataset_id=? ORDER BY ant1,ant2,time,channel",
        (dataset_id,)).fetchall()
    return [dict(r) for r in rows]


def add_flag(conn, dataset_id, source, reason, scope, ant1, ant2, t0, t1, c0, c1):
    cur = conn.execute(
        "INSERT INTO flags(dataset_id,source,reason,scope,ant1,ant2,t0,t1,c0,c1,created_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (dataset_id, source, reason, scope, ant1, ant2,
         float(t0), float(t1), int(c0), int(c1), time.time()))
    conn.commit()
    return cur.lastrowid


def get_flags(conn, dataset_id, flag_ids=None):
    sql = "SELECT * FROM flags WHERE dataset_id=?"
    args = [dataset_id]
    if flag_ids is not None:
        if not flag_ids:
            return []
        sql += " AND id IN (%s)" % ",".join("?" * len(flag_ids))
        args += list(flag_ids)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args).fetchall()]


def flag_matches(flag, sample):
    if not (flag["t0"] <= sample["time"] <= flag["t1"]):
        return False
    if not (flag["c0"] <= sample["channel"] <= flag["c1"]):
        return False
    if flag["scope"] == "antenna":
        return sample["ant1"] == flag["ant1"] or sample["ant2"] == flag["ant1"]
    return {sample["ant1"], sample["ant2"]} == {flag["ant1"], flag["ant2"]}


def flagged_mask(samples, flags):
    """每个样本命中的旗标 id 列表 (叠加时保留全部来源)。"""
    hits = []
    for s in samples:
        hits.append([f["id"] for f in flags if flag_matches(f, s)])
    return hits


def create_run(conn, dataset_id, params, flag_ids, status, result):
    rh = result_hash(result)
    cur = conn.execute(
        "INSERT INTO runs(dataset_id,params,flag_ids,status,result,result_hash,created_at)"
        " VALUES(?,?,?,?,?,?,?)",
        (dataset_id, json.dumps(params, sort_keys=True),
         json.dumps(sorted(flag_ids)), status, json.dumps(result), rh, time.time()))
    conn.commit()
    return cur.lastrowid, rh


def get_run(conn, run_id):
    r = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return dict(r) if r else None


def list_runs(conn, dataset_id=None):
    sql = "SELECT id,dataset_id,params,flag_ids,status,result_hash,created_at FROM runs"
    args = []
    if dataset_id is not None:
        sql += " WHERE dataset_id=?"
        args.append(dataset_id)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args).fetchall()]
