"""视界标定室验收测试。"""
import cmath
import importlib
import math
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DB", str(tmp_path / "test.db"))
    import app as app_mod
    importlib.reload(app_mod)
    c = TestClient(app_mod.create_app(str(tmp_path / "test.db")))
    r = c.post("/api/datasets/import-fixture")
    assert r.status_code == 200
    c.dataset_id = r.json()["dataset_id"]
    return c


def solve(client, **over):
    body = {"dataset_id": client.dataset_id, "ref_ant": "A1",
            "t0": 0, "t1": 3, "time_bin": 1, "chan_group": 8, "c0": 0, "c1": 7}
    body.update(over)
    r = client.post("/api/runs", json=body)
    assert r.status_code == 200
    return r.json()


# ---------- 数据口径与导入顺序 ----------

def test_import_order_invariant(client):
    h1 = client.get("/api/datasets").json()[0]["hash"]
    r = client.post("/api/datasets/import-fixture?order_seed=7")
    assert r.json()["created"] is False           # 乱序导入被去重
    r2 = client.post("/api/datasets/import-fixture?order_seed=99")
    assert r2.json()["hash"] == h1
    assert len(client.get("/api/datasets").json()) == 1


def test_fixture_contents(client):
    data = client.get(f"/api/datasets/{client.dataset_id}/visibilities").json()
    assert data["antennas"] == ["A1", "A2", "A3", "A4"]
    assert len(data["samples"]) == 6 * 4 * 8      # 6 基线 × 4 时间 × 8 频道
    # A4 全部被 ingest 质量位旗标
    a4 = [s for s in data["samples"] if "A4" in (s["ant1"], s["ant2"])]
    assert a4 and all(s["flag_ids"] for s in a4)
    assert all("ingest" in s["flag_sources"] for s in a4)
    # 旗标携带来源与理由
    f0 = data["flags"][0]
    assert f0["source"] == "ingest" and f0["reason"]


# ---------- 求解与诊断 ----------

def test_solve_ok_and_a4_uncalibrated(client):
    run = solve(client)
    assert run["status"] == "degraded"            # A4 不可求 -> 降级而非失败
    detail = client.get(f"/api/runs/{run['run_id']}").json()["result"]
    for g in detail["groups"].values():
        assert g["status"] == "ok"
        assert g["uncalibrated"] == ["A4"]
        assert g["solvable_subgraph"] == ["A1", "A2", "A3"]
        # 增益非零, 且数值条件有限
        assert all(abs(complex(*v)) > 0.5 for v in g["gains"].values())
        assert math.isfinite(g["diagnostics"]["cond_phase"])
    # A2 在 t>=2 有 +0.5 rad 突跳, 逐频道逐时窗求解应能恢复
    run1 = solve(client, chan_group=1)
    d1 = client.get(f"/api/runs/{run1['run_id']}").json()["result"]
    ch0 = {g["tbin"]: g for g in d1["groups"].values() if g["c0"] == 0}
    ph = {t: cmath.phase(complex(*g["gains"]["A2"])) for t, g in ch0.items()}
    assert abs(ph[2] - ph[0] - 0.5) < 0.05


def test_reference_fully_flagged(client):
    run = solve(client, ref_ant="A4")
    assert run["status"] == "failed"
    detail = client.get(f"/api/runs/{run['run_id']}").json()["result"]
    for g in detail["groups"].values():
        assert g["status"] == "reference_no_data"
        assert g["gains"] == {}                   # 不伪造全零增益


def test_disconnected_graph(client):
    # 手工旗标 A2 的全部数据 -> 基线图断开
    client.post("/api/flags", json={
        "dataset_id": client.dataset_id, "reason": "测试断图", "scope": "antenna",
        "ant1": "A2", "t0": 0, "t1": 3, "c0": 0, "c1": 7})
    run = solve(client)
    detail = client.get(f"/api/runs/{run['run_id']}").json()["result"]
    for g in detail["groups"].values():
        assert g["solvable_subgraph"] == ["A1", "A3"]
        assert g["uncalibrated"] == ["A2", "A4"]
        assert g["diagnostics"]["components"]


def test_degenerate_not_faked(client, monkeypatch):
    import solver
    # 人为制造病态: 岭正则化趋零 + 阈值趋零 -> 退化路径
    run = solve(client, ref_ant="A1")
    assert run["status"] != "failed"
    monkeypatch.setattr(solver, "COND_THRESHOLD", 1.0)
    body = {"dataset_id": client.dataset_id, "ref_ant": "A1",
            "t0": 0, "t1": 3, "time_bin": 1, "chan_group": 8, "c0": 0, "c1": 7}
    r = client.post("/api/runs", json=body).json()
    assert r["status"] == "failed"
    detail = client.get(f"/api/runs/{r['run_id']}").json()["result"]
    for g in detail["groups"].values():
        assert g["status"] == "degenerate"
        assert g["gains"] == {}
        assert g["solvable_subgraph"] and g["diagnostics"]["degenerate"]


# ---------- 旗标语义 ----------

def test_flagged_excluded_but_locatable(client):
    client.post("/api/flags", json={
        "dataset_id": client.dataset_id, "reason": "A2 相位突跳", "scope": "antenna",
        "ant1": "A2", "t0": 2, "t1": 3, "c0": 0, "c1": 7})
    run = solve(client)
    detail = client.get(f"/api/runs/{run['run_id']}").json()["result"]
    flagged = [r for r in detail["residuals"] if r["flag_ids"]]
    assert flagged                                # 被旗标样本仍在结果中可定位
    assert all(r["time"] >= 2 or "A4" in (r["ant1"], r["ant2"]) or r["flag_ids"]
               for r in flagged)
    # 旗标区间保留来源与理由
    flags = client.get(f"/api/flags?dataset_id={client.dataset_id}").json()
    manual = [f for f in flags if f["source"] == "manual"]
    assert manual and manual[0]["reason"] == "A2 相位突跳"


def test_auto_suggestion_finds_rfi(client):
    r = client.post(f"/api/flags/suggest?dataset_id={client.dataset_id}&apply=true")
    sugs = r.json()["suggestions"]
    assert sugs, "应发现 A2-A3 的局部干扰"
    assert all(s["reason"].startswith("自动建议") for s in sugs)
    hits = [s for s in sugs if {s["ant1"], s["ant2"]} == {"A2", "A3"}
            and s["t0"] == 1.0 and s["c0"] <= 4 and s["c1"] >= 6]
    assert hits
    flags = client.get(f"/api/flags?dataset_id={client.dataset_id}").json()
    assert any(f["source"] == "auto" for f in flags)


# ---------- 相位绕回 ----------

def test_phase_unwrap_crosses_negative_axis(client):
    r = client.post(f"/api/datasets/{client.dataset_id}/unwrap",
                    params={"ant1": "A1", "ant2": "A3", "time": 0})
    d = r.json()
    raw = d["raw_phase"]
    assert max(raw) > 3.0 and min(raw) < -2.0    # 原始相位确实跨过 ±π
    unw = d["unwrapped_phase"]
    diffs = [unw[i + 1] - unw[i] for i in range(len(unw) - 1)]
    assert all(abs(d) < 1.0 for d in diffs)       # 解缠后无 2π 跳变
    assert abs(unw[-1] - unw[0]) > 2.0                 # 且保留真实累积相位


# ---------- 闭相位 ----------

def test_closure_phase_triangle(client):
    client.post(f"/api/flags/suggest?dataset_id={client.dataset_id}&apply=true")
    run = solve(client)
    detail = client.get(f"/api/runs/{run['run_id']}").json()["result"]
    tri = [c for c in detail["closures"]
           if c["triangle"] == ["A1", "A2", "A3"] and not c["flagged"]]
    assert tri
    # 点源模型下闭相位应接近 0 (模 2π)
    for c in tri:
        p = (c["closure_phase"] + math.pi) % (2 * math.pi) - math.pi
        assert abs(p) < 0.05


# ---------- 运行固定与重放 ----------

def test_flags_do_not_rewrite_old_runs(client):
    run1 = solve(client)
    # 求解后新增旗标
    client.post("/api/flags", json={
        "dataset_id": client.dataset_id, "reason": "事后旗标", "scope": "antenna",
        "ant1": "A2", "t0": 0, "t1": 3, "c0": 0, "c1": 7})
    rep = client.post(f"/api/runs/{run1['run_id']}/replay").json()
    assert rep["matches_original"] is True        # 重放不读取新旗标
    # 新运行则使用新旗标, 结果不同
    run2 = solve(client)
    assert run2["result_hash"] != run1["result_hash"]


def test_unsolvable_subgraph_order_invariant(tmp_path, monkeypatch):
    """不同基线导入顺序 -> 不可求子图相同。"""
    results = []
    for i, seed in enumerate([None, 42]):
        dbp = tmp_path / f"ord{i}.db"
        import app as app_mod
        c = TestClient(app_mod.create_app(str(dbp)))
        c.post("/api/datasets/import-fixture" + ("" if seed is None else f"?order_seed={seed}"))
        c.post("/api/flags", json={
            "dataset_id": 1, "reason": "断图", "scope": "antenna",
            "ant1": "A2", "t0": 0, "t1": 3, "c0": 0, "c1": 7})
        run = c.post("/api/runs", json={
            "dataset_id": 1, "ref_ant": "A1", "t0": 0, "t1": 3,
            "time_bin": 1, "chan_group": 8, "c0": 0, "c1": 7}).json()
        detail = c.get(f"/api/runs/{run['run_id']}").json()["result"]
        results.append(detail)
    for g1, g2 in zip(results[0]["groups"].values(), results[1]["groups"].values()):
        assert g1["uncalibrated"] == g2["uncalibrated"] == ["A2", "A4"]
        assert g1["solvable_subgraph"] == g2["solvable_subgraph"]


def test_export_wipe_reimport(client):
    run = solve(client)
    rec = client.get(f"/api/runs/{run['run_id']}/export").json()
    assert rec["format"] == "calib-lab-run@v1"
    client.post("/api/dev/wipe")
    assert client.get("/api/datasets").json() == []
    r = client.post("/api/runs/import", json=rec).json()
    assert r["verified"] is True                  # 清库重放结果一致
    assert r["result_hash"] == run["result_hash"]
