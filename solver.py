"""天线增益求解、闭相位与数值诊断。

约定:
- 可见度样本 V_ij = g_i * conj(g_j) * M_ij + 噪声, 模型 M_ij = 1 (点源)。
- 幅度方程: ln|V_ij| = a_i - a_j; 相位方程: arg(V_ij) = p_i - p_j。
- 规范: 参考天线相位 p_ref = 0, 平均对数幅度为 0。
- 任何失败路径都不返回全零增益, 只返回可求子图 / 未定标天线 / 条件数诊断。
"""
from __future__ import annotations

import numpy as np

RIDGE = 1e-6
COND_THRESHOLD = 1e10


def unwrap_phases(phases):
    """沿频道轴解缠相位, 正确跨过负实轴 (±π) 分支。"""
    return np.unwrap(np.asarray(phases, dtype=float)).tolist()


def _components(ants, edges):
    parent = {a: a for a in ants}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    groups = {}
    for a in ants:
        groups.setdefault(find(a), []).append(a)
    return [sorted(v) for v in groups.values()]


def solve_group(samples, ref, ridge=None, cond_threshold=None, ants_hint=None):
    ridge = RIDGE if ridge is None else ridge
    cond_threshold = COND_THRESHOLD if cond_threshold is None else cond_threshold
    """对单个 (时间窗, 频率组) 求解复增益。

    samples: [{"ant1","ant2","vis"(complex),"weight"}] (已剔除旗标样本, 向量平均后)
    返回 dict: status / gains / uncalibrated / diagnostics。
    """
    data_ants = sorted({s["ant1"] for s in samples} | {s["ant2"] for s in samples})
    ants = sorted(set(data_ants) | set(ants_hint or []))
    edges = [(s["ant1"], s["ant2"]) for s in samples]
    comps = _components(ants, edges) if ants else []
    diag = {"components": comps, "cond_amplitude": None, "cond_phase": None,
            "degenerate": False, "ridge": ridge}

    if not samples:
        return {"status": "no_data", "gains": {}, "uncalibrated": ants,
                "diagnostics": diag}
    if ref not in data_ants:
        # 参考天线无可用数据: 不伪造增益
        return {"status": "reference_no_data", "gains": {}, "uncalibrated": ants,
                "diagnostics": diag}

    ref_comp = next(c for c in comps if ref in c)
    uncalibrated = sorted(a for c in comps if ref not in c for a in c)
    idx = {a: k for k, a in enumerate(ref_comp)}
    n = len(ref_comp)

    rows, rhs_a, rhs_p, weights = [], [], [], []
    for s in samples:
        row = np.zeros(n)
        row[idx[s["ant1"]]] = 1.0
        row[idx[s["ant2"]]] = -1.0
        rows.append(row)
        rhs_a.append(np.log(abs(s["vis"])))
        rhs_p.append(np.angle(s["vis"]))
        weights.append(float(s.get("weight", 1.0)))
    A = np.asarray(rows)
    W = np.diag(weights)

    def _solve(b, gauge_row):
        # 岭正则化 + 规范行, 消除整体相位/幅度简并
        M = A.T @ W @ A + ridge * np.eye(n) + np.outer(gauge_row, gauge_row)
        cond = float(np.linalg.cond(M))
        x = np.linalg.solve(M, A.T @ W @ b)
        return x, cond

    gauge_phase = np.zeros(n)
    gauge_phase[idx[ref]] = 1.0
    gauge_amp = np.ones(n) / np.sqrt(n)

    a_sol, cond_a = _solve(np.asarray(rhs_a), gauge_amp)
    p_sol, cond_p = _solve(np.asarray(rhs_p), gauge_phase)
    diag["cond_amplitude"] = cond_a
    diag["cond_phase"] = cond_p
    cond = max(cond_a, cond_p)

    if not np.isfinite(cond) or cond > cond_threshold:
        diag["degenerate"] = True
        # 方程退化: 不伪造增益, 只报告可求子图与诊断
        return {"status": "degenerate", "gains": {}, "uncalibrated": ants,
                "solvable_subgraph": ref_comp, "diagnostics": diag}

    gains = {a: complex(np.exp(a_sol[idx[a]]) * np.exp(1j * p_sol[idx[a]]))
             for a in ref_comp}
    return {"status": "ok", "gains": gains, "uncalibrated": uncalibrated,
            "solvable_subgraph": ref_comp, "diagnostics": diag}


def closure_phase(vis_ij, vis_jk, vis_ki):
    """闭合相位 arg(V_ij * V_jk * V_ki), 与天线增益无关。"""
    return float(np.angle(vis_ij * vis_jk * vis_ki))
