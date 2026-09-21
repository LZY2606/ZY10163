"""求解编排: 分组、旗标应用、增益求解、残差与闭相位、自动建议。"""
from __future__ import annotations

import numpy as np

import db
import solver


def ingest_flags(samples):
    """把质量位坏样本转成 ingest 旗标区间。

    天线所有样本均坏 -> 单个天线区间; 其余按基线对合并 (时间, 连续频道)。
    """
    bad = [s for s in samples if s["quality"] & 1]
    ants = sorted({s["ant1"] for s in samples} | {s["ant2"] for s in samples})
    out = []
    covered = set()
    for ant in ants:
        touching = [s for s in samples if ant in (s["ant1"], s["ant2"])]
        if touching and all(s["quality"] & 1 for s in touching):
            times = sorted({s["time"] for s in touching})
            chans = sorted({s["channel"] for s in touching})
            out.append({"scope": "antenna", "ant1": ant, "ant2": None,
                        "t0": times[0], "t1": times[-1],
                        "c0": chans[0], "c1": chans[-1],
                        "reason": "质量位标记: 天线数据不可用"})
            covered.update(id(s) for s in touching)
    pairs = {}
    for s in bad:
        if id(s) not in covered:
            pairs.setdefault((s["ant1"], s["ant2"]), []).append(s)
    for (a1, a2), ss in sorted(pairs.items()):
        by_time = {}
        for s in ss:
            by_time.setdefault(s["time"], []).append(s["channel"])
        for t, chans in sorted(by_time.items()):
            chans.sort()
            start = prev = chans[0]
            for c in chans[1:] + [None]:
                if c != (prev + 1 if prev is not None else None):
                    out.append({"scope": "baseline", "ant1": a1, "ant2": a2,
                                "t0": t, "t1": t, "c0": start, "c1": prev,
                                "reason": "质量位标记: 局部坏样本"})
                    start = c
                prev = c
    return out


def _group_key(sample, params):
    t0 = params["t0"]
    tb = params["time_bin"]
    cg = params["chan_group"]
    tbin = 0 if not tb else int((sample["time"] - t0) // tb)
    cgroup = sample["channel"] // cg
    return tbin, cgroup


def _group_label(key, params):
    tbin, cgroup = key
    cg = params["chan_group"]
    return {"tbin": tbin, "c0": cgroup * cg, "c1": cgroup * cg + cg - 1}


def solve_run(samples, flags, params):
    """完整求解一次运行。samples/flags 已按运行固定, 不读取外部状态。"""
    hits = db.flagged_mask(samples, flags)
    sel = [s for s, h in zip(samples, hits)
           if params["t0"] <= s["time"] <= params["t1"]
           and params["c0"] <= s["channel"] <= params["c1"]]
    sel_hits = [h for s, h in zip(samples, hits)
                if params["t0"] <= s["time"] <= params["t1"]
                and params["c0"] <= s["channel"] <= params["c1"]]

    # 分组 + 组内向量平均 (对跨 ±π 绕回安全)
    buckets = {}
    for s, h in zip(sel, sel_hits):
        if h:
            continue  # 被旗标的样本不进入方程
        key = (s["ant1"], s["ant2"], _group_key(s, params))
        buckets.setdefault(key, []).append(complex(s["real"], s["imag"]))

    all_ants = sorted({s["ant1"] for s in sel} | {s["ant2"] for s in sel})
    groups = {}
    for (a1, a2, gkey), vals in buckets.items():
        groups.setdefault(gkey, []).append(
            {"ant1": a1, "ant2": a2, "vis": sum(vals) / len(vals),
             "weight": float(len(vals))})

    group_results = {}
    gains_by_group = {}
    statuses = []
    for gkey in sorted(groups):
        res = solver.solve_group(groups[gkey], params["ref_ant"],
                                 ants_hint=all_ants)
        label = _group_label(gkey, params)
        gains = {a: [g.real, g.imag] for a, g in res["gains"].items()}
        gains_by_group[gkey] = {a: complex(*v) for a, v in gains.items()}
        group_results[str(gkey)] = {
            **label, "status": res["status"], "gains": gains,
            "uncalibrated": res.get("uncalibrated", []),
            "solvable_subgraph": res.get("solvable_subgraph", []),
            "diagnostics": res["diagnostics"],
        }
        statuses.append(res["status"])

    # 残差: 所有选区样本 (含被旗标者) 都可定位
    residuals = []
    for s, h in zip(sel, sel_hits):
        gkey = _group_key(s, params)
        gains = gains_by_group.get(gkey, {})
        g1, g2 = gains.get(s["ant1"]), gains.get(s["ant2"])
        r = None
        if g1 is not None and g2 is not None:
            v = complex(s["real"], s["imag"])
            res = v - g1 * g2.conjugate()
            r = [res.real, res.imag]
        residuals.append({
            "ant1": s["ant1"], "ant2": s["ant2"],
            "time": s["time"], "channel": s["channel"],
            "flag_ids": h, "residual": r,
        })

    closures = closure_phases(sel, sel_hits)

    if not groups:
        status = "failed"
    elif any(st in ("reference_no_data", "degenerate") for st in statuses):
        status = "failed"
    elif any(st == "no_data" for st in statuses) or any(
            group_results[k]["uncalibrated"] for k in group_results):
        status = "degraded"
    else:
        status = "ok"

    return {"status": status, "params": params, "groups": group_results,
            "closures": closures, "residuals": residuals}


def closure_phases(samples, hits):
    """对每组天线三角形, 按 (时间, 频道) 计算闭相位 (原始数据量, 与增益无关)。"""
    by_baseline = {}
    for s, h in zip(samples, hits):
        by_baseline[(s["ant1"], s["ant2"], s["time"], s["channel"])] = (
            complex(s["real"], s["imag"]), h)

    def get(a, b, t, c):
        if (a, b, t, c) in by_baseline:
            return by_baseline[(a, b, t, c)]
        v, h = by_baseline[(b, a, t, c)]
        return v.conjugate(), h

    ants = sorted({s["ant1"] for s in samples} | {s["ant2"] for s in samples})
    out = []
    for x in range(len(ants)):
        for y in range(x + 1, len(ants)):
            for z in range(y + 1, len(ants)):
                tri = (ants[x], ants[y], ants[z])
                times = sorted({s["time"] for s in samples})
                chans = sorted({s["channel"] for s in samples})
                for t in times:
                    for c in chans:
                        try:
                            v1, h1 = get(tri[0], tri[1], t, c)
                            v2, h2 = get(tri[1], tri[2], t, c)
                            v3, h3 = get(tri[2], tri[0], t, c)
                        except KeyError:
                            continue
                        out.append({
                            "triangle": list(tri), "time": t, "channel": c,
                            "closure_phase": solver.closure_phase(v1, v2, v3),
                            "flagged": bool(h1 or h2 or h3),
                        })
    return out


def suggest_flags(samples, flags, sigma=5.0):
    """自动建议: 每个 (基线, 频道) 沿时间取中值, 残差超阈者建议旗标 (不落库)。"""
    hits = db.flagged_mask(samples, flags)
    cells = {}
    for s, h in zip(samples, hits):
        if h:
            continue
        key = (s["ant1"], s["ant2"], s["channel"])
        cells.setdefault(key, []).append(s)
    suggestions = []
    for (a1, a2, chan), ss in sorted(cells.items()):
        vals = np.array([complex(s["real"], s["imag"]) for s in ss])
        med = np.median(vals)
        dev = np.abs(vals - med)
        mad = float(np.median(dev)) or 1e-12
        for s, d in zip(ss, dev):
            if d > sigma * 1.4826 * mad:
                suggestions.append({
                    "scope": "baseline", "ant1": a1, "ant2": a2,
                    "t0": s["time"], "t1": s["time"],
                    "c0": s["channel"], "c1": s["channel"],
                    "reason": "自动建议: 残差 %.1fσ 超阈 (疑似局部干扰)" % (
                        d / (1.4826 * mad)),
                })
    return _merge_suggestions(suggestions)


def _merge_suggestions(sugs):
    """把同基线同时间的连续频道建议合并为区间。"""
    merged = []
    by_key = {}
    for s in sugs:
        by_key.setdefault((s["ant1"], s["ant2"], s["t0"]), []).append(s)
    for (a1, a2, t), items in sorted(by_key.items()):
        items.sort(key=lambda x: x["c0"])
        cur = dict(items[0])
        for nxt in items[1:]:
            if nxt["c0"] <= cur["c1"] + 1:
                cur["c1"] = max(cur["c1"], nxt["c1"])
            else:
                merged.append(cur)
                cur = dict(nxt)
        merged.append(cur)
    return merged
