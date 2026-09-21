"""Antenna-gain calibration solver.

Equation (per visibility cell, log domain):
    V_ab = conj(g_a) * g_b * S_ab
    log|V_ab/S_ab| = x_a + x_b + noise          (amplitudes multiply)
    arg(V_ab/S_ab) = p_b - p_a + noise mod 2pi  (phases subtract)

Phases are solved on each connected component after anchoring one node;
branch cuts (the negative real axis) are handled by wrapping every residual
into (-pi, pi].  Amplitude and phase use the same unflagged cells, so the
baseline graph is identical for both.  Components the reference cannot reach
are reported in diagnostics and never receive fabricated zero gains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np

from .flags import Flag, wrap_phase


@dataclass
class GroupDiagnostic:
    group_index: int
    channels: list[int]
    components: list[list[str]]
    ref_component: list[str] | None
    uncalibrated: list[str]
    singular_values: list[float]
    rank: int
    n_equations: int
    n_unknowns: int
    condition_number: float | None
    rank_deficiency: int
    ref_has_data: bool


@dataclass
class SolveResult:
    status: str
    ref_antenna: str
    groups: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    residuals: list[dict[str, Any]] = field(default_factory=list)
    closures: list[dict[str, Any]] = field(default_factory=list)


def _connected_components(
    antennas: list[str], edges: set[tuple[str, str]]
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {a: set() for a in antennas}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(antennas):
        if start in seen:
            continue
        stack = [start]
        comp: list[str] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            comp.append(node)
            stack.extend(sorted(adjacency[node] - seen))
        components.append(sorted(comp))
    return components


def _weighted_lstsq(
    matrix: np.ndarray, vector: np.ndarray, rcond: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """SVD least squares; returns solution, singular values, rank, condition."""
    if matrix.size == 0:
        return np.zeros(0), np.zeros(0), 0, None
    u, svals, vt = np.linalg.svd(matrix, full_matrices=False)
    tol = rcond * float(svals[0]) if svals.size else 0.0
    rank = int(np.sum(svals > tol))
    condition = float(svals[0] / svals[-1]) if svals.size and svals[-1] > 0 else None
    inv = np.zeros_like(svals)
    inv[:rank] = 1.0 / svals[:rank]
    solution = vt.T @ (inv * (u.T @ vector))
    return solution, svals, rank, condition


def _component_system(
    samples: list[dict[str, Any]],
    antennas: list[str],
    component: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Incidence rows for one component.

    Amplitude rows put +1 at both endpoints (log-sum: x_a + x_b); phase rows
    put -1 at a and +1 at b (phase difference p_b - p_a).
    Returns the unsigned incidence matrix, log-amplitude vector and phase
    vector of V/S.
    """
    index = {name: i for i, name in enumerate(component)}
    amp_rows: list[np.ndarray] = []
    phase_rows: list[np.ndarray] = []
    amps: list[float] = []
    phases: list[float] = []
    for sample in samples:
        a, b = sample["antenna_a"], sample["antenna_b"]
        if a not in index or b not in index:
            continue
        amp_row = np.zeros(len(component))
        amp_row[index[a]] = 1.0
        amp_row[index[b]] = 1.0
        amp_rows.append(amp_row)
        phase_row = np.zeros(len(component))
        phase_row[index[a]] = -1.0
        phase_row[index[b]] = 1.0
        amps.append(sample["log_ratio"])
        phases.append(sample["phase_ratio"])
        phase_rows.append(phase_row)
    if not amp_rows:
        return (
            np.zeros((0, len(component))),
            np.zeros((0, len(component))),
            np.zeros(0),
            np.zeros(0),
        )
    return (
        np.vstack(amp_rows),
        np.vstack(phase_rows),
        np.asarray(amps),
        np.asarray(phases),
    )


def _solve_anchored(
    matrix: np.ndarray,
    vector: np.ndarray,
    anchor_idx: int,
    iterative_phase: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """Solve B x = y with x[anchor] = 0 (gauge fix, not a fabricated gain)."""
    n_nodes = matrix.shape[1]
    keep = [i for i in range(n_nodes) if i != anchor_idx]
    reduced = matrix[:, keep]
    target = vector.copy()
    solution = np.zeros(n_nodes)
    svals = np.zeros(0)
    rank = 0
    condition: float | None = None
    if reduced.shape[0]:
        x_red, svals, rank, condition = _weighted_lstsq(reduced, target)
        solution[np.asarray(keep)] = x_red
    if iterative_phase:
        # re-wrap residuals and re-solve until stable: this is what makes
        # phases cross the negative-real-axis branch correctly.
        x_full = solution.copy()
        for _ in range(32):
            residual_row = target - reduced @ x_full[keep]
            y_tilde = reduced @ x_full[keep] + wrap_phase(residual_row)
            x_red, svals, rank, condition = _weighted_lstsq(reduced, y_tilde)
            x_new = np.zeros(n_nodes)
            x_new[keep] = x_red
            delta = (
                float(np.max(np.abs(x_new - x_full))) if x_full.size else 0.0
            )
            x_full = x_new
            if delta < 1e-11:
                break
        solution = x_full
    return solution, svals, rank, condition


def _solve_anchored_sum(
    matrix: np.ndarray,
    vector: np.ndarray,
    anchor_idx: int,
) -> tuple[np.ndarray, np.ndarray, int, float | None]:
    """Solve the log-amplitude sum system with x[anchor] = 0.

    Rows of ``matrix`` are +1 at both endpoints, so the anchor column is moved
    to the right-hand side instead of simply dropped.
    """
    n_nodes = matrix.shape[1]
    keep = [i for i in range(n_nodes) if i != anchor_idx]
    reduced = matrix[:, keep]
    target = vector - matrix[:, anchor_idx] * 0.0  # anchor log-gain is zero
    solution = np.zeros(n_nodes)
    if reduced.shape[0]:
        x_red, svals, rank, condition = _weighted_lstsq(reduced, target)
        solution[np.asarray(keep)] = x_red
    else:
        svals, rank, condition = np.zeros(0), 0, None
    return solution, svals, rank, condition


def closure_phase(
    z_ab: complex, z_bc: complex, z_ca: complex
) -> float:
    """arg(V_ab V_bc V_ca); the product is formed before taking the angle.

    Forming the complex product first keeps the result branch-safe instead of
    summing three independently wrapped angles.
    """
    return float(np.angle(z_ab * z_bc * z_ca))


def solve(
    antennas: list[str],
    rows: list[Any],
    flags: list[Flag],
    ntime: int,
    nchan: int,
    ref_antenna: str,
    t_start: int,
    t_end: int,
    group_size: int,
) -> SolveResult:
    """Run the calibration.

    ``rows`` must be DB rows (or mapping rows) in any order; canonical
    ordering is applied here so import order cannot change the answer.
    """
    antennas = sorted(antennas)
    if ref_antenna not in antennas:
        raise ValueError(f"unknown reference antenna: {ref_antenna}")
    if not (0 <= t_start <= t_end < ntime) or group_size <= 0:
        raise ValueError("invalid time window or group size")

    ordered = sorted(
        rows,
        key=lambda r: (r["antenna_a"], r["antenna_b"], r["t_idx"], r["channel"]),
    )
    applied = [f for f in flags if f.status == "applied"]

    def _is_flagged(a: str, b: str, t: int, ch: int) -> bool:
        return any(f.matches(a, b, t, ch) for f in applied)

    # pre-stage cell data
    cells: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    for row in ordered:
        a, b, t, ch = row["antenna_a"], row["antenna_b"], row["t_idx"], row["channel"]
        obs = complex(row["re"], row["im"])
        model = complex(row["model_re"], row["model_im"])
        cells[(a, b, t, ch)] = {
            "antenna_a": a,
            "antenna_b": b,
            "t_idx": t,
            "channel": ch,
            "obs": obs,
            "model": model,
            "quality": int(row["quality"]),
            "flagged": _is_flagged(a, b, t, ch),
        }

    group_edges: list[tuple[int, int, set[tuple[str, str]]]] = []
    for g0 in range(0, nchan, group_size):
        g1 = min(nchan - 1, g0 + group_size - 1)
        edges: set[tuple[str, str]] = set()
        for a, b in combinations(antennas, 2):
            for t in range(t_start, t_end + 1):
                for ch in range(g0, g1 + 1):
                    cell = cells.get((a, b, t, ch))
                    if cell is not None and not cell["flagged"]:
                        edges.add((a, b))
        group_edges.append((g0, g1, edges))

    result = SolveResult(status="ok", ref_antenna=ref_antenna)
    gains_by_group: list[dict[str, complex | None]] = []
    group_diags: list[dict[str, Any]] = []
    overall_uncalibrated: set[str] = set(antennas)

    for g_index, (g0, g1, edges) in enumerate(group_edges):
        channels = list(range(g0, g1 + 1))
        components = _connected_components(antennas, edges)
        ref_component: list[str] | None = next(
            (comp for comp in components if ref_antenna in comp), None
        )
        ref_has_data = any(
            tuple(sorted((ref_antenna, other))) in edges
            for other in antennas
            if other != ref_antenna
        )
        gains: dict[str, complex | None] = {a: None for a in antennas}
        x_amp: dict[str, float] = {}
        p_phase: dict[str, float] = {}

        samples_by_comp: dict[str, list[dict[str, Any]]] = {}
        for comp in components:
            members = set(comp)
            samples: list[dict[str, Any]] = []
            for a, b in combinations(sorted(comp), 2):
                for t in range(t_start, t_end + 1):
                    for ch in channels:
                        cell = cells[(a, b, t, ch)]
                        if cell["flagged"]:
                            continue
                        ratio = cell["obs"] / cell["model"]
                        samples.append(
                            {
                                "antenna_a": a,
                                "antenna_b": b,
                                "t_idx": t,
                                "channel": ch,
                                "log_ratio": float(np.log(max(abs(ratio), 1e-12))),
                                "phase_ratio": float(np.angle(ratio)),
                            }
                        )
            samples_by_comp["".join(comp)] = samples

        component_diags: list[dict[str, Any]] = []
        max_condition: float | None = None
        eq_total = 0

        for comp in components:
            amp_matrix, phase_matrix, amp_vec, phase_vec = _component_system(
                samples_by_comp["".join(comp)], antennas, comp
            )
            eq_total += amp_matrix.shape[0]
            is_ref_comp = ref_component is not None and comp == ref_component
            anchor_idx = comp.index(ref_antenna) if is_ref_comp else 0

            expected_rank = max(0, len(comp) - 1)
            amp_sol = np.zeros(0)
            phase_sol = np.zeros(0)
            # SVD diagnostics run for every component (including solvable
            # subgraphs a dead reference cannot reach); solutions are only
            # accepted for the component containing the reference.
            if len(comp) > 1:
                amp_sol, svals_a, rank_a, cond_a = _solve_anchored_sum(
                    amp_matrix, amp_vec, anchor_idx
                )
                phase_sol, _svp, _rp, cond_p = _solve_anchored(
                    phase_matrix, phase_vec, anchor_idx, iterative_phase=True
                )
            else:
                svals_a = np.zeros(0)
                rank_a = 0
                cond_a = None
                cond_p = None
            component_diags.append(
                {
                    "nodes": comp,
                    "n_nodes": len(comp),
                    "n_edges": int(
                        sum(
                            1
                            for a, b in combinations(comp, 2)
                            if tuple(sorted((a, b))) in edges
                        )
                    ),
                    "is_reference_component": is_ref_comp,
                    "singular_values": [float(s) for s in svals_a],
                    "rank": int(rank_a),
                    "expected_rank": expected_rank,
                    "rank_deficiency": int(max(0, expected_rank - rank_a)),
                    "condition_number": cond_a,
                }
            )
            for cond in (cond_a, cond_p):
                if cond is not None:
                    max_condition = (
                        cond if max_condition is None else max(max_condition, cond)
                    )
            if not is_ref_comp:
                # disconnected from the reference: gauge relative to the
                # reference is unobservable; do not fabricate gains
                continue

            if len(comp) > 1:
                for i, name in enumerate(comp):
                    x_amp[name] = float(amp_sol[i])
                    p_phase[name] = float(wrap_phase(phase_sol[i]))

        if ref_component is not None and len(ref_component) > 1:
            for name in ref_component:
                gain = math.exp(x_amp.get(name, 0.0)) * complex(
                    math.cos(p_phase.get(name, 0.0)),
                    math.sin(p_phase.get(name, 0.0)),
                )
                gains[name] = gain

        uncalibrated = sorted(a for a in antennas if gains[a] is None)
        ref_diag = next(
            (d for d in component_diags if d["is_reference_component"]), None
        )
        rank_deficiency = int(ref_diag["rank_deficiency"]) if ref_diag else 0
        diag = GroupDiagnostic(
            group_index=g_index,
            channels=channels,
            components=components,
            ref_component=ref_component,
            uncalibrated=uncalibrated,
            singular_values=sorted(
                (s for d in component_diags for s in d["singular_values"]),
                reverse=True,
            ),
            rank=int(sum(d["rank"] for d in component_diags)),
            n_equations=eq_total,
            n_unknowns=int(max(0, len(antennas) - len(components))),
            condition_number=max_condition,
            rank_deficiency=rank_deficiency,
            ref_has_data=bool(ref_has_data),
        )
        diag_dict = vars(diag)
        diag_dict["component_diagnostics"] = component_diags
        group_diags.append(diag_dict)
        gains_by_group.append(gains)
        if ref_component is not None:
            overall_uncalibrated -= set(ref_component)

    # residuals for the window/frequency layout (flagged kept but null values)
    residual_rows: list[dict[str, Any]] = []
    for g_index, (g0, g1, _edges) in enumerate(group_edges):
        gains = gains_by_group[g_index]
        for a, b in combinations(antennas, 2):
            for t in range(t_start, t_end + 1):
                for ch in range(g0, g1 + 1):
                    cell = cells[(a, b, t, ch)]
                    entry: dict[str, Any] = {
                        "antenna_a": a,
                        "antenna_b": b,
                        "t_idx": t,
                        "channel": ch,
                        "group_index": g_index,
                        "flagged": cell["flagged"],
                    }
                    if cell["flagged"] or gains.get(a) is None or gains.get(b) is None:
                        entry.update(
                            residual_re=None,
                            residual_im=None,
                            phase_residual=None,
                            amp_residual=None,
                        )
                    else:
                        predicted = np.conj(gains[a]) * gains[b] * cell["model"]
                        residual = cell["obs"] - predicted
                        entry.update(
                            residual_re=float(residual.real),
                            residual_im=float(residual.imag),
                            phase_residual=float(
                                wrap_phase(np.angle(cell["obs"]) - np.angle(predicted))
                            ),
                            amp_residual=float(abs(cell["obs"]) - abs(predicted)),
                        )
                    residual_rows.append(entry)

    # raw closure phases (observable; independent of gain solution)
    closure_rows: list[dict[str, Any]] = []
    for triplet in combinations(antennas, 3):
        a, b, c = triplet
        for t in range(t_start, t_end + 1):
            for g_index, (g0, g1, _edges) in enumerate(group_edges):
                values = []
                for ch in range(g0, g1 + 1):
                    z_ab = cells[tuple(sorted((a, b))) + (t, ch)]["obs"]
                    z_bc = cells[tuple(sorted((b, c))) + (t, ch)]["obs"]
                    z_ac = cells[tuple(sorted((a, c))) + (t, ch)]["obs"]
                    z_ca = np.conj(z_ac)
                    values.append(closure_phase(z_ab, z_bc, z_ca))
                # average on the unit circle so channels straddling the
                # negative-real-axis branch combine correctly
                circular_mean = float(
                    np.angle(np.mean(np.exp(1j * np.asarray(values))))
                )
                closure_rows.append(
                    {
                        "triplet": "-".join(triplet),
                        "t_idx": t,
                        "group_index": g_index,
                        "closure_raw": circular_mean,
                    }
                )

    any_ref_data = any(d["ref_has_data"] for d in group_diags)
    if not any_ref_data:
        status = "failed_ref_no_data"
    elif any(d["rank_deficiency"] > 0 for d in group_diags):
        status = "degenerate"
    elif overall_uncalibrated:
        status = "partial"
    else:
        status = "ok"

    result.status = status
    result.groups = [
        {
            "group_index": g_index,
            "channels": diag["channels"],
            "gains": {
                name: (
                    None
                    if gains_by_group[g_index][name] is None
                    else {
                        "re": float(gains_by_group[g_index][name].real),
                        "im": float(gains_by_group[g_index][name].imag),
                        "amp": float(abs(gains_by_group[g_index][name])),
                        "phase": float(np.angle(gains_by_group[g_index][name])),
                    }
                )
                for name in antennas
            },
        }
        for g_index, diag in enumerate(group_diags)
    ]
    result.diagnostics = {
        "status": status,
        "ref_antenna": ref_antenna,
        "ref_has_data": any_ref_data,
        "solvable_subgraph": group_diags[0]["components"]
        if group_diags
        else [],
        "uncalibrated_antennas": sorted(overall_uncalibrated),
        "groups": group_diags,
    }
    result.residuals = residual_rows
    result.closures = closure_rows
    return result
