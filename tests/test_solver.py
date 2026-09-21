from __future__ import annotations

import math

import numpy as np

from calib.fixture import ANTENNAS, TRUE_GAINS
from calib.flags import Flag
from calib.solver import closure_phase, solve, wrap_phase


class Row(dict):
    pass


def _rows(payload):
    return [Row(**r) for r in payload["visibilities"]]


def _a3_flag():
    return Flag(
        id=0,
        source="manual",
        status="applied",
        reason="a3 offline",
        antenna_a="A3",
        antenna_b=None,
        t_start=None,
        t_end=None,
        ch_start=None,
        ch_end=None,
    )


def test_clean_window_recovers_true_gains(fixture_payload):
    meta = fixture_payload["meta"]
    res = solve(
        ANTENNAS,
        _rows(fixture_payload),
        [_a3_flag()],
        meta["ntime"],
        meta["nchan"],
        "A0",
        0,
        0,
        1,
    )
    assert res.status == "partial"
    gains = res.groups[0]["gains"]
    for name, (amp, phase) in TRUE_GAINS.items():
        if name == "A3":
            assert gains[name] is None
            continue
        assert abs(gains[name]["amp"] - amp) < 0.01
        assert abs(wrap_phase(gains[name]["phase"] - phase)) < 0.01


def test_fully_flagged_reference_returns_diagnostics_not_zero_gains(
    fixture_payload,
):
    meta = fixture_payload["meta"]
    res = solve(
        ANTENNAS,
        _rows(fixture_payload),
        [_a3_flag()],
        meta["ntime"],
        meta["nchan"],
        "A3",
        0,
        5,
        2,
    )
    assert res.status == "failed_ref_no_data"
    assert res.diagnostics["ref_has_data"] is False
    assert res.diagnostics["uncalibrated_antennas"] == ["A0", "A1", "A2"]
    # the still-observable subgraph is reported separately
    assert ["A0", "A1", "A2"] in res.diagnostics["solvable_subgraph"]
    assert ["A3"] in res.diagnostics["solvable_subgraph"]
    # no fabricated gains anywhere
    assert all(
        gain is None
        for group in res.groups
        for gain in group["gains"].values()
    )
    diag = res.diagnostics["groups"][0]
    assert diag["n_equations"] > 0
    assert diag["condition_number"] is not None


def test_solver_is_independent_of_import_order(fixture_payload):
    meta = fixture_payload["meta"]
    rows_a = _rows(fixture_payload)
    rng = np.random.default_rng(7)
    rows_b = list(rows_a)
    rng.shuffle(rows_b)
    kwargs = dict(
        antennas=ANTENNAS,
        flags=[_a3_flag()],
        ntime=meta["ntime"],
        nchan=meta["nchan"],
        ref_antenna="A0",
        t_start=0,
        t_end=3,
        group_size=2,
    )
    ra = solve(rows=rows_a, **kwargs)
    rb = solve(rows=rows_b, **kwargs)
    assert ra.status == rb.status
    for ga, gb in zip(ra.groups, rb.groups):
        for name in ANTENNAS:
            a, b = ga["gains"][name], gb["gains"][name]
            assert (a is None) == (b is None)
            if a is not None:
                assert abs(a["re"] - b["re"]) < 1e-12
                assert abs(a["im"] - b["im"]) < 1e-12
    assert ra.diagnostics["solvable_subgraph"] == rb.diagnostics["solvable_subgraph"]


def test_disconnected_reference_subgraph_uses_nulls(fixture_payload):
    meta = fixture_payload["meta"]
    flags = [_a3_flag()]
    # isolate A2 as well (flag every baseline touching A2), keep A3 off
    flags.append(
        Flag(
            id=1,
            source="manual",
            status="applied",
            reason="isolate A2",
            antenna_a="A2",
            antenna_b=None,
            t_start=None,
            t_end=None,
            ch_start=None,
            ch_end=None,
        )
    )
    res = solve(
        ANTENNAS,
        _rows(fixture_payload),
        flags,
        meta["ntime"],
        meta["nchan"],
        "A0",
        0,
        5,
        2,
    )
    assert res.status == "partial"
    gains = res.groups[0]["gains"]
    assert gains["A2"] is None
    assert gains["A3"] is None
    assert gains["A0"] is not None and gains["A1"] is not None
    assert "A2" in res.diagnostics["uncalibrated_antennas"]


def test_wrap_phase_handles_branch_cut():
    assert abs(float(wrap_phase(np.pi)) - np.pi) < 1e-9
    assert abs(float(wrap_phase(-np.pi)) - (-np.pi)) < 1e-9
    assert abs(float(wrap_phase(np.pi + 0.2)) - (-np.pi + 0.2)) < 1e-9
    assert abs(float(wrap_phase(3 * np.pi + 0.1)) - (-np.pi + 0.1)) < 1e-9


def test_closure_phase_uses_complex_product():
    z1 = complex(math.cos(3.0), math.sin(3.0))
    z2 = complex(math.cos(3.0), math.sin(3.0))
    z3 = complex(math.cos(3.0), math.sin(3.0))
    expected = 9.0 - 2 * math.pi  # 9 rad lies in the second turn
    assert abs(closure_phase(z1, z2, z3) - expected) < 1e-12


def test_residuals_keep_flagged_cells_locatable(fixture_payload):
    meta = fixture_payload["meta"]
    res = solve(
        ANTENNAS,
        _rows(fixture_payload),
        [_a3_flag()],
        meta["ntime"],
        meta["nchan"],
        "A0",
        0,
        1,
        2,
    )
    flagged = [r for r in res.residuals if r["flagged"]]
    assert flagged
    assert all(r["residual_re"] is None for r in flagged)
    assert {(r["antenna_a"], r["antenna_b"]) for r in flagged} <= {
        ("A0", "A3"),
        ("A1", "A3"),
        ("A2", "A3"),
    }
