from __future__ import annotations

import math

import numpy as np

from calib.fixture import (
    ANTENNAS,
    NCHAN,
    NTIME,
    build_fixture,
    source_vis,
    true_gain,
)


def test_fixture_shape_and_determinism(fixture_payload):
    again = build_fixture()
    assert len(again["visibilities"]) == 6 * NTIME * NCHAN
    assert fixture_payload["meta"]["seed"] == again["meta"]["seed"]
    first = {
        (r["antenna_a"], r["antenna_b"], r["t_idx"], r["channel"]): (r["re"], r["im"])
        for r in again["visibilities"]
    }
    second = {
        (r["antenna_a"], r["antenna_b"], r["t_idx"], r["channel"]): (r["re"], r["im"])
        for r in build_fixture()["visibilities"]
    }
    assert first == second


def test_fixture_has_four_antennas_and_a3_manual_flag(fixture_payload):
    assert [a["name"] for a in fixture_payload["antennas"]] == ANTENNAS
    a3 = [f for f in fixture_payload["flags"] if f["antenna_a"] == "A3"]
    assert len(a3) == 1
    assert a3[0]["source"] == "manual"


def test_source_phase_wraps_across_negative_real_axis():
    # phase slope 0.95/channel: consecutive channels cross the +/-pi branch
    phases = [np.angle(source_vis(0, ch)) for ch in range(NCHAN)]
    diffs = [float(wrap(phases[i + 1] - phases[i])) for i in range(NCHAN - 1)]
    # wrapped consecutive phase difference stays the unwrapped slope
    assert all(abs(d - 0.95) < 1e-9 for d in diffs)
    raw = [math.atan2(source_vis(0, ch).imag, source_vis(0, ch).real)
           for ch in range(NCHAN)]
    assert min(raw) < 0 < max(raw)  # branch cut is crossed


def wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def test_phase_jump_is_present_from_t_four():
    before = np.angle(true_gain(1, 3))
    after = np.angle(true_gain(1, 4))
    assert abs(wrap(after - before) - math.radians(70)) < 1e-9
