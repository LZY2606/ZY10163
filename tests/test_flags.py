from __future__ import annotations

from calib.flags import Flag, build_match_map, suggest_flags


class Row(dict):
    pass


def _rows(payload):
    return [Row(**r) for r in payload["visibilities"]]


def _a3_flag():
    return Flag(
        id=0,
        source="manual",
        status="applied",
        reason="a3",
        antenna_a="A3",
        antenna_b=None,
        t_start=None,
        t_end=None,
        ch_start=None,
        ch_end=None,
    )


def test_interval_stacks_multiple_sources(fixture_payload):
    extra = Flag(
        id=1,
        source="auto",
        status="applied",
        reason="rfi",
        antenna_a="A0",
        antenna_b="A3",
        t_start=0,
        t_end=2,
        ch_start=0,
        ch_end=3,
    )
    match_map = build_match_map(
        [_a3_flag(), extra],
        ["A0", "A1", "A2", "A3"],
        6,
        8,
    )
    covered = match_map[("A0", "A3", 1, 2)]
    reasons = {f.reason for f in covered}
    assert reasons == {"a3", "rfi"}
    assert ("A0", "A1", 0, 0) not in match_map


def test_suggestions_cover_rfi_and_phase_jump(fixture_payload):
    meta = fixture_payload["meta"]
    suggestions = suggest_flags(
        _rows(fixture_payload),
        [_a3_flag()],
        meta["ntime"],
        meta["nchan"],
    )
    rfi = [s for s in suggestions if "RFI" in s["reason"]]
    jump = [s for s in suggestions if "突跳" in s["reason"]]
    quality = [s for s in suggestions if "quality" in s["reason"]]
    assert rfi and jump and quality
    rfi_cells = {
        (s["antenna_a"], s["antenna_b"], t, c)
        for s in rfi
        for t in range(s["t_start"], s["t_end"] + 1)
        for c in range(s["ch_start"], s["ch_end"] + 1)
    }
    assert ("A0", "A2", 2, 5) in rfi_cells
    assert ("A0", "A2", 2, 6) in rfi_cells


def test_suggestions_do_not_overlap_existing_flags(fixture_payload):
    meta = fixture_payload["meta"]
    # if A3 were not flagged the detector would never see those baselines
    suggestions = suggest_flags(
        _rows(fixture_payload),
        [_a3_flag()],
        meta["ntime"],
        meta["nchan"],
    )
    assert all(s["antenna_a"] != "A3" and s["antenna_b"] != "A3"
               for s in suggestions)
