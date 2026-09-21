from __future__ import annotations


def _apply_all_suggestions(client):
    suggestions = client.post("/api/suggest").json()["suggestions"]
    for item in suggestions:
        assert client.post(f"/api/flags/{item['id']}/apply").status_code == 200
    return suggestions


def test_health_and_index(client):
    assert client.get("/api/health").json()["service"] == "视界标定室"
    page = client.get("/").text
    assert "视界标定室" in page


def test_state_and_data(client):
    state = client.get("/api/state").json()
    assert state["version"] == "fixture-v1"
    assert len(state["antennas"]) == 4
    data = client.get("/api/data").json()
    assert len(data["visibilities"]) == 288
    assert len(data["flags"]) == 1  # A3 manual flag


def test_full_solve_recovers_gains(client):
    _apply_all_suggestions(client)
    resp = client.post(
        "/api/solve",
        json={"ref_antenna": "A0", "t_start": 0, "t_end": 5, "group_size": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["status"] == "partial"  # A3 offline
    diag = body["result"]["diagnostics"]
    assert diag["uncalibrated_antennas"] == ["A3"]
    gains = body["result"]["groups"][0]["gains"]
    assert gains["A3"] is None
    assert abs(gains["A1"]["amp"] - 1.12) < 0.02
    assert abs(gains["A2"]["amp"] - 0.9) < 0.02


def test_failed_reference_diagnostics(client):
    resp = client.post(
        "/api/solve",
        json={"ref_antenna": "A3", "t_start": 0, "t_end": 5, "group_size": 2},
    )
    body = resp.json()
    assert body["result"]["status"] == "failed_ref_no_data"
    assert body["result"]["diagnostics"]["uncalibrated_antennas"] == [
        "A0",
        "A1",
        "A2",
    ]


def test_manual_flag_crud(client):
    created = client.post(
        "/api/flags",
        json={
            "source": "manual",
            "reason": "test interval",
            "antenna_a": "A0",
            "antenna_b": "A1",
            "t_start": 0,
            "t_end": 1,
            "ch_start": 0,
            "ch_end": 2,
        },
    ).json()
    flag_id = created["id"]
    flags = client.get("/api/data").json()["flags"]
    assert any(f["id"] == flag_id for f in flags)
    assert client.delete(f"/api/flags/{flag_id}").status_code == 200
    assert client.delete(f"/api/flags/{flag_id}").status_code == 404


def test_replay_ignores_later_flags(client):
    _apply_all_suggestions(client)
    run_id = client.post(
        "/api/solve",
        json={"ref_antenna": "A0", "t_start": 0, "t_end": 5, "group_size": 2},
    ).json()["run_id"]
    first = client.post(f"/api/runs/{run_id}/replay").json()
    assert first["checks"]["replay_matches"] is True
    assert first["checks"]["data_hash_match"] is True
    assert first["checks"]["flagset_hash_match"] is True

    # add a brand new flag AFTER the run was frozen
    client.post(
        "/api/flags",
        json={
            "source": "manual",
            "reason": "later addition",
            "antenna_a": "A0",
            "antenna_b": "A2",
            "t_start": 0,
            "t_end": 0,
            "ch_start": 0,
            "ch_end": 0,
        },
    )
    second = client.post(f"/api/runs/{run_id}/replay").json()
    assert second["checks"]["replay_matches"] is True
    snapshot_sources = {f["reason"] for f in second["snapshot_flags"]}
    assert "later addition" not in snapshot_sources


def test_export_run_payload(client):
    run_id = client.post(
        "/api/solve",
        json={"ref_antenna": "A0", "t_start": 0, "t_end": 5, "group_size": 2},
    ).json()["run_id"]
    export = client.get(f"/api/runs/{run_id}/export").json()
    assert set(export) == {"run", "flag_snapshot", "gains", "residuals", "closures"}
    assert export["run"]["ref_antenna"] == "A0"
    assert export["flag_snapshot"]  # A3 snapshot present


def test_reimport_clears_runs_and_flags(client):
    client.post(
        "/api/solve",
        json={"ref_antenna": "A0", "t_start": 0, "t_end": 5, "group_size": 2},
    )
    assert len(client.get("/api/runs").json()["runs"]) == 1
    out = client.post("/api/reimport").json()
    assert out["ok"] is True
    assert client.get("/api/runs").json()["runs"] == []
    state = client.get("/api/state").json()
    assert len(state["flags"]) == 1  # fixture flag restored


def test_invalid_reference_and_window(client):
    bad_ref = client.post(
        "/api/solve",
        json={"ref_antenna": "ZZ", "t_start": 0, "t_end": 5, "group_size": 2},
    )
    assert bad_ref.status_code == 400
    bad_window = client.post(
        "/api/solve",
        json={"ref_antenna": "A0", "t_start": 4, "t_end": 1, "group_size": 2},
    )
    assert bad_window.status_code == 400
