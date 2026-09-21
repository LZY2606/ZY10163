from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("CALIB_DB", str(db_path))
    import app as appmod

    monkeypatch.setattr(appmod, "DB_PATH", db_path)
    from fastapi.testclient import TestClient

    with TestClient(appmod.app) as c:
        yield c


@pytest.fixture()
def fixture_payload():
    from calib.fixture import load_fixture

    return load_fixture()
