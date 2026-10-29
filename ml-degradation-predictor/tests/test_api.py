import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client(small_run, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "paths", small_run["paths"])
    main.artifacts.cache_clear()
    yield TestClient(main.app)
    main.artifacts.cache_clear()


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_booking_returns_probability(client):
    r = client.post("/predict/booking", json={"lead_time": 200, "deposit_type": "Non Refund"})
    assert r.status_code == 200 and 0 <= r.json()["cancellation_probability"] <= 1


def test_latest_risk_and_windows(client):
    r = client.get("/risk/latest").json()
    assert r["status"] in {"normal", "investigate", "high_risk"} and 0 <= r["risk"] <= 1
    assert len(client.get("/windows?limit=5").json()) == 5


def test_score_window_with_new_bookings(client, small_run):
    sample = small_run["bookings"].tail(300).drop(columns=["arrival_date", "period", "prediction", "window_id"])
    r = client.post("/risk/window", json={"bookings": sample.where(sample.notna(), None).to_dict("records"),
                                          "outcomes_known": True})
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["risk"] <= 1


def test_score_window_rejects_tiny_windows(client):
    assert client.post("/risk/window", json={"bookings": [{"lead_time": 1}] * 3}).status_code == 422
