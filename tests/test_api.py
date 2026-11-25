import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(smoke_artifacts, tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setenv("REELRANK_ARTIFACTS", str(smoke_artifacts))
    monkeypatch_module.setenv("REELRANK_LOG_DIR", str(tmp_path_factory.mktemp("logs")))
    import reelrank.serving.api as api

    importlib.reload(api)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_recommendations_contract(client):
    r = client.get("/recommendations/4", params={"k": 10, "explain": True})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 10 and len(body["items"]) <= 10
    it = body["items"][0]
    for key in ("video_id", "category", "retrieval_sources", "predicted_engagement", "final_score", "rank"):
        assert key in it
    assert "pool_by_source" in body["explain"]
    assert "X-Response-Time-ms" in r.headers


@pytest.mark.parametrize("params", [{"k": 0}, {"k": 101}, {"algorithm": "bogus"}, {"lam": 1.5}])
def test_invalid_parameters_are_422(client, params):
    assert client.get("/recommendations/4", params=params).status_code == 422


def test_unknown_user_is_served_with_fallback(client):
    r = client.get("/recommendations/99999999")
    assert r.status_code == 200 and r.json()["fallback"] == "unknown_user"


def test_similar(client):
    assert client.get("/videos/3/similar", params={"k": 4}).status_code == 200
    assert client.get("/videos/999999/similar").status_code == 404


def test_metrics_and_feedback(client):
    assert "monitoring (live requests, simulation only)" in client.get("/models/metrics").json()
    r = client.post("/feedback", json={"user_id": 4, "video_id": 3, "event": "complete", "watch_ratio": 1.2})
    assert r.status_code == 202 and r.json()["accepted"]
    assert client.post("/feedback", json={"user_id": 4, "video_id": 3, "event": "teleport"}).status_code == 422
