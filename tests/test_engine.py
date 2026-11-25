"""Serving contracts on the smoke-run artifacts: alignment, fallbacks, uniqueness, feature parity, refusals."""

import shutil

import numpy as np
import polars as pl
import pytest

from reelrank.config import load_config
from reelrank.ranking.dataset import FEATURES
from reelrank.serving.engine import Engine, ManifestError


@pytest.fixture(scope="module")
def eng(smoke_artifacts):
    return Engine(smoke_artifacts)


def test_recommendations_unique_ranked_and_eligible(eng):
    for u in (0, 5, 17):
        res = eng.recommend(u, k=15)
        ids = [it["video_id"] for it in res["items"]]
        assert len(ids) == len(set(ids)) <= 15
        assert [it["rank"] for it in res["items"]] == list(range(1, len(ids) + 1))
        assert all(eng.eligible[i] for i in ids)
        assert not set(ids) & set(eng.seen[u].indices.tolist())  # novel feed
        assert res["model_version"] == eng.manifest["model_version"]


def test_unknown_user_falls_back_to_trending(eng):
    res = eng.recommend(10**7, k=5)
    assert res["fallback"] == "unknown_user" and len(res["items"]) == 5
    assert all(it["retrieval_sources"] == ["trending"] for it in res["items"])


def test_invalid_k_and_algorithm(eng):
    with pytest.raises(ValueError):
        eng.recommend(1, k=0)
    with pytest.raises(ValueError):
        eng.recommend(1, algorithm="nope")


def test_k_larger_than_pool_returns_fewer(eng):
    res = eng.recommend(3, k=5000)
    assert 0 < len(res["items"]) < 5000


def test_every_algorithm_runs(eng):
    for a in ("reelrank", "ranker_only", "two_tower", "als", "trending", "popularity"):
        assert eng.recommend(2, k=5, algorithm=a)["items"]


def test_serving_features_match_offline_pipeline(eng, smoke_artifacts):
    """The engine's ranker features equal those the offline pipeline computed for the same candidates."""
    cfg = load_config("configs/smoke.yaml")
    cand = pl.read_parquet(cfg.processed / "ranker" / "test_candidates.parquet")
    for u in cand["user_id"].unique().sort().to_list()[:5]:
        sub = cand.filter(pl.col("user_id") == u)
        items = sub["video_id"].to_numpy().astype(np.int64)
        X = eng.features(u, items, eng.user_vector(u))
        np.testing.assert_allclose(X, sub.select(FEATURES).to_numpy(), rtol=1e-4, atol=1e-4)


def test_candidates_match_offline_generator(eng):
    cfg = load_config("configs/smoke.yaml")
    cand = pl.read_parquet(cfg.processed / "candidates" / "test.parquet")
    u = int(cand["user_id"][0])
    offline = set(cand.filter(pl.col("user_id") == u)["video_id"].to_list())
    items, _, _ = eng.candidates(u)
    assert set(items.tolist()) == offline


def test_similar_excludes_query_and_404s(eng):
    res = eng.similar(3, k=5)
    assert 3 not in [it["video_id"] for it in res["items"]]
    with pytest.raises(KeyError):
        eng.similar(-1)


def test_manifest_mismatch_is_refused(smoke_artifacts, tmp_path):
    dst = tmp_path / "artifacts"
    shutil.copytree(smoke_artifacts / "serving", dst / "serving")
    shutil.copy(smoke_artifacts / "manifest.json", dst / "manifest.json")
    vec = np.load(dst / "serving" / "item_vectors.npy")
    np.save(dst / "serving" / "item_vectors.npy", vec[::-1])  # same shape, different model
    with pytest.raises(ManifestError, match="does not match"):
        Engine(dst)


def test_empty_catalog_returns_empty_feed(smoke_artifacts):
    e = Engine(smoke_artifacts)
    e.eligible[:] = False
    e.trend_order = e.trend_order[:0]
    e.cat_pools = {}
    res = e.recommend(1, k=10)
    assert res["items"] == []


def test_monitoring_counters(eng):
    eng.recommend(1, k=5)
    m = eng.monitoring()
    assert m["requests"] >= 1 and m["latency_p95_ms"] > 0


def test_every_algorithm_returns_only_eligible_videos(eng):
    for a in ("reelrank", "trending", "popularity", "two_tower", "als"):
        for u in (0, 7, 10**7):
            assert all(eng.eligible[it["video_id"]] for it in eng.recommend(u, k=50, algorithm=a)["items"])


def test_served_ranker_scores_equal_offline_scores(eng):
    """The engine's ranker reproduces the offline candidate scores of whichever ranker the manifest names."""
    cfg = load_config("configs/smoke.yaml")
    name = eng.manifest["ranker"]
    src = "scores_nn_test" if name == "nn_utility" else "scores_lgb_test"
    if eng.ranker_kind.startswith("feature:"):
        pytest.skip("a raw retrieval score was selected")
    cand = pl.read_parquet(cfg.processed / "ranker" / "test_candidates.parquet")
    scores = pl.read_parquet(cfg.processed / "ranker" / f"{src}.parquet")
    u = int(cand["user_id"][0])
    sub = cand.filter(pl.col("user_id") == u).join(scores, on=["user_id", "video_id"])
    items = sub["video_id"].to_numpy().astype(np.int64)
    got = eng.rank_scores(sub.select(FEATURES).to_numpy().astype(np.float32), items, eng.user_vector(u))
    np.testing.assert_allclose(got, sub[name].to_numpy(), rtol=1e-3, atol=1e-4)
