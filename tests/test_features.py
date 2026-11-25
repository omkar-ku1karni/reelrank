"""Data QA and feature-cutoff (leakage) tests on the synthetic fixture."""

import numpy as np
import polars as pl
import pytest

from reelrank.config import read_json
from reelrank.evaluation.protocol import load_protocol


def _load(cfg):
    f = cfg.processed / "features"
    ev = pl.read_parquet(f / "events.parquet")
    return f, ev, np.load(f / "snap_days.npy")


def test_ingest_dedupes_and_counts(fixture_cfg):
    qa = read_json(fixture_cfg.processed / "qa_report.json")["tables"]
    t = qa["interactions"]
    assert t["exact_duplicate_rows_removed"] >= 1
    assert t["rows_after_cleaning"] == t["raw_rows"] - t["exact_duplicate_rows_removed"] - t["same_key_conflicts_removed"]
    assert qa["dense"]["pairs_also_in_big_matrix"] == 0
    assert qa["dense"]["null_timestamp"] > 0
    assert qa["items"]["caption_rows_malformed_dropped"] == 1
    ev = pl.read_parquet(fixture_cfg.processed / "features" / "events.parquet")
    assert ev["event_id"].is_unique().all()
    assert ev.select("user_id", "video_id", "ts").is_duplicated().sum() == 0


def test_user_snapshot_uses_only_strictly_earlier_days(fixture_cfg):
    f, ev, snap_days = _load(fixture_cfg)
    stats = np.load(f / "user_stats.npy")
    hist = np.load(f / "user_hist.npy")
    rng = np.random.default_rng(0)
    for s in rng.choice(len(snap_days), 6, replace=False):
        d = int(snap_days[s])
        for u in rng.choice(ev["user_id"].unique().to_numpy(), 5, replace=False):
            prior = ev.filter((pl.col("user_id") == u) & (pl.col("day") < d))
            assert stats[s, u, 0] == pytest.approx(np.log1p(prior.height), abs=1e-5)
            if prior.height:
                assert stats[s, u, 2] == pytest.approx(prior["wr_c"].mean(), rel=1e-4)
            pos = prior.filter(pl.col("pos")).sort(["ts", "video_id"])["video_id"].to_numpy()
            h = hist[s, u]
            h = h[h >= 0]
            assert h.tolist() == pos[-len(h):].tolist() if len(h) else len(pos) == 0
            # nothing from the snapshot day or later can appear in the history
            later = set(ev.filter((pl.col("user_id") == u) & (pl.col("day") >= d) & pl.col("pos"))["video_id"].to_list())
            earlier = set(pos.tolist())
            assert set(h.tolist()) <= earlier
            assert not (set(h.tolist()) & (later - earlier))


def test_item_stats_exclude_cutoff_day(fixture_cfg):
    f, ev, snap_days = _load(fixture_cfg)
    pop = np.load(f / "item_pop_before.npy")
    s = len(snap_days) // 2
    d = int(snap_days[s])
    cnt = np.bincount(ev.filter(pl.col("day") < d)["video_id"].to_numpy(), minlength=pop.shape[1])
    np.testing.assert_array_equal(pop[s], cnt)


def test_eval_targets_are_novel_eligible_positives(fixture_cfg):
    p = load_protocol(fixture_cfg, "test")
    _, ev, _ = _load(fixture_cfg)
    seen = ev.filter(pl.col("day") < p.cutoff)
    r, c = np.nonzero(p.rel >= 2)
    users = p.users[r]
    pairs = set(zip(seen["user_id"].to_list(), seen["video_id"].to_list()))
    assert not any((int(u), int(i)) in pairs for u, i in zip(users, c))
    assert p.candidate_mask[c].all()


def test_mask_scores_removes_seen_and_ineligible(fixture_cfg):
    p = load_protocol(fixture_cfg, "test")
    u = p.users[:3]
    s = np.zeros((3, p.n_items), dtype=np.float32)
    p.mask_scores(s, u)
    assert np.isinf(s[:, ~p.candidate_mask]).all()
    for j, user in enumerate(u):
        assert np.isinf(s[j, p.seen[user].indices]).all()


def test_dense_protocol_labels_cover_dense_catalog(fixture_cfg):
    p = load_protocol(fixture_cfg, "dense")
    assert p.candidate_mask.sum() == 60
    assert ((p.rel[:, ~p.candidate_mask]) == 0).all()


def test_rolling_window_feature_matches_brute_force(fixture_cfg):
    f, ev, snap_days = _load(fixture_cfg)
    rec = np.load(f / "user_recent_wr.npy")
    rng = np.random.default_rng(1)
    for s in rng.choice(len(snap_days), 8, replace=False):
        d = int(snap_days[s])
        for u in rng.choice(ev["user_id"].unique().to_numpy(), 6, replace=False):
            prior = ev.filter((pl.col("user_id") == u) & (pl.col("day") < d)).sort(["ts", "event_id"])
            expected = prior["wr_c"].tail(20).mean() if prior.height else -1.0
            assert rec[s, u] == pytest.approx(expected, rel=1e-4, abs=1e-5)
