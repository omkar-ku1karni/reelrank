import math

import numpy as np
import polars as pl
import pytest

from reelrank.evaluation.metrics import (
    bootstrap_ci,
    category_entropy,
    exposure_summary,
    gini,
    intra_list_diversity,
    paired_bootstrap,
    ranking_metrics,
    topk,
)
from reelrank.ranking.metrics import calibration, impression_metrics


def test_topk_orders_and_breaks_ties_by_lower_index():
    s = np.array([[0.1, 0.9, 0.9, -np.inf, 0.5]])
    assert topk(s, 3).tolist() == [[1, 2, 4]]
    assert topk(s, 10).shape == (1, 5)  # k larger than catalog is clipped


def test_recall_precision_ndcg_mrr_hand_computed():
    # one user, 6 items; relevant (grade >= 2): items 1 (grade 3) and 4 (grade 2); item 2 is partial (grade 1)
    rel = np.array([[0, 3, 1, 0, 2, 0]], dtype=np.int8)
    rec = np.array([[2, 1, 0, 4]])
    m = ranking_metrics(rec, rel, ks=[2, 4], ndcg_ks=[2, 4])
    assert m["recall@2"][0] == pytest.approx(0.5)
    assert m["recall@4"][0] == pytest.approx(1.0)
    assert m["precision@2"][0] == pytest.approx(0.5)
    assert m["mrr"][0] == pytest.approx(0.5)  # first relevant at rank 2
    dcg4 = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3) + 0 + (2**2 - 1) / math.log2(5)
    idcg4 = (2**3 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3) + (2**1 - 1) / math.log2(4)
    assert m["ndcg@4"][0] == pytest.approx(dcg4 / idcg4)


def test_users_without_relevant_items_are_excluded_not_zero():
    rel = np.zeros((2, 4), dtype=np.int8)
    rel[0, 1] = 2
    m = ranking_metrics(np.array([[1, 0], [0, 1]]), rel, [2], [2])
    assert m["recall@2"][0] == 1.0 and np.isnan(m["recall@2"][1])
    assert np.isnan(m["ndcg@2"][1])


def test_empty_slots_count_as_misses():
    rel = np.array([[0, 2, 0]], dtype=np.int8)
    m = ranking_metrics(np.array([[-1, 1]]), rel, [2], [2])
    assert m["recall@2"][0] == 1.0
    assert m["ndcg@2"][0] == pytest.approx((1 / math.log2(3)) / 1.0)


def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.array([0, 0, 0, 10.0])) == pytest.approx(0.75)


def test_diversity_metrics():
    rec = np.array([[0, 1, 2], [0, 0, -1]])
    cat = np.array([1, 1, 2])
    ild = intra_list_diversity(rec, item_cat=cat)
    assert ild[0] == pytest.approx(2 / 3)
    assert ild[1] == pytest.approx(0.0)
    ent = category_entropy(rec, cat)
    assert ent[0] == pytest.approx(-(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3)))
    expo = exposure_summary(np.array([[0, 1]]), 4, np.array([10.0, 1, 1, 1]), np.ones(4, bool),
                            np.array([False, True, True, True]))
    assert expo["coverage"] == 0.5 and expo["long_tail_share"] == 0.5


def test_bootstrap_ci_contains_mean_and_paired_delta():
    rng = np.random.default_rng(0)
    a = rng.normal(0.3, 0.1, 500)
    b = a + 0.05
    mean, lo, hi = bootstrap_ci(a, 300, seed=1)
    assert lo < mean < hi
    d = paired_bootstrap(a, b, 300, seed=1)
    assert d["delta"] == pytest.approx(0.05) and d["lo"] == pytest.approx(0.05) and d["users"] == 500


def _impressions(grades, scores, wr=None):
    n = len(grades)
    return pl.DataFrame({"user_id": [1] * n, "day": [10] * n, "grade": grades,
                         "wr": wr or [0.5] * n, "complete": [0] * n, "skip": [0] * n, "duration_ms": [1000] * n}), \
        np.array(scores, dtype=float)


def test_impression_metrics_perfect_and_reversed_orderings():
    df, s = _impressions([3, 2, 0, 1], [4, 3, 1, 2])
    assert impression_metrics(df, s)["ndcg@20"] == pytest.approx(1.0)
    df, s = _impressions([3, 2, 0, 1], [1, 2, 4, 3])
    m = impression_metrics(df, s)
    assert m["ndcg@20"] < 1.0 and m["mrr"] == pytest.approx(1 / 3)


def test_calibration_perfect_predictor_has_zero_ece():
    y = np.array([0, 0, 1, 1])
    c = calibration(np.array([0.0, 0.0, 1.0, 1.0]), y)
    assert c["ece"] == pytest.approx(0.0) and c["brier"] == pytest.approx(0.0)
