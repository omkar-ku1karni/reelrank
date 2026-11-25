"""Ranking, diversity and uncertainty metrics. Pure NumPy so every process (torch, FAISS, LightGBM) can use them.

Definitions (all per user, then macro-averaged over users that have at least one relevant item):
  Recall@K    = |relevant ∩ top-K| / |relevant|
  Precision@K = |relevant ∩ top-K| / K
  DCG@K       = Σ_{r=1..K} (2^grade_r - 1) / log2(r + 1);  NDCG@K = DCG@K / ideal DCG@K
  MRR         = 1 / rank of the first relevant item (0 if none in the list)
"""

from __future__ import annotations

import numpy as np


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores per row, best first; ties break towards the lower item index."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(scores, part, axis=1)
    order = np.lexsort((part, -vals), axis=1)
    return np.take_along_axis(part, order, axis=1)


def ranking_metrics(rec: np.ndarray, rel: np.ndarray, ks: list[int], ndcg_ks: list[int],
                    valid: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-user metrics.

    rec:   [n, K] recommended item indices (best first); -1 marks an empty slot.
    rel:   [n, I] graded relevance (0 = not relevant). Binary relevance is grade >= 2 (watch_ratio >= 0.8).
    valid: [n, K] optional mask; False slots count as misses.
    """
    n, K = rec.shape
    safe = np.where(rec >= 0, rec, 0)
    g = np.take_along_axis(rel, safe, axis=1).astype(np.float64)
    g[rec < 0] = 0
    if valid is not None:
        g[~valid] = 0
    hit = g >= 2
    n_rel = (rel >= 2).sum(1)
    out: dict[str, np.ndarray] = {"n_relevant": n_rel}
    cum = np.cumsum(hit, axis=1)
    for k in ks:
        kk = min(k, K)
        h = cum[:, kk - 1]
        out[f"recall@{k}"] = np.where(n_rel > 0, h / np.maximum(n_rel, 1), np.nan)
        out[f"precision@{k}"] = h / k
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    gains = (2.0 ** g - 1.0) * disc
    ideal_sorted = -np.sort(-rel.astype(np.float64), axis=1)[:, :K]
    ideal = (2.0 ** ideal_sorted - 1.0) * disc[: ideal_sorted.shape[1]]
    for k in ndcg_ks:
        kk = min(k, K)
        dcg = gains[:, :kk].sum(1)
        idcg = ideal[:, :kk].sum(1)
        out[f"ndcg@{k}"] = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), np.nan)
    first = np.where(hit.any(1), hit.argmax(1) + 1, 0)
    out["mrr"] = np.where(n_rel > 0, np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0), np.nan)
    return out


def ndcg_from_lists(gains_per_list: list[np.ndarray], ideal_per_list: list[np.ndarray], k: int) -> np.ndarray:
    """NDCG@k for variable-length lists of grades (e.g. observed impressions per user-day)."""
    out = np.full(len(gains_per_list), np.nan)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for i, (g, ideal) in enumerate(zip(gains_per_list, ideal_per_list)):
        gg = (2.0 ** np.asarray(g[:k], dtype=float) - 1) @ disc[: min(k, len(g))]
        ii = np.sort(np.asarray(ideal, dtype=float))[::-1][:k]
        idcg = (2.0 ** ii - 1) @ disc[: len(ii)]
        out[i] = gg / idcg if idcg > 0 else np.nan
    return out


# ---------------------------------------------------------------------------------------------- diversity
def intra_list_diversity(rec: np.ndarray, item_emb: np.ndarray | None = None,
                         item_cat: np.ndarray | None = None) -> np.ndarray:
    """Mean pairwise dissimilarity in each list: 1 - cosine (embeddings) or 1[different category]."""
    out = np.full(len(rec), np.nan)
    for i, r in enumerate(rec):
        r = r[r >= 0]
        m = len(r)
        if m < 2:
            continue
        if item_emb is not None:
            e = item_emb[r]
            e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            sim = e @ e.T
        else:
            c = item_cat[r]
            sim = (c[:, None] == c[None, :]).astype(float)
        out[i] = (1 - sim)[np.triu_indices(m, 1)].mean()
    return out


def category_entropy(rec: np.ndarray, item_cat: np.ndarray) -> np.ndarray:
    out = np.zeros(len(rec))
    for i, r in enumerate(rec):
        r = r[r >= 0]
        if len(r) == 0:
            continue
        _, c = np.unique(item_cat[r], return_counts=True)
        p = c / c.sum()
        out[i] = -(p * np.log2(p)).sum()
    return out


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative exposure vector (0 = equal exposure, 1 = one item gets everything)."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum.sum() / cum[-1])) / n)


def exposure_summary(rec: np.ndarray, n_items: int, pop: np.ndarray, catalog: np.ndarray,
                     tail_mask: np.ndarray) -> dict[str, float]:
    """Catalog coverage, exposure Gini, long-tail share and novelty (mean -log2 popularity share)."""
    flat = rec[rec >= 0].ravel()
    expo = np.bincount(flat, minlength=n_items).astype(float)
    p = (pop + 1) / (pop + 1)[catalog].sum()
    return {
        "coverage": float((expo[catalog] > 0).sum() / catalog.sum()),
        "exposure_gini": gini(expo[catalog]),
        "long_tail_share": float(tail_mask[flat].mean()) if len(flat) else 0.0,
        "novelty_bits": float((-np.log2(p[flat])).mean()) if len(flat) else 0.0,
    }


# --------------------------------------------------------------------------------------------- uncertainty
def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile CI over users (NaN users dropped)."""
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0,
                     alpha: float = 0.05) -> dict[str, float]:
    """Paired user bootstrap of mean(b - a) on users where both are defined."""
    m = ~(np.isnan(a) | np.isnan(b))
    d = b[m] - a[m]
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "users": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(0, n_boot, 100):
        idx = rng.integers(0, len(d), size=(min(100, n_boot - i), len(d)))
        means[i: i + len(idx)] = d[idx].mean(1)
    base = float(a[m].mean())
    return {"delta": float(d.mean()), "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)), "users": int(m.sum()),
            "relative": float(d.mean() / base) if base else float("nan"),
            "p_le_0": float((means <= 0).mean())}
