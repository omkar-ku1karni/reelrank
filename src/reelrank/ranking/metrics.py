"""Logged-impression ranking metrics: re-order each (user, day) group of *observed* events by a score.

These measure conditional ranking quality on items the logging policy actually showed; they say nothing about
unshown items. Reported per group, macro-averaged over groups with at least one positive grade.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def impression_metrics(df: pl.DataFrame, score: np.ndarray, top: int = 10) -> dict[str, float]:
    """NDCG@10/20 (graded), MRR (first grade >= 2), and observed outcomes of each group's top-`top`."""
    d = pl.DataFrame({"g": df["user_id"].cast(pl.Int64) * 100000 + df["day"].cast(pl.Int64),
                      "grade": df["grade"], "wr": df["wr"], "complete": df["complete"], "skip": df["skip"],
                      "dur": df["duration_ms"], "s": np.asarray(score, dtype=np.float64),
                      # deterministic tie-break: stable row order
                      "r": np.arange(df.height)})
    d = d.sort(["g", "s", "r"], descending=[False, True, False])
    d = d.with_columns(pos=pl.int_range(pl.len()).over("g") + 1)
    disc = 1.0 / np.log2(d["pos"].to_numpy() + 1)
    gain = (2.0 ** d["grade"].to_numpy() - 1.0)
    d = d.with_columns(dg=pl.Series(gain * disc))
    ideal = (d.select("g", "grade").sort(["g", "grade"], descending=[False, True])
             .with_columns(pos=pl.int_range(pl.len()).over("g") + 1))
    ideal = ideal.with_columns(ig=(2.0 ** pl.col("grade") - 1.0) / (pl.col("pos") + 1).log(2))
    out = {}
    agg_ideal = ideal.group_by("g").agg(i10=pl.col("ig").filter(pl.col("pos") <= 10).sum(),
                                        i20=pl.col("ig").filter(pl.col("pos") <= 20).sum())
    agg = d.group_by("g").agg(
        d10=pl.col("dg").filter(pl.col("pos") <= 10).sum(), d20=pl.col("dg").filter(pl.col("pos") <= 20).sum(),
        first=pl.col("pos").filter(pl.col("grade") >= 2).min(),
        top_wr=pl.col("wr").filter(pl.col("pos") <= top).mean(),
        top_complete=pl.col("complete").filter(pl.col("pos") <= top).cast(pl.Float64).mean(),
        top_skip=pl.col("skip").filter(pl.col("pos") <= top).cast(pl.Float64).mean(),
        top_dur=pl.col("dur").filter(pl.col("pos") <= top).cast(pl.Float64).mean(),
        n=pl.len()).join(agg_ideal, on="g")
    valid = agg.filter(pl.col("i20") > 0)
    out["ndcg@10"] = float((valid["d10"] / valid["i10"]).mean())
    out["ndcg@20"] = float((valid["d20"] / valid["i20"]).mean())
    out["mrr"] = float((1.0 / valid["first"].cast(pl.Float64)).fill_null(0.0).mean())
    out["top10_watch_ratio"] = float(agg["top_wr"].mean())
    out["top10_completion"] = float(agg["top_complete"].mean())
    out["top10_skip"] = float(agg["top_skip"].mean())
    out["top10_duration_s"] = float(agg["top_dur"].mean() / 1000)
    out["groups"] = int(valid.height)
    return out


def per_group_ndcg(df: pl.DataFrame, score: np.ndarray, k: int = 20) -> np.ndarray:
    """Per-group NDCG@k aligned to sorted group keys (for paired bootstraps between scorers)."""
    d = pl.DataFrame({"g": df["user_id"].cast(pl.Int64) * 100000 + df["day"].cast(pl.Int64), "grade": df["grade"],
                      "s": np.asarray(score, dtype=np.float64), "r": np.arange(df.height)})
    d = d.sort(["g", "s", "r"], descending=[False, True, False]).with_columns(
        pos=pl.int_range(pl.len()).over("g") + 1)
    d = d.with_columns(dg=(2.0 ** pl.col("grade") - 1.0) / (pl.col("pos") + 1).log(2))
    ideal = d.select("g", "grade").sort(["g", "grade"], descending=[False, True]).with_columns(
        pos=pl.int_range(pl.len()).over("g") + 1).with_columns(
        ig=(2.0 ** pl.col("grade") - 1.0) / (pl.col("pos") + 1).log(2))
    a = d.group_by("g").agg(dk=pl.col("dg").filter(pl.col("pos") <= k).sum()).join(
        ideal.group_by("g").agg(ik=pl.col("ig").filter(pl.col("pos") <= k).sum()), on="g").sort("g")
    return np.where(a["ik"].to_numpy() > 0, a["dk"].to_numpy() / np.maximum(a["ik"].to_numpy(), 1e-12), np.nan)


def calibration(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Expected calibration error, Brier score and a reliability table (equal-width bins)."""
    prob = np.clip(np.asarray(prob, dtype=float), 0, 1)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = prob[m].mean(), y[m].mean()
            ece += m.mean() * abs(conf - acc)
            table.append({"bin": b, "mean_pred": float(conf), "observed": float(acc), "n": int(m.sum())})
    return {"ece": float(ece), "brier": float(((prob - y) ** 2).mean()), "base_rate": float(y.mean()),
            "reliability": table}
