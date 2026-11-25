"""Offline monitoring simulation (LightGBM-side process). This is a replay, not production monitoring.

    python -m reelrank.serving.simulate

1. Replays every test user (plus 2% unknown IDs) through the serving engine: request latency p50/p95/p99, stage
   latencies, fallback rate.
2. Drift between the validation and test snapshots: PSI of ranker scores on the candidate pools, feature
   missingness (no platform stats, no history, no creator history), and Jensen-Shannon divergence of the
   category mix and item exposure of the final feeds.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("lightgbm")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from reelrank.config import Config, cli, write_json  # noqa: E402
from reelrank.serving.engine import Engine  # noqa: E402


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """Population stability index of b relative to a (quantile bins of a)."""
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(((pb - pa) * np.log(pb / pa)).sum())


def js(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(x, y):
        nz = x > 0
        return float((x[nz] * np.log2(x[nz] / y[nz])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run(cfg: Config) -> dict:
    eng = Engine(cfg.artifacts)
    p_users = np.load(cfg.artifacts / "runs" / "test" / "e2e_reranked.npy", mmap_mode="r")  # shape check only
    users = pl.read_parquet(cfg.artifacts / "runs" / "test" / "per_user.parquet").filter(
        pl.col("system") == "e2e_reranked")["user_id"].to_numpy()
    assert len(users) == p_users.shape[0]
    rng = np.random.default_rng(cfg.seed)
    reqs = list(users) + list(rng.integers(eng.U + 1, eng.U + 10**6, max(1, len(users) // 50)))
    rng.shuffle(reqs)
    for u in reqs:
        eng.recommend(int(u), k=20)
    lat = np.array(eng.stats["latency_ms"])
    out: dict = {"note": "offline replay of the test users through the serving engine; not production traffic",
                 "requests": len(reqs), "latency_ms": {f"p{q}": float(np.percentile(lat, q)) for q in (50, 95, 99)},
                 "stage_p95_ms": {k: float(np.percentile(np.array(v), 95)) for k, v in eng.stats["stage_ms"].items()},
                 "fallback_rate": {k: v / len(reqs) for k, v in eng.stats["fallbacks"].items()}}
    rdir = cfg.processed / "ranker"
    sv = pl.read_parquet(rdir / "scores_lgb_val.parquet")["lgb_rank"].to_numpy()
    st = pl.read_parquet(rdir / "scores_lgb_test.parquet")["lgb_rank"].to_numpy()
    cv = pl.read_parquet(rdir / "val_candidates.parquet", columns=["p_has", "u_has_history", "author_mean_wr"])
    ct = pl.read_parquet(rdir / "test_candidates.parquet", columns=["p_has", "u_has_history", "author_mean_wr"])
    out["score_psi_val_to_test"] = psi(sv, st)
    out["missingness"] = {name: {"val": float(f(cv)), "test": float(f(ct))} for name, f in {
        "no_platform_stats": lambda d: (d["p_has"] == 0).mean(),
        "user_without_history": lambda d: (d["u_has_history"] == 0).mean(),
        "no_creator_history": lambda d: (d["author_mean_wr"] < 0).mean()}.items()}
    cats = eng.st["first_cat"]
    n_cat = int(cats.max()) + 1
    feeds = {p: np.load(cfg.artifacts / "runs" / p / "e2e_reranked.npy") for p in ("val", "test")}
    mix = {p: np.bincount(cats[f[f >= 0]], minlength=n_cat).astype(float) for p, f in feeds.items()}
    expo = {p: np.bincount(f[f >= 0], minlength=eng.I).astype(float) for p, f in feeds.items()}
    out["category_mix_js_bits"] = js(mix["val"], mix["test"])
    out["item_exposure_js_bits"] = js(expo["val"], expo["test"])
    out["thresholds"] = {"score_psi_alert": 0.25, "fallback_rate_alert": 0.05, "latency_p95_ms_alert": 100}
    out["alerts"] = [k for k, bad in {
        "score_psi": out["score_psi_val_to_test"] > 0.25,
        "fallback_rate": sum(out["fallback_rate"].values()) > 0.05,
        "latency_p95": out["latency_ms"]["p95"] > 100}.items() if bad]
    write_json(cfg.reports / "metrics" / "monitoring_sim.json", out)
    print({k: v for k, v in out.items() if k not in ("note",)})
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
