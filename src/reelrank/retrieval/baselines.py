"""Fixed retrieval baselines every neural model is compared against.

    python -m reelrank.retrieval.baselines

Random (seeded), global popularity, recent trending, category affinity, item-item cosine, implicit ALS.
ALS is tuned on validation (factors x alpha x recency half-life, then regularization) and refit with the
chosen setting on the data before each model cutoff (R0 < Aug 6, RG < Aug 27, R1 < Aug 29, R2 < Sep 1).
Confidence comes from observed engagement strength (summed capped watch ratio), never from explicit dislike.

Outputs: artifacts/runs/<protocol>/*.{npy,json}, artifacts/als/<R0|R1|R2>.npz, artifacts/itemknn/<tag>.npz,
reports/metrics/{baseline_metrics,als_tuning}.json, popularity_by_cutoff.parquet and
data/processed/features/matrix_<tag>.npz (sparse interaction matrix; ids are already contiguous indices).
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import polars as pl
import scipy.sparse as sp

from reelrank.config import Config, cli, read_json, write_json
from reelrank.evaluation.protocol import Protocol, load_protocol, recommend
from reelrank.evaluation.store import print_table, save_run
from reelrank.tracking import mlflow_run


class Baselines:
    def __init__(self, cfg: Config, events: pl.DataFrame):
        self.cfg = cfg
        self.events = events
        fdir = cfg.processed / "features"
        self.man = read_json(fdir / "manifest.json")
        self.U, self.I = self.man["users"], self.man["items"]
        self.static = dict(np.load(fdir / "item_static.npz"))
        self.item_stats = np.load(fdir / "item_stats.npy", mmap_mode="r")
        self.user_cat = np.load(fdir / "user_cat.npy", mmap_mode="r")
        self.user_hist = np.load(fdir / "user_hist.npy", mmap_mode="r")
        self.pop_before = np.load(fdir / "item_pop_before.npy", mmap_mode="r")
        self.stat_idx = {n: i for i, n in enumerate(self.man["item_stats"])}

    # matrices
    def strength_matrix(self, cutoff: int, half_life: float | None = None) -> sp.csr_matrix:
        """Summed capped watch ratio per (user, video) before cutoff, optionally decayed by event age."""
        w = pl.col("wr_c")
        if half_life:
            w = w * (0.5 ** ((cutoff - pl.col("day")) / half_life))
        e = (self.events.filter(pl.col("day") < cutoff).group_by(["user_id", "video_id"])
             .agg(s=w.sum()))
        return sp.csr_matrix((e["s"].to_numpy().astype(np.float32),
                              (e["user_id"].to_numpy(), e["video_id"].to_numpy())), shape=(self.U, self.I))

    def positive_matrix(self, cutoff: int) -> sp.csr_matrix:
        e = self.events.filter((pl.col("day") < cutoff) & pl.col("pos")).select("user_id", "video_id").unique()
        return sp.csr_matrix((np.ones(e.height, dtype=np.float32),
                              (e["user_id"].to_numpy(), e["video_id"].to_numpy())), shape=(self.U, self.I))

    # simple scorers
    def random(self, p: Protocol):
        rng = np.random.default_rng(self.cfg.seed)
        return lambda u: rng.random((len(u), self.I), dtype=np.float32)

    def popularity(self, p: Protocol):
        s = np.asarray(self.item_stats[p.snap, :, self.stat_idx["log_pos"]], dtype=np.float32)
        return lambda u: np.broadcast_to(s, (len(u), self.I))

    def trending(self, p: Protocol):
        s = np.asarray(self.item_stats[p.snap, :, self.stat_idx["log_recent_pop"]], dtype=np.float32)
        s = s + 1e-3 * np.asarray(self.item_stats[p.snap, :, self.stat_idx["log_pos"]])  # tie-break
        return lambda u: np.broadcast_to(s, (len(u), self.I))

    def category(self, p: Protocol):
        """log P(category | user) + log P(item | category): category affinity times within-category popularity."""
        cat = self.static["first_cat"]
        pos = np.exp(np.asarray(self.item_stats[p.snap, :, self.stat_idx["log_pos"]], dtype=np.float64)) - 1
        pos = pos * p.candidate_mask + 1e-3 * p.candidate_mask
        cat_tot = np.bincount(cat, weights=pos, minlength=self.user_cat.shape[2])
        log_item_given_cat = np.log(np.maximum(pos, 1e-9) / np.maximum(cat_tot[cat], 1e-9)).astype(np.float32)
        uc = np.log(np.asarray(self.user_cat[p.snap], dtype=np.float32))
        return lambda u: uc[u][:, cat] + log_item_given_cat[None, :]

    # item-item cosine
    def itemknn(self, p: Protocol, tag: str):
        X = self.positive_matrix(p.cutoff).tocsc()
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0))).ravel()
        Xn = (X @ sp.diags(1.0 / np.maximum(norms, 1e-9))).tocsr().astype(np.float32)
        XnT = Xn.T.tocsr()
        k = int(self.cfg["baselines"]["item_knn_neighbors"])
        rows, cols, vals = [], [], []
        for lo in range(0, self.I, 1024):
            blk = (XnT[lo: lo + 1024] @ Xn).toarray()
            np.fill_diagonal(blk[:, lo: lo + 1024], 0.0)
            kk = min(k, self.I - 1)
            idx = np.argpartition(-blk, kk - 1, axis=1)[:, :kk]
            v = np.take_along_axis(blk, idx, axis=1)
            keep = v > 0
            rows.append(np.repeat(np.arange(lo, lo + len(blk)), kk)[keep.ravel()])
            cols.append(idx.ravel()[keep.ravel()])
            vals.append(v.ravel()[keep.ravel()])
        S = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.I, self.I), dtype=np.float32)
        d = self.cfg.artifacts / "itemknn"
        d.mkdir(parents=True, exist_ok=True)
        sp.save_npz(d / f"{tag}.npz", S)
        hist = np.asarray(self.user_hist[p.snap])

        def score(u):
            h = hist[u]
            r, c = np.nonzero(h >= 0)
            H = sp.csr_matrix((np.ones(len(r), np.float32), (r, h[r, c])), shape=(len(u), self.I))
            return (H @ S).toarray()

        return score

    # ALS
    def fit_als(self, cutoff: int, factors: int, reg: float, alpha: float, half_life: float | None = None,
                seed: int | None = None):
        from implicit.als import AlternatingLeastSquares

        C = self.strength_matrix(cutoff, half_life)
        m = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha,
                                    iterations=int(self.cfg["baselines"]["als_iterations"]),
                                    random_state=seed if seed is not None else self.cfg.seed,
                                    calculate_training_loss=False)
        t = time.time()
        m.fit(C, show_progress=False)
        uf, vf = np.asarray(m.user_factors), np.asarray(m.item_factors)
        return uf, vf, time.time() - t


def run(cfg: Config) -> dict:
    fdir = cfg.processed / "features"
    events = pl.read_parquet(fdir / "events.parquet", columns=["user_id", "video_id", "day", "wr_c", "pos"])
    B = Baselines(cfg, events)
    ks, nks = cfg["eval"]["ks"], cfg["eval"]["ndcg_ks"]
    K = max(ks)
    out_metrics: dict = {}
    mdir = cfg.reports / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)

    names = [n for n in cfg.windows() if n != "r0"] + ["dense"]
    protocols = {n: load_protocol(cfg, n, events) for n in names}

    # ALS tuning on validation (two stages)
    grid = cfg["baselines"]["als_grid"]
    val = protocols["val"]
    tuning = []

    def trial(f, r, a, hl):
        uf, vf, secs = B.fit_als(val.cutoff, f, r, a, hl)
        rec = recommend(val, lambda u: uf[u] @ vf.T, K)
        m = val.evaluate(rec, ks, nks)
        row = {"factors": f, "regularization": r, "alpha": a, "half_life_days": hl, "fit_seconds": secs,
               **{k: float(np.nanmean(m[k])) for k in ("recall@100", "recall@300", "ndcg@20")}}
        tuning.append(row)
        print(f"  ALS f={f:<4d} reg={r:<5} alpha={a:<4} half-life={hl}  recall@100={row['recall@100']:.4f}"
              f" ({secs:.0f}s)")
        return row

    for f, a, hl in itertools.product(grid["factors"], grid["alpha"], grid["half_life_days"]):
        trial(f, 0.1, a, hl)
    b1 = max(tuning, key=lambda r: r["recall@100"])
    for r in grid["regularization"]:
        if r != 0.1:
            trial(b1["factors"], r, b1["alpha"], b1["half_life_days"])
    best = max(tuning, key=lambda r: r["recall@100"])
    write_json(mdir / "als_tuning.json", {"grid": tuning, "selected": best, "criterion": "val recall@100"})
    with mlflow_run(cfg, "baselines", "als_tuning", {"grid": grid}) as log:
        log({"best_val_recall_at_100": best["recall@100"]})

    # refit ALS per protocol and persist factors
    adir = cfg.artifacts / "als"
    adir.mkdir(parents=True, exist_ok=True)
    als = {}
    for tag, c in cfg.model_cutoffs().items():
        uf, vf, secs = B.fit_als(c, best["factors"], best["regularization"], best["alpha"], best["half_life_days"])
        np.savez(adir / f"{tag}.npz", user=uf, item=vf, cutoff=c, factors=best["factors"],
                 regularization=best["regularization"], alpha=best["alpha"],
                 half_life_days=best["half_life_days"] or 0)
        als[tag] = (uf, vf)
        sp.save_npz(fdir / f"matrix_{tag}.npz", B.strength_matrix(c))

    # evaluate every baseline on val / test / dense
    for pname, p in protocols.items():
        uf, vf = als[p.model_tag]
        systems = {
            "random": B.random(p), "popularity": B.popularity(p), "trending": B.trending(p),
            "category": B.category(p), "itemknn": B.itemknn(p, p.model_tag), "als": lambda u: uf[u] @ vf.T,
        }
        out_metrics[pname] = {}
        for name, fn in systems.items():
            t = time.time()
            rec = recommend(p, fn, K)
            per_user = p.evaluate(rec, ks, nks)
            meta = save_run(cfg, p, name, rec, per_user, {"stage": "baseline", "seconds": time.time() - t})
            out_metrics[pname][name] = meta["metrics"]
        print(f"\n[{pname}] {len(p.users):,} users")
        print_table(out_metrics[pname], ["recall@50", "recall@100", "recall@300", "precision@20", "ndcg@20",
                                         "coverage@20"])
        with mlflow_run(cfg, "baselines", f"baselines_{pname}") as log:
            for name, m in out_metrics[pname].items():
                log({f"{name}/{k}": v for k, v in m.items()})

    # popularity by cutoff (deciles from training history only)
    rows = []
    for pname, p in protocols.items():
        if pname == "dense":
            continue
        pop = p.pop.copy()
        elig = np.where(p.candidate_mask)[0]
        order = elig[np.argsort(-pop[elig], kind="stable")]
        dec = np.zeros(B.I, dtype=np.int8) - 1
        dec[order] = (np.arange(len(order)) * 10 // len(order)).astype(np.int8)
        rows.append(pl.DataFrame({"protocol": pname, "video_id": np.arange(B.I), "pop_before": pop,
                                  "eligible": p.candidate_mask, "decile": dec}))
    pl.concat(rows).write_parquet(mdir / "popularity_by_cutoff.parquet")
    write_json(mdir / "baseline_metrics.json", {"als_selected": best, "metrics": out_metrics})
    return out_metrics


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
