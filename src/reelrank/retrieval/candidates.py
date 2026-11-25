"""Candidate generation: union of two-tower, ALS, trending and category-fallback pools (exact NumPy search).

    python -m reelrank.retrieval.candidates

For each evaluated user at the protocol cutoff: top-200 two-tower + top-100 ALS + top-30 trending + top-20 from
the user's favourite category, de-duplicated with source bit flags. Users without history fall back to the
trending + category pools. Candidate recall (before ranking) is reported per source and for the union so a
retrieval bottleneck is visible.

Outputs: data/processed/candidates/<protocol>.parquet (user_id, video_id, sources, tt_score, tt_rank, als_score,
als_rank) and reports/metrics/candidate_recall.json.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from reelrank.config import Config, cli, read_json, write_json
from reelrank.evaluation.metrics import topk
from reelrank.evaluation.protocol import Protocol, load_protocol
from reelrank.retrieval.evaluate import vectors

SRC_TT, SRC_ALS, SRC_TREND, SRC_CAT = 1, 2, 4, 8
SOURCE_NAMES = {SRC_TT: "two_tower", SRC_ALS: "als", SRC_TREND: "trending", SRC_CAT: "category"}


def selected_variant(cfg: Config) -> str:
    sel = cfg.reports / "metrics" / "selection.json"
    return read_json(sel)["retrieval_variant"] if sel.exists() else "main"


class CandidateGenerator:
    def __init__(self, cfg: Config, p: Protocol, variant: str):
        self.cfg, self.p = cfg, p
        c = cfg["candidates"]
        self.n_tt, self.n_als, self.n_trend, self.n_cat = (int(c[k]) for k in ("two_tower", "als", "trending",
                                                                            "category"))
        fdir = cfg.processed / "features"
        man = read_json(fdir / "manifest.json")
        si = {n: i for i, n in enumerate(man["item_stats"])}
        stats = np.load(fdir / "item_stats.npy", mmap_mode="r")[p.snap]
        self.trend_score = (np.asarray(stats[:, si["log_recent_pop"]]) + 1e-3 * np.asarray(stats[:, si["log_pos"]]))
        self.user_cat = np.load(fdir / "user_cat.npy", mmap_mode="r")[p.snap]
        self.has_hist = np.load(fdir / "user_stats.npy", mmap_mode="r")[p.snap, :, man["user_stats"].index(
            "has_history")] > 0
        self.cat = p.extra["static"]["first_cat"]
        vec = vectors(cfg, p.model_tag, variant, p.cutoff)
        if vec is None:
            raise FileNotFoundError(f"two-tower vectors for {p.model_tag}/{variant} at the {p.name} cutoff")
        self.uv, self.iv = vec
        als = np.load(cfg.artifacts / "als" / f"{p.model_tag}.npz")
        self.au, self.ai = als["user"], als["item"]
        # trending pool is global; the category pool is per category (most trending items inside it)
        t = self.trend_score[None, :].astype(np.float32).copy()
        p.mask_scores(t, np.array([0]), exclude_seen=False)
        order = np.argsort(-t[0], kind="stable")
        self.trend_order = order[p.candidate_mask[order]]  # eligible videos only
        self.cat_pools = {}
        for c_ in np.unique(self.cat):
            items = self.trend_order[self.cat[self.trend_order] == c_]
            self.cat_pools[int(c_)] = items[: 4 * self.n_cat]

    def generate(self, users: np.ndarray, batch: int = 512) -> pl.DataFrame:
        frames = []
        for lo in range(0, len(users), batch):
            u = users[lo: lo + batch]
            tt = (self.uv[u] @ self.iv.T).astype(np.float32)
            al = (self.au[u] @ self.ai.T).astype(np.float32)
            self.p.mask_scores(tt, u)
            self.p.mask_scores(al, u)
            tt_top = topk(tt, self.n_tt)
            al_top = topk(al, self.n_als)
            seen = self.p.seen[u]
            for r, user in enumerate(u):
                src: dict[int, int] = {}
                if self.has_hist[user]:
                    for it in tt_top[r]:
                        src[int(it)] = src.get(int(it), 0) | SRC_TT
                    for it in al_top[r]:
                        src[int(it)] = src.get(int(it), 0) | SRC_ALS
                sv = set(seen[r].indices.tolist())
                n = 0
                for it in self.trend_order:
                    if n >= self.n_trend:
                        break
                    if int(it) not in sv:
                        src[int(it)] = src.get(int(it), 0) | SRC_TREND
                        n += 1
                fav = int(np.argmax(self.user_cat[user]))
                n = 0
                for it in self.cat_pools.get(fav, []):
                    if n >= self.n_cat:
                        break
                    if int(it) not in sv:
                        src[int(it)] = src.get(int(it), 0) | SRC_CAT
                        n += 1
                items = np.fromiter(src.keys(), dtype=np.int32, count=len(src))
                frames.append((np.full(len(items), user, dtype=np.int32), items,
                               np.fromiter(src.values(), dtype=np.int8, count=len(src)),
                               tt[r, items], al[r, items]))
        uu, ii, ss, ts, as_ = (np.concatenate(x) for x in zip(*frames))
        df = pl.DataFrame({"user_id": uu, "video_id": ii, "sources": ss, "tt_score": ts, "als_score": as_})
        df = df.with_columns(
            tt_score=pl.when(pl.col("tt_score").is_infinite()).then(None).otherwise(pl.col("tt_score")),
            als_score=pl.when(pl.col("als_score").is_infinite()).then(None).otherwise(pl.col("als_score")),
        )
        return df.with_columns(
            tt_rank=pl.col("tt_score").rank("ordinal", descending=True).over("user_id"),
            als_rank=pl.col("als_score").rank("ordinal", descending=True).over("user_id"),
        )


def candidate_recall(p: Protocol, cand: pl.DataFrame) -> dict:
    """Share of each user's relevant items present in the pool (per source and for the union)."""
    upos = {int(u): i for i, u in enumerate(p.users)}
    rows = cand["user_id"].to_numpy()
    r = np.array([upos[int(x)] for x in rows])
    rel = p.rel[r, cand["video_id"].to_numpy()] >= 2
    n_rel = (p.rel >= 2).sum(1)
    out = {"mean_pool_size": float(cand.height / len(p.users))}
    srcs = cand["sources"].to_numpy()
    for bit, name in [(0, "union"), *SOURCE_NAMES.items()]:
        m = rel if bit == 0 else rel & ((srcs & bit) > 0)
        hits = np.bincount(r[m], minlength=len(p.users))
        rec = np.where(n_rel > 0, hits / np.maximum(n_rel, 1), np.nan)
        out[f"recall_{name}"] = float(np.nanmean(rec))
    out["source_share"] = {name: float(((srcs & bit) > 0).mean()) for bit, name in SOURCE_NAMES.items()}
    return out


def run(cfg: Config) -> dict:
    variant = selected_variant(cfg)
    out_dir = cfg.processed / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    events = pl.read_parquet(cfg.processed / "features" / "events.parquet", columns=["user_id", "video_id", "day"])
    summary = {"retrieval_variant": variant}
    for pname in ("val", "test", "dense"):
        p = load_protocol(cfg, pname, events)
        gen = CandidateGenerator(cfg, p, variant)
        cand = gen.generate(p.users)
        cand.write_parquet(out_dir / f"{pname}.parquet")
        summary[pname] = candidate_recall(p, cand)
        print(f"[{pname}] {cand.height:,} candidates for {len(p.users):,} users: {summary[pname]}")
    write_json(cfg.reports / "metrics" / "candidate_recall.json", summary)
    return summary


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
