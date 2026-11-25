"""Ranker feature tables built from snapshot state only (no PyTorch / LightGBM in this process).

    python -m reelrank.ranking.dataset

The same `pair_features` function serves three uses so train/serve features cannot drift:
  <window>_observed.parquet   observed events in r0 / val / test windows (ranker training + logged-impression eval)
  <protocol>_candidates.parquet  retrieval candidate pools for val / test / dense (end-to-end funnel)

Retrieval scores come from a *frozen* model trained before the window: R0 for the Aug 6-10 training window,
R1 for Aug 29-31, R2 for Sep 1-5 and the dense matrix. Every feature for an example on day d reads the
snapshot at the start of d. Unobserved candidates are never labelled as skips: they only appear in the
candidate tables, which are used for evaluation, not training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from reelrank.config import Config, cli, day_to_date, read_json, write_json
from reelrank.evaluation.protocol import load_protocol
from reelrank.retrieval.candidates import selected_variant

USER_COLS = ["u_log_events", "u_log_pos", "u_mean_wr", "u_std_wr", "u_complete_rate", "u_skip_rate",
             "u_pref_log_dur", "u_log_days_since", "u_has_history"]
USER_STATIC = ["u_recent20_wr", "u_active_degree", "u_log_register_days", "u_is_author", "u_log_follow", "u_log_fans"]
ITEM_COLS = ["i_log_pop", "i_log_pos", "i_mean_wr", "i_complete_rate", "i_skip_rate", "i_log_recent_pop"]
PLAT_COLS = ["p_log_show", "p_play_rate", "p_complete_rate", "p_like_rate", "p_share_rate", "p_progress", "p_has"]
ITEM_EXTRA = ["i_log_age", "i_log_dur", "i_first_cat", "i_n_tags", "i_is_ad"]
PAIR_COLS = ["tt_score", "als_score", "cat_aff", "dur_gap", "log_prior_uv", "log_author_cnt", "author_mean_wr"]
FEATURES = USER_COLS + USER_STATIC + ITEM_COLS + PLAT_COLS + ITEM_EXTRA + PAIR_COLS
CATEGORICAL = ["i_first_cat"]
LABELS = ["grade", "wr", "log_play_s", "complete", "skip", "observed", "duration_ms"]


def sample_train_groups(df: pl.DataFrame, max_rows: int | None, seed: int) -> pl.DataFrame:
    """Keep whole (user, day) groups, chosen by a seeded hash, until about `max_rows` rows remain."""
    if not max_rows or df.height <= max_rows:
        return df
    g = df.select("user_id", "day").unique().sort(["user_id", "day"])
    g = g.with_columns(h=(pl.col("user_id").cast(pl.UInt64) * 1000003 + pl.col("day").cast(pl.UInt64) * 7919 + seed)
                       .hash(seed))
    sizes = df.group_by(["user_id", "day"]).len()
    # joins do not preserve order: sort after joining, with explicit tie-breakers, before the running total
    g = g.join(sizes, on=["user_id", "day"]).sort(["h", "user_id", "day"]).with_columns(cum=pl.col("len").cum_sum())
    keep = g.filter(pl.col("cum") <= max_rows).select("user_id", "day")
    return df.join(keep, on=["user_id", "day"], how="semi").sort(["user_id", "day", "video_id"])


class FeatureBuilder:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        fdir = cfg.processed / "features"
        self.man = read_json(fdir / "manifest.json")
        self.snap_days = np.load(fdir / "snap_days.npy")
        self.user_stats = np.load(fdir / "user_stats.npy", mmap_mode="r")
        self.user_cat = np.load(fdir / "user_cat.npy", mmap_mode="r")
        self.user_recent = np.load(fdir / "user_recent_wr.npy", mmap_mode="r")
        self.item_stats = np.load(fdir / "item_stats.npy", mmap_mode="r")
        self.item_platform = np.load(fdir / "item_platform.npy", mmap_mode="r")
        pop = np.load(fdir / "item_pop_before.npy")
        self.static = dict(np.load(fdir / "item_static.npz"))
        st = self.static
        first_seen = np.where(pop > 0, 1, 0).argmax(0)
        first_day = np.where(pop.max(0) > 0, self.snap_days[np.maximum(first_seen - 1, 0)], 10**6)
        self.born = np.minimum(np.where(st["upload_day"] >= 0, st["upload_day"], 10**6), first_day)
        users = pd.read_parquet(cfg.processed / "users.parquet").sort_values("user_id")
        self.user_static = np.stack([
            users["active_degree"].to_numpy(dtype=float), np.log1p(users["register_days"].to_numpy(dtype=float)),
            users["is_video_author"].to_numpy(dtype=float), np.log1p(users["follow_user_num"].to_numpy(dtype=float)),
            np.log1p(users["fans_user_num"].to_numpy(dtype=float)),
        ], 1).astype(np.float32)
        self.events = pl.read_parquet(fdir / "events.parquet",
                                      columns=["user_id", "video_id", "day", "wr_c", "watch_ratio", "play_ms",
                                               "duration_ms", "grade", "complete", "skip"])
        author = st["author"]
        self.author = author
        ev = self.events.select("user_id", "video_id", "day", "wr_c").with_columns(
            author=pl.col("video_id").map_batches(lambda s: pl.Series(author[s.to_numpy()]), return_dtype=pl.Int64))
        # cumulative (user, video) and (user, author) history; state from day d is visible from day d + 1
        self.uv_hist = (ev.group_by(["user_id", "video_id", "day"]).agg(n=pl.len()).sort("day")
                        .with_columns(cum=pl.col("n").cum_sum().over(["user_id", "video_id"]),
                                      visible=pl.col("day") + 1).select("user_id", "video_id", "visible", "cum")
                        .sort("visible"))
        self.ua_hist = (ev.filter(pl.col("author") >= 0).group_by(["user_id", "author", "day"])
                        .agg(n=pl.len(), s=pl.col("wr_c").sum()).sort("day")
                        .with_columns(cum=pl.col("n").cum_sum().over(["user_id", "author"]),
                                      cum_s=pl.col("s").cum_sum().over(["user_id", "author"]),
                                      visible=pl.col("day") + 1)
                        .select("user_id", "author", "visible", "cum", "cum_s").sort("visible"))
        self._vec_cache: dict = {}

    def snap(self, day: int) -> int:
        s = int(np.searchsorted(self.snap_days, day))
        assert self.snap_days[s] == day
        return s

    def vectors(self, tag: str, variant: str, day: int):
        key = (tag, variant, day)
        if key not in self._vec_cache:
            d = self.cfg.artifacts / "two_tower" / tag / variant
            self._vec_cache[key] = (np.load(d / f"users_{day_to_date(day)}.npy"),
                                    np.load(d / f"items_{day_to_date(day)}.npy"))
        return self._vec_cache[key]

    def pair_features(self, u: np.ndarray, v: np.ndarray, day: np.ndarray, tag: str, variant: str) -> pl.DataFrame:
        n = len(u)
        s = np.array([self.snap(int(d)) for d in np.unique(day)])
        smap = dict(zip(np.unique(day).tolist(), s.tolist()))
        si = np.array([smap[int(d)] for d in day]) if n else np.zeros(0, dtype=int)
        cols: dict[str, np.ndarray] = {}
        us = np.asarray(self.user_stats[si, u])
        for j, c in enumerate(USER_COLS):
            cols[c] = us[:, j]
        ust = np.concatenate([np.asarray(self.user_recent[si, u])[:, None], self.user_static[u]], 1)
        for j, c in enumerate(USER_STATIC):
            cols[c] = ust[:, j]
        it = np.asarray(self.item_stats[si, v])
        for j, c in enumerate(ITEM_COLS):
            cols[c] = it[:, j]
        pf = np.asarray(self.item_platform[si, v])
        for j, c in enumerate(PLAT_COLS):
            cols[c] = pf[:, j]
        st = self.static
        cols["i_log_age"] = np.log1p(np.clip(day - self.born[v], 0, 400)).astype(np.float32)
        cols["i_log_dur"] = st["log_dur"][v]
        cols["i_first_cat"] = st["first_cat"][v].astype(np.float32)
        cols["i_n_tags"] = st["tags"][v].sum(1)
        cols["i_is_ad"] = st["is_ad"][v].astype(np.float32)
        # retrieval scores from the frozen models of this window
        tt = np.zeros(n, dtype=np.float32)
        for d in np.unique(day):
            m = day == d
            uv, iv = self.vectors(tag, variant, int(d))
            tt[m] = np.einsum("ij,ij->i", uv[u[m]], iv[v[m]])
        cols["tt_score"] = tt
        als = np.load(self.cfg.artifacts / "als" / f"{tag}.npz")
        cols["als_score"] = np.einsum("ij,ij->i", als["user"][u], als["item"][v]).astype(np.float32)
        uc = np.asarray(self.user_cat[si, u])
        cols["cat_aff"] = uc[np.arange(n), st["first_cat"][v]]
        cols["dur_gap"] = np.abs(st["log_dur"][v] - cols["u_pref_log_dur"])
        q = pl.DataFrame({"row": np.arange(n), "user_id": u.astype(np.int32), "video_id": v.astype(np.int32),
                          "author": self.author[v], "visible": day.astype(np.int32)}).sort("visible")
        q = q.join_asof(self.uv_hist, on="visible", by=["user_id", "video_id"], strategy="backward")
        q = q.join_asof(self.ua_hist.rename({"cum": "a_cum", "cum_s": "a_s"}), on="visible",
                        by=["user_id", "author"], strategy="backward").sort("row")
        cols["log_prior_uv"] = np.log1p(q["cum"].fill_null(0).to_numpy()).astype(np.float32)
        a_cum = q["a_cum"].fill_null(0).to_numpy().astype(np.float32)
        cols["log_author_cnt"] = np.log1p(a_cum)
        cols["author_mean_wr"] = np.where(a_cum > 0, q["a_s"].fill_null(0).to_numpy() / np.maximum(a_cum, 1), -1.0
                                          ).astype(np.float32)
        return pl.DataFrame({c: np.asarray(cols[c], dtype=np.float32) for c in FEATURES})

    # ------------------------------------------------------------------------------------------ tables
    def observed(self, window: str, variant: str) -> pl.DataFrame:
        start, end, tag = self.cfg.windows()[window]
        cap = int(self.cfg["ranker"]["max_events_per_user_day"])
        ev = self.events.filter((pl.col("day") >= start) & (pl.col("day") <= end)).with_row_index("r")
        # cap very heavy user-days with a deterministic hash sample
        ev = ev.with_columns(h=(pl.col("r") * 2654435761 % 4294967296)).with_columns(
            k=pl.col("h").rank("ordinal").over(["user_id", "day"])).filter(pl.col("k") <= cap).drop("h", "k", "r")
        ev = ev.sort(["user_id", "day", "video_id"])
        u, v, d = (ev[c].to_numpy() for c in ("user_id", "video_id", "day"))
        X = self.pair_features(u, v, d, tag, variant)
        lab = ev.select(
            pl.col("user_id"), pl.col("video_id"), pl.col("day"), pl.col("grade"),
            wr=pl.col("wr_c"), log_play_s=(pl.col("play_ms") / 1000).log1p().cast(pl.Float32),
            complete=pl.col("complete").cast(pl.Int8), skip=pl.col("skip").cast(pl.Int8),
            observed=pl.lit(1, pl.Int8), duration_ms=pl.col("duration_ms"), watch_ratio=pl.col("watch_ratio"))
        return pl.concat([lab, X], how="horizontal").with_columns(window=pl.lit(window), tag=pl.lit(tag))

    def candidates(self, pname: str, variant: str) -> pl.DataFrame:
        p = load_protocol(self.cfg, pname)
        cand = pl.read_parquet(self.cfg.processed / "candidates" / f"{pname}.parquet")
        u, v = cand["user_id"].to_numpy(), cand["video_id"].to_numpy()
        d = np.full(len(u), p.cutoff, dtype=np.int32)
        X = self.pair_features(u, v, d, p.model_tag, variant)
        upos = {int(x): i for i, x in enumerate(p.users)}
        grade = p.rel[np.array([upos[int(x)] for x in u]), v]
        lab = cand.select("user_id", "video_id", "sources", "tt_rank", "als_rank").with_columns(
            day=pl.lit(p.cutoff, pl.Int32), grade=pl.Series(grade.astype(np.int8)),
            observed=pl.Series((grade > 0).astype(np.int8)),
            duration_ms=pl.Series(np.round(np.expm1(self.static["log_dur"][v]) * 1000).astype(np.int64)))
        return pl.concat([lab, X], how="horizontal").with_columns(window=pl.lit(pname), tag=pl.lit(p.model_tag))


def run(cfg: Config) -> dict:
    variant = selected_variant(cfg)
    out = cfg.processed / "ranker"
    out.mkdir(parents=True, exist_ok=True)
    fb = FeatureBuilder(cfg)
    summary = {"retrieval_variant": variant, "features": FEATURES}
    for w in ("r0", "val", "test"):
        df = fb.observed(w, variant)
        df.write_parquet(out / f"{w}_observed.parquet")
        summary[f"{w}_observed"] = {"rows": df.height, "user_days": int(df.select("user_id", "day").n_unique()),
                                    "grade_dist": df["grade"].value_counts().sort("grade")["count"].to_list()}
        print(f"[{w}] observed: {df.height:,} rows")
    for pname in ("val", "test", "dense"):
        df = fb.candidates(pname, variant)
        df.write_parquet(out / f"{pname}_candidates.parquet")
        summary[f"{pname}_candidates"] = {"rows": df.height, "relevant_share": float((df["grade"] >= 2).mean())}
        print(f"[{pname}] candidates: {df.height:,} rows")
    write_json(cfg.reports / "metrics" / "ranker_dataset.json", summary)
    return summary


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
