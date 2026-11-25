"""Leakage-safe daily feature snapshots.

    python -m reelrank.features.build

A snapshot at day D summarises every event with day < D (strictly earlier calendar days, UTC+8). Same-day
events never see each other, which settles timestamp ties conservatively. One snapshot exists for each day
that has events plus each protocol cutoff, so any (user, day) training example reads the state at the start
of its own day.

Outputs (data/processed/features/):
  events.parquet         cleaned big-matrix events + labels + snapshot index
  snap_days.npy          [S] day index of each snapshot
  user_stats.npy         [S, U, F_u]   activity / watch-ratio / recency statistics
  user_cat.npy           [S, U, C]     smoothed first-level category affinity over positives
  user_hist.npy          [S, U, L]     last L positively-engaged video ids (oldest -> newest, pad -1)
  user_hist_wr.npy       [S, U, L]     watch ratio of those events
  user_recent_wr.npy     [S, U]        mean capped watch ratio of the last 20 events before D (DuckDB window)
  item_stats.npy         [S, I, F_i]   in-dataset popularity and engagement rates (shrunk)
  item_platform.npy      [S, I, F_p]   platform-wide daily statistics as of D-1 or earlier
  item_static.npz        tags multi-hot, category, log duration, duration bucket, creator, upload day
  eval/*.parquet         retrieval targets for the validation and test protocols, dense labels
  manifest.json          feature names, shapes, cutoffs, content hashes
"""

from __future__ import annotations

import hashlib

import duckdb
import numpy as np
import pandas as pd
import polars as pl

from reelrank.config import Config, cli, day_to_date, write_json

USER_STATS = ["log_events", "log_pos", "mean_wr", "std_wr", "complete_rate", "skip_rate",
              "pref_log_dur", "log_days_since_last", "has_history"]
ITEM_STATS = ["log_pop", "log_pos", "mean_wr", "complete_rate", "skip_rate", "log_recent_pop"]
PLATFORM = ["log_show", "play_rate", "complete_rate", "like_rate", "share_rate", "progress", "has_platform"]
DUR_EDGES_S = [7, 10, 15, 20, 30, 45, 60, 120]


def load_events(cfg: Config) -> pl.DataFrame:
    lab = cfg["labels"]
    e = pl.scan_parquet(str(cfg.processed / "interactions" / "**" / "*.parquet"), hive_partitioning=True)
    edges = lab["grade_edges"]
    wr = pl.col("watch_ratio")
    return (
        e.drop("date")
        .with_columns(
            pos=(wr >= lab["positive_wr"]),
            complete=(wr >= lab["complete_wr"]),
            skip=(wr < lab["skip_wr"]),
            grade=pl.when(wr < edges[0]).then(0).when(wr < edges[1]).then(1).when(wr < edges[2]).then(2)
            .otherwise(3).cast(pl.Int8),
            wr_c=wr.clip(0, lab["wr_cap"]),
        )
        .sort(["user_id", "ts", "video_id", "play_ms"])
        .collect()
    )


def snapshot_days(cfg: Config, events: pl.DataFrame) -> np.ndarray:
    days = set(events["day"].unique().to_list())
    days.update(cfg.model_cutoffs().values())
    days.add(cfg.day("end") + 1)
    return np.array(sorted(days), dtype=np.int32)


def _cum_before(per_day: np.ndarray) -> np.ndarray:
    """per_day[s] holds sums for snapshot-day s; return state *before* each snapshot (exclusive cumsum)."""
    out = np.cumsum(per_day, axis=0)
    out = np.concatenate([np.zeros_like(out[:1]), out[:-1]], axis=0)
    return out


def build_item_static(items: pd.DataFrame, n_tags: int) -> dict[str, np.ndarray]:
    n = len(items)
    tags = np.zeros((n, n_tags), dtype=np.float32)
    for i, t in enumerate(items["tags"]):
        tags[i, list(t)] = 1.0
    dur_s = items["duration_ms"].to_numpy() / 1000.0
    authors = items["author_id"].to_numpy()
    # Creator vocabulary: creators with >= 2 videos get their own id; the rest share id 0 ("rare/unknown").
    vc = pd.Series(authors[authors >= 0]).value_counts()
    frequent = vc[vc >= 2].index.to_numpy()
    author_idx = np.zeros(n, dtype=np.int32)
    lookup = {a: i + 1 for i, a in enumerate(np.sort(frequent))}
    for i, a in enumerate(authors):
        author_idx[i] = lookup.get(a, 0)
    return {
        "tags": tags,
        "first_cat": items["first_cat"].to_numpy().astype(np.int64),
        "log_dur": np.log1p(dur_s).astype(np.float32),
        "dur_bucket": np.digitize(dur_s, DUR_EDGES_S).astype(np.int64),
        "author": authors.astype(np.int64),
        "author_idx": author_idx.astype(np.int64),
        "upload_day": items["upload_day"].to_numpy().astype(np.int32),
        "is_ad": items["is_ad"].to_numpy().astype(bool),
    }


def build(cfg: Config) -> dict:
    out = cfg.processed / "features"
    (out / "eval").mkdir(parents=True, exist_ok=True)
    fcfg = cfg["features"]
    items = pd.read_parquet(cfg.processed / "items.parquet")
    n_cat = int(items["first_cat"].max()) + 1
    n_tags = int(max(max(t) for t in items["tags"])) + 1
    ev = load_events(cfg)
    U = int(ev["user_id"].max()) + 1
    I = len(items)
    assert int(ev["video_id"].max()) < I
    S_days = snapshot_days(cfg, ev)
    S = len(S_days)
    day_to_snap = {int(d): i for i, d in enumerate(S_days)}
    ev = ev.with_columns(snap=pl.col("day").replace_strict(day_to_snap, return_dtype=pl.Int32))
    static = build_item_static(items, n_tags)
    log_dur = static["log_dur"]
    first_cat = static["first_cat"]

    u = ev["user_id"].to_numpy()
    v = ev["video_id"].to_numpy()
    s = ev["snap"].to_numpy()
    pos = ev["pos"].to_numpy()
    wr = ev["wr_c"].to_numpy().astype(np.float64)
    comp = ev["complete"].to_numpy()
    skip = ev["skip"].to_numpy()

    # user statistics
    def acc(idx, val, shape):
        a = np.zeros(shape, dtype=np.float64)
        np.add.at(a, idx, val)
        return a

    n_ev = _cum_before(acc((s, u), 1.0, (S, U)))
    n_pos = _cum_before(acc((s, u), pos.astype(float), (S, U)))
    s_wr = _cum_before(acc((s, u), wr, (S, U)))
    s_wr2 = _cum_before(acc((s, u), wr * wr, (S, U)))
    n_comp = _cum_before(acc((s, u), comp.astype(float), (S, U)))
    n_skip = _cum_before(acc((s, u), skip.astype(float), (S, U)))
    s_ldur = _cum_before(acc((s, u), np.where(pos, log_dur[v], 0.0), (S, U)))
    # last active day before each snapshot
    active = acc((s, u), 1.0, (S, U)) > 0
    last_snap = np.full((S, U), -1, dtype=np.int64)
    run = np.full(U, -1, dtype=np.int64)
    for si in range(S):
        last_snap[si] = run
        run = np.where(active[si], si, run)
    days_since = np.where(last_snap >= 0, S_days[:, None] - S_days[np.maximum(last_snap, 0)], 60).clip(0, 60)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_wr = np.where(n_ev > 0, s_wr / np.maximum(n_ev, 1), 0.0)
        std_wr = np.sqrt(np.maximum(np.where(n_ev > 0, s_wr2 / np.maximum(n_ev, 1), 0.0) - mean_wr**2, 0.0))
        pref = np.where(n_pos > 0, s_ldur / np.maximum(n_pos, 1), float(np.mean(log_dur)))
    user_stats = np.stack([
        np.log1p(n_ev), np.log1p(n_pos), mean_wr, std_wr,
        n_comp / np.maximum(n_ev, 1), n_skip / np.maximum(n_ev, 1), pref,
        np.log1p(days_since), (n_ev > 0).astype(float),
    ], axis=-1).astype(np.float32)

    pos_ev = ev.filter(pl.col("pos"))
    pu, pv, ps = (pos_ev[c].to_numpy() for c in ("user_id", "video_id", "snap"))
    cat_cnt = _cum_before(acc((ps, pu, first_cat[pv]), 1.0, (S, U, n_cat)))
    prior = np.bincount(first_cat[pv], minlength=n_cat).astype(np.float64)
    prior = prior / prior.sum()
    alpha = 5.0  # Dirichlet smoothing towards the global category mix
    user_cat = ((cat_cnt + alpha * prior) / (cat_cnt.sum(-1, keepdims=True) + alpha)).astype(np.float32)
    del cat_cnt

    # last-L positive history per snapshot
    L = int(fcfg["history_len"])
    hist = np.full((S, U, L), -1, dtype=np.int32)
    hist_wr = np.zeros((S, U, L), dtype=np.float16)
    pdays = pos_ev["day"].to_numpy()
    pwr = pos_ev["wr_c"].to_numpy()
    bounds = np.searchsorted(pu, np.arange(U + 1))
    ar = np.arange(L)
    for user in range(U):
        lo, hi = bounds[user], bounds[user + 1]
        if hi == lo:
            continue
        ends = lo + np.searchsorted(pdays[lo:hi], S_days, side="left")  # first index with day >= D
        idx = ends[:, None] - L + ar[None, :]
        ok = idx >= lo
        idx = np.clip(idx, lo, max(hi - 1, lo))
        hist[:, user, :] = np.where(ok, pv[idx], -1)
        hist_wr[:, user, :] = np.where(ok, pwr[idx], 0.0)

    # item statistics
    k = float(fcfg["shrink"])
    i_n = acc((s, v), 1.0, (S, I))
    i_pos = acc((s, v), pos.astype(float), (S, I))
    i_wr = acc((s, v), wr, (S, I))
    i_comp = acc((s, v), comp.astype(float), (S, I))
    i_skip = acc((s, v), skip.astype(float), (S, I))
    c_n, c_pos, c_wr, c_comp, c_skip = (_cum_before(a) for a in (i_n, i_pos, i_wr, i_comp, i_skip))
    g_wr, g_comp, g_skip = wr.mean(), comp.mean(), skip.mean()
    # "Trending": events in the last `recent_days` days *that have data* before D. The big matrix has
    # multi-week gaps, so a calendar lookback would often be empty.
    day_volume = i_n.sum(1)
    busy = day_volume >= 0.05 * day_volume.max()
    recent = np.zeros((S, I))
    rd = int(fcfg["recent_days"])
    for si in range(S):
        prev = np.where(busy[:si])[0][-rd:]
        if len(prev):
            recent[si] = i_n[prev].sum(0)
    item_stats = np.stack([
        np.log1p(c_n), np.log1p(c_pos),
        (c_wr + k * g_wr) / (c_n + k), (c_comp + k * g_comp) / (c_n + k), (c_skip + k * g_skip) / (c_n + k),
        np.log1p(recent),
    ], axis=-1).astype(np.float32)
    pop_before = c_n.astype(np.float32)  # raw counts, for popularity baselines and logQ
    del i_wr, i_comp, i_skip, c_wr, c_comp, c_skip

    # platform daily statistics, as of the latest day strictly before D
    daily = pd.read_parquet(cfg.processed / "item_daily.parquet").sort_values(["video_id", "day"])
    d0, d1 = int(min(daily["day"].min(), S_days.min())), int(max(daily["day"].max(), S_days.max()))
    grid = np.full((d1 - d0 + 2, I, len(PLATFORM)), np.nan, dtype=np.float32)
    show = daily["show_cnt"].to_numpy().astype(float)
    play = daily["play_cnt"].to_numpy().astype(float)
    vals = np.stack([
        np.log1p(show), play / np.maximum(show, 1), daily["complete_play_cnt"] / np.maximum(play, 1),
        daily["like_cnt"] / np.maximum(play, 1), daily["share_cnt"] / np.maximum(play, 1),
        daily["play_progress"].fillna(0).to_numpy(), np.ones(len(daily)),
    ], axis=-1).astype(np.float32)
    # a row dated d becomes visible on day d+1
    grid[daily["day"].to_numpy() - d0 + 1, daily["video_id"].to_numpy()] = vals
    df = pd.DataFrame(grid.reshape(grid.shape[0], -1)).ffill().to_numpy().reshape(grid.shape)
    item_platform = np.nan_to_num(df[np.clip(S_days - d0, 0, grid.shape[0] - 1)], nan=0.0).astype(np.float32)

    user_recent = recent_watch_ratio(ev, S_days, U, int(fcfg.get("rolling_events", 20)))

    # save
    arrays = {"snap_days": S_days, "user_stats": user_stats, "user_cat": user_cat, "user_hist": hist,
              "user_hist_wr": hist_wr, "item_stats": item_stats, "item_platform": item_platform,
              "item_pop_before": pop_before, "user_recent_wr": user_recent}
    hashes = {}
    for name, arr in arrays.items():
        np.save(out / f"{name}.npy", arr)
        hashes[name] = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    np.savez(out / "item_static.npz", **static)
    ev.write_parquet(out / "events.parquet")

    evals = build_eval_sets(cfg, ev, static, S_days)
    manifest = {
        "config": cfg.name, "config_hash": cfg.hash(), "users": U, "items": I, "categories": n_cat,
        "tags": n_tags, "authors": int(static["author_idx"].max()) + 1, "history_len": L,
        "snapshots": S, "snap_dates": [day_to_date(d) for d in S_days],
        "user_stats": USER_STATS, "item_stats": ITEM_STATS, "item_platform": PLATFORM,
        "windows": cfg["protocol"]["windows"], "dense_model": cfg["protocol"]["dense_model"],
        "events": ev.height, "positives": int(ev["pos"].sum()), "hashes": hashes, "eval": evals,
    }
    write_json(out / "manifest.json", manifest)
    print(f"features: {S} snapshots x {U} users x {I} items; {ev.height:,} events")
    for k_, v_ in evals.items():
        print(f"  {k_}: {v_}")
    return manifest


def recent_watch_ratio(ev: pl.DataFrame, snap_days: np.ndarray, n_users: int, n: int = 20) -> np.ndarray:
    """[S, U] mean capped watch ratio of each user's last `n` events from days strictly before each snapshot day.

    SQL window over each user's events ordered by (ts, event_id): `ROWS BETWEEN n PRECEDING AND 1 PRECEDING`.
    Timestamp ties are handled by reading the window at the user's *first* event of a day, which only covers
    rows from earlier days. For a snapshot day D the value is read at the first event on or after D; after a
    user's last event day it is the mean of their final n events. -1 means no history.
    """
    con = duckdb.connect()
    con.register("ev", ev.select("event_id", "user_id", "day", "ts", "wr_c").to_arrow())
    first = con.sql(f"""
        WITH w AS (
            SELECT user_id, day,
                   avg(wr_c) OVER (PARTITION BY user_id ORDER BY ts, event_id
                                   ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING) AS hist_wr,
                   row_number() OVER (PARTITION BY user_id, day ORDER BY ts, event_id) AS k
            FROM ev)
        SELECT user_id, day, hist_wr FROM w WHERE k = 1 ORDER BY user_id, day""").pl()
    final = con.sql(f"""
        SELECT user_id, avg(wr_c) AS v FROM (
            SELECT user_id, wr_c,
                   row_number() OVER (PARTITION BY user_id ORDER BY ts DESC, event_id DESC) AS r FROM ev)
        WHERE r <= {n} GROUP BY user_id""").pl()
    final_v = dict(zip(final["user_id"].to_list(), final["v"].to_list()))
    out = np.full((len(snap_days), n_users), -1.0, dtype=np.float32)
    for (u,), g in first.group_by(["user_id"], maintain_order=True):
        days = g["day"].to_numpy()
        vals = np.append(g["hist_wr"].fill_null(-1.0).to_numpy(), final_v[u])
        out[:, u] = vals[np.searchsorted(days, snap_days, side="left")]
    return out


def eligible_items(static: dict, pop_before_row: np.ndarray, day: int) -> np.ndarray:
    """Catalog eligible at the start of `day`: uploaded before it, or already watched before it.

    `upload_dt` is mostly reliable (median first view = upload day) but 8.8% of videos are watched up to 27
    days before their listed upload date, so an earlier observed view also makes a video eligible.
    """
    up = static["upload_day"]
    return ((up >= 0) & (up < day)) | (pop_before_row > 0)


def build_eval_sets(cfg: Config, ev: pl.DataFrame, static: dict, S_days: np.ndarray) -> dict:
    """Retrieval targets per protocol: novel (not previously consumed) positives inside the target window."""
    out = cfg.processed / "features" / "eval"
    pop = np.load(cfg.processed / "features" / "item_pop_before.npy")
    summary = {}
    for name, (start, last_day, model) in cfg.windows().items():
        si = int(np.searchsorted(S_days, start))
        elig = eligible_items(static, pop[si], start)
        seen = ev.filter(pl.col("day") < start).select("user_id", "video_id").unique()
        win = ev.filter((pl.col("day") >= start) & (pl.col("day") <= last_day))
        obs = (win.group_by(["user_id", "video_id"])
               .agg(grade=pl.col("grade").max(), max_wr=pl.col("watch_ratio").max(), n=pl.len(),
                    pos=pl.col("pos").any()))
        elig_df = pl.DataFrame({"video_id": np.arange(len(elig), dtype=np.int32), "eligible": elig})
        obs = (obs.join(seen.with_columns(seen=pl.lit(True)), on=["user_id", "video_id"], how="left")
               .with_columns(seen=pl.col("seen").fill_null(False))
               .join(elig_df, on="video_id", how="left"))
        obs.write_parquet(out / f"{name}_observed.parquet")
        targets = obs.filter(pl.col("pos") & ~pl.col("seen") & pl.col("eligible"))
        targets.write_parquet(out / f"{name}_targets.parquet")
        repeat_share = float(obs.filter(pl.col("pos"))["seen"].mean() or 0.0)
        summary[name] = {
            "cutoff": day_to_date(start), "last_day": day_to_date(last_day), "model": model,
            "window_events": win.height, "users_with_events": int(win["user_id"].n_unique()),
            "users_with_novel_positive": int(targets["user_id"].n_unique()),
            "novel_positive_pairs": targets.height, "repeat_share_of_positive_pairs": repeat_share,
            "eligible_items": int(elig.sum()),
            "ineligible_positive_pairs_dropped": int(obs.filter(pl.col("pos") & ~pl.col("eligible")).height),
        }
    dense = pl.read_parquet(cfg.processed / "dense.parquet")
    edges = cfg["labels"]["grade_edges"]
    wr = pl.col("watch_ratio")
    dense = dense.with_columns(
        pos=wr >= cfg["labels"]["positive_wr"],
        grade=pl.when(wr < edges[0]).then(0).when(wr < edges[1]).then(1).when(wr < edges[2]).then(2)
        .otherwise(3).cast(pl.Int8))
    dense.write_parquet(out / "dense_labels.parquet")
    summary["dense"] = {"users": int(dense["user_id"].n_unique()), "items": int(dense["video_id"].n_unique()),
                        "pairs": dense.height, "positive_share": float(dense["pos"].mean())}
    return summary


def main() -> None:
    cfg, _ = cli(__doc__)
    build(cfg)


if __name__ == "__main__":
    main()
