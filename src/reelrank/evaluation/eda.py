"""Exploratory analysis of the cleaned KuaiRec tables -> reports/figures/eda_*.png + reports/metrics/eda.json.

    python -m reelrank.evaluation.eda
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from reelrank.config import Config, cli, date_to_day, day_to_date, read_json, write_json
from reelrank.evaluation.metrics import gini
from reelrank.evaluation.plots import ACCENT, INK, MUTED, SERIES, plt, save, style


def run(cfg: Config) -> dict:
    style()
    fig_dir = cfg.reports / "figures"
    fdir = cfg.processed / "features"
    ev = pl.read_parquet(fdir / "events.parquet",
                         columns=["user_id", "video_id", "day", "watch_ratio", "duration_ms", "pos", "complete", "skip"])
    items = pd.read_parquet(cfg.processed / "items.parquet")
    dense = pl.read_parquet(cfg.processed / "dense.parquet")
    qa = read_json(cfg.processed / "qa_report.json")
    U, I = ev["user_id"].n_unique(), ev["video_id"].n_unique()
    out: dict = {"events": ev.height, "users": U, "items": I,
                 "density_big": ev.select("user_id", "video_id").n_unique() / (U * I),
                 "density_dense": qa["tables"]["dense"]["density"],
                 "positive_rate_big": float(ev["pos"].mean()), "positive_rate_dense": float(
                     (dense["watch_ratio"] >= cfg["labels"]["positive_wr"]).mean()),
                 "completion_rate_big": float(ev["complete"].mean()), "skip_rate_big": float(ev["skip"].mean())}

    # 1. daily volume: the three bursts and the evaluation windows
    daily = ev.group_by("day").len().sort("day")
    daily = daily.filter(pl.col("len") > 1000)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar([pd.Timestamp(day_to_date(d)) for d in daily["day"]], daily["len"].to_numpy() / 1e3, color=ACCENT, width=0.8)
    for i, (name, w) in enumerate(cfg["protocol"]["windows"].items()):
        ax.axvspan(pd.Timestamp(w["start"]) - pd.Timedelta(hours=12), pd.Timestamp(w["end"]) + pd.Timedelta(hours=12),
                   color=SERIES[(i + 1) % len(SERIES)], alpha=0.15)
        ax.text(pd.Timestamp(w["start"]) - pd.Timedelta(hours=10), 700 + 45 * (i % 2), name, fontsize=8, color=INK)
    ax.set(title="Big-matrix events per day (UTC+8): three bursts, evaluation windows shaded",
           ylabel="events (thousands)", ylim=(0, 790))
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    save(fig, fig_dir / "eda_daily_volume.png")
    out["active_days"] = int(daily.height)

    # 2. long tail: Lorenz curve of item interactions
    cnt = np.sort(ev.group_by("video_id").len()["len"].to_numpy())[::-1]
    share = np.cumsum(cnt) / cnt.sum()
    out["item_interaction_gini"] = gini(cnt)
    out["top_1pct_items_share"] = float(share[int(0.01 * len(cnt))])
    out["top_10pct_items_share"] = float(share[int(0.10 * len(cnt))])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot(np.arange(1, len(cnt) + 1) / len(cnt), share, color=ACCENT)
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=0.8)
    ax.set(title=f"Item exposure concentration (Gini {out['item_interaction_gini']:.2f})",
           xlabel="share of videos (most watched first)", ylabel="share of interactions")
    save(fig, fig_dir / "eda_long_tail.png")

    # 3. user history length
    ul = ev.group_by("user_id").len()["len"].to_numpy()
    out["user_history"] = {q: float(np.quantile(ul, x)) for q, x in (("p10", .1), ("median", .5), ("p90", .9))}
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.hist(np.log10(ul), bins=40, color=ACCENT)
    ax.set(title="User history length", xlabel="log10(events per user)", ylabel="users")
    save(fig, fig_dir / "eda_user_history.png")

    # 4. watch ratio vs duration (duration bias of watch-ratio labels)
    dur_s = ev["duration_ms"].to_numpy() / 1000
    wr = ev["watch_ratio"].to_numpy()
    edges = np.array([0, 5, 7, 10, 15, 20, 30, 45, 60, 120, 600])
    b = np.clip(np.digitize(dur_s, edges) - 1, 0, len(edges) - 2)
    def q(i, x):
        v = wr[b == i]
        return float(np.quantile(v, x)) if len(v) else float("nan")

    med = [q(i, .5) for i in range(len(edges) - 1)]
    q25 = [q(i, .25) for i in range(len(edges) - 1)]
    q75 = [q(i, .75) for i in range(len(edges) - 1)]
    posr = [float((wr[b == i] >= cfg["labels"]["positive_wr"]).mean()) if (b == i).any() else float("nan")
            for i in range(len(edges) - 1)]
    out["watch_ratio_by_duration"] = {f"{edges[i]}-{edges[i + 1]}s": {"median": float(med[i]), "positive_rate": posr[i]}
                                      for i in range(len(edges) - 1)}
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(med))
    ax.fill_between(x, q25, q75, color=ACCENT, alpha=0.2, label="25-75%")
    ax.plot(x, med, color=ACCENT, marker="o", label="median watch ratio")
    ax.axhline(cfg["labels"]["positive_wr"], color=SERIES[1], ls="--", lw=1, label="positive threshold 0.8")
    ax.set_xticks(x, [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)], rotation=30)
    ax.set(title="Watch ratio falls with video length", xlabel="video duration (s)", ylabel="watch ratio")
    ax.legend(fontsize=8)
    save(fig, fig_dir / "eda_watch_ratio_duration.png")

    # 5. category entropy of each user's positives
    cat = items.set_index("video_id")["first_cat"].reindex(range(items["video_id"].max() + 1)).to_numpy()
    pos = ev.filter(pl.col("pos")).select("user_id", "video_id").with_columns(
        c=pl.Series(cat[ev.filter(pl.col("pos"))["video_id"].to_numpy()]))
    ce = pos.group_by(["user_id", "c"]).len().with_columns(p=pl.col("len") / pl.col("len").sum().over("user_id")).group_by(
        "user_id").agg(h=-(pl.col("p") * pl.col("p").log(2)).sum())
    out["user_category_entropy_bits"] = {"median": float(ce["h"].median()),
                                         "max_possible": float(np.log2(items["first_cat"].nunique()))}
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.hist(ce["h"].to_numpy(), bins=40, color=ACCENT)
    ax.set(title="Category entropy of each user's positives", xlabel="bits", ylabel="users")
    save(fig, fig_dir / "eda_category_entropy.png")

    # 6. activity cohorts
    stats = ev.group_by("user_id").agg(n=pl.len(), wr=pl.col("watch_ratio").clip(0, 5).mean(), pos=pl.col("pos").mean())
    bins = [0, 500, 1000, 2000, 4000, 10**9]
    labels = ["<500", "500-1k", "1k-2k", "2k-4k", "4k+"]
    st = stats.to_pandas()
    st["cohort"] = pd.cut(st["n"], bins, labels=labels, right=False)
    coh = st.groupby("cohort", observed=False).agg(users=("n", "size"), mean_wr=("wr", "mean"), pos_rate=("pos", "mean"))
    out["activity_cohorts"] = coh.reset_index().astype({"cohort": str}).to_dict("records")

    # 7. popularity deciles (share of interactions)
    dec_share = [float(cnt[int(i * len(cnt) / 10): int((i + 1) * len(cnt) / 10)].sum() / cnt.sum()) for i in range(10)]
    out["interaction_share_by_popularity_decile"] = dec_share
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.bar(np.arange(1, 11), dec_share, color=ACCENT)
    ax.set(title="Interactions by item popularity decile", xlabel="decile (1 = most watched)", ylabel="share")
    save(fig, fig_dir / "eda_popularity_deciles.png")

    # 8. big vs dense matrix
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bins_wr = np.linspace(0, 4, 81)
    ax.hist(np.clip(wr, 0, 4), bins=bins_wr, density=True, alpha=0.6, label="big matrix (logged feed)", color=ACCENT)
    ax.hist(np.clip(dense["watch_ratio"].to_numpy(), 0, 4), bins=bins_wr, density=True, alpha=0.6,
            label="small matrix (dense, 99.6% observed)", color=SERIES[1])
    ax.set(title="Watch ratio: logged vs fully observed exposure", xlabel="watch ratio (clipped at 4)", ylabel="density")
    ax.legend(fontsize=8)
    save(fig, fig_dir / "eda_big_vs_dense.png")
    out["median_watch_ratio"] = {"big": float(np.median(wr)), "dense": float(dense["watch_ratio"].median())}

    # fresh content: share of videos first watched on their upload day
    first = ev.group_by("video_id").agg(first=pl.col("day").min()).sort("video_id")
    up = items.set_index("video_id").loc[first["video_id"].to_numpy(), "upload_day"].to_numpy()
    out["videos_first_watched_on_upload_day"] = float((first["first"].to_numpy() == up).mean())
    out["videos_watched_before_listed_upload"] = float((first["first"].to_numpy() < up).mean())
    out["window_dates"] = {k: [w["start"], w["end"]] for k, w in cfg["protocol"]["windows"].items()}
    out["days"] = {"first": day_to_date(int(ev["day"].min())), "last": day_to_date(int(ev["day"].max())),
                   "test_end": cfg["protocol"]["end"], "test_end_day": date_to_day(cfg["protocol"]["end"])}
    write_json(cfg.reports / "metrics" / "eda.json", out)
    print({k: v for k, v in out.items() if not isinstance(v, (dict, list))})
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
