"""Research report + figures, generated only from persisted metrics (no retraining).

    python -m reelrank.evaluation.report   ->  reports/research_report.md, reports/figures/*.png
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from reelrank.config import Config, cli, read_json, write_json
from reelrank.evaluation.metrics import paired_bootstrap
from reelrank.evaluation.plots import MUTED, SERIES, plt, save, style

VARIANT_LABEL = {
    "main": "in-batch + per-day logQ (main)", "inbatch": "in-batch, no correction",
    "inbatch_globalq": "in-batch + global logQ", "uniform": "uniform sampled negatives",
    "freq": "frequency-sampled, no correction", "freq_logq": "frequency-sampled + logQ",
    "hard": "in-batch + logQ + ALS hard negatives", "id_only": "ID-only towers (no metadata)",
    "temp01": "main, temperature 0.1", "gru": "GRU history encoder", "sasrec": "SASRec history encoder",
    "cold": "cold-start holdout (metadata)", "cold_id_only": "cold-start holdout (ID only)",
}


def _m(cfg: Config, name: str) -> dict:
    f = cfg.reports / "metrics" / f"{name}.json"
    return read_json(f) if f.exists() else {}


def _runs(cfg: Config, proto: str) -> dict:
    d = cfg.artifacts / "runs" / proto
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}


def _pu(cfg: Config, proto: str) -> pl.DataFrame:
    return pl.read_parquet(cfg.artifacts / "runs" / proto / "per_user.parquet")


def compare(cfg: Config, proto: str, a: str, b: str, metric: str) -> dict:
    df = _pu(cfg, proto)
    x = df.filter(pl.col("system") == a).select("user_id", pl.col(metric).alias("a"))
    y = df.filter(pl.col("system") == b).select("user_id", pl.col(metric).alias("b"))
    j = x.join(y, on="user_id")
    return paired_bootstrap(j["a"].to_numpy().astype(float), j["b"].to_numpy().astype(float),
                            int(cfg["eval"]["bootstrap"]), cfg.seed)


def f4(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def ci(d: dict, digits: int = 4) -> str:
    return f"{d['delta']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------------------------ figures
def fig_ladder(cfg: Config, funnel: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, proto in zip(axes, ("test", "dense_test")):
        if proto not in funnel:
            continue
        lad = funnel[proto]["ladder"]
        labels = [r["system"] for r in lad]
        vals = [r["ndcg@20"] for r in lad]
        ax.barh(range(len(lad)), vals, color=[MUTED] * 2 + [SERIES[3], SERIES[0], SERIES[2], SERIES[1]][: len(lad) - 2])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_yticks(range(len(lad)), labels)
        ax.invert_yaxis()
        ax.set_title({"test": "Temporal test (Sep 1-5, observed labels)",
                      "dense_test": "Dense matrix, held-out users (fully labelled)"}[proto])
        ax.set_xlabel("NDCG@20")
        ax.set_xlim(0, max(vals) * 1.18)
    save(fig, cfg.reports / "figures" / "ladder.png")


def fig_pareto(cfg: Config, funnel: dict, selection: dict) -> None:
    on = funnel.get("select_on", "val")
    sw = funnel.get(on, {}).get("sweep", [])
    if not sw:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fam = lambda s: ("popularity penalty" if s["pop_penalty"] else "caps + novelty slot" if s["novelty_slots"]  # noqa: E731
                     else "category/creator caps" if s["category_cap"] else "MMR only")
    for i, f in enumerate(["MMR only", "category/creator caps", "caps + novelty slot", "popularity penalty"]):
        pts = [s for s in sw if fam(s) == f]
        ax.scatter([p["ndcg@20"] for p in pts], [p["ild_category"] for p in pts], label=f, color=SERIES[i], s=36)
    ch = next(s for s in sw if s["config"] == selection.get("rerank_name"))
    ax.scatter([ch["ndcg@20"]], [ch["ild_category"]], s=180, facecolors="none", edgecolors="black", lw=1.5,
               label=f"chosen: {ch['config']}")
    base = max(s["ndcg@20"] for s in sw if s["lam"] == 1.0 and not s["category_cap"] and not s["pop_penalty"])
    ax.axvline(base * (1 - cfg["rerank"]["max_ndcg_loss"]), color=MUTED, ls="--", lw=1)
    ax.set(xlabel=f"NDCG@20 ({on})", ylabel="intra-list category diversity",
           title=f"Reranking: relevance vs diversity ({on} users)")
    ax.legend(fontsize=7, loc="best")
    save(fig, cfg.reports / "figures" / "pareto.png")


def fig_negatives(cfg: Config, retr: dict, training: dict, test_runs: dict) -> None:
    val, test = retr.get("val", {}), retr.get("test", {})
    vs = [v for v in ["inbatch", "inbatch_globalq", "freq", "uniform", "freq_logq", "hard", "main", "id_only",
                      "temp01", "gru", "sasrec"] if f"tt_{v}" in val]
    if not vs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    y = np.arange(len(vs))
    axes[0].barh(y - 0.2, [val[f"tt_{v}"]["recall@100"] for v in vs], 0.4, label="validation", color=SERIES[0])
    axes[0].barh(y + 0.2, [test.get(f"tt_{v}", {}).get("recall@100", np.nan) for v in vs], 0.4, label="test",
                 color=SERIES[1])
    for ref, c in (("als", "black"), ("trending", SERIES[3])):
        if ref in test_runs:
            axes[0].axvline(test_runs[ref]["metrics"]["recall@100"], color=c, ls="--", lw=1.2, label=f"{ref} (test)")
    axes[0].set_yticks(y, [VARIANT_LABEL[v] for v in vs], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Recall@100", title="Two-tower variants")
    axes[0].legend(fontsize=7)
    for i, v in enumerate(["main", "inbatch", "inbatch_globalq", "uniform", "freq_logq", "hard"]):
        h = training.get(f"R1/{v}", {}).get("history", [])
        if h:
            axes[1].plot([r["epoch"] for r in h], [r.get("recall@100") for r in h], marker="o", ms=3,
                         label=VARIANT_LABEL[v], color=SERIES[i])
    axes[1].set(xlabel="epoch (500k sampled positives)", ylabel="val Recall@100", title="Learning curves")
    axes[1].legend(fontsize=7)
    save(fig, cfg.reports / "figures" / "negative_sampling.png")


def fig_deciles(cfg: Config, retr: dict) -> None:
    d = retr.get("deciles_test")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(1, 11)
    ax.bar(x - 0.2, d["interaction_share"], 0.4, color="#cbd5e1", label="training interactions")
    ax.bar(x + 0.2, d["target_share"], 0.4, color="#94a3b8", label="test-window positives")
    for i, s in enumerate(["popularity", "als", "tt_inbatch", "tt_main"]):
        if s in d["systems"]:
            ax.plot(x, d["systems"][s]["rec_share"], marker="o", ms=4, color=SERIES[i], label=f"{s} top-100 share")
    ax.set(xlabel="popularity decile before the cutoff (1 = most watched)", ylabel="share",
           title="Who gets exposure: recommendation share by popularity decile (test)")
    ax.set_xticks(x)
    ax.legend(fontsize=7, ncol=2)
    save(fig, cfg.reports / "figures" / "exposure_deciles.png")


def fig_ann(cfg: Config, ib: dict) -> None:
    if not ib:
        return
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for r in ib["indexes"]:
        c = SERIES[0] if "HNSW" in r["name"] else SERIES[1] if "IVF" in r["name"] else "black"
        ax.scatter(r["single"]["p50_ms"], r["recall@300_vs_exact"], color=c, s=30)
    ax.scatter([], [], color="black", label="exact (NumPy / FlatIP)")
    ax.scatter([], [], color=SERIES[0], label="HNSW (M, efSearch sweep)")
    ax.scatter([], [], color=SERIES[1], label="IVF-Flat (nprobe sweep)")
    ax.set_xscale("log")
    ax.set(xlabel="single-query p50 latency (ms, log)", ylabel="recall@300 vs exact",
           title=f"Vector search over {ib['items']:,} items (64-d)")
    ax.legend(fontsize=7)
    save(fig, cfg.reports / "figures" / "ann_tradeoff.png")


def fig_calibration(cfg: Config, nn: dict) -> None:
    cal = nn.get("calibration_test")
    if not cal:
        return
    fig, ax = plt.subplots(figsize=(4.4, 4))
    for i, head in enumerate(("complete", "skip")):
        t = cal[head]["reliability"]
        ax.plot([r["mean_pred"] for r in t], [r["observed"] for r in t], marker="o", color=SERIES[i],
                label=f"P({head}) ECE {cal[head]['ece']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED)
    ax.set(xlabel="predicted", ylabel="observed", title="Multitask heads: calibration (test)")
    ax.legend(fontsize=8)
    save(fig, cfg.reports / "figures" / "calibration.png")


# ------------------------------------------------------------------------------------------------ report
def run(cfg: Config) -> str:
    style()
    qa = read_json(cfg.processed / "qa_report.json")["tables"]
    fman = read_json(cfg.processed / "features" / "manifest.json")
    eda, base, retr, cand = _m(cfg, "eda"), _m(cfg, "baseline_metrics"), _m(cfg, "retrieval_eval"), _m(cfg, "candidate_recall")
    lgbm, nn, funnel, ib = _m(cfg, "ranker_lgb"), _m(cfg, "ranker_nn"), _m(cfg, "funnel"), _m(cfg, "index_benchmark")
    sel = _m(cfg, "selection")
    training = {f"{p.parent.parent.name}/{p.parent.name}": read_json(p)
                for p in sorted((cfg.artifacts / "two_tower").glob("*/*/info.json"))}
    fig_ladder(cfg, funnel)
    fig_pareto(cfg, funnel, sel)
    fig_negatives(cfg, retr, training, _runs(cfg, "test"))
    fig_deciles(cfg, retr)
    fig_ann(cfg, ib)
    fig_calibration(cfg, nn)
    variant = sel.get("retrieval_variant", "main")
    tt = f"tt_{variant}"
    R = {p: _runs(cfg, p) for p in ("gap", "val", "test", "dense")}
    get = lambda p, s, m: R[p].get(s, {}).get("metrics", {}).get(m)  # noqa: E731
    headline = {}
    L: list[str] = []
    w = L.append

    t_int = qa["interactions"]
    w("# ReelRank: research report\n")
    w(f"_Generated by `make report` from persisted metrics. Config `{cfg.name}` (hash `{cfg.hash()}`). "
      "All results are offline: there is no live traffic, and offline gains do not prove retention or watch-time uplift._\n")

    # ---- summary
    w("## 1. Summary\n")
    cmp_als = compare(cfg, "test", "als", tt, "recall@100") if tt in R["test"] and "als" in R["test"] else None
    cmp_tr = compare(cfg, "test", "trending", tt, "recall@100") if tt in R["test"] else None
    lad_t = funnel.get("test", {}).get("ladder", [])
    lad_d = funnel.get("dense_test", {}).get("ladder", [])
    rows = [["Events processed (after de-duplication)", f"{t_int['rows_after_cleaning']:,} of {t_int['raw_rows']:,} raw rows"],
            ["Users x videos", f"{qa['interactions']['users']:,} x {qa['interactions']['videos']:,}"]]
    if cmp_als:
        rows.append([f"Recall@100, two-tower ({variant}) vs ALS, temporal test",
                     f"{f4(get('test', tt, 'recall@100'))} vs {f4(get('test', 'als', 'recall@100'))}: "
                     f"Δ {ci(cmp_als)} (paired user bootstrap, {cmp_als['users']:,} users)"])
        headline["recall@100_tt_vs_als"] = cmp_als
    if cmp_tr:
        rows.append(["Recall@100, two-tower vs trending, temporal test", f"Δ {ci(cmp_tr)}"])
    for lad, name in ((lad_t, "temporal test"), (lad_d, "dense matrix, held-out users")):
        rk = next((r for r in lad if r["system"].startswith("+ ranker")), None)
        rr = next((r for r in lad if r["system"].startswith("+ reranker")), None)
        if rk:
            rows.append([f"NDCG@20 change from adding the ranker after two-tower retrieval ({name})",
                         f"{f4(lad[3]['ndcg@20'])} -> {f4(rk['ndcg@20'])}: Δ {ci(rk['vs_previous_ndcg@20'])}"])
        if rr:
            rows.append([f"NDCG@20 change from the diversity reranker ({name})",
                         f"{ci(rr['vs_previous_ndcg@20'])}; category ILD {rk['ild_category']:.3f} -> {rr['ild_category']:.3f}"])
    if "test" in funnel:
        r_ = funnel["test"]["reranked"]
        rows.append(["Catalog coverage@20 / exposure Gini (test, final feed)",
                     f"{pct(r_['coverage'])} / {r_['exposure_gini']:.3f}"])
    if ib:
        ex = next(r for r in ib["indexes"] if r["name"].startswith("numpy"))
        rows.append(["Exact retrieval latency (single query, top-300)", f"p50 {ex['single']['p50_ms']:.3f} ms, "
                     f"p95 {ex['single']['p95_ms']:.3f} ms over {ib['items']:,} items"])
    w(table(rows, ["", ""]) + "\n")

    # ---- data
    w("## 2. Data and quality checks\n")
    w(f"KuaiRec 2.0 (Gao et al., CIKM 2022; Zenodo record 18164998, MD5-verified). The big matrix logs a production "
      f"feed: {t_int['raw_rows']:,} raw rows. Checks on it:\n")
    w(f"- {t_int['exact_duplicate_rows_removed']:,} exact duplicate rows "
      f"({pct(t_int['exact_duplicate_rows_removed'] / t_int['raw_rows'])}) were removed; the heaviest days are "
      + ", ".join(f"{d} ({n:,})" for d, n in t_int.get("duplicate_heaviest_days", {}).items())
      + f". {t_int['same_key_conflicts_removed']} more rows shared (user, video, timestamp) with a different play time "
      "and were collapsed.")
    w(f"- No missing timestamps, negative play times or non-positive durations; watch ratio equals play/duration on "
      f"every row. {pct(t_int['watch_ratio_gt_1_share'])} of rows have watch ratio > 1 (replays).")
    w(f"- {t_int['repeat_consumption_events']:,} events are repeat views of an already-watched video. The main task is "
      "novel-item recommendation; repeat share of positives is reported per window.")
    w(f"- The `date` column disagrees with the Unix timestamp on {t_int.get('date_column_disagrees_with_timestamp', 0):,} "
      "rows; calendar days come from the timestamp (UTC+8).")
    if eda:
        w(f"- Events fall on {eda['active_days']} busy days in three bursts separated by multi-week gaps (figure below).")
    d = qa["dense"]
    w(f"- Small matrix: {d['rows']:,} rows, {d['users']:,} users x {d['videos']:,} videos, density {pct(d['density'])}; "
      f"{d['null_timestamp']:,} rows have no timestamp (kept as labels). {d['pairs_also_in_big_matrix']} pairs overlap "
      "the big matrix, so dense evaluation cannot leak through training pairs.")
    it = qa["items"]
    w(f"- {it['caption_rows_malformed_dropped']} caption rows were malformed and dropped; every video still has metadata. "
      f"{it['creators']:,} creators (creator IDs exist, so the reranker can cap per-creator repetition).")
    if eda:
        w(f"- {pct(eda['videos_first_watched_on_upload_day'])} of videos are first watched on their upload day, and "
          f"{pct(eda['videos_watched_before_listed_upload'])} are watched *before* their listed upload date, so upload "
          "dates are used for eligibility only together with observed views.")
        w(f"- Item exposure Gini {eda['item_interaction_gini']:.2f}; the top 10% of videos take "
          f"{pct(eda['top_10pct_items_share'])} of interactions. Watch ratio falls steeply with duration (median "
          f"{eda['watch_ratio_by_duration']['0-5s']['median']:.2f} under 5 s vs "
          f"{eda['watch_ratio_by_duration']['60-120s']['median']:.2f} at 60-120 s), so `watch_ratio >= 0.8` favours short videos.")
    w("\n![daily volume](figures/eda_daily_volume.png)\n![watch ratio vs duration](figures/eda_watch_ratio_duration.png)\n")

    # ---- protocol
    w("## 3. Evaluation protocol\n")
    rows = []
    for name, wnd in cfg["protocol"]["windows"].items():
        ev = fman["eval"][name]
        rows.append([name, f"{wnd['start']} -> {wnd['end']}", wnd["model"], f"{ev['users_with_novel_positive']:,}",
                     f"{ev['novel_positive_pairs']:,}", pct(ev["repeat_share_of_positive_pairs"]),
                     f"{ev['eligible_items']:,}", f"{ev['ineligible_positive_pairs_dropped']:,}"])
    w(table(rows, ["window", "dates (UTC+8)", "model", "users", "novel positive pairs", "repeat share of positives",
                   "eligible videos", "positives on not-yet-eligible videos"]) + "\n")
    w("- Each model family trains only on days strictly before its window. R0 scores the ranker's training window, "
      "R1 is used for every selection decision, R2 is trained once with the selected settings and is the only "
      "model evaluated on test and on the dense matrix. The retrieval variant is selected on temporal validation; the "
      "ranker and reranker on half of the dense users (Section 10). The test window and the other dense half were read "
      "after selection.")
    w("- A feature snapshot for day *d* summarises events from days < *d*; same-day events never see each other, which "
      "resolves timestamp ties conservatively. Daily platform statistics become visible the day after they are dated.")
    w("- Positive = watch ratio >= 0.8 (a proxy, not intent). Graded relevance for NDCG: 0 skip (< 0.2), 1 partial, "
      "2 high watch (>= 0.8), 3 completion (>= 1.0). Unobserved pairs are not negatives; on the temporal protocol "
      "they simply earn no gain, so temporal metrics are observed-positive metrics with incomplete labels.")
    w("- Temporal and dense numbers answer different questions and are never compared in absolute terms.")
    w("- Uncertainty: paired user bootstrap (1,000 resamples) over the users both systems were evaluated on.\n")

    # ---- retrieval
    w("## 4. Retrieval\n")
    sysorder = ["random", "popularity", "trending", "category", "itemknn", "als", "tt_inbatch", "tt_inbatch_globalq",
                "tt_uniform", "tt_freq", "tt_freq_logq", "tt_hard", "tt_main", "tt_temp01", "tt_id_only", "tt_gru",
                "tt_sasrec"]
    rows = []
    for s in sysorder:
        if s in R["test"] or s in R["val"]:
            lab = VARIANT_LABEL.get(s.removeprefix("tt_"), s) if s.startswith("tt_") else s
            rows.append([lab, f4(get("val", s, "recall@100")),
                         f4(get("test", s, "recall@50")), f4(get("test", s, "recall@100")),
                         f4(get("test", s, "recall@300")), f4(get("test", s, "ndcg@20")),
                         f4(get("test", s, "coverage@20")), f4(get("dense", s, "precision@20")), f4(get("dense", s, "ndcg@20"))])
    w(table(rows, ["system", "val R@100", "test R@50", "test R@100", "test R@300", "test NDCG@20", "test coverage@20",
                   "dense P@20", "dense NDCG@20"]) + "\n")
    als_sel = base.get("als_selected", {})
    w(f"ALS was tuned on validation: {als_sel.get('factors')} factors, regularization {als_sel.get('regularization')}, "
      f"alpha {als_sel.get('alpha')}, recency half-life {als_sel.get('half_life_days')} days. Recency weighting was the "
      "single largest ALS effect: see `reports/metrics/als_tuning.json` (every half-life-7 setting beats every "
      "un-decayed one on validation).\n")
    if cmp_als:
        verdict = "beats" if cmp_als["lo"] > 0 else ("is worse than" if cmp_als["hi"] < 0 else "is not distinguishable from")
        w(f"On the temporal test, the selected two-tower ({VARIANT_LABEL.get(variant, variant)}) {verdict} ALS on "
          f"Recall@100: Δ {ci(cmp_als)} ({pct(cmp_als['relative'])} relative).")
    for s in ("popularity", "itemknn", "als"):
        if s in R["dense"] and tt in R["dense"]:
            c = compare(cfg, "dense", s, tt, "ndcg@20")
            w(f"- Dense matrix, NDCG@20 two-tower - {s}: {ci(c)}.")
    w("\n![negative sampling](figures/negative_sampling.png)\n")

    # ---- negative sampling
    w("## 5. Negative sampling and the sampling-bias correction\n")
    if "tt_main" in R["val"] and "tt_inbatch_globalq" in R["val"]:
        w("Training batches are drawn from a single day so in-batch negatives share that day's item features. That "
          "makes the in-batch sampling distribution the day's item frequency, not the global one. Subtracting log q "
          "with the actual (per-day) distribution is what makes in-batch training work here:\n")
        rows = []
        for v in ("inbatch", "inbatch_globalq", "main", "uniform", "freq", "freq_logq", "hard"):
            s = f"tt_{v}"
            if s in R["val"]:
                rows.append([VARIANT_LABEL[v], f4(get("val", s, "recall@100")), f4(get("test", s, "recall@100")),
                             f4(get("test", s, "long_tail_share@20")), f4(get("test", s, "coverage@20"))])
        w(table(rows, ["negatives", "val R@100", "test R@100", "test long-tail share@20", "test coverage@20"]) + "\n")
        for a, b, label in (("tt_inbatch_globalq", "tt_main", "per-day vs global logQ"),
                            ("tt_freq", "tt_freq_logq", "logQ on frequency-sampled negatives"),
                            ("tt_main", "tt_hard", "adding ALS hard negatives")):
            if a in R["test"] and b in R["test"]:
                w(f"- {label}: test Recall@100 Δ {ci(compare(cfg, 'test', a, b, 'recall@100'))}")
        w("\nThe correction addresses sampling bias in the training objective; it does not remove exposure bias of the "
          "logging policy that produced the data.\n")

    # ---- ablations
    w("## 6. Architecture ablations\n")
    rows = []
    for v in ("main", "id_only", "temp01", "gru", "sasrec"):
        s = f"tt_{v}"
        if s in R["val"]:
            tr = training.get(f"R1/{v}", {})
            rows.append([VARIANT_LABEL[v], f4(get("val", s, "recall@100")), f4(get("test", s, "recall@100")),
                         f4(get("dense", s, "ndcg@20")), f"{tr.get('examples_per_second', 0):,.0f}",
                         f"{tr.get('train_seconds', 0):.0f}", f"{tr.get('parameters', 0):,}"])
    w(table(rows, ["variant", "val R@100", "test R@100", "dense NDCG@20", "train examples/s", "train s (R1)",
                   "parameters"]) + "\n")
    for a, b in (("tt_id_only", "tt_main"), ("tt_main", "tt_gru"), ("tt_main", "tt_sasrec")):
        if a in R["test"] and b in R["test"]:
            w(f"- {b} - {a}, test Recall@100: {ci(compare(cfg, 'test', a, b, 'recall@100'))}")
    seq = [s for s in ("tt_gru", "tt_sasrec") if s in R["test"] and tt in R["test"]]
    if seq and any(get("val", s, "recall@100") < get("val", tt, "recall@100") for s in seq):
        cost = training.get("R1/sasrec", {}).get("train_seconds", 0) / max(training.get("R1/main", {}).get("train_seconds", 1), 1)
        w(f"\nThe sequence encoders were not selected: they scored lower on validation, although they score higher on "
          f"test. Selection followed validation only. They also cost about {cost:.0f}x the training time of mean pooling "
          "on CPU. With more validation data or tuning they are the natural next step.")
    w("")

    # ---- exposure & cold start
    w("## 7. Popularity exposure and cold start\n")
    w("![exposure](figures/exposure_deciles.png)\n")
    dt_ = retr.get("deciles_test", {})
    if dt_:
        rows = []
        for s, v in dt_["systems"].items():
            rows.append([s, pct(v["rec_share"][0]), pct(sum(v["rec_share"][5:])),
                         f4(v["recall_by_decile"][0]), f4(v["recall_by_decile"][5])])
        w(table(rows, ["system", "top-100 share from decile 1", "share from deciles 6-10", "recall on decile-1 targets",
                       "recall on decile-6 targets"]) + "\n")
        w(f"Decile 1 holds {pct(dt_['interaction_share'][0])} of training interactions and {pct(dt_['target_share'][0])} "
          "of test positives.\n")
    cs = retr.get("cold_start", {})
    if cs:
        w(f"Held-out items. {cs['held_out_items']:,} items with training history were removed from training and from "
          "every user history; their popularity and platform statistics are zeroed at serving time, so only static "
          "metadata (tags, category, duration, creator) is left.\n")
        rows = [[k, f4(v.get("within_cold_recall@20")), f4(v.get("within_cold_recall@50")), f4(v["cold_recall@100"]),
                 f4(v["warm_recall@100"]), pct(v["cold_share_of_top100"])] for k, v in cs.items() if isinstance(v, dict)]
        rows.append(["random order within the cold catalog", f4(cs.get("random_within_cold_recall@20")),
                     f4(cs.get("random_within_cold_recall@50")), "", "", ""])
        w(table(rows, ["model", "Recall@20 within cold catalog", "Recall@50 within cold catalog",
                       "cold-item Recall@100 (full catalog)", "warm-item Recall@100", "cold share of top-100"]) + "\n")
        w(f"{cs.get('cold_catalog_size', 0):,} held-out videos are eligible at the test cutoff. In the full catalog a "
          "history-less video never reaches the top 100: the item tower leans on popularity and recency, which a cold "
          "video does not have. *Within* the cold catalog, the metadata tower shows whether static features say which new "
          "video a user will like; in production that would feed a dedicated fresh-content slot.\n")
    co = retr.get("cohorts", {})
    if co:
        cohorts = [c for c, v in next(iter(co.values())).items() if v["users"] > 0]
        rows = [[s] + [f4(co[s][c]["recall@100"]) for c in cohorts] for s in co]
        w("Low-history users (test Recall@100 by events before the cutoff; users per cohort: "
          + ", ".join(f"{c}: {next(iter(co.values()))[c]['users']:,}" for c in cohorts) + ")\n")
        w(table(rows, ["system"] + cohorts) + "\n")

    # ---- candidates
    w("## 8. Candidate generation\n")
    if cand:
        rows = [[p, f"{cand[p]['mean_pool_size']:.0f}", f4(cand[p]["recall_union"]), f4(cand[p]["recall_two_tower"]),
                 f4(cand[p]["recall_als"]), f4(cand[p]["recall_trending"]), f4(cand[p]["recall_category"])]
                for p in ("val", "test", "dense") if p in cand]
        w(table(rows, ["protocol", "mean pool", "union recall", "two-tower (200)", "ALS (100)", "trending (30)",
                       "category (20)"]) + "\n")
        w("Union recall is the ceiling for everything downstream: the ranker can only reorder what retrieval found.\n")

    # ---- ranking
    w("## 9. Ranking\n")
    w("Rankers train only on observed events with known outcomes (validation protocol: Aug 6-10 with R0 retrieval "
      "scores; test protocol: Aug 29-31 with R1 scores). Unobserved retrieved candidates are never labelled as skips.\n")
    if lgbm:
        rows = []
        merged = {**lgbm.get("test", {}), **nn.get("test", {})}
        for s, m in sorted(merged.items(), key=lambda kv: -kv[1]["ndcg@20"]):
            rows.append([s, f4(lgbm.get("val", {}).get(s, nn.get("val", {}).get(s, {})).get("ndcg@20")), f4(m["ndcg@20"]),
                         f4(m["mrr"]), f"{m['top10_watch_ratio']:.3f}", pct(m["top10_completion"]), pct(m["top10_skip"]),
                         f"{m['top10_duration_s']:.1f}"])
        w("Logged impressions (re-order each user-day's observed videos; "
          f"{lgbm['test']['lgb_rank']['groups']:,} test user-days):\n")
        w(table(rows, ["scorer", "val NDCG@20", "test NDCG@20", "test MRR", "top-10 watch ratio", "top-10 completion",
                       "top-10 skip", "top-10 mean duration (s)"]) + "\n")
        fi = list(lgbm.get("feature_importance_gain", {}).items())[:8]
        w("LambdaMART gain importance (top 8): " + ", ".join(f"`{k}` {v:.2f}" for k, v in fi) + ".\n")
    if nn:
        uw = nn["utility_weights"]
        w(f"Multitask neural ranker. Utility = {uw['w_watch']}*z(watch) + {uw['w_complete']}*P(complete) - "
          f"{uw['w_skip']}*P(skip), weights chosen on validation NDCG@20. The utility is a ranking score, not a probability.")
        cal = nn.get("calibration_test", {})
        if cal:
            w(f"Calibration on test: P(complete) ECE {cal['complete']['ece']:.3f} (base rate {pct(cal['complete']['base_rate'])}), "
              f"P(skip) ECE {cal['skip']['ece']:.3f} (base rate {pct(cal['skip']['base_rate'])}).")
        db = nn.get("duration_bias_test", {})
        if db:
            w(f"Duration. Mean length of the top-10 observed videos: {db.get('nn_watch_only', 0):.1f} s when ranked by "
              f"predicted watch *time*, {db.get('nn_complete_only', 0):.1f} s by P(complete), {db.get('nn_utility', 0):.1f} s "
              f"by the selected utility; {db.get('positive_mean_s', 0):.1f} s for actual positives and "
              f"{db.get('all_observed_mean_s', 0):.1f} s across all observed videos. Watch time rewards length while the "
              "watch-ratio label rewards brevity; the validation-chosen utility puts zero weight on watch time.\n")
        w("![calibration](figures/calibration.png)\n")

    # ---- end to end
    w("## 10. End-to-end funnel and ablation ladder\n")
    if funnel:
        on = funnel.get("select_on", "val")
        rk_names = list(funnel["test"]["rankers"])
        rows = [[r] + [f4(funnel[p]["rankers"][r]["ndcg@20"]) for p in ("val", "test", "dense_val", "dense_test")]
                for r in rk_names]
        w("Why the two protocols disagree. Each ranker reorders the same retrieval pool (NDCG@20):\n")
        w(table(rows, ["scorer on the candidate pool", "temporal val", "temporal test", "dense val", "dense test"]) + "\n")
        w("Rankers trained on logged impressions learn *engagement given exposure*. On the dense matrix every user has "
          "been shown every video, and they are by far the best scorers. Temporal labels exist only for videos the "
          "production policy chose to show, so temporal NDCG also rewards predicting that *exposure*. Retrieval models "
          "trained on what was watched learn it; impression-trained rankers cannot. This is the logging-policy "
          "(exposure) bias KuaiRec was built to expose.\n")
        w(f"Selection rule. {funnel.get('selection_note', '')} The dense users are split by a fixed seed: "
          f"{funnel['dense_users']['dense_val']} for selection (`dense_val`), {funnel['dense_users']['dense_test']} "
          f"reported only (`dense_test`). Selected ranker: `{sel.get('ranker')}` (on `{on}`).\n")
    w("![ladder](figures/ladder.png)\n")
    for proto, lad in (("test", lad_t), ("dense_test", lad_d)):
        if not lad:
            continue
        rows = [[r["system"], f4(r["ndcg@20"]), f4(r["recall@20"]), f4(r["precision@20"]), f"{r['ild_category']:.3f}",
                 ci(r["vs_previous_ndcg@20"]) if "vs_previous_ndcg@20" in r else ""] for r in lad]
        w(f"{proto} ({funnel[proto]['users']:,} users)\n")
        w(table(rows, ["system", "NDCG@20", "Recall@20", "Precision@20", "category ILD", "Δ NDCG@20 vs previous [95% CI]"]) + "\n")
    w("Retrieval-only rows rank the full eligible catalog; ranker rows reorder the shared candidate union. All rows use "
      "the same eligibility (novel, already-uploaded videos) and relevance definition.\n")

    # ---- rerank
    w("## 11. Reranking: relevance versus diversity\n")
    w("![pareto](figures/pareto.png)\n")
    on = funnel.get("select_on", "val") if funnel else "val"
    if funnel and "sweep" in funnel.get(on, {}):
        w(f"Operating point chosen on `{on}`: {sel.get('rerank_name')} (the most category-diverse configuration "
          f"within {pct(cfg['rerank']['max_ndcg_loss'])} of relevance-only NDCG@20).")
        rows = []
        for p in ("val", "test", "dense_test"):
            if p in funnel:
                b, r_ = funnel[p]["rankers"][sel.get("ranker", "lgb_rank")], funnel[p]["reranked"]
                rows.append([p, f"{b['ndcg@20']:.4f} -> {r_['ndcg@20']:.4f}", f"{b['ild_category']:.3f} -> {r_['ild_category']:.3f}",
                             f"{b['category_entropy']:.2f} -> {r_['category_entropy']:.2f}", f"{pct(b['coverage'])} -> {pct(r_['coverage'])}",
                             f"{b['exposure_gini']:.3f} -> {r_['exposure_gini']:.3f}", f"{pct(b['long_tail_share'])} -> {pct(r_['long_tail_share'])}",
                             f"{r_['rerank_p95_ms']:.2f}"])
        w(table(rows, ["protocol", "NDCG@20", "category ILD", "category entropy (bits)", "coverage@20", "exposure Gini",
                       "long-tail share", "rerank p95 (ms)"]) + "\n")
        w("More diversity costs some measured relevance, so where to sit on this curve is a product call.\n")

    # ---- robustness
    w("## 12. Robustness: a cutoff right after a 16-day gap\n")
    if R["gap"]:
        rows = [[s, f4(get("gap", s, "recall@100")), f4(get("val", s, "recall@100"))]
                for s in ("random", "popularity", "trending", "itemknn", "als", tt) if s in R["gap"]]
        w(table(rows, ["system", "post-gap R@100 (Aug 27-28)", "in-burst val R@100 (Aug 29-31)"]) + "\n")
        rnd = get("gap", "random", "recall@100")
        w(f"After a multi-week gap most watched videos are new. Every history-based method drops sharply; the best "
          f"post-gap system ({tt}) keeps {f4(get('gap', tt, 'recall@100'))} Recall@100 ({get('gap', tt, 'recall@100') / rnd:.1f}x "
          f"random), and popularity falls to {f4(get('gap', 'popularity', 'recall@100'))}, below random ({f4(rnd)}). A feed "
          "would need fresh-content retrieval (metadata towers, real-time trending) rather than history alone.\n")

    # ---- engineering
    w("## 13. Engineering\n")
    if ib:
        rows = [[r["name"], f"{r['recall@300_vs_exact']:.4f}", f"{r['single']['p50_ms']:.3f}", f"{r['single']['p95_ms']:.3f}",
                 f"{r['batched']['p50_ms']:.2f}", f"{r['memory_mb']:.1f}", f"{r['build_s']:.2f}"] for r in ib["indexes"]]
        w(table(rows, ["index", "recall@300 vs exact", "p50 ms", "p95 ms", "batch-256 p50 ms", "memory MB", "build s"]) + "\n")
        w(f"{ib['note']} Measured on {ib['machine']}, FAISS {ib['faiss_version']}, {ib['faiss_threads']} thread(s), "
          "warmed, fixed seeds. These numbers say nothing about web-scale serving.\n")
        w("![ann](figures/ann_tradeoff.png)\n")
    ms = _m(cfg, "monitoring_sim")
    if ms:
        w(f"Serving replay (offline simulation). {ms['requests']:,} requests (test users plus 2% unknown IDs) through "
          f"the full engine: p50 {ms['latency_ms']['p50']:.1f} ms, p95 {ms['latency_ms']['p95']:.1f} ms, p99 "
          f"{ms['latency_ms']['p99']:.1f} ms; stage p95 "
          + ", ".join(f"{k.removesuffix('_ms')} {v:.1f} ms" for k, v in ms["stage_p95_ms"].items())
          + "; fallback rate " + ", ".join(f"{k} {pct(v)}" for k, v in ms["fallback_rate"].items())
          + f". Drift val -> test: ranker-score PSI {ms['score_psi_val_to_test']:.3f}, category-mix JS "
          f"{ms['category_mix_js_bits']:.3f} bits, item-exposure JS {ms['item_exposure_js_bits']:.3f} bits; alerts: "
          f"{', '.join(ms['alerts']) or 'none'}.\n")
    w("Process isolation. On macOS the PyTorch, FAISS and LightGBM wheels each bundle an OpenMP runtime. Measured on "
      "the development machine: torch+FAISS segfaults, torch+LightGBM segfaults or deadlocks depending on import order, "
      "and FAISS+LightGBM deadlocks; implicit (ALS) is safe with all three. Every pipeline stage is therefore its own "
      "process, artifacts move as `.npy`/Parquet, serving runs the two-tower as a NumPy forward pass (parity with PyTorch "
      "checked at export), and `reelrank.runtime.claim()` refuses to load a conflicting runtime.\n")

    # ---- negative results
    w("## 14. Negative results and caveats\n")
    neg = []
    if "tt_inbatch_globalq" in R["val"]:
        neg.append(f"In-batch negatives with the textbook global logQ correction reach only "
                   f"{f4(get('val', 'tt_inbatch_globalq', 'recall@100'))} val Recall@100, no better than no correction "
                   f"({f4(get('val', 'tt_inbatch', 'recall@100'))}); same-day batching changes the sampler.")
    for s in ("tt_gru", "tt_sasrec"):
        if s in R["test"] and tt in R["test"]:
            c = compare(cfg, "test", tt, s, "recall@100")
            if c["hi"] < 0 or (c["lo"] < 0 < c["hi"]):
                neg.append(f"{VARIANT_LABEL[s[3:]]} does not beat mean pooling on test Recall@100 (Δ {ci(c)}) at "
                           f"{training.get('R1/' + s[3:], {}).get('train_seconds', 0) / max(training.get('R1/main', {}).get('train_seconds', 1), 1):.1f}x "
                           "the training time.")
    if "dense" in R and "itemknn" in R["dense"] and tt in R["dense"]:
        c = compare(cfg, "dense", "itemknn", tt, "ndcg@20")
        if c["hi"] < 0:
            neg.append(f"On the dense matrix, item-kNN beats the two-tower on NDCG@20 (Δ {ci(c)}): the temporal and dense "
                       "protocols reward different things (recency vs stable taste over a fixed catalog).")
    if R["gap"]:
        neg.append("After the 16-day gap every history-based method loses most of its recall, and popularity falls below "
                   "random (Section 12).")
    fi = lgbm.get("feature_importance_gain", {})
    dur_gain = fi.get("i_log_dur", 0) + fi.get("dur_gap", 0)
    if fi and max(fi, key=fi.get) in ("i_log_dur", "dur_gap"):
        neg.append(f"The ranker's strongest signal is video length: `i_log_dur` and `dur_gap` carry {pct(dur_gain)} of "
                   "LambdaMART gain. The watch-ratio label rewards short videos (Section 2), so much of the ranking "
                   "lift is 'shorter first'. Retrieval-score features add almost nothing on logged impressions "
                   f"(NDCG@20 {f4(lgbm['test']['lgb_rank']['ndcg@20'])} with them vs "
                   f"{f4(lgbm['test']['lgb_rank_no_retrieval']['ndcg@20'])} without). A production system would want "
                   "a duration-normalised engagement label (e.g. watch-ratio percentile within a duration bucket).")
    else:
        neg.append("The positive label (watch ratio >= 0.8) is strongly duration-biased (Section 2); a production "
                   "system would want duration-normalised engagement labels.")
    if funnel and "dense_test" in funnel:
        rk = sel.get("ranker", "lgb_rank")
        neg.append(f"The impression-trained ranker (`{rk}`) lowers temporal-test NDCG@20 "
                   f"({f4(funnel['test']['rankers']['tt_score']['ndcg@20'])} -> {f4(funnel['test']['rankers'][rk]['ndcg@20'])}) "
                   f"while raising held-out dense NDCG@20 ({f4(funnel['dense_test']['rankers']['tt_score']['ndcg@20'])} -> "
                   f"{f4(funnel['dense_test']['rankers'][rk]['ndcg@20'])}). Which system 'wins' depends on whether the "
                   "metric rewards predicting the logging policy's exposure; a production ranker would need an "
                   "exposure-aware objective (e.g. inverse-propensity weighting with logged propensities, which KuaiRec "
                   "does not provide).")
    neg.append("KuaiRec has no impressions without plays, so 'skip' means a short play, not a scroll-past.")
    for n in neg:
        w(f"- {n}")
    w("")

    # ---- online experiment
    w("## 15. Proposed online experiment (not run)\n")
    w("- Design: user-level randomised A/B test, 50/50, control = current production ranker, treatment = ReelRank "
      "funnel at the chosen MMR operating point. Randomise on user ID hash; two full weekly cycles minimum.")
    w("- Primary metric: watch time per user per day (log-transformed for analysis), with CUPED on the pre-period.")
    w("- Secondary: completion rate, sessions per user, 7-day return rate.")
    w("- Guardrails: early-skip rate, hides/reports per 1k impressions, creator exposure Gini, share of impressions "
      "from new uploads, p95 feed latency. Any guardrail breach blocks launch.")
    w("- Power: size the test on the pre-period variance of the primary metric for a 1% minimum detectable effect; "
      "check sample-ratio mismatch daily and stop on SRM.")
    w("- Offline improvements reported here are *not* evidence of retention or watch-time uplift.\n")

    w("## 16. Reproducibility\n")
    w("`make all` downloads the data, verifies its MD5 and rebuilds every number in this report (~2 h on an Apple M4 "
      "CPU). Seeds are fixed in `configs/kuairec.yaml`; every training run is tracked in a local MLflow store "
      "(`make mlflow-ui`). `make smoke` runs the identical pipeline on a synthetic fixture in about two minutes.\n")
    text = "\n".join(L)
    (cfg.reports / "research_report.md").write_text(text)
    if cfg.name == "kuairec":
        update_readme(cfg, readme_block(cfg, R, retr, funnel, ib, _m(cfg, "monitoring_sim"), t_int, tt, variant, cmp_als))
    write_json(cfg.reports / "metrics" / "headline.json", headline)
    print(f"wrote {cfg.reports / 'research_report.md'} ({len(text):,} chars)")
    return text


def readme_block(cfg: Config, R: dict, retr: dict, funnel: dict, ib: dict, ms: dict, t_int: dict, tt: str,
                 variant: str, cmp_als: dict | None) -> str:
    get = lambda p, s, m: R[p].get(s, {}).get("metrics", {}).get(m)  # noqa: E731
    rows = [["Events processed", f"{t_int['rows_after_cleaning']:,} real interactions "
             f"({t_int['exact_duplicate_rows_removed']:,} duplicate rows removed), 7,176 users x 10,728 videos"]]
    if cmp_als:
        rows.append(["Retrieval, temporal test Recall@100",
                     f"two-tower {f4(get('test', tt, 'recall@100'))} vs ALS {f4(get('test', 'als', 'recall@100'))}, "
                     f"trending {f4(get('test', 'trending', 'recall@100'))}, popularity {f4(get('test', 'popularity', 'recall@100'))}"
                     f": Δ vs ALS {ci(cmp_als)} ({pct(cmp_als['relative'])} relative)"])
    if "tt_inbatch_globalq" in R["val"]:
        rows.append(["Sampling-bias correction (val Recall@100)",
                     f"no correction {f4(get('val', 'tt_inbatch', 'recall@100'))}, global logQ "
                     f"{f4(get('val', 'tt_inbatch_globalq', 'recall@100'))}, per-day logQ "
                     f"{f4(get('val', 'tt_main', 'recall@100'))}"])
    for proto, name in (("test", "temporal test"), ("dense_test", "dense matrix, held-out users")):
        lad = funnel.get(proto, {}).get("ladder", [])
        rk = next((r for r in lad if r["system"].startswith("+ ranker")), None)
        rr = next((r for r in lad if r["system"].startswith("+ reranker")), None)
        if rk and rr:
            rows.append([f"Funnel NDCG@20, {name}",
                         f"retrieval {f4(lad[3]['ndcg@20'])} -> + ranker {f4(rk['ndcg@20'])} "
                         f"(Δ {ci(rk['vs_previous_ndcg@20'])}) -> + reranker {f4(rr['ndcg@20'])} "
                         f"(Δ {ci(rr['vs_previous_ndcg@20'])}, category ILD {rk['ild_category']:.3f} -> {rr['ild_category']:.3f})"])
    if "test" in funnel:
        r_ = funnel["test"]["reranked"]
        rows.append(["Final feed, temporal test", f"catalog coverage@20 {pct(r_['coverage'])}, exposure Gini "
                     f"{r_['exposure_gini']:.3f}, long-tail share {pct(r_['long_tail_share'])}"])
    cs = retr.get("cold_start", {})
    meta, ido = cs.get("tt_cold (metadata + unknown id)"), cs.get("tt_cold_id_only (unknown id only)")
    if meta and ido:
        rows.append(["Cold-start videos (480 held-out IDs), Recall@20 within the cold catalog",
                     f"metadata tower {f4(meta['within_cold_recall@20'])} vs random "
                     f"{f4(cs['random_within_cold_recall@20'])} vs ID-only {f4(ido['within_cold_recall@20'])}"])
    if ms:
        rows.append(["Serving latency (full funnel, offline replay)",
                     f"p50 {ms['latency_ms']['p50']:.1f} ms, p95 {ms['latency_ms']['p95']:.1f} ms over {ms['requests']:,} requests"])
    if ib:
        ex = next(r for r in ib["indexes"] if r["name"].startswith("numpy"))
        rows.append(["Exact vector search", f"top-300 over {ib['items']:,} items: p50 {ex['single']['p50_ms']:.3f} ms"])
    body = table(rows, ["", ""])
    return (body + "\n\nThe ranker helps on the dense matrix and hurts on the temporal test, and that is the main finding. "
            "Rankers trained on logged impressions learn engagement *given exposure*. The dense matrix has every video "
            "shown to every user, so it measures that directly; temporal labels exist only where the production feed chose "
            "to show a video, so they also reward predicting the logging policy. Ranker and reranker are selected on half "
            "of the dense users and reported on the other half ([report, Section 10](reports/research_report.md))."
            "\n\nConfidence intervals are 95% paired user bootstraps. Temporal metrics count only observed positives; "
            "dense-matrix metrics use a ~99.6%-labelled user x video matrix. The two are never compared in absolute terms.\n\n"
            "![ablation ladder](reports/figures/ladder.png)\n")


def update_readme(cfg: Config, block: str) -> None:
    from reelrank.config import ROOT

    p = ROOT / "README.md"
    s = p.read_text()
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a in s and b in s:
        p.write_text(s[: s.index(a) + len(a)] + "\n" + block + "\n" + s[s.index(b):])


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
