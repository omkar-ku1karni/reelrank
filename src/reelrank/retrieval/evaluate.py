"""Evaluate exported two-tower vectors with exact (NumPy) search; no PyTorch in this process.

    python -m reelrank.retrieval.evaluate

For every trained variant and protocol: top-300 lists, Recall/Precision/NDCG, exposure. Also the cold-start
experiment (warm vs held-out item targets, low-history user cohorts) and exposure by popularity decile.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from reelrank.config import Config, cli, day_to_date, read_json, write_json
from reelrank.evaluation.metrics import ranking_metrics, topk
from reelrank.evaluation.protocol import Protocol, load_protocol, recommend
from reelrank.evaluation.store import load_summaries, print_table, save_run


def vectors(cfg: Config, tag: str, variant: str, day: int) -> tuple[np.ndarray, np.ndarray] | None:
    d = cfg.artifacts / "two_tower" / tag / variant
    f = d / f"items_{day_to_date(day)}.npy"
    if not f.exists():
        return None
    return np.load(d / f"users_{day_to_date(day)}.npy"), np.load(f)


def tt_scorer(uv: np.ndarray, iv: np.ndarray):
    return lambda u: uv[u] @ iv.T


def cold_start(cfg: Config, p: Protocol, K: int) -> dict:
    """Warm vs held-out-item recall for the cold variants, against the same items left warm in `main`."""
    info = cfg.artifacts / "two_tower" / p.model_tag / "cold" / "weights.npz"
    if not info.exists():
        return {}
    held = np.load(info)["held_out"]
    out = {"held_out_items": int(held.sum())}
    systems = {"tt_main (items warm)": "main", "tt_cold (metadata + unknown id)": "cold",
               "tt_cold_id_only (unknown id only)": "cold_id_only"}
    rel_cold = np.where(held[None, :], p.rel, 0)
    rel_warm = np.where(held[None, :], 0, p.rel)
    cold_catalog = held & p.candidate_mask
    has_cold = (rel_cold >= 2).any(1)
    n_cold = int(cold_catalog.sum())
    out["cold_catalog_size"] = n_cold
    out["random_within_cold_recall@20"] = min(20 / max(n_cold, 1), 1.0)
    out["random_within_cold_recall@50"] = min(50 / max(n_cold, 1), 1.0)
    for label, variant in systems.items():
        vec = vectors(cfg, p.model_tag, variant, p.cutoff)
        if vec is None:
            continue
        rec = recommend(p, tt_scorer(*vec), K)
        # ranking *within* the cold catalog: does metadata say which new video a user will like?
        uv, iv = vec
        within = []
        for lo in range(0, len(p.users), 512):
            u = p.users[lo: lo + 512]
            s = (uv[u] @ iv.T).astype(np.float32)
            p.mask_scores(s, u)
            s[:, ~cold_catalog] = -np.inf
            within.append(topk(s, 50))
        mw_cold = ranking_metrics(np.concatenate(within)[has_cold], rel_cold[has_cold], [20, 50], [20])
        mc = ranking_metrics(rec, rel_cold, [100, 300], [20])
        mw = ranking_metrics(rec, rel_warm, [100, 300], [20])
        out[label] = {"within_cold_recall@20": float(np.nanmean(mw_cold["recall@20"])),
                      "within_cold_recall@50": float(np.nanmean(mw_cold["recall@50"])),
                      "cold_recall@100": float(np.nanmean(mc["recall@100"])),
                      "cold_recall@300": float(np.nanmean(mc["recall@300"])),
                      "warm_recall@100": float(np.nanmean(mw["recall@100"])),
                      "warm_recall@300": float(np.nanmean(mw["recall@300"])),
                      "users_with_cold_targets": int((mc["n_relevant"] > 0).sum()),
                      "cold_share_of_top100": float(held[rec[:, :100]].mean())}
    # reference: popularity can never surface a held-out item that has no history
    out["catalog_share_held_out"] = float(held[p.candidate_mask].mean())
    return out


def cohorts(cfg: Config, p: Protocol, systems: list[str]) -> dict:
    """Recall@100 by user activity cohort (interactions before the cutoff)."""
    stats = np.load(cfg.processed / "features" / "user_stats.npy", mmap_mode="r")
    n_events = np.expm1(np.asarray(stats[p.snap, p.users, 0]))
    edges = [0, 50, 200, 1000, 3000, np.inf]
    labels = ["<50", "50-200", "200-1k", "1k-3k", "3k+"]
    df = pl.read_parquet(cfg.artifacts / "runs" / p.name / "per_user.parquet")
    out = {}
    for s in systems:
        sub = df.filter(pl.col("system") == s)
        if sub.is_empty():
            continue
        r = dict(zip(sub["user_id"].to_list(), sub["recall@100"].to_list()))
        vals = np.array([r.get(int(u), np.nan) for u in p.users], dtype=float)
        out[s] = {}
        for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
            m = (n_events >= lo) & (n_events < hi)
            out[s][lab] = {"users": int(m.sum()), "recall@100": float(np.nanmean(vals[m])) if m.any() else None}
    return out


def decile_exposure(cfg: Config, p: Protocol, systems: list[str], k: int = 100) -> dict:
    pop = pl.read_parquet(cfg.reports / "metrics" / "popularity_by_cutoff.parquet").filter(
        pl.col("protocol") == p.name).sort("video_id")
    dec = pop["decile"].to_numpy()
    obs = pl.read_parquet(cfg.processed / "features" / "events.parquet", columns=["video_id", "day"]).filter(
        pl.col("day") < p.cutoff)
    inter = np.bincount(obs["video_id"].to_numpy(), minlength=p.n_items).astype(float)
    tgt = (p.rel >= 2)
    out = {"catalog_share": [], "interaction_share": [], "target_share": []}
    for d in range(10):
        m = dec == d
        out["catalog_share"].append(float(m.sum() / (dec >= 0).sum()))
        out["interaction_share"].append(float(inter[m].sum() / inter[dec >= 0].sum()))
        out["target_share"].append(float(tgt[:, m].sum() / max(tgt.sum(), 1)))
    out["systems"] = {}
    for s in systems:
        f = cfg.artifacts / "runs" / p.name / f"{s}.npy"
        if not f.exists():
            continue
        rec = np.load(f)[:, :k]
        expo = np.bincount(dec[rec.ravel()] + 1, minlength=11)[1:].astype(float)
        hits = np.take_along_axis(p.rel, rec, 1) >= 2
        rec_by_dec = []
        for d in range(10):
            in_d = dec[rec] == d
            tot = tgt[:, dec == d].sum()
            rec_by_dec.append(float((hits & in_d).sum() / tot) if tot else None)
        out["systems"][s] = {"rec_share": (expo / expo.sum()).tolist(), "recall_by_decile": rec_by_dec}
    return out


def run(cfg: Config) -> dict:
    ks, nks = cfg["eval"]["ks"], cfg["eval"]["ndcg_ks"]
    K = max(ks)
    events = pl.read_parquet(cfg.processed / "features" / "events.parquet", columns=["user_id", "video_id", "day"])
    summary: dict = {}
    for pname in [n for n in cfg.windows() if n != "r0"] + ["dense"]:
        p = load_protocol(cfg, pname, events)
        base = cfg.artifacts / "two_tower" / p.model_tag
        variants = sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []
        rows = {}
        for variant in variants:
            vec = vectors(cfg, p.model_tag, variant, p.cutoff)
            if vec is None:
                continue
            rec = recommend(p, tt_scorer(*vec), K)
            per_user = p.evaluate(rec, ks, nks)
            meta = save_run(cfg, p, f"tt_{variant}", rec, per_user, {"stage": "two_tower", "variant": variant})
            rows[f"tt_{variant}"] = meta
        summary[pname] = {k: v["metrics"] for k, v in rows.items()}
        if pname == "val" and rows:
            # model selection uses validation only; cold-start variants answer a different question
            eligible = {k: v for k, v in rows.items() if not k.startswith("tt_cold")}
            best = max(eligible, key=lambda k: eligible[k]["metrics"]["recall@100"])
            sel_path = cfg.reports / "metrics" / "selection.json"
            sel = read_json(sel_path) if sel_path.exists() else {}
            sel.update({"retrieval_variant": best.removeprefix("tt_"), "retrieval_criterion": "val recall@100",
                        "retrieval_val_recall@100": eligible[best]["metrics"]["recall@100"]})
            write_json(sel_path, sel)
            print(f"selected retrieval variant: {best}")
        allrows = load_summaries(cfg, pname)
        print(f"\n[{pname}] {len(p.users):,} users")
        print_table(allrows, ["recall@50", "recall@100", "recall@300", "ndcg@20", "coverage@20",
                              "long_tail_share@20"])
        if pname == "test":
            summary["cold_start"] = cold_start(cfg, p, K)
            summary["cohorts"] = cohorts(cfg, p, ["popularity", "trending", "category", "als", "tt_main"])
        if pname in ("val", "test"):
            summary[f"deciles_{pname}"] = decile_exposure(
                cfg, p, ["popularity", "trending", "als", "itemknn", "tt_inbatch", "tt_main", "tt_uniform",
                         "tt_freq", "tt_freq_logq", "tt_hard"])
    write_json(cfg.reports / "metrics" / "retrieval_eval.json", summary)
    if "cold_start" in summary:
        print("\ncold start:", summary["cold_start"])
    return summary


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
