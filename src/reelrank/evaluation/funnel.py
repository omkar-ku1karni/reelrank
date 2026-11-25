"""End-to-end funnel evaluation, reranking tradeoffs and the ablation ladder (NumPy only).

    python -m reelrank.evaluation.funnel

For val / test / dense:
  1. retrieval ceiling: candidate-pool recall (from candidates.py)
  2. conditional ranking: each ranker orders the same candidate pool -> top-20
  3. selection on dense_val (a seeded half of the fully-labelled dense users): the best ranker, then a reranking
     sweep (MMR λ x feed constraints x popularity penalty) -> Pareto front; operating point = most
     category-diverse config within `max_ndcg_loss` of relevance-only NDCG@20. dense_test (the other half),
     temporal val and temporal test are reported with the selected configuration.
  4. ladder: popularity -> ALS -> two-tower (uncorrected) -> two-tower (logQ) -> + ranker -> + reranker,
     with paired user-bootstrap intervals for every step. The test protocol is read once, after selection.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import polars as pl

from reelrank.config import Config, cli, read_json, write_json
from reelrank.evaluation.metrics import category_entropy, intra_list_diversity, paired_bootstrap
from reelrank.evaluation.protocol import Protocol, load_protocol
from reelrank.evaluation.store import save_run
from reelrank.reranking.mmr import RerankConfig, rerank
from reelrank.retrieval.candidates import selected_variant
from reelrank.retrieval.evaluate import vectors

RANKERS = ["tt_score", "als_score", "lgb_rank", "lgb_regression", "lgb_rank_no_retrieval", "nn_utility"]


def load_pool(cfg: Config, pname: str) -> pl.DataFrame:
    rdir = cfg.processed / "ranker"
    cand = pl.read_parquet(rdir / f"{pname}_candidates.parquet",
                           columns=["user_id", "video_id", "sources", "tt_score", "als_score", "i_log_pos", "grade"])
    for f in ("scores_lgb", "scores_nn"):
        path = rdir / f"{f}_{pname}.parquet"
        if path.exists():
            cand = cand.join(pl.read_parquet(path), on=["user_id", "video_id"], how="left")
    return cand.sort(["user_id", "video_id"])


class Funnel:
    def __init__(self, cfg: Config, p: Protocol, pool: pl.DataFrame, item_emb: np.ndarray):
        self.cfg, self.p, self.emb = cfg, p, item_emb
        st = p.extra["static"]
        self.cat, self.author = st["first_cat"], st["author"]
        self.log_pop = np.log1p(p.pop)
        self.tail = p.tail_mask
        u = pool["user_id"].to_numpy()
        bounds = np.flatnonzero(np.diff(u)) + 1
        self.starts = np.concatenate([[0], bounds])
        self.ends = np.concatenate([bounds, [len(u)]])
        self.pool_users = u[self.starts]
        self.items = pool["video_id"].to_numpy()
        self.cols = {c: pool[c].to_numpy() for c in pool.columns if c in RANKERS}
        upos = {int(x): i for i, x in enumerate(p.users)}
        self.row_of = np.array([upos[int(x)] for x in self.pool_users])

    def lists(self, score_col: str, rc: RerankConfig | None, k: int) -> tuple[np.ndarray, np.ndarray]:
        rec = np.full((len(self.p.users), k), -1, dtype=np.int32)
        lat = np.zeros(len(self.pool_users))
        s_all = self.cols[score_col]
        for j, (a, b) in enumerate(zip(self.starts, self.ends)):
            it = self.items[a:b]
            s = np.nan_to_num(s_all[a:b].astype(np.float64), nan=-1e9)
            t = time.perf_counter()
            if rc is None:
                order = np.lexsort((np.arange(len(it)), -s))[:k]
                out = it[order]
            else:
                out = rerank(it, s, k, rc, emb=self.emb[it], cats=self.cat[it], authors=self.author[it],
                             log_pop=self.log_pop[it], tail=self.tail[it])
            lat[j] = time.perf_counter() - t
            rec[self.row_of[j], : len(out)] = out
        return rec, lat

    def measure(self, rec: np.ndarray, lat: np.ndarray | None = None) -> tuple[dict, dict]:
        per_user = self.p.evaluate(rec, [20], [10, 20])
        ild_cat = intra_list_diversity(rec, item_cat=self.cat)
        ild_emb = intra_list_diversity(rec, item_emb=self.emb)
        ent = category_entropy(rec, self.cat)
        expo = self.p.exposure(rec, rec.shape[1])
        m = {k: float(np.nanmean(v)) for k, v in per_user.items() if k != "n_relevant"}
        m.update({"ild_category": float(np.nanmean(ild_cat)), "ild_embedding": float(np.nanmean(ild_emb)),
                  "category_entropy": float(ent.mean()), **expo})
        if lat is not None:
            m["rerank_p50_ms"] = float(np.percentile(lat, 50) * 1000)
            m["rerank_p95_ms"] = float(np.percentile(lat, 95) * 1000)
        per_user.update({"ild_category": ild_cat, "category_entropy": ent})
        return m, per_user


def sweep_configs(cfg: Config) -> list[RerankConfig]:
    r = cfg["rerank"]
    cons = [dict(), dict(category_cap=r["category_cap"], author_cap=r["author_cap"]),
            dict(category_cap=r["category_cap"], author_cap=r["author_cap"], novelty_slots=r["novelty_slots"])]
    out = [RerankConfig(lam=float(lam), **c) for lam in r["lambdas"] for c in cons]
    out += [RerankConfig(lam=1.0, pop_penalty=float(pp)) for pp in r["pop_penalty"] if pp > 0]
    return out


def subset(p: Protocol, mask: np.ndarray, name: str) -> Protocol:
    return replace(p, name=name, users=p.users[mask], rel=p.rel[mask])


def dense_split(cfg: Config, p: Protocol) -> np.ndarray:
    """Seeded half of the dense users for selection (dense_val); the other half (dense_test) is report-only."""
    rng = np.random.default_rng(cfg.seed + 17)
    m = np.zeros(len(p.users), dtype=bool)
    m[rng.permutation(len(p.users))[: len(p.users) // 2]] = True
    return m


def pareto_front(sweep: list[dict]) -> list[str]:
    out = []
    for s in sweep:  # non-dominated in (ndcg@20, ild_category)
        if not any(o["ndcg@20"] >= s["ndcg@20"] and o["ild_category"] >= s["ild_category"]
                   and (o["ndcg@20"] > s["ndcg@20"] or o["ild_category"] > s["ild_category"]) for o in sweep):
            out.append(s["config"])
    return out


def run(cfg: Config) -> dict:
    k = int(cfg["rerank"]["k"])
    variant = selected_variant(cfg)
    select_on = cfg["rerank"].get("select_on", "dense_val")
    events = pl.read_parquet(cfg.processed / "features" / "events.parquet", columns=["user_id", "video_id", "day"])
    sel_path = cfg.reports / "metrics" / "selection.json"
    selection = read_json(sel_path) if sel_path.exists() else {}
    base = {n: load_protocol(cfg, n, events) for n in ("val", "test", "dense")}
    dmask = dense_split(cfg, base["dense"])
    P = {"val": base["val"], "test": base["test"],
         "dense_val": subset(base["dense"], dmask, "dense_val"), "dense_test": subset(base["dense"], ~dmask, "dense_test")}
    source = {"val": ("val", None), "test": ("test", None), "dense_val": ("dense", dmask), "dense_test": ("dense", ~dmask)}
    F: dict[str, Funnel] = {}
    for name, p in P.items():
        src, _ = source[name]
        pool = load_pool(cfg, src).filter(pl.col("user_id").is_in(p.users))
        _, iv = vectors(cfg, p.model_tag, variant, p.cutoff)
        F[name] = Funnel(cfg, p, pool, iv)
    out: dict = {"retrieval_variant": variant, "k": k, "select_on": select_on,
                 "selection_note": ("Ranker and reranker are selected on dense_val: half of the fully-labelled small-matrix "
                                    "users. Temporal labels exist only for videos the production policy chose to show, so "
                                    "temporal NDCG rewards reproducing that exposure; the first version of this rule "
                                    "selected on temporal validation and picked 'no ranker' for that reason."),
                 "dense_users": {"dense_val": int(dmask.sum()), "dense_test": int((~dmask).sum())}}
    res: dict = {n: {"users": int(len(p.users)), "rankers": {}} for n, p in P.items()}
    per_user: dict = {n: {} for n in P}
    # conditional ranking of the shared pool, every protocol
    for name in P:
        for col in [c for c in RANKERS if c in F[name].cols]:
            rec, _ = F[name].lists(col, None, k)
            m, pu = F[name].measure(rec)
            res[name]["rankers"][col] = m
            per_user[name][f"pool_{col}"] = pu
            save_run(cfg, P[name], f"e2e_{col}", rec, pu, {"stage": "funnel"}, k_expo=k)
    # Feature ablations are scored for the report but can't be served (the engine needs the full
    # feature schema), so only servable rankers compete.
    servable = [c for c in res[select_on]["rankers"] if not c.startswith("lgb_rank_no_")]
    ranker = max(servable, key=lambda c: res[select_on]["rankers"][c]["ndcg@20"])
    selection["ranker"] = ranker
    selection["ranker_selected_on"] = select_on
    # reranking sweep on the selection protocol (and temporal validation, for the tradeoff figure)
    for name in dict.fromkeys([select_on, "val"]):
        sweep = []
        for rc in sweep_configs(cfg):
            rec, lat = F[name].lists(ranker, rc, k)
            m, _ = F[name].measure(rec, lat)
            sweep.append({"config": rc.name, **rc.__dict__, **m})
        res[name]["sweep"] = sweep
        res[name]["pareto"] = pareto_front(sweep)
    sweep = res[select_on]["sweep"]
    floor = (1 - float(cfg["rerank"]["max_ndcg_loss"])) * res[select_on]["rankers"][ranker]["ndcg@20"]
    ok = [s for s in sweep if s["ndcg@20"] >= floor]
    chosen = max(ok, key=lambda s: (s["ild_category"], s["ndcg@20"])) if ok else sweep[0]
    selection["rerank"] = {k_: chosen[k_] for k_ in ("lam", "category_cap", "author_cap", "novelty_slots",
                                                     "pop_penalty", "head")}
    selection["rerank_name"] = chosen["config"]
    selection["rerank_selected_on"] = select_on
    write_json(sel_path, selection)
    rc = RerankConfig(**selection["rerank"])
    # reranked feeds + ablation ladder for every protocol
    for name, p in P.items():
        rec_rr, lat = F[name].lists(ranker, rc, k)
        m_rr, pu_rr = F[name].measure(rec_rr, lat)
        res[name]["reranked"] = {"config": selection["rerank_name"], **m_rr}
        save_run(cfg, p, "e2e_reranked", rec_rr, pu_rr, {"stage": "funnel", "rerank": selection["rerank"]}, k_expo=k)
        per_user[name]["e2e_reranked"] = pu_rr
        src, mask = source[name]
        ladder_systems = [("popularity", "popularity"), ("ALS", "als"),
                          ("two-tower, in-batch (no correction)", "tt_inbatch"),
                          ("two-tower, in-batch + per-day logQ" if variant == "main" else f"two-tower ({variant})",
                           f"tt_{variant}"),
                          (f"+ ranker ({ranker})", f"pool_{ranker}"),
                          (f"+ reranker ({selection['rerank_name']})", "e2e_reranked")]
        ladder, prev = [], None
        for label, key in ladder_systems:
            if key.startswith("pool_") or key == "e2e_reranked":
                pu = per_user[name][key]
            else:
                f = cfg.artifacts / "runs" / src / f"{key}.npy"
                if not f.exists():
                    continue
                rec = np.load(f)[:, :k]
                pu = F[name].measure(rec[mask] if mask is not None else rec)[1]
            row = {"system": label, "key": key,
                   **{m_: float(np.nanmean(pu[m_])) for m_ in ("ndcg@20", "recall@20", "precision@20", "mrr",
                                                              "ild_category")}}
            if prev is not None:
                row["vs_previous_ndcg@20"] = paired_bootstrap(prev["ndcg@20"], pu["ndcg@20"],
                                                              int(cfg["eval"]["bootstrap"]), cfg.seed)
                row["vs_previous_recall@20"] = paired_bootstrap(prev["recall@20"], pu["recall@20"],
                                                                int(cfg["eval"]["bootstrap"]), cfg.seed)
            ladder.append(row)
            prev = pu
        res[name]["ladder"] = ladder
        print(f"\n[{name}] {len(p.users):,} users, ranker={ranker}, rerank={selection['rerank_name']}")
        for r in ladder:
            d = r.get("vs_previous_ndcg@20")
            ci = f"  Δ={d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]" if d else ""
            print(f"  {r['system']:45s} ndcg@20={r['ndcg@20']:.4f} recall@20={r['recall@20']:.4f} "
                  f"ild={r['ild_category']:.3f}{ci}")
    out.update(res)
    write_json(cfg.reports / "metrics" / "funnel.json", out)
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
