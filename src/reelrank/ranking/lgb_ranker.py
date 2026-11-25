"""LightGBM rankers on observed interactions (own process: LightGBM never shares one with PyTorch or FAISS).

    python -m reelrank.ranking.lgb_ranker

Validation protocol: train on Aug 6-10 events (R0 retrieval scores), early-stop on Aug 29-31 (R1 scores).
Test protocol: retrain on Aug 29-31 (R1 scores) with the selected number of trees, score Sep 1-5 (R2) and the
dense matrix. Queries are (user, day) groups of observed events; graded relevance 0 skip / 1 partial /
2 high watch / 3 completion.

Models: lambdarank (LambdaMART, graded), regression (pointwise, capped watch ratio), and feature ablations
(no two-tower score, no ALS score, no retrieval scores). Logged-impression metrics compare them with
single-signal orderings (two-tower score, ALS score, popularity, trending, random) on the same groups.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("lightgbm")

import time  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from reelrank.config import Config, cli, file_sha256, write_json  # noqa: E402
from reelrank.ranking.dataset import CATEGORICAL, FEATURES, sample_train_groups  # noqa: E402
from reelrank.ranking.metrics import impression_metrics  # noqa: E402
from reelrank.tracking import mlflow_run  # noqa: E402

ABLATIONS = {
    "lgb_rank": FEATURES,
    "lgb_rank_no_tt": [f for f in FEATURES if f != "tt_score"],
    "lgb_rank_no_als": [f for f in FEATURES if f != "als_score"],
    "lgb_rank_no_retrieval": [f for f in FEATURES if f not in ("tt_score", "als_score")],
}


def groups(df: pl.DataFrame) -> np.ndarray:
    g = df.select("user_id", "day").with_columns(k=pl.col("user_id") * 100000 + pl.col("day"))["k"].to_numpy()
    change = np.flatnonzero(np.diff(g)) + 1
    return np.diff(np.concatenate([[0], change, [len(g)]]))


def params(cfg: Config, objective: str) -> dict:
    c = cfg["ranker"]["lgb"]
    p = {"objective": objective, "learning_rate": c["learning_rate"], "num_leaves": c["num_leaves"],
         "min_child_samples": c["min_child_samples"], "subsample": c["subsample"], "subsample_freq": 1,
         "colsample_bytree": c["colsample_bytree"], "verbose": -1, "seed": cfg.seed, "deterministic": True,
         "force_row_wise": True, "num_threads": 8}
    if objective == "lambdarank":
        p.update({"metric": "ndcg", "eval_at": [10, 20], "lambdarank_truncation_level": 30,
                  "label_gain": [0, 1, 3, 7]})
    else:
        p.update({"metric": "l2"})
    return p


def train(cfg: Config, train_df: pl.DataFrame, feats: list[str], objective: str, n_trees: int,
          valid_df: pl.DataFrame | None = None) -> tuple[lgb.Booster, int]:
    label = "grade" if objective == "lambdarank" else "wr"
    cats = [f for f in feats if f in CATEGORICAL]
    dtr = lgb.Dataset(train_df.select(feats).to_numpy(), label=train_df[label].to_numpy(), feature_name=feats,
                      categorical_feature=cats, group=groups(train_df) if objective == "lambdarank" else None,
                      free_raw_data=False)
    cbs, valid = [], []
    if valid_df is not None:
        dva = lgb.Dataset(valid_df.select(feats).to_numpy(), label=valid_df[label].to_numpy(), feature_name=feats,
                          categorical_feature=cats, reference=dtr,
                          group=groups(valid_df) if objective == "lambdarank" else None)
        valid = [dva]
        cbs = [lgb.early_stopping(int(cfg["ranker"]["lgb"]["early_stopping"]), verbose=False)]
    b = lgb.train(params(cfg, objective), dtr, num_boost_round=n_trees, valid_sets=valid, callbacks=cbs)
    best = b.best_iteration if valid_df is not None and b.best_iteration > 0 else n_trees
    return b, best


def single_signal_scores(df: pl.DataFrame, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"random": rng.random(df.height), "two_tower_score": df["tt_score"].to_numpy(),
            "als_score": df["als_score"].to_numpy(), "popularity": df["i_log_pos"].to_numpy(),
            "trending": df["i_log_recent_pop"].to_numpy()}


def run(cfg: Config) -> dict:
    rdir = cfg.processed / "ranker"
    adir = cfg.artifacts / "ranker"
    adir.mkdir(parents=True, exist_ok=True)
    r0 = pl.read_parquet(rdir / "r0_observed.parquet")
    val = pl.read_parquet(rdir / "val_observed.parquet")
    test = pl.read_parquet(rdir / "test_observed.parquet")
    n_max = int(cfg["ranker"]["lgb"]["n_estimators"])
    cap = cfg["ranker"].get("max_train_rows")
    r0_train = sample_train_groups(r0, cap, cfg.seed)
    val_train = sample_train_groups(val, cap, cfg.seed)
    out: dict = {"val": {}, "test": {}, "selected_trees": {}, "train_seconds": {},
                 "train_rows": {"r0": r0_train.height, "val": val_train.height}}

    for name, s in single_signal_scores(val, cfg.seed).items():
        out["val"][name] = impression_metrics(val, s)
    for name, s in single_signal_scores(test, cfg.seed).items():
        out["test"][name] = impression_metrics(test, s)

    models = {**{k: ("lambdarank", v) for k, v in ABLATIONS.items()}, "lgb_regression": ("regression", FEATURES)}
    final: dict[str, lgb.Booster] = {}
    for name, (obj, feats) in models.items():
        t = time.time()
        b, best = train(cfg, r0_train, feats, obj, n_max, valid_df=val)
        out["val"][name] = impression_metrics(val, b.predict(val.select(feats).to_numpy(), num_iteration=best))
        out["selected_trees"][name] = int(best)
        # test protocol: retrain on the window just before test with the selected tree count
        b2, _ = train(cfg, val_train, feats, obj, best)
        out["train_seconds"][name] = time.time() - t
        out["test"][name] = impression_metrics(test, b2.predict(test.select(feats).to_numpy()))
        final[name] = b2
        b.save_model(str(adir / f"R1_{name}.txt"))
        b2.save_model(str(adir / f"R2_{name}.txt"))
        print(f"  {name:24s} trees={best:4d} val ndcg@20={out['val'][name]['ndcg@20']:.4f} "
              f"test ndcg@20={out['test'][name]['ndcg@20']:.4f}", flush=True)

    imp = final["lgb_rank"].feature_importance("gain")
    out["feature_importance_gain"] = dict(sorted(zip(FEATURES, (imp / imp.sum()).round(5).tolist()),
                                                 key=lambda kv: -kv[1]))
    # candidate pools: score with the protocol's model (R1 for val, R2 for test / dense)
    for pname, tag in (("val", "R1"), ("test", "R2"), ("dense", "R2")):
        cand = pl.read_parquet(rdir / f"{pname}_candidates.parquet")
        cols = {}
        for name, (_obj, feats) in models.items():
            b = lgb.Booster(model_file=str(adir / f"{tag}_{name}.txt"))
            t = time.time()
            cols[name] = b.predict(cand.select(feats).to_numpy()).astype(np.float32)
            if name == "lgb_rank":
                out.setdefault("candidate_scoring_seconds", {})[pname] = time.time() - t
        pl.DataFrame({"user_id": cand["user_id"], "video_id": cand["video_id"], **cols}).write_parquet(
            rdir / f"scores_lgb_{pname}.parquet")
    out["model_hashes"] = {p.name: file_sha256(p)[:16] for p in sorted(adir.glob("R2_*.txt"))}
    write_json(cfg.reports / "metrics" / "ranker_lgb.json", out)
    with mlflow_run(cfg, "ranker", "lightgbm", {"features": FEATURES, **cfg["ranker"]["lgb"]}) as log:
        for split in ("val", "test"):
            for name, m in out[split].items():
                log({f"{split}/{name}/{k}": v for k, v in m.items()})
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
