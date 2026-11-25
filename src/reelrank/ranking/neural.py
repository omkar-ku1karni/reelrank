"""Multitask neural ranker (PyTorch process).

    python -m reelrank.ranking.neural

Input: frozen two-tower user and item vectors for the example's day, concatenated with the standardised ranker features ⊕ a
category embedding, through a shared MLP (256 -> 128) with three heads:
  watch     Huber regression on log1p(play seconds)
  complete  P(watch_ratio >= 1), BCE
  skip      P(watch_ratio < 0.2), BCE
Total loss = weighted sum. Ranking utility = w_watch*z(watch) + w_complete*P(complete) - w_skip*P(skip), with
weights chosen on validation NDCG@20 (the utility is a score, not a probability). Calibration of the two
probability heads and the duration bias of ranking by raw watch time are reported separately.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("torch")

import itertools  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from reelrank.config import Config, cli, day_to_date, write_json  # noqa: E402
from reelrank.ranking.dataset import FEATURES, sample_train_groups  # noqa: E402
from reelrank.ranking.metrics import calibration, impression_metrics  # noqa: E402
from reelrank.retrieval.candidates import selected_variant  # noqa: E402
from reelrank.tracking import mlflow_run  # noqa: E402

NUMERIC = [f for f in FEATURES if f != "i_first_cat"]


class MultiTask(nn.Module):
    def __init__(self, d_num: int, n_cat: int, d_vec: int, hidden: list[int], dropout: float = 0.1):
        super().__init__()
        self.cat = nn.Embedding(n_cat, 8)
        layers, d = [], d_num + 8 + 2 * d_vec
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.watch = nn.Linear(d, 1)
        self.complete = nn.Linear(d, 1)
        self.skip = nn.Linear(d, 1)

    def forward(self, num, cat, uvec, ivec):
        h = self.trunk(torch.cat([num, self.cat(cat), uvec, ivec], -1))
        return self.watch(h).squeeze(-1), self.complete(h).squeeze(-1), self.skip(h).squeeze(-1)


class Data:
    def __init__(self, cfg: Config, df: pl.DataFrame, variant: str, mean=None, std=None):
        num = df.select(NUMERIC).to_numpy().astype(np.float32)
        if mean is None:
            mean, std = num.mean(0), num.std(0) + 1e-6
        self.mean, self.std = mean, std
        self.num = torch.from_numpy((num - mean) / std)
        self.cat = torch.from_numpy(df["i_first_cat"].to_numpy().astype(np.int64))
        u, v, d = (df[c].to_numpy() for c in ("user_id", "video_id", "day"))
        tag = df["tag"][0]
        uvec = np.zeros((len(u), cfg["two_tower"]["dim"]), np.float32)
        ivec = np.zeros_like(uvec)
        base = cfg.artifacts / "two_tower" / tag / variant
        for day in np.unique(d):
            m = d == day
            uvec[m] = np.load(base / f"users_{day_to_date(int(day))}.npy")[u[m]]
            ivec[m] = np.load(base / f"items_{day_to_date(int(day))}.npy")[v[m]]
        self.uvec, self.ivec = torch.from_numpy(uvec), torch.from_numpy(ivec)
        self.y_watch = torch.from_numpy(df["log_play_s"].to_numpy().astype(np.float32)) if "log_play_s" in df.columns else None
        self.y_comp = torch.from_numpy(df["complete"].to_numpy().astype(np.float32)) if "complete" in df.columns else None
        self.y_skip = torch.from_numpy(df["skip"].to_numpy().astype(np.float32)) if "skip" in df.columns else None
        self.n = len(u)

    def batch(self, idx):
        return self.num[idx], self.cat[idx], self.uvec[idx], self.ivec[idx]


def fit(cfg: Config, tr: Data, epochs: int, val: Data | None = None, val_df: pl.DataFrame | None = None):
    c = cfg["ranker"]["nn"]
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(8)
    n_cat = int(tr.cat.max()) + 1 + 64
    model = MultiTask(len(NUMERIC), n_cat, tr.uvec.shape[1], list(c["hidden"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["lr"]), weight_decay=1e-5)
    w = c["head_weights"]
    rng = np.random.default_rng(cfg.seed)
    hist, best, best_state, best_ep = [], -1.0, None, epochs
    B = int(c["batch_size"])
    for ep in range(1, epochs + 1):
        model.train()
        t = time.time()
        perm = rng.permutation(tr.n)
        losses = []
        for i in range(0, tr.n, B):
            idx = torch.from_numpy(perm[i: i + B])
            pw, pc, ps = model(*tr.batch(idx))
            loss = (w["watch"] * F.huber_loss(pw, tr.y_watch[idx])
                    + w["complete"] * F.binary_cross_entropy_with_logits(pc, tr.y_comp[idx])
                    + w["skip"] * F.binary_cross_entropy_with_logits(ps, tr.y_skip[idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        row = {"epoch": ep, "loss": float(np.mean(losses)), "seconds": time.time() - t}
        if val is not None:
            pr = predict(model, val)
            row["val_ndcg@20_default_utility"] = impression_metrics(val_df, utility(pr, 1.0, 1.0, 1.0))["ndcg@20"]
            if row["val_ndcg@20_default_utility"] > best:
                best, best_ep = row["val_ndcg@20_default_utility"], ep
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        hist.append(row)
        print(f"    epoch {ep}: loss={row['loss']:.4f} {row.get('val_ndcg@20_default_utility', '')}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_ep


@torch.no_grad()
def predict(model: MultiTask, d: Data) -> dict[str, np.ndarray]:
    model.eval()
    out = {"watch": [], "complete": [], "skip": []}
    for i in range(0, d.n, 65536):
        idx = torch.arange(i, min(i + 65536, d.n))
        pw, pc, ps = model(*d.batch(idx))
        out["watch"].append(pw.numpy())
        out["complete"].append(torch.sigmoid(pc).numpy())
        out["skip"].append(torch.sigmoid(ps).numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def utility(pr: dict, w_watch: float, w_complete: float, w_skip: float, stats: tuple | None = None) -> np.ndarray:
    mu, sd = stats if stats else (pr["watch"].mean(), pr["watch"].std() + 1e-9)
    return w_watch * (pr["watch"] - mu) / sd + w_complete * pr["complete"] - w_skip * pr["skip"]


def run(cfg: Config) -> dict:
    variant = selected_variant(cfg)
    rdir = cfg.processed / "ranker"
    cap = cfg["ranker"].get("max_train_rows")
    r0 = sample_train_groups(pl.read_parquet(rdir / "r0_observed.parquet"), cap, cfg.seed)
    val = pl.read_parquet(rdir / "val_observed.parquet")
    test = pl.read_parquet(rdir / "test_observed.parquet")
    epochs = int(cfg["ranker"]["nn"]["epochs"])
    out: dict = {"retrieval_variant": variant}

    # validation protocol: train on r0 window, select epochs and utility weights on val
    d_r0 = Data(cfg, r0, variant)
    d_val = Data(cfg, val, variant, d_r0.mean, d_r0.std)
    t = time.time()
    model, hist, best_ep = fit(cfg, d_r0, epochs, d_val, val)
    out["val_history"], out["selected_epochs"], out["train_seconds_val"] = hist, best_ep, time.time() - t
    pv = predict(model, d_val)
    g = cfg["ranker"]["utility_grid"]
    grid = []
    for a, b, c in itertools.product(g["w_watch"], g["w_complete"], g["w_skip"]):
        if a == b == c == 0:
            continue
        m = impression_metrics(val, utility(pv, a, b, c))
        grid.append({"w_watch": a, "w_complete": b, "w_skip": c, **m})
    best = max(grid, key=lambda r: r["ndcg@20"])
    W = (best["w_watch"], best["w_complete"], best["w_skip"])
    out["utility_grid"] = grid
    out["utility_weights"] = dict(zip(["w_watch", "w_complete", "w_skip"], W))
    out["val"] = {
        "nn_utility": impression_metrics(val, utility(pv, *W)),
        "nn_watch_only": impression_metrics(val, pv["watch"]),
        "nn_complete_only": impression_metrics(val, pv["complete"]),
        "nn_neg_skip_only": impression_metrics(val, -pv["skip"]),
    }
    out["calibration_val"] = {"complete": calibration(pv["complete"], val["complete"].to_numpy()),
                              "skip": calibration(pv["skip"], val["skip"].to_numpy())}

    # test protocol: retrain on val window with the selected epochs
    d_tr2 = Data(cfg, sample_train_groups(val, cap, cfg.seed), variant)
    t = time.time()
    model2, _, _ = fit(cfg, d_tr2, best_ep)
    out["train_seconds_test"] = time.time() - t
    d_te = Data(cfg, test, variant, d_tr2.mean, d_tr2.std)
    pt = predict(model2, d_te)
    wstats = (float(predict(model2, d_tr2)["watch"].mean()), float(predict(model2, d_tr2)["watch"].std() + 1e-9))
    out["test"] = {
        "nn_utility": impression_metrics(test, utility(pt, *W, stats=wstats)),
        "nn_watch_only": impression_metrics(test, pt["watch"]),
        "nn_complete_only": impression_metrics(test, pt["complete"]),
        "nn_neg_skip_only": impression_metrics(test, -pt["skip"]),
    }
    out["calibration_test"] = {"complete": calibration(pt["complete"], test["complete"].to_numpy()),
                               "skip": calibration(pt["skip"], test["skip"].to_numpy())}
    # duration bias: what ranking by each signal does to the length of what gets shown first
    out["duration_bias_test"] = {
        "all_observed_mean_s": float(test["duration_ms"].mean() / 1000),
        "positive_mean_s": float(test.filter(pl.col("grade") >= 2)["duration_ms"].mean() / 1000),
        **{k: v["top10_duration_s"] for k, v in out["test"].items()},
    }
    out["watch_pred_duration_corr_test"] = float(np.corrcoef(pt["watch"], np.log1p(test["duration_ms"].to_numpy()))[0, 1])

    # candidate pools
    for pname, (mdl, mean_std, stats) in {"val": (model, (d_r0.mean, d_r0.std), None),
                                          "test": (model2, (d_tr2.mean, d_tr2.std), wstats),
                                          "dense": (model2, (d_tr2.mean, d_tr2.std), wstats)}.items():
        cand = pl.read_parquet(rdir / f"{pname}_candidates.parquet")
        parts = [predict(mdl, Data(cfg, cand.slice(lo, 500_000), variant, *mean_std))
                 for lo in range(0, cand.height, 500_000)]
        pc = {k: np.concatenate([p_[k] for p_ in parts]) for k in ("watch", "complete", "skip")}
        pl.DataFrame({"user_id": cand["user_id"], "video_id": cand["video_id"],
                      "nn_utility": utility(pc, *W, stats=stats if stats else (pv["watch"].mean(), pv["watch"].std())),
                      "nn_watch": pc["watch"], "nn_complete": pc["complete"], "nn_skip": pc["skip"]}).write_parquet(
            rdir / f"scores_nn_{pname}.parquet")
    adir = cfg.artifacts / "ranker"
    adir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model2.state_dict(), "numeric": NUMERIC, "mean": d_tr2.mean, "std": d_tr2.std,
                "utility": W, "watch_stats": wstats, "variant": variant}, adir / "R2_nn.pt")
    np.savez(adir / "R2_nn_weights.npz", **{k: v.numpy() for k, v in model2.state_dict().items()},
             mean=d_tr2.mean, std=d_tr2.std, utility=np.array(W), watch_stats=np.array(wstats))
    write_json(cfg.reports / "metrics" / "ranker_nn.json", out)
    with mlflow_run(cfg, "ranker", "multitask_nn", {**cfg["ranker"]["nn"], "utility": W}) as log:
        for split in ("val", "test"):
            for name, m in out[split].items():
                log({f"{split}/{name}/{k}": v for k, v in m.items()})
    print("utility weights", W, "val", out["val"]["nn_utility"]["ndcg@20"], "test", out["test"]["nn_utility"]["ndcg@20"])
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
