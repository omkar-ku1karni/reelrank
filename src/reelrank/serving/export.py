"""Export a self-contained serving bundle for one snapshot, with a hash-pinned artifact manifest.

    python -m reelrank.serving.export

The bundle holds everything the API needs and nothing else (no event log, no PyTorch):
  artifacts/serving/snapshot.npz      user/item state at the serving cutoff + sparse history counts
  artifacts/serving/items_meta.parquet  metadata cards (caption, category, creator, duration)
  artifacts/serving/demo_truth.parquet  what each user actually watched after the cutoff (offline demo only)
  artifacts/manifest.json             file hashes, model versions, id-map version, feature schema

Before writing, the NumPy towers are run over every user and item and compared with the vectors PyTorch
exported; the export fails if they disagree (max abs diff > 1e-4).
"""

from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp

from reelrank.config import Config, cli, day_to_date, file_sha256, read_json, write_json
from reelrank.evaluation.protocol import load_protocol
from reelrank.ranking.dataset import FEATURES
from reelrank.retrieval.candidates import selected_variant
from reelrank.serving.numpy_model import NumpyTwoTower

MODEL_VERSION_PREFIX = "reelrank"


def build_bundle(cfg: Config) -> dict:
    variant = selected_variant(cfg)
    tag = cfg.protocol_model("test")
    day = cfg.protocol_cutoff("test")
    fdir = cfg.processed / "features"
    sdir = cfg.artifacts / "serving"
    sdir.mkdir(parents=True, exist_ok=True)
    man = read_json(fdir / "manifest.json")
    snap_days = np.load(fdir / "snap_days.npy")
    s = int(np.searchsorted(snap_days, day))
    p = load_protocol(cfg, "test")
    st = dict(np.load(fdir / "item_static.npz"))
    pop_all = np.load(fdir / "item_pop_before.npy")
    first_seen = np.where(pop_all > 0, 1, 0).argmax(0)
    first_day = np.where(pop_all.max(0) > 0, snap_days[np.maximum(first_seen - 1, 0)], 10**6)
    born = np.minimum(np.where(st["upload_day"] >= 0, st["upload_day"], 10**6), first_day)
    item_age = np.log1p(np.clip(day - born, 0, 400)).astype(np.float32)
    item_stats = np.load(fdir / "item_stats.npy", mmap_mode="r")[s].copy()
    item_platform = np.load(fdir / "item_platform.npy", mmap_mode="r")[s].copy()
    users = pd.read_parquet(cfg.processed / "users.parquet").sort_values("user_id")
    user_static = np.stack([
        users["active_degree"].to_numpy(dtype=float), np.log1p(users["register_days"].to_numpy(dtype=float)),
        users["is_video_author"].to_numpy(dtype=float), np.log1p(users["follow_user_num"].to_numpy(dtype=float)),
        np.log1p(users["fans_user_num"].to_numpy(dtype=float))], 1).astype(np.float32)
    ev = pl.read_parquet(fdir / "events.parquet", columns=["user_id", "video_id", "day", "wr_c"]).filter(
        pl.col("day") < day)
    U, I = man["users"], man["items"]
    uv = ev.group_by(["user_id", "video_id"]).agg(n=pl.len())
    prior_uv = sp.csr_matrix((uv["n"].to_numpy().astype(np.float32), (uv["user_id"].to_numpy(), uv["video_id"].to_numpy())),
                             shape=(U, I))
    authors = st["author"]
    a_ids, a_idx = np.unique(authors, return_inverse=True)
    ev_a = ev.with_columns(a=pl.Series(a_idx[ev["video_id"].to_numpy()]))
    ua = ev_a.filter(pl.Series(authors[ev["video_id"].to_numpy()] >= 0)).group_by(["user_id", "a"]).agg(
        n=pl.len(), s=pl.col("wr_c").sum())
    shape_a = (U, len(a_ids))
    author_cnt = sp.csr_matrix((ua["n"].to_numpy().astype(np.float32), (ua["user_id"].to_numpy(), ua["a"].to_numpy())),
                               shape=shape_a)
    author_sum = sp.csr_matrix((ua["s"].to_numpy().astype(np.float32), (ua["user_id"].to_numpy(), ua["a"].to_numpy())),
                               shape=shape_a)
    si = {n: i for i, n in enumerate(man["item_stats"])}
    trend = item_stats[:, si["log_recent_pop"]] + 1e-3 * item_stats[:, si["log_pos"]]
    arrays = {
        "day": np.array(day), "snap": np.array(s),
        "user_stats": np.load(fdir / "user_stats.npy", mmap_mode="r")[s].copy(),
        "user_cat": np.load(fdir / "user_cat.npy", mmap_mode="r")[s].copy(),
        "user_hist": np.load(fdir / "user_hist.npy", mmap_mode="r")[s].copy(),
        "user_static": np.concatenate([np.load(fdir / "user_recent_wr.npy", mmap_mode="r")[s][:, None],
                                       user_static], 1).astype(np.float32),
        "item_stats": item_stats, "item_platform": item_platform,
        "item_age": item_age, "eligible": p.candidate_mask, "pop": p.pop.astype(np.float32),
        "trend": trend.astype(np.float32), "tail": p.tail_mask, "author_index": a_idx.astype(np.int64),
        **{f"static_{k}": v for k, v in st.items()},
    }
    for name, m in (("prior_uv", prior_uv), ("author_cnt", author_cnt), ("author_sum", author_sum),
                    ("seen", p.seen.astype(np.float32))):
        m = m.tocsr()
        arrays[f"{name}_data"], arrays[f"{name}_indices"], arrays[f"{name}_indptr"] = m.data, m.indices, m.indptr
        arrays[f"{name}_shape"] = np.array(m.shape)
    np.savez_compressed(sdir / "snapshot.npz", **arrays)

    # parity: NumPy towers vs the PyTorch-exported vectors
    tdir = cfg.artifacts / "two_tower" / tag / variant
    info = read_json(tdir / "info.json")
    iv_t = np.load(tdir / f"items_{day_to_date(day)}.npy")
    uv_t = np.load(tdir / f"users_{day_to_date(day)}.npy")
    if info["spec"]["encoder"] == "mean":
        nt = NumpyTwoTower(tdir / "weights.npz", info["spec"])
        dyn = np.concatenate([item_stats, item_platform, item_age[:, None]], 1)
        iv_np = nt.item_forward(np.arange(I), st["first_cat"], st["dur_bucket"], st["author_idx"], st["tags"],
                                st["log_dur"], dyn)
        uv_np = nt.user_forward(np.arange(U), arrays["user_stats"], arrays["user_cat"], arrays["user_hist"])
        parity = {"item_max_abs_diff": float(np.abs(iv_np - iv_t).max()),
                  "user_max_abs_diff": float(np.abs(uv_np - uv_t).max())}
        if max(parity.values()) > 1e-4:
            raise SystemExit(f"NumPy/PyTorch tower mismatch: {parity}")
        user_mode = "numpy_tower"
    else:  # sequential encoders: serve the user vectors PyTorch computed at the snapshot
        parity = {"note": "sequential encoder: precomputed user vectors are served"}
        np.save(sdir / "user_vectors.npy", uv_t)
        user_mode = "precomputed"
    np.save(sdir / "item_vectors.npy", iv_t)
    shutil.copy(tdir / "weights.npz", sdir / "two_tower_weights.npz")
    shutil.copy(cfg.artifacts / "als" / f"{tag}.npz", sdir / "als.npz")
    selection = read_json(cfg.reports / "metrics" / "selection.json")
    ranker = selection.get("ranker", "lgb_rank")
    for old in sdir.glob("ranker_*"):
        old.unlink()
    if ranker.startswith("lgb_"):
        shutil.copy(cfg.artifacts / "ranker" / f"{tag}_{ranker}.txt", sdir / "ranker_lgb.txt")
        ranker_kind = "lightgbm"
    elif ranker == "nn_utility":
        shutil.copy(cfg.artifacts / "ranker" / f"{tag}_nn_weights.npz", sdir / "ranker_nn.npz")
        ranker_kind = "numpy_multitask"
    else:  # a retrieval score won validation: rank by that feature
        ranker_kind = f"feature:{ranker}"

    items = pd.read_parquet(cfg.processed / "items.parquet")
    items[["video_id", "caption", "cover_text", "first_cat", "first_cat_name", "first_cat_en", "second_cat_name",
           "author_id", "duration_ms", "upload_day", "is_ad"]].to_parquet(sdir / "items_meta.parquet", index=False)
    truth = pl.read_parquet(fdir / "eval" / "test_observed.parquet").filter(pl.col("pos"))
    truth.select("user_id", "video_id", "max_wr", "seen").write_parquet(sdir / "demo_truth.parquet")

    files = {}
    for f in sorted(sdir.iterdir()):
        files[f.name] = {"sha256": file_sha256(f), "bytes": f.stat().st_size}
    id_map = hashlib.sha256(np.arange(U).tobytes() + items["video_id"].to_numpy().astype(np.int64).tobytes()).hexdigest()[:16]
    ckpt = file_sha256(tdir / "model.pt")[:16]
    manifest = {
        "model_version": f"{MODEL_VERSION_PREFIX}-{tag}-{variant}-{ckpt[:8]}",
        "snapshot_date": day_to_date(day), "model_tag": tag, "retrieval_variant": variant,
        "two_tower_spec": info["spec"], "two_tower_checkpoint_sha256": ckpt, "user_vectors": user_mode,
        "id_map_version": id_map, "users": U, "items": I, "feature_schema": FEATURES,
        "feature_manifest_hash": man["config_hash"], "ranker": ranker, "ranker_kind": ranker_kind,
        "rerank": selection["rerank"],
        "rerank_name": selection["rerank_name"], "candidates": cfg["candidates"], "parity": parity,
        "files": files,
    }
    write_json(cfg.artifacts / "manifest.json", manifest)
    print(f"serving bundle for {day_to_date(day)} ({manifest['model_version']}); parity {parity}")
    return manifest


def main() -> None:
    cfg, _ = cli(__doc__)
    build_bundle(cfg)


if __name__ == "__main__":
    main()
