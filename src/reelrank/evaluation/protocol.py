"""Evaluation protocols shared by every model so candidate eligibility and relevance are identical.

temporal (r0 / val / test): users with >= 1 novel positive in the target window; candidates are videos
    eligible at the cutoff that the user had not consumed before it (novel-item task). Unobserved pairs
    count as non-relevant, so these are *observed-positive* metrics with incomplete labels.
dense: the 1,411 KuaiRec small-matrix users ranked over its 3,327 videos, with ~99.6% of pairs labelled.
    Models are trained on the big matrix only (no pair overlaps the small matrix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import polars as pl
import scipy.sparse as sp

from reelrank.config import Config, read_json
from reelrank.evaluation.metrics import exposure_summary, ranking_metrics, topk
from reelrank.features.build import eligible_items


@dataclass
class Protocol:
    name: str
    model_tag: str                  # trained model family serving this protocol (R0 / RG / R1 / R2)
    cutoff: int                     # day index; models for this protocol train on day < cutoff
    snap: int                       # snapshot index whose state is "everything before cutoff"
    users: np.ndarray               # evaluated user ids
    candidate_mask: np.ndarray      # [I] bool: items that may be recommended at all
    rel: np.ndarray                 # [n_users, I] int8 graded relevance
    seen: sp.csr_matrix             # [U, I] consumed before cutoff
    pop: np.ndarray                 # [I] interactions before cutoff
    n_items: int
    extra: dict = field(default_factory=dict)

    def mask_scores(self, scores: np.ndarray, users: np.ndarray, exclude_seen: bool = True) -> np.ndarray:
        """Set ineligible and (optionally) already-consumed items to -inf, in place."""
        scores[:, ~self.candidate_mask] = -np.inf
        if exclude_seen:
            sub = self.seen[users]
            rows = np.repeat(np.arange(len(users)), np.diff(sub.indptr))
            scores[rows, sub.indices] = -np.inf
        return scores

    @cached_property
    def tail_mask(self) -> np.ndarray:
        """Long tail = eligible items outside the most popular 20% (by interactions before cutoff)."""
        p = self.pop.copy()
        p[~self.candidate_mask] = -1
        n_head = int(np.ceil(0.2 * self.candidate_mask.sum()))
        head = np.argsort(-p, kind="stable")[:n_head]
        m = self.candidate_mask.copy()
        m[head] = False
        return m

    def evaluate(self, rec: np.ndarray, ks: list[int], ndcg_ks: list[int]) -> dict[str, np.ndarray]:
        return ranking_metrics(rec, self.rel, ks, ndcg_ks)

    def exposure(self, rec: np.ndarray, k: int) -> dict[str, float]:
        return exposure_summary(rec[:, :k], self.n_items, self.pop, self.candidate_mask, self.tail_mask)


def seen_matrix(events: pl.DataFrame, cutoff: int, n_users: int, n_items: int) -> sp.csr_matrix:
    e = events.filter(pl.col("day") < cutoff).select("user_id", "video_id").unique()
    return sp.csr_matrix((np.ones(e.height, dtype=np.int8), (e["user_id"].to_numpy(), e["video_id"].to_numpy())),
                         shape=(n_users, n_items))


def load_protocol(cfg: Config, name: str, events: pl.DataFrame | None = None, max_users: int | None = None) -> Protocol:
    fdir = cfg.processed / "features"
    man = read_json(fdir / "manifest.json")
    U, I = man["users"], man["items"]
    snap_days = np.load(fdir / "snap_days.npy")
    cutoff = cfg.protocol_cutoff(name)
    snap = int(np.searchsorted(snap_days, cutoff))
    assert snap_days[snap] == cutoff
    pop = np.load(fdir / "item_pop_before.npy", mmap_mode="r")[snap].astype(np.float64)
    static = dict(np.load(fdir / "item_static.npz"))
    if events is None:
        events = pl.read_parquet(fdir / "events.parquet", columns=["user_id", "video_id", "day"])
    seen = seen_matrix(events, cutoff, U, I)
    max_users = max_users or cfg["eval"].get("max_eval_users")
    if name == "dense":
        lab = pl.read_parquet(fdir / "eval" / "dense_labels.parquet")
        users = np.sort(lab["user_id"].unique().to_numpy())
        cand = np.zeros(I, dtype=bool)
        cand[lab["video_id"].unique().to_numpy()] = True
        uix = np.searchsorted(users, lab["user_id"].to_numpy())
        rel = np.zeros((len(users), I), dtype=np.int8)
        rel[uix, lab["video_id"].to_numpy()] = lab["grade"].to_numpy()
    else:
        tg = pl.read_parquet(fdir / "eval" / f"{name}_observed.parquet").filter(
            ~pl.col("seen") & pl.col("eligible"))
        users = np.sort(tg.filter(pl.col("pos"))["user_id"].unique().to_numpy())
        if max_users and len(users) > max_users:
            users = np.sort(np.random.default_rng(cfg.seed).choice(users, max_users, replace=False))
        tg = tg.filter(pl.col("user_id").is_in(users))
        cand = eligible_items(static, pop, cutoff)
        uix = np.searchsorted(users, tg["user_id"].to_numpy())
        rel = np.zeros((len(users), I), dtype=np.int8)
        rel[uix, tg["video_id"].to_numpy()] = tg["grade"].to_numpy()
    return Protocol(name=name, model_tag=cfg.protocol_model(name), cutoff=cutoff, snap=snap, users=users,
                    candidate_mask=cand, rel=rel, seen=seen,
                    pop=pop, n_items=I, extra={"static": static})


def recommend(protocol: Protocol, score_fn, k: int, batch: int = 512, exclude_seen: bool = True) -> np.ndarray:
    """Run `score_fn(user_ids) -> [n, I]` over evaluated users and return masked top-k item indices."""
    out = np.empty((len(protocol.users), k), dtype=np.int32)
    for i in range(0, len(protocol.users), batch):
        u = protocol.users[i: i + batch]
        s = np.asarray(score_fn(u), dtype=np.float32).copy()
        protocol.mask_scores(s, u, exclude_seen=exclude_seen)
        out[i: i + len(u)] = topk(s, k)
    return out


def summarize(per_user: dict[str, np.ndarray]) -> dict[str, float]:
    return {k: float(np.nanmean(v)) for k, v in per_user.items() if k != "n_relevant"}
