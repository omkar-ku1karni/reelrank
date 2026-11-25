"""Online recommendation engine over an exported serving bundle (NumPy + LightGBM; never PyTorch or FAISS).

Pipeline per request: user vector (NumPy tower over snapshot features) -> exact-search candidates from
two-tower / ALS / trending / category pools -> LightGBM ranker on the shared feature schema -> MMR reranker
with feed constraints. Every file is verified against artifacts/manifest.json before use; a mismatched
model/index/id-map version refuses to load.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("lightgbm")

import collections  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

from reelrank.config import file_sha256  # noqa: E402
from reelrank.evaluation.metrics import topk  # noqa: E402
from reelrank.ranking.dataset import FEATURES  # noqa: E402
from reelrank.reranking.mmr import RerankConfig, rerank  # noqa: E402
from reelrank.serving.numpy_model import NumpyTwoTower  # noqa: E402
from reelrank.serving.numpy_ranker import NumpyMultiTask  # noqa: E402

SOURCE_BITS = {"two_tower": 1, "als": 2, "trending": 4, "category": 8}
ALGORITHMS = ("reelrank", "ranker_only", "two_tower", "als", "trending", "popularity")


class ManifestError(RuntimeError):
    pass


@dataclass
class Rec:
    video_id: int
    rank: int
    score: float
    ranker_score: float | None
    sources: list[str]


class Engine:
    def __init__(self, artifacts: Path, verify: bool = True):
        self.root = Path(artifacts)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise ManifestError(f"{mpath} not found: run `make export` first")
        self.manifest = json.loads(mpath.read_text())
        sdir = self.root / "serving"
        if verify:
            for name, meta in self.manifest["files"].items():
                f = sdir / name
                if not f.exists():
                    raise ManifestError(f"missing serving file {name}")
                if file_sha256(f) != meta["sha256"]:
                    raise ManifestError(f"{name} does not match the manifest (sha256): refusing to serve a "
                                        "mismatched model/index version")
        z = np.load(sdir / "snapshot.npz")
        self.snap = {k: z[k] for k in z.files}
        S = self.snap
        self.U, self.I = int(self.manifest["users"]), int(self.manifest["items"])
        self.item_vectors = np.load(sdir / "item_vectors.npy")
        self.als = np.load(sdir / "als.npz")
        self.als_user, self.als_item = self.als["user"], self.als["item"]
        kind = self.manifest.get("ranker_kind", "lightgbm")
        self.ranker_kind = kind
        self.ranker = lgb.Booster(model_file=str(sdir / "ranker_lgb.txt")) if kind == "lightgbm" else None
        self.nn_ranker = NumpyMultiTask(sdir / "ranker_nn.npz") if kind == "numpy_multitask" else None
        if self.manifest.get("user_vectors", "numpy_tower") == "numpy_tower":
            self.tower, self.user_vectors = NumpyTwoTower(sdir / "two_tower_weights.npz",
                                                          self.manifest["two_tower_spec"]), None
        else:  # sequential encoders are served from vectors precomputed at the snapshot
            self.tower, self.user_vectors = None, np.load(sdir / "user_vectors.npy")
        self._check_alignment()
        self.meta = pd.read_parquet(sdir / "items_meta.parquet").set_index("video_id")
        self.truth = pd.read_parquet(sdir / "demo_truth.parquet")

        def csr(name):
            return sp.csr_matrix((S[f"{name}_data"], S[f"{name}_indices"], S[f"{name}_indptr"]),
                                 shape=tuple(S[f"{name}_shape"]))

        self.prior_uv, self.author_cnt, self.author_sum, self.seen = (csr(n) for n in ("prior_uv", "author_cnt",
                                                                                      "author_sum", "seen"))
        st = {k.removeprefix("static_"): v for k, v in S.items() if k.startswith("static_")}
        self.st = st
        self.item_dyn = np.concatenate([S["item_stats"], S["item_platform"], S["item_age"][:, None]], 1)
        self.eligible = S["eligible"].astype(bool)
        order = np.argsort(-np.where(self.eligible, S["trend"], -np.inf), kind="stable")
        self.trend_order = order[self.eligible[order]]  # eligible videos only
        order = np.argsort(-np.where(self.eligible, S["pop"], -np.inf), kind="stable")
        self.pop_order = order[self.eligible[order]]
        self.cat_pools = {int(c): self.trend_order[st["first_cat"][self.trend_order] == c] for c in
                          np.unique(st["first_cat"])}
        self.rerank_cfg = RerankConfig(**self.manifest["rerank"])
        self.cand_cfg = self.manifest["candidates"]
        self.log_pop = np.log1p(S["pop"])
        self.lock = threading.Lock()
        self.stats = {"requests": 0, "fallbacks": collections.Counter(), "latency_ms": collections.deque(maxlen=5000),
                      "stage_ms": collections.defaultdict(lambda: collections.deque(maxlen=5000)),
                      "missing_platform": collections.deque(maxlen=5000), "top_score": collections.deque(maxlen=5000),
                      "exposure": collections.Counter()}

    # ------------------------------------------------------------------------------------- contracts
    def _check_alignment(self) -> None:
        S = self.snap
        checks = {
            "item vectors rows": (self.item_vectors.shape[0], self.I),
            "ALS item rows": (self.als_item.shape[0], self.I),
            "ALS user rows": (self.als_user.shape[0], self.U),
            "tower / user-vector users": ((self.tower.U if self.tower else self.user_vectors.shape[0]), self.U),
            "snapshot users": (S["user_stats"].shape[0], self.U),
            "snapshot items": (S["item_stats"].shape[0], self.I),
        }
        if self.ranker is not None:
            checks["ranker features"] = (self.ranker.num_feature(), len(FEATURES))
        if self.nn_ranker is not None:
            checks["nn ranker numeric features"] = (len(self.nn_ranker.mean), len(FEATURES) - 1)
        bad = {k: v for k, v in checks.items() if v[0] != v[1]}
        if bad:
            raise ManifestError(f"artifact shapes disagree with the manifest: {bad}")
        if self.manifest["feature_schema"] != FEATURES or (
                self.ranker is not None and self.ranker.feature_name() != FEATURES):
            raise ManifestError("ranker feature schema differs from the code's FEATURES")

    def rank_scores(self, X: np.ndarray, items: np.ndarray, uvec: np.ndarray) -> np.ndarray:
        if not len(items):
            return np.zeros(0)
        if self.ranker is not None:
            return self.ranker.predict(X, num_threads=1)
        if self.nn_ranker is not None:
            return self.nn_ranker.score(X, uvec, self.item_vectors[items])
        return X[:, FEATURES.index(self.ranker_kind.removeprefix("feature:"))].astype(np.float64)

    @property
    def model_version(self) -> str:
        return self.manifest["model_version"]

    def known_user(self, user_id: int) -> bool:
        return 0 <= user_id < self.U

    def has_history(self, user_id: int) -> bool:
        return self.known_user(user_id) and bool(self.snap["user_stats"][user_id, -1] > 0)

    # ------------------------------------------------------------------------------------- features
    def user_vector(self, user_id: int) -> np.ndarray:
        S = self.snap
        if self.tower is None:
            return self.user_vectors[user_id]
        u = np.array([user_id])
        return self.tower.user_forward(u, S["user_stats"][u], S["user_cat"][u], S["user_hist"][u].astype(np.int64))[0]

    def features(self, user_id: int, items: np.ndarray, uvec: np.ndarray) -> np.ndarray:
        """Ranker features for one user and many items; identical to ranking.dataset.pair_features."""
        S, st = self.snap, self.st
        n = len(items)
        us = np.broadcast_to(S["user_stats"][user_id], (n, S["user_stats"].shape[1]))
        ust = np.broadcast_to(S["user_static"][user_id], (n, S["user_static"].shape[1]))
        a = S["author_index"][items]
        cnt = np.asarray(self.author_cnt[user_id, a].todense()).ravel()
        ssum = np.asarray(self.author_sum[user_id, a].todense()).ravel()
        cols = [us, ust, S["item_stats"][items], S["item_platform"][items],
                S["item_age"][items, None], st["log_dur"][items, None],
                st["first_cat"][items, None].astype(np.float32), st["tags"][items].sum(1, keepdims=True),
                st["is_ad"][items, None].astype(np.float32),
                (self.item_vectors[items] @ uvec)[:, None],
                (self.als_item[items] @ self.als_user[user_id])[:, None],
                S["user_cat"][user_id, st["first_cat"][items]][:, None],
                np.abs(st["log_dur"][items] - S["user_stats"][user_id, 6])[:, None],
                np.log1p(np.asarray(self.prior_uv[user_id, items].todense()).ravel())[:, None],
                np.log1p(cnt)[:, None], np.where(cnt > 0, ssum / np.maximum(cnt, 1), -1.0)[:, None]]
        return np.concatenate([np.asarray(c, dtype=np.float32) for c in cols], 1)

    # ------------------------------------------------------------------------------------- pipeline
    def _mask(self, scores: np.ndarray, user_id: int, novel: bool) -> np.ndarray:
        scores[~self.eligible] = -np.inf
        if novel and self.known_user(user_id):
            scores[self.seen[user_id].indices] = -np.inf
        return scores

    def candidates(self, user_id: int, novel: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        c = self.cand_cfg
        src: dict[int, int] = {}
        uvec = None
        seen = set(self.seen[user_id].indices.tolist()) if (novel and self.known_user(user_id)) else set()
        if self.has_history(user_id):
            uvec = self.user_vector(user_id)
            tt = self._mask((self.item_vectors @ uvec).astype(np.float32), user_id, novel)
            al = self._mask((self.als_item @ self.als_user[user_id]).astype(np.float32), user_id, novel)
            for it in topk(tt[None, :], int(c["two_tower"]))[0]:
                if np.isfinite(tt[it]):
                    src[int(it)] = src.get(int(it), 0) | SOURCE_BITS["two_tower"]
            for it in topk(al[None, :], int(c["als"]))[0]:
                if np.isfinite(al[it]):
                    src[int(it)] = src.get(int(it), 0) | SOURCE_BITS["als"]
        for pool, n, bit in ((self.trend_order, int(c["trending"]), "trending"),):
            taken = 0
            for it in pool:
                if taken >= n:
                    break
                if int(it) not in seen:
                    src[int(it)] = src.get(int(it), 0) | SOURCE_BITS[bit]
                    taken += 1
        fav = int(np.argmax(self.snap["user_cat"][user_id])) if self.known_user(user_id) else int(
            self.st["first_cat"][self.trend_order[0]])
        taken = 0
        for it in self.cat_pools.get(fav, []):
            if taken >= int(c["category"]):
                break
            if int(it) not in seen:
                src[int(it)] = src.get(int(it), 0) | SOURCE_BITS["category"]
                taken += 1
        items = np.fromiter(src.keys(), dtype=np.int64, count=len(src))
        bits = np.fromiter(src.values(), dtype=np.int64, count=len(src))
        return items, bits, uvec

    def recommend(self, user_id: int, k: int = 20, algorithm: str = "reelrank", lam: float | None = None,
                  novel: bool = True, explain: bool = False) -> dict:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algorithm!r}; choose one of {ALGORITHMS}")
        if k < 1:
            raise ValueError("k must be >= 1")
        t0 = time.perf_counter()
        stage = {}
        fallback = None
        if not self.known_user(user_id):
            fallback = "unknown_user"
        elif not self.has_history(user_id):
            fallback = "no_history"
        out: list[Rec] = []
        debug: dict = {}
        if fallback == "unknown_user" or algorithm in ("trending", "popularity"):
            order = self.trend_order if algorithm != "popularity" else self.pop_order
            seen = set(self.seen[user_id].indices.tolist()) if (novel and self.known_user(user_id)) else set()
            picks = [int(i) for i in order[: k + len(seen) + 1] if int(i) not in seen][:k]
            base = self.snap["trend"] if algorithm != "popularity" else self.log_pop
            out = [Rec(i, r + 1, float(base[i]), None, [algorithm if algorithm in ("trending", "popularity")
                                                         else "trending"]) for r, i in enumerate(picks)]
        elif algorithm in ("two_tower", "als"):
            if algorithm == "two_tower":
                s = (self.item_vectors @ self.user_vector(user_id)).astype(np.float32)
            else:
                s = (self.als_item @ self.als_user[user_id]).astype(np.float32)
            s = self._mask(s, user_id, novel)
            top = [int(i) for i in topk(s[None, :], k)[0] if np.isfinite(s[i])]
            out = [Rec(i, r + 1, float(s[i]), None, [algorithm]) for r, i in enumerate(top)]
        else:
            t = time.perf_counter()
            items, bits, uvec = self.candidates(user_id, novel)
            stage["candidates_ms"] = (time.perf_counter() - t) * 1e3
            if uvec is None:
                uvec = np.zeros(self.item_vectors.shape[1], dtype=np.float32)
            t = time.perf_counter()
            X = self.features(user_id, items, uvec)
            stage["features_ms"] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            score = self.rank_scores(X, items, uvec)
            stage["ranker_ms"] = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            if algorithm == "ranker_only":
                rc = RerankConfig(lam=1.0)
            else:
                rc = self.rerank_cfg if lam is None else RerankConfig(**{**self.rerank_cfg.__dict__, "lam": float(lam)})
            chosen = rerank(items, score, k, rc, emb=self.item_vectors[items], cats=self.st["first_cat"][items],
                            authors=self.st["author"][items], log_pop=self.log_pop[items],
                            tail=self.snap["tail"][items])
            stage["rerank_ms"] = (time.perf_counter() - t) * 1e3
            pos = {int(i): j for j, i in enumerate(items)}
            rel_order = np.lexsort((np.arange(len(items)), -score))
            rel_rank = {int(items[j]): r + 1 for r, j in enumerate(rel_order)}
            for r, i in enumerate(chosen):
                j = pos[int(i)]
                out.append(Rec(int(i), r + 1, float(score[j]), float(score[j]),
                               [n for n, b in SOURCE_BITS.items() if bits[j] & b]))
            if explain:
                debug = {"pool_size": int(len(items)),
                         "pool_by_source": {n: int(((bits & b) > 0).sum()) for n, b in SOURCE_BITS.items()},
                         "ranker_rank": {int(i): rel_rank[int(i)] for i in chosen},
                         "rerank": rc.__dict__}
            self.stats["missing_platform"].append(float((X[:, FEATURES.index("p_has")] == 0).mean()) if len(X) else 0)
        total = (time.perf_counter() - t0) * 1e3
        with self.lock:
            self.stats["requests"] += 1
            if fallback:
                self.stats["fallbacks"][fallback] += 1
            self.stats["latency_ms"].append(total)
            for kk, v in stage.items():
                self.stats["stage_ms"][kk].append(v)
            if out:
                self.stats["top_score"].append(out[0].score)
            self.stats["exposure"].update(r.video_id for r in out)
        res = {"user_id": user_id, "algorithm": algorithm, "k": k, "model_version": self.model_version,
               "snapshot_date": self.manifest["snapshot_date"], "fallback": fallback, "latency_ms": total,
               "stage_ms": stage, "items": [self.card(r) for r in out]}
        if explain:
            res["explain"] = debug
        return res

    def card(self, r: Rec) -> dict:
        m = self.meta.loc[r.video_id]
        return {"video_id": r.video_id, "rank": r.rank, "final_score": r.score, "predicted_engagement": r.ranker_score,
                "retrieval_sources": r.sources, "category": m["first_cat_en"], "category_zh": m["first_cat_name"],
                "subcategory_zh": m["second_cat_name"], "caption": m["caption"], "creator_id": int(m["author_id"]),
                "duration_s": round(float(m["duration_ms"]) / 1000, 1), "is_ad": bool(m["is_ad"])}

    def similar(self, video_id: int, k: int = 20) -> dict:
        if not (0 <= video_id < self.I):
            raise KeyError(video_id)
        v = self.item_vectors[video_id]
        s = (self.item_vectors @ v).astype(np.float32)
        s[video_id] = -np.inf
        s[~self.eligible] = -np.inf
        top = [int(i) for i in topk(s[None, :], k)[0] if np.isfinite(s[i])]
        return {"video_id": video_id, "model_version": self.model_version,
                "items": [{**self.card(Rec(i, r + 1, float(s[i]), None, ["two_tower"])), "cosine": float(s[i])}
                          for r, i in enumerate(top)]}

    def history(self, user_id: int, n: int = 20) -> list[dict]:
        if not self.known_user(user_id):
            return []
        h = self.snap["user_hist"][user_id]
        h = h[h >= 0][::-1][:n]
        return [self.card(Rec(int(i), j + 1, 0.0, None, [])) for j, i in enumerate(h)]

    def watched_after(self, user_id: int) -> set[int]:
        """Videos this user positively watched in the 5 days after the snapshot (offline evaluation data)."""
        t = self.truth
        return set(t.loc[(t["user_id"] == user_id) & (~t["seen"]), "video_id"].astype(int).tolist())

    def monitoring(self) -> dict:
        s = self.stats
        lat = np.array(s["latency_ms"]) if s["latency_ms"] else np.zeros(1)
        exp = np.array(list(s["exposure"].values()), dtype=float)
        from reelrank.evaluation.metrics import gini

        return {"requests": s["requests"], "fallback_rate": {k: v / max(s["requests"], 1) for k, v in s["fallbacks"].items()},
                "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
                "stage_p95_ms": {k: float(np.percentile(np.array(v), 95)) for k, v in s["stage_ms"].items() if v},
                "candidate_missing_platform_stats": float(np.mean(s["missing_platform"])) if s["missing_platform"] else None,
                "mean_top_score": float(np.mean(s["top_score"])) if s["top_score"] else None,
                "distinct_items_served": len(s["exposure"]), "exposure_gini_served": float(gini(exp)) if len(exp) else 0.0}
