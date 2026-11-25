"""NumPy forward pass of the multitask neural ranker (ranking/neural.py) for PyTorch-free serving."""

from __future__ import annotations

import re

import numpy as np

from reelrank.ranking.dataset import FEATURES

NUMERIC_IDX = [i for i, f in enumerate(FEATURES) if f != "i_first_cat"]
CAT_IDX = FEATURES.index("i_first_cat")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyMultiTask:
    def __init__(self, path):
        z = np.load(path)
        self.w = {k: z[k] for k in z.files}
        idx = sorted({int(m.group(1)) for k in self.w if (m := re.match(r"trunk\.(\d+)\.weight$", k))})
        self.trunk = [(self.w[f"trunk.{i}.weight"], self.w[f"trunk.{i}.bias"]) for i in idx]
        self.mean, self.std = self.w["mean"], self.w["std"]
        self.utility = self.w["utility"]
        self.watch_mu, self.watch_sd = self.w["watch_stats"]

    def heads(self, X: np.ndarray, uvec: np.ndarray, ivec: np.ndarray) -> dict[str, np.ndarray]:
        num = (X[:, NUMERIC_IDX] - self.mean) / self.std
        cat = self.w["cat.weight"][X[:, CAT_IDX].astype(np.int64)]
        h = np.concatenate([num, cat, np.broadcast_to(uvec, (len(X), len(uvec))), ivec], 1).astype(np.float32)
        for W, b in self.trunk:
            h = np.maximum(h @ W.T + b, 0.0)
        head = lambda n: (h @ self.w[f"{n}.weight"].T + self.w[f"{n}.bias"]).ravel()  # noqa: E731
        return {"watch": head("watch"), "complete": _sigmoid(head("complete")), "skip": _sigmoid(head("skip"))}

    def score(self, X: np.ndarray, uvec: np.ndarray, ivec: np.ndarray) -> np.ndarray:
        p = self.heads(X, uvec, ivec)
        a, b, c = self.utility
        return a * (p["watch"] - self.watch_mu) / self.watch_sd + b * p["complete"] - c * p["skip"]
