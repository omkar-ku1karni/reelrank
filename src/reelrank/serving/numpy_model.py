"""NumPy forward pass of the mean-pool two-tower model, so serving never imports PyTorch.

Parity with the PyTorch module is checked by tests/test_numpy_parity.py (run in its own process).
"""

from __future__ import annotations

import re

import numpy as np


def _mlp_layers(w: dict, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = sorted({int(m.group(1)) for k in w if (m := re.match(rf"{prefix}\.(\d+)\.weight$", k))})
    return [(w[f"{prefix}.{i}.weight"], w[f"{prefix}.{i}.bias"]) for i in idx]


def _run_mlp(x: np.ndarray, layers) -> np.ndarray:
    for j, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if j < len(layers) - 1:
            x = np.maximum(x, 0.0)
    return x


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class NumpyTwoTower:
    def __init__(self, weights_path, spec: dict):
        z = np.load(weights_path)
        self.w = {k: z[k] for k in z.files}
        if spec["encoder"] != "mean":
            raise ValueError("NumPy serving supports the mean-pool encoder; sequential encoders serve "
                             "precomputed user vectors")
        self.metadata = bool(spec["metadata"])
        self.user_layers = _mlp_layers(self.w, "user_mlp")
        self.item_layers = _mlp_layers(self.w, "item_mlp")
        self.item_emb = self.w["item_id.weight"]
        self.user_emb = self.w["user_id.weight"]
        self.I = self.item_emb.shape[0] - 2
        self.U = self.user_emb.shape[0] - 1
        self.PAD, self.UNK = self.I + 1, self.I
        self.held_out = self.w.get("held_out", np.zeros(self.I, dtype=bool)).astype(bool)

    def user_forward(self, uid: np.ndarray, stats: np.ndarray, cat: np.ndarray, hist: np.ndarray) -> np.ndarray:
        uid = np.where((uid >= 0) & (uid < self.U), uid, self.U)  # unknown users -> the unknown-user row
        if self.held_out.any():
            hist = np.where((hist >= 0) & self.held_out[np.maximum(hist, 0)], -1, hist)
        valid = hist >= 0
        emb = self.item_emb[np.where(valid, hist, self.PAD)]
        pooled = emb.sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
        parts = [self.user_emb[uid], pooled]
        if self.metadata:
            parts += [(stats - self.w["user_norm_mean"]) / self.w["user_norm_std"], cat]
        return _l2(_run_mlp(np.concatenate(parts, -1).astype(np.float32), self.user_layers))

    def item_forward(self, iid: np.ndarray, first_cat, dur_bucket, author_idx, tags, log_dur, dyn) -> np.ndarray:
        ids = iid.copy()
        dyn = dyn.copy()
        if self.held_out.any():
            cold = self.held_out[iid]
            ids[cold] = self.UNK
            dyn[cold] = 0.0
        parts = [self.item_emb[ids]]
        if self.metadata:
            parts += [self.w["cat.weight"][first_cat], self.w["dur.weight"][dur_bucket],
                      self.w["author.weight"][author_idx], tags @ self.w["tags.weight"].T, log_dur[:, None],
                      (dyn - self.w["item_norm_mean"]) / self.w["item_norm_std"]]
        return _l2(_run_mlp(np.concatenate(parts, -1).astype(np.float32), self.item_layers))
