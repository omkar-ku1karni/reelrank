"""FAISS-side test, run as its own pytest process (`make test-faiss`)."""

import faiss
import numpy as np

from reelrank.evaluation.metrics import topk


def test_flat_ip_index_matches_numpy_exact_search():
    rng = np.random.default_rng(0)
    items = rng.normal(size=(2000, 64)).astype(np.float32)
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    q = rng.normal(size=(50, 64)).astype(np.float32)
    idx = faiss.IndexFlatIP(64)
    idx.add(items)
    _, fi = idx.search(q, 100)
    ni = topk(q @ items.T, 100)
    # identical sets (order may differ only on exact score ties)
    assert all(set(a) == set(b) for a, b in zip(fi.tolist(), ni.tolist()))
