"""Diversity-aware reranking: maximal marginal relevance with feed constraints. Pure NumPy, deterministic.

MMR(i) = λ * rel(i) - (1 - λ) * max over selected j of sim(i, j)

rel is the ranker score min-max normalised within the list (minus an optional popularity penalty), sim is
embedding cosine mapped to [0, 1]. Constraints apply to the first `head` positions: at most `category_cap`
items per first-level category and `author_cap` per creator (KuaiRec does expose creator ids), plus
`novelty_slots` positions reserved for long-tail items when any are available. Ties break by input order
(the ranker's order), output items are unique, and fewer than k items come back if the pool runs out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RerankConfig:
    lam: float = 1.0
    category_cap: int | None = None
    author_cap: int | None = None
    novelty_slots: int = 0
    pop_penalty: float = 0.0
    head: int = 10

    @property
    def name(self) -> str:
        parts = [f"mmr{self.lam:g}"]
        if self.category_cap:
            parts.append(f"cat{self.category_cap}")
        if self.author_cap:
            parts.append(f"auth{self.author_cap}")
        if self.novelty_slots:
            parts.append(f"nov{self.novelty_slots}")
        if self.pop_penalty:
            parts.append(f"pop{self.pop_penalty:g}")
        return "_".join(parts)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    return np.zeros_like(x, dtype=np.float64) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def rerank(items: np.ndarray, scores: np.ndarray, k: int, rc: RerankConfig, emb: np.ndarray | None = None,
           cats: np.ndarray | None = None, authors: np.ndarray | None = None, log_pop: np.ndarray | None = None,
           tail: np.ndarray | None = None) -> np.ndarray:
    """Return up to k unique items. `items` and per-item arrays are aligned; arrays are indexed by position."""
    items = np.asarray(items)
    n = len(items)
    if n == 0 or k <= 0:
        return items[:0]
    # unique-ify defensively, keeping the first (highest-ranked) occurrence
    _, first = np.unique(items, return_index=True)
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    order = np.lexsort((np.arange(n), -np.asarray(scores, dtype=np.float64)))  # by score, ties by input order
    order = order[keep[order]]
    items = items[order]
    rel = _minmax(np.asarray(scores, dtype=np.float64)[order])
    if rc.pop_penalty and log_pop is not None:
        rel = rel - rc.pop_penalty * _minmax(np.asarray(log_pop, dtype=np.float64)[order])
    take = lambda a: None if a is None else np.asarray(a)[order]  # noqa: E731
    emb, cats, authors, tail = take(emb), take(cats), take(authors), take(tail)
    m = len(items)
    use_sim = rc.lam < 1.0 and emb is not None
    if use_sim:
        e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    max_sim = np.zeros(m)
    avail = np.ones(m, dtype=bool)
    chosen: list[int] = []
    cat_count: dict = {}
    auth_count: dict = {}
    k = min(k, m)
    while len(chosen) < k and avail.any():
        pos = len(chosen)
        val = rc.lam * rel - (1 - rc.lam) * max_sim if use_sim else rel.copy()
        ok = avail.copy()
        if pos < rc.head:
            if rc.category_cap and cats is not None:
                full = [c for c, n_ in cat_count.items() if n_ >= rc.category_cap]
                if full:
                    ok &= ~np.isin(cats, full)
            if rc.author_cap and authors is not None:
                full = [a for a, n_ in auth_count.items() if n_ >= rc.author_cap and a >= 0]
                if full:
                    ok &= ~np.isin(authors, full)
            # reserve the last `novelty_slots` head positions for long-tail items
            if rc.novelty_slots and tail is not None and pos >= rc.head - rc.novelty_slots:
                have_tail = sum(bool(tail[j]) for j in chosen)
                if have_tail < rc.novelty_slots and (ok & tail).any():
                    ok &= tail
        if not ok.any():  # constraints unsatisfiable: relax them rather than return a short list
            ok = avail
        cand = np.flatnonzero(ok)
        best = cand[np.argmax(val[cand])]  # argmax returns the first max -> stable tie-break by rank
        chosen.append(int(best))
        avail[best] = False
        if cats is not None:
            cat_count[cats[best]] = cat_count.get(cats[best], 0) + 1
        if authors is not None:
            auth_count[authors[best]] = auth_count.get(authors[best], 0) + 1
        if use_sim:
            sim = (1.0 + e @ e[best]) / 2.0
            np.maximum(max_sim, sim, out=max_sim)
    return items[np.array(chosen, dtype=int)]
