import numpy as np

from reelrank.reranking.mmr import RerankConfig, rerank


def _pool(n=30, seed=0):
    rng = np.random.default_rng(seed)
    items = np.arange(100, 100 + n)
    scores = rng.random(n)
    emb = rng.normal(size=(n, 8))
    cats = rng.integers(0, 3, n)
    authors = rng.integers(0, 5, n)
    tail = rng.random(n) < 0.2
    return items, scores, emb, cats, authors, tail


def test_relevance_only_is_a_stable_sort():
    items = np.array([5, 6, 7, 8])
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    assert rerank(items, scores, 4, RerankConfig(lam=1.0)).tolist() == [6, 5, 7, 8]


def test_unique_items_and_short_pool():
    items = np.array([1, 2, 2, 3])
    out = rerank(items, np.array([0.3, 0.9, 0.8, 0.1]), 10, RerankConfig(lam=0.7), emb=np.eye(4))
    assert len(out) == 3 and len(set(out.tolist())) == 3


def test_empty_pool_and_zero_k():
    assert rerank(np.array([], dtype=int), np.array([]), 5, RerankConfig()).size == 0
    assert rerank(np.array([1, 2]), np.array([1.0, 0.5]), 0, RerankConfig()).size == 0


def test_deterministic():
    args = _pool()
    rc = RerankConfig(lam=0.6, category_cap=2, author_cap=1, novelty_slots=1)
    a = rerank(args[0], args[1], 10, rc, emb=args[2], cats=args[3], authors=args[4], tail=args[5])
    b = rerank(args[0], args[1], 10, rc, emb=args[2], cats=args[3], authors=args[4], tail=args[5])
    assert a.tolist() == b.tolist()


def test_caps_respected_in_head():
    items, scores, emb, _, authors, tail = _pool(60, seed=3)
    cats = np.random.default_rng(4).integers(0, 5, 60)  # 5 categories x cap 3 can fill a head of 10
    rc = RerankConfig(lam=1.0, category_cap=3, author_cap=2, head=10)
    out = rerank(items, scores, 20, rc, emb=emb, cats=cats, authors=authors)
    pos = {it: j for j, it in enumerate(items)}
    head = [pos[i] for i in out[:10]]
    assert max(np.bincount(cats[head])) <= 3
    assert max(np.bincount(authors[head])) <= 2


def test_novelty_slot_inserts_a_tail_item():
    items = np.arange(12)
    scores = np.linspace(1, 0, 12)
    tail = np.zeros(12, bool)
    tail[-1] = True  # the only long-tail item is ranked last
    out = rerank(items, scores, 10, RerankConfig(lam=1.0, novelty_slots=1, head=10), tail=tail)
    assert 11 in out[:10].tolist() and out[0] == 0


def test_lower_lambda_increases_diversity():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 8))
    emb = np.vstack([base + 0.01 * rng.normal(size=(20, 8)), rng.normal(size=(20, 8))])
    scores = np.concatenate([np.linspace(1, 0.8, 20), np.linspace(0.7, 0.5, 20)])
    items = np.arange(40)

    def ild(sel):
        e = emb[sel] / np.linalg.norm(emb[sel], axis=1, keepdims=True)
        s = e @ e.T
        return (1 - s)[np.triu_indices(len(sel), 1)].mean()

    rel = rerank(items, scores, 10, RerankConfig(lam=1.0), emb=emb)
    div = rerank(items, scores, 10, RerankConfig(lam=0.5), emb=emb)
    assert ild(div) > ild(rel)


def test_unsatisfiable_caps_are_relaxed_not_truncated():
    items = np.arange(5)
    out = rerank(items, np.linspace(1, 0, 5), 5, RerankConfig(category_cap=1, head=10), cats=np.zeros(5, int))
    assert len(out) == 5
