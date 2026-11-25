"""Vector-retrieval engineering benchmark (own process: FAISS never shares one with PyTorch or LightGBM).

    python -m reelrank.retrieval.faiss_bench

Exact search first: NumPy matmul and FAISS IndexFlatIP over the 64-d item vectors at the test cutoff. HNSW
(M x efSearch sweep) and IVF-Flat (nprobe sweep) are measured as engineering experiments only: build time,
index memory, warmed p50/p95 latency for single and batched queries, and recall@K relative to exact search.
Results are specific to this catalog (~10.7k items) and this machine; they are not web-scale claims.
"""

from __future__ import annotations

from reelrank.runtime import claim

claim("faiss")

import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from reelrank.config import Config, cli, day_to_date, write_json  # noqa: E402
from reelrank.retrieval.candidates import selected_variant  # noqa: E402


def _lat(fn, queries: np.ndarray, reps: int) -> dict:
    for q in queries[:20]:
        fn(q[None, :])  # warm-up
    t = np.empty(reps)
    for i in range(reps):
        q = queries[i % len(queries)][None, :]
        s = time.perf_counter()
        fn(q)
        t[i] = time.perf_counter() - s
    return {"p50_ms": float(np.percentile(t, 50) * 1e3), "p95_ms": float(np.percentile(t, 95) * 1e3)}


def _batch(fn, queries: np.ndarray, bs: int, reps: int = 20) -> dict:
    fn(queries[:bs])
    t = np.empty(reps)
    for i in range(reps):
        lo = (i * bs) % max(len(queries) - bs, 1)
        s = time.perf_counter()
        fn(queries[lo: lo + bs])
        t[i] = time.perf_counter() - s
    return {"batch": bs, "p50_ms": float(np.percentile(t, 50) * 1e3), "p95_ms": float(np.percentile(t, 95) * 1e3),
            "queries_per_s": float(bs / np.median(t))}


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.mean([len(np.intersect1d(a[a >= 0], e)) / len(e) for a, e in zip(approx, exact)]))


def index_bytes(index) -> int:
    return int(faiss.serialize_index(index).nbytes)


def run(cfg: Config) -> dict:
    variant = selected_variant(cfg)
    tag = cfg.protocol_model("test")
    day = cfg.protocol_cutoff("test")
    d = cfg.artifacts / "two_tower" / tag / variant
    items = np.ascontiguousarray(np.load(d / f"items_{day_to_date(day)}.npy").astype(np.float32))
    users = np.ascontiguousarray(np.load(d / f"users_{day_to_date(day)}.npy").astype(np.float32))
    rng = np.random.default_rng(cfg.seed)
    q = users[rng.choice(len(users), min(2000, len(users)), replace=False)]
    K = 300
    threads = int(os.environ.get("REELRANK_FAISS_THREADS", "1"))
    faiss.omp_set_num_threads(threads)
    reps = 1000
    out = {"items": int(items.shape[0]), "dim": int(items.shape[1]), "k": K, "queries": int(len(q)),
           "faiss_threads": threads, "machine": f"{platform.machine()} / {platform.platform()}",
           "faiss_version": faiss.__version__, "indexes": []}

    def np_search(x):
        s = x @ items.T
        part = np.argpartition(-s, K - 1, axis=1)[:, :K]
        return np.take_along_axis(part, np.argsort(-np.take_along_axis(s, part, 1), 1), 1)

    exact = np_search(q)
    out["indexes"].append({"name": "numpy exact (matmul + argpartition)", "build_s": 0.0,
                           "memory_mb": items.nbytes / 2**20, "recall@300_vs_exact": 1.0,
                           "single": _lat(np_search, q, reps), "batched": _batch(np_search, q, 256)})
    t = time.perf_counter()
    flat = faiss.IndexFlatIP(items.shape[1])
    flat.add(items)
    build = time.perf_counter() - t
    _, fi = flat.search(q, K)
    out["indexes"].append({"name": "faiss IndexFlatIP (exact)", "build_s": build,
                           "memory_mb": index_bytes(flat) / 2**20, "recall@300_vs_exact": recall_vs_exact(fi, exact),
                           "single": _lat(lambda x: flat.search(x, K), q, reps),
                           "batched": _batch(lambda x: flat.search(x, K), q, 256)})
    for M in (16, 32):
        t = time.perf_counter()
        h = faiss.IndexHNSWFlat(items.shape[1], M, faiss.METRIC_INNER_PRODUCT)
        h.hnsw.efConstruction = 200
        h.add(items)
        build = time.perf_counter() - t
        for ef in (64, 128, 300, 600):
            h.hnsw.efSearch = ef
            _, hi = h.search(q, K)
            out["indexes"].append({"name": f"faiss HNSW M={M} efSearch={ef}", "build_s": build,
                                   "memory_mb": index_bytes(h) / 2**20,
                                   "recall@300_vs_exact": recall_vs_exact(hi, exact),
                                   "single": _lat(lambda x: h.search(x, K), q, reps // 2),
                                   "batched": _batch(lambda x: h.search(x, K), q, 256)})
    nlist = 100
    t = time.perf_counter()
    quant = faiss.IndexFlatIP(items.shape[1])
    ivf = faiss.IndexIVFFlat(quant, items.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    ivf.train(items)
    ivf.add(items)
    build = time.perf_counter() - t
    for nprobe in (5, 10, 20, 50):
        ivf.nprobe = nprobe
        _, vi = ivf.search(q, K)
        out["indexes"].append({"name": f"faiss IVF-Flat nlist={nlist} nprobe={nprobe}", "build_s": build,
                               "memory_mb": index_bytes(ivf) / 2**20, "recall@300_vs_exact": recall_vs_exact(vi, exact),
                               "single": _lat(lambda x: ivf.search(x, K), q, reps // 2),
                               "batched": _batch(lambda x: ivf.search(x, K), q, 256)})
    idx_dir = cfg.artifacts / "index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(flat, str(idx_dir / f"{tag}_{variant}_flatip.faiss"))
    out["note"] = ("Exact search over this catalog takes well under a millisecond per query, so serving uses exact "
                   "NumPy search; approximate indexes trade recall for speed they are not needed for here.")
    write_json(cfg.reports / "metrics" / "index_benchmark.json", out)
    for r in out["indexes"]:
        print(f"  {r['name']:42s} recall={r['recall@300_vs_exact']:.4f} p50={r['single']['p50_ms']:.3f}ms "
              f"p95={r['single']['p95_ms']:.3f}ms batch256={r['batched']['p50_ms']:.2f}ms "
              f"mem={r['memory_mb']:.1f}MB build={r['build_s']:.2f}s")
    return out


def main() -> None:
    cfg, _ = cli(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
