"""Latency, vector-index benchmark, artifact manifest and live monitoring counters."""

import pandas as pd
import plotly.express as px
import streamlit as st
from common import engine, header, metric, sample_users

header("Engineering", "Measured on the development machine (Apple M4, CPU only). Exact search is used in serving.")
e = engine()
ib = metric("index_benchmark")
if ib:
    df = pd.DataFrame([{"index": r["name"], "recall@300 vs exact": r["recall@300_vs_exact"],
                        "p50 ms": r["single"]["p50_ms"], "p95 ms": r["single"]["p95_ms"],
                        "batch-256 p50 ms": r["batched"]["p50_ms"], "memory MB": r["memory_mb"],
                        "build s": r["build_s"]} for r in ib["indexes"]])
    st.subheader(f"Vector retrieval over {ib['items']:,} × {ib['dim']} item vectors (top-{ib['k']})")
    df["family"] = df["index"].map(lambda s: "HNSW" if "HNSW" in s else "IVF-Flat" if "IVF" in s else "exact")
    fig = px.scatter(df, x="p50 ms", y="recall@300 vs exact", color="family", hover_name="index", log_x=True,
                     height=360, color_discrete_map={"exact": "#111827", "HNSW": "#2563eb", "IVF-Flat": "#f97316"})
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(df.round(4), hide_index=True, width="stretch")
    st.caption(ib.get("note", ""))

st.subheader("End-to-end request latency (this process)")
n = st.slider("Requests to time", 20, 300, 100, 20)
if st.button("Run latency check"):
    import numpy as np

    us = sample_users(e, n)
    lat, stages = [], {}
    for u in us:
        res = e.recommend(int(u), k=20)
        lat.append(res["latency_ms"])
        for k_, v in res["stage_ms"].items():
            stages.setdefault(k_, []).append(v)
    st.write({"p50 ms": round(float(np.percentile(lat, 50)), 2), "p95 ms": round(float(np.percentile(lat, 95)), 2),
              **{f"{k_} p95": round(float(np.percentile(v, 95)), 2) for k_, v in stages.items()}})

st.subheader("Monitoring (live counters; offline simulation, not production)")
st.json(e.monitoring())
st.subheader("Artifact manifest")
m = dict(e.manifest)
m["files"] = {k: v["sha256"][:16] for k, v in m["files"].items()}
st.json(m, expanded=False)
