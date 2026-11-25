"""Offline evaluation: ablation ladder with paired bootstrap intervals, retrieval and ranking tables."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import header, metric

header("Offline results", "Temporal test = observed positives in Sep 1-5 (incomplete labels). Dense = KuaiRec small "
                          "matrix, ~99.6% of user-video pairs labelled. Do not compare absolute numbers across the two.")
funnel = metric("funnel")
proto = st.radio("Protocol", ["dense_test", "test", "dense_val", "val"], horizontal=True,
                 format_func=lambda p: {"test": "Temporal test (Sep 1-5)", "dense_test": "Dense matrix, held-out users",
                                        "dense_val": "Dense matrix, selection users",
                                        "val": "Temporal validation (Aug 29-31)"}[p])
if not funnel or proto not in funnel:
    st.warning("Run `make pipeline` first.")
    st.stop()
lad = funnel[proto]["ladder"]
df = pd.DataFrame([{"system": r["system"], "NDCG@20": r["ndcg@20"], "Recall@20": r["recall@20"],
                    "Precision@20": r["precision@20"], "MRR": r["mrr"], "category ILD": r["ild_category"],
                    "Δ NDCG@20 vs previous": (f"{r['vs_previous_ndcg@20']['delta']:+.4f} "
                                             f"[{r['vs_previous_ndcg@20']['lo']:+.4f}, {r['vs_previous_ndcg@20']['hi']:+.4f}]"
                                             if "vs_previous_ndcg@20" in r else "")} for r in lad])
st.subheader("Ablation ladder")
st.dataframe(df, hide_index=True, width="stretch")
fig = go.Figure(go.Bar(x=df["NDCG@20"], y=df["system"], orientation="h", marker_color="#2563eb",
                       text=[f"{v:.3f}" for v in df["NDCG@20"]], textposition="outside"))
fig.update_layout(height=320, margin=dict(l=10, r=40, t=10, b=10), yaxis=dict(autorange="reversed"),
                  xaxis_title=f"NDCG@20 (users: {funnel[proto]['users']})")
st.plotly_chart(fig, width="stretch")

st.subheader("Rankers on the same candidate pool")
st.dataframe(pd.DataFrame(funnel[proto]["rankers"]).T[["ndcg@20", "ndcg@10", "recall@20", "precision@20", "mrr",
                                                       "ild_category", "coverage", "long_tail_share"]].round(4),
             width="stretch")
ret = metric("retrieval_eval")
rproto = "dense" if proto.startswith("dense") else proto
if rproto in ret:
    st.subheader("Retrieval (full eligible catalog, top-300)")
    st.dataframe(pd.DataFrame(ret[rproto]).T[["recall@50", "recall@100", "recall@300", "ndcg@20", "coverage@20"]]
                 .sort_values("recall@100", ascending=False).round(4), width="stretch")
lgb = metric("ranker_lgb")
if lgb:
    st.subheader("Logged-impression ranking (re-order each user-day's observed videos)")
    split = "val" if proto == "val" else "test"
    nn = metric("ranker_nn")
    rows = {**lgb[split], **(nn.get(split, {}) if nn else {})}
    st.dataframe(pd.DataFrame(rows).T[["ndcg@10", "ndcg@20", "mrr", "top10_watch_ratio", "top10_completion",
                                       "top10_skip", "top10_duration_s"]].sort_values("ndcg@20", ascending=False).round(4),
                 width="stretch")
