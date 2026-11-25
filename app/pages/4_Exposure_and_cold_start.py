"""Popularity exposure by decile, cold-start items and low-history users."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import header, metric

header("Exposure & cold start", "Popularity deciles are computed from training history only (decile 1 = most watched "
                                "before the cutoff). Recommendation share uses each system's top-100.")
r = metric("retrieval_eval")
if not r:
    st.warning("Run `make pipeline` first.")
    st.stop()
proto = st.radio("Protocol", ["test", "val"], horizontal=True)
d = r.get(f"deciles_{proto}")
if d:
    systems = st.multiselect("Systems", list(d["systems"]), default=[s for s in ("popularity", "als", "tt_inbatch", "tt_main")
                                                                  if s in d["systems"]])
    x = [f"D{i}" for i in range(1, 11)]
    fig = go.Figure()
    fig.add_bar(x=x, y=d["interaction_share"], name="training interactions", marker_color="#cbd5e1")
    fig.add_bar(x=x, y=d["target_share"], name="test-window positives", marker_color="#94a3b8")
    palette = ["#2563eb", "#f97316", "#10b981", "#a855f7", "#ef4444", "#0ea5e9", "#eab308", "#64748b"]
    for i, s in enumerate(systems):
        fig.add_scatter(x=x, y=d["systems"][s]["rec_share"], name=f"{s} rec share", mode="lines+markers",
                        line=dict(color=palette[i % len(palette)], width=2.5))
    fig.update_layout(height=420, barmode="group", yaxis_title="share", margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({s: d["systems"][s]["recall_by_decile"] for s in systems}, index=x).round(4).T,
                 width="stretch")
cs = r.get("cold_start", {})
if cs:
    st.subheader("Cold-start items (held-out IDs, test protocol)")
    st.caption(f"{cs['held_out_items']} items were removed from training and from every user history. Their in-dataset "
               "and platform statistics are zeroed, so only static metadata remains.")
    st.dataframe(pd.DataFrame({k: v for k, v in cs.items() if isinstance(v, dict)}).T.round(4), width="stretch")
co = r.get("cohorts", {})
if co:
    st.subheader("Recall@100 by user activity (events before the cutoff)")
    st.dataframe(pd.DataFrame({s: {c: v["recall@100"] for c, v in row.items()} for s, row in co.items()}).round(4),
                 width="stretch")
    st.caption("Users per cohort: " + ", ".join(f"{c}: {v['users']}" for c, v in next(iter(co.values())).items()))
