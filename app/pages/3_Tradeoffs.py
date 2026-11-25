"""Relevance vs diversity: the reranking sweep on validation and the chosen operating point."""

import pandas as pd
import plotly.express as px
import streamlit as st
from common import header, metric

header("Relevance vs diversity", "Each point is one reranker configuration. The operating point is the most "
                                 "category-diverse setting within 2% of relevance-only NDCG@20 on the selection users.")
f = metric("funnel")
on = f.get("select_on", "val") if f else "val"
if not f or "sweep" not in f.get(on, {}):
    st.warning("Run `make pipeline` first.")
    st.stop()
which = st.radio("Users", [on] + (["val"] if on != "val" and "sweep" in f.get("val", {}) else []), horizontal=True,
                 format_func=lambda p: {"dense_val": "Dense matrix, selection users (chooses the operating point)",
                                        "val": "Temporal validation"}.get(p, p))
sw = pd.DataFrame(f[which]["sweep"])
sel = metric("selection").get("rerank_name")


def family(r) -> str:
    on = lambda v: pd.notna(v) and bool(v)  # noqa: E731  (JSON null loads as NaN, which is truthy)
    if on(r["pop_penalty"]):
        return "popularity penalty"
    if on(r["novelty_slots"]):
        return "caps + novelty slot"
    return "category/creator caps" if on(r["category_cap"]) else "MMR only"


sw["family"] = sw.apply(family, axis=1)
y = st.radio("Diversity axis", ["ild_category", "ild_embedding", "coverage", "category_entropy", "long_tail_share"],
             horizontal=True)
fig = px.scatter(sw, x="ndcg@20", y=y, color="family", hover_name="config",
                 hover_data={"lam": True, "rerank_p95_ms": ":.3f"}, height=460,
                 category_orders={"family": ["MMR only", "category/creator caps", "caps + novelty slot",
                                             "popularity penalty"]})
fig.update_traces(marker=dict(size=11))
ch = sw[sw["config"] == sel]
fig.add_scatter(x=ch["ndcg@20"], y=ch[y], mode="markers", name=f"chosen: {sel}",
                marker=dict(symbol="star", size=20, color="black"))
st.plotly_chart(fig, width="stretch")
st.caption(f"Chosen on {on}: **{sel}**. Pareto front (NDCG@20 vs category ILD): {', '.join(f[which].get('pareto', []))}")
st.dataframe(sw[["config", "ndcg@20", "recall@20", "ild_category", "ild_embedding", "category_entropy", "coverage",
                 "exposure_gini", "long_tail_share", "rerank_p95_ms"]].sort_values("ndcg@20", ascending=False).round(4),
             hide_index=True, width="stretch")
for p in ("test", "dense_test"):
    if p in f:
        r = f[p]
        base = r["rankers"][metric("selection").get("ranker", "lgb_rank")]
        rr = r["reranked"]
        st.markdown(f"**{p}**: NDCG@20 {base['ndcg@20']:.4f} → {rr['ndcg@20']:.4f}, category ILD "
                    f"{base['ild_category']:.3f} → {rr['ild_category']:.3f}, coverage {base['coverage']:.3f} → "
                    f"{rr['coverage']:.3f}, long-tail share {base['long_tail_share']:.3f} → {rr['long_tail_share']:.3f}")
