"""ReelRank feed: pick a historical KuaiRec user and see the multi-stage recommender's feed."""

import numpy as np
import streamlit as st
from common import cards, engine, header, sample_users

header("Feed", "Recommendations as of the 2020-09-01 snapshot (UTC+8) of the public KuaiRec log. Cards show "
               "metadata only: KuaiRec has no playable videos. A green border means the user actually watched "
               "that video (watch ratio ≥ 0.8) in the next five days, which is offline evaluation data. Those labels exist "
               "only where the production feed showed a video, so they favour two-tower retrieval; the ranker optimises "
               "engagement given exposure and wins on the fully-labelled matrix instead (see Offline results).")
e = engine()
users = sample_users(e)
with st.sidebar:
    st.subheader("User")
    uid = st.selectbox("Historical user", users, index=users.index(3) if 3 in users else 0)
    if st.button("Random user"):
        uid = int(np.random.default_rng().choice(users))
    st.subheader("Pipeline")
    algo = st.radio("Algorithm", ["reelrank", "ranker_only", "two_tower", "als", "trending", "popularity"],
                    format_func=lambda a: {"reelrank": "Full funnel (retrieve → rank → rerank)",
                                           "ranker_only": "Retrieve → rank (no diversity)",
                                           "two_tower": "Two-tower only", "als": "ALS only",
                                           "trending": "Trending", "popularity": "Popularity"}[a])
    lam = st.slider("Diversity: MMR λ (1 = relevance only)", 0.3, 1.0, float(e.rerank_cfg.lam), 0.05,
                    disabled=algo != "reelrank")
    k = st.slider("Feed length", 10, 50, 20, 5)
    novel = st.toggle("Hide already-watched videos", True)
    st.caption(f"Model `{e.model_version}` · rerank `{e.manifest['rerank_name']}`")

res = e.recommend(int(uid), k=k, algorithm=algo, lam=lam if algo == "reelrank" else None, novel=novel, explain=True)
hits = e.watched_after(int(uid))
got = sum(1 for it in res["items"] if it["video_id"] in hits)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Watched later", f"{got} / {len(res['items'])}")
c2.metric("Categories in feed", len({it["category"] for it in res["items"]}))
c3.metric("Latency", f"{res['latency_ms']:.1f} ms")
c4.metric("Fallback", res["fallback"] or "none")
if res.get("explain"):
    ex = res["explain"]
    st.caption(f"Candidate pool {ex['pool_size']} videos · by source "
               + ", ".join(f"{k}: {v}" for k, v in ex["pool_by_source"].items())
               + " · stage ms " + ", ".join(f"{k.removesuffix('_ms')}: {v:.1f}" for k, v in res["stage_ms"].items()))
left, right = st.columns([3, 1])
with left:
    cards(res["items"], hits)
with right:
    st.markdown("**Recent positive history**")
    for h in e.history(int(uid), 12):
        st.markdown(f"<div style='font-size:.8rem;margin-bottom:6px'><b>{h['category']}</b> · {h['duration_s']:.0f}s"
                    f"<br><span style='opacity:.7'>{(h['caption'] or '(no caption)')[:40]}</span></div>",
                    unsafe_allow_html=True)
