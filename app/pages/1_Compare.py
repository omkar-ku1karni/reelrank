"""Side-by-side comparison of algorithms for one user."""

import streamlit as st
from common import cards, engine, header, sample_users

header("Compare algorithms", "Same user, same snapshot, four systems. Green = watched in the next five days on the logged "
                             "feed, a label that rewards reproducing what the production policy showed (see Offline results).")
e = engine()
users = sample_users(e)
uid = st.selectbox("User", users, index=min(10, len(users) - 1))
k = st.select_slider("Items per list", [10, 20], 10)
hits = e.watched_after(int(uid))
systems = [("reelrank", "Full funnel"), ("two_tower", "Two-tower"), ("als", "ALS"), ("trending", "Trending")]
cols = st.columns(len(systems))
for col, (algo, label) in zip(cols, systems):
    res = e.recommend(int(uid), k=k, algorithm=algo)
    got = sum(1 for it in res["items"] if it["video_id"] in hits)
    with col:
        st.markdown(f"#### {label}")
        st.caption(f"{got}/{k} watched later · {len({it['category'] for it in res['items']})} categories · "
                   f"{res['latency_ms']:.1f} ms")
        cards(res["items"], hits, show_scores=False)
