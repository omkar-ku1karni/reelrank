"""Shared Streamlit helpers: cached engine, metric loaders, card rendering."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reelrank.serving.engine import Engine  # noqa: E402

ARTIFACTS = Path(os.environ.get("REELRANK_ARTIFACTS", ROOT / "artifacts"))
METRICS = Path(os.environ.get("REELRANK_REPORTS", ROOT / "reports")) / "metrics"

CSS = """
<style>
.rr-grid {display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:12px;}
.rr-card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:10px 12px; background:rgba(128,128,128,.04);}
.rr-card.hit {border:2px solid #10b981; background:rgba(16,185,129,.07);}
.rr-rank {font-weight:700; font-size:.8rem; color:#6b7280;}
.rr-cat {font-weight:700; font-size:.95rem; margin:2px 0;}
.rr-cap {font-size:.82rem; min-height:2.4em; overflow:hidden; display:-webkit-box; -webkit-line-clamp:2;
         -webkit-box-orient:vertical; opacity:.85;}
.rr-meta {font-size:.75rem; color:#6b7280; margin-top:4px;}
.rr-badge {display:inline-block; font-size:.68rem; padding:1px 6px; border-radius:999px; margin:2px 3px 0 0;
           background:rgba(37,99,235,.12); color:#2563eb;}
.rr-badge.als {background:rgba(249,115,22,.14); color:#c2410c;}
.rr-badge.trending {background:rgba(168,85,247,.14); color:#7e22ce;}
.rr-badge.category {background:rgba(100,116,139,.16); color:#475569;}
.rr-hit {font-size:.72rem; color:#059669; font-weight:700;}
</style>
"""


@st.cache_resource(show_spinner="Loading serving bundle…")
def engine() -> Engine:
    return Engine(ARTIFACTS)


@st.cache_data
def metric(name: str) -> dict:
    f = METRICS / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def header(title: str, caption: str | None = None) -> None:
    st.set_page_config(page_title=f"ReelRank · {title}", page_icon="🎞️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if caption:
        st.caption(caption)


def cards(items: list[dict], hits: set[int] | None = None, show_scores: bool = True) -> None:
    hits = hits or set()
    out = ['<div class="rr-grid">']
    for it in items:
        hit = it["video_id"] in hits
        badges = "".join(f'<span class="rr-badge {s}">{s.replace("_", "-")}</span>' for s in it.get("retrieval_sources", []))
        score = ""
        if show_scores and it.get("predicted_engagement") is not None:
            score = f" · ranker {it['predicted_engagement']:.2f}"
        cap = html.escape(it.get("caption") or "(no caption)")
        out.append(
            f'<div class="rr-card{" hit" if hit else ""}"><div class="rr-rank">#{it.get("rank", "")}'
            f'{" · <span class=rr-hit>watched ✓</span>" if hit else ""}</div>'
            f'<div class="rr-cat">{html.escape(it["category"])} <span style="font-weight:400;opacity:.6">'
            f'{html.escape(it.get("category_zh", ""))}</span></div><div class="rr-cap">{cap}</div>'
            f'<div class="rr-meta">video {it["video_id"]} · creator {it["creator_id"]} · {it["duration_s"]:.0f}s{score}</div>'
            f'<div>{badges}</div></div>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)


def sample_users(e: Engine, n: int = 400) -> list[int]:
    import numpy as np

    users = sorted(set(e.truth["user_id"].astype(int).tolist()))
    rng = np.random.default_rng(7)
    return sorted(rng.choice(users, min(n, len(users)), replace=False).tolist())
