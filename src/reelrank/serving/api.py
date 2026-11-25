"""FastAPI service over the exported serving bundle.

    uvicorn reelrank.serving.api:app --port 8000        (or `make api`)

GET  /recommendations/{user_id}?k=20&algorithm=reelrank&lam=&novel=true&explain=false
GET  /videos/{video_id}/similar?k=20
GET  /models/metrics      offline evaluation summary + live request monitoring
POST /feedback            append a feedback event to a local JSONL log (no online retraining)
GET  /health

Recommendations reflect the state of the KuaiRec log at the snapshot date (2020-09-01, UTC+8); they are
metadata cards for public research data, not playable videos.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from reelrank.config import ROOT
from reelrank.serving.engine import ALGORITHMS, Engine

ARTIFACTS = Path(os.environ.get("REELRANK_ARTIFACTS", ROOT / "artifacts"))
LOG_DIR = Path(os.environ.get("REELRANK_LOG_DIR", ARTIFACTS / "logs"))
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["engine"] = Engine(ARTIFACTS)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    yield
    STATE.clear()


app = FastAPI(title="ReelRank", version="0.1.0", lifespan=lifespan,
              description="Multi-stage short-video recommender on KuaiRec (offline research prototype).")


def engine() -> Engine:
    return STATE["engine"]


@app.middleware("http")
async def request_log(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t) * 1e3
    response.headers["X-Response-Time-ms"] = f"{ms:.2f}"
    if request.url.path.startswith("/recommendations"):
        with open(LOG_DIR / "requests.jsonl", "a") as f:
            f.write(json.dumps({"ts": time.time(), "path": request.url.path, "query": str(request.url.query),
                                "status": response.status_code, "ms": round(ms, 3)}) + "\n")
    return response


@app.get("/health")
def health() -> dict:
    e = engine()
    return {"status": "ok", "model_version": e.model_version, "snapshot_date": e.manifest["snapshot_date"],
            "users": e.U, "items": e.I}


@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, k: int = Query(20, ge=1, le=100),
                    algorithm: Literal[ALGORITHMS] = "reelrank",  # type: ignore[valid-type]
                    lam: float | None = Query(None, ge=0.0, le=1.0, description="override the MMR λ"),
                    novel: bool = Query(True, description="exclude videos the user already watched"),
                    explain: bool = False) -> dict:
    return engine().recommend(user_id, k=k, algorithm=algorithm, lam=lam, novel=novel, explain=explain)


@app.get("/videos/{video_id}/similar")
def similar(video_id: int, k: int = Query(20, ge=1, le=100)) -> dict:
    try:
        return engine().similar(video_id, k)
    except KeyError:
        raise HTTPException(404, f"unknown video_id {video_id}") from None


@app.get("/models/metrics")
def metrics() -> dict:
    e = engine()
    rep = ARTIFACTS.parent / "reports" / "metrics"
    offline = {}
    for name in ("funnel", "candidate_recall", "index_benchmark"):
        f = rep / f"{name}.json"
        if f.exists():
            offline[name] = json.loads(f.read_text())
    summary = {}
    if "funnel" in offline:
        summary = {p: {"ladder": [{k: v for k, v in r.items() if not k.startswith("vs_")}
                                  for r in offline["funnel"][p]["ladder"]]}
                   for p in ("test", "dense_test") if p in offline["funnel"]}
    return {"model_version": e.model_version, "snapshot_date": e.manifest["snapshot_date"],
            "rerank": e.manifest["rerank_name"], "offline": summary,
            "monitoring (live requests, simulation only)": e.monitoring()}


class Feedback(BaseModel):
    user_id: int
    video_id: int
    event: Literal["impression", "play", "complete", "skip", "like", "hide"]
    watch_ratio: float | None = Field(None, ge=0)
    model_version: str | None = None


@app.post("/feedback", status_code=202)
def feedback(fb: Feedback) -> dict:
    e = engine()
    if not (0 <= fb.video_id < e.I):
        raise HTTPException(422, "unknown video_id")
    rec = {"ts": time.time(), **fb.model_dump(), "served_model": e.model_version}
    with open(LOG_DIR / "feedback.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return {"accepted": True, "note": "logged locally for offline analysis; models are not retrained online"}
