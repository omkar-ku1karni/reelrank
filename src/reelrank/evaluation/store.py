"""Persist recommendation lists and metrics so later stages (report, app, exposure analysis) never retrain.

artifacts/runs/<protocol>/<system>.npy          top-K item indices for each evaluated user
artifacts/runs/<protocol>/<system>.json         summary metrics + metadata
artifacts/runs/<protocol>/per_user.parquet      long table of per-user metrics (one row per user/system)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from reelrank.config import Config, write_json
from reelrank.evaluation.protocol import Protocol, summarize


def run_dir(cfg: Config, protocol: str) -> Path:
    d = cfg.artifacts / "runs" / protocol
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(cfg: Config, protocol: Protocol, system: str, rec: np.ndarray, per_user: dict[str, np.ndarray],
             meta: dict | None = None, k_expo: int = 20) -> dict:
    d = run_dir(cfg, protocol.name)
    np.save(d / f"{system}.npy", rec.astype(np.int32))
    summary = summarize(per_user)
    summary.update({f"{k}@{k_expo}": v for k, v in protocol.exposure(rec, k_expo).items()})
    rec_meta = {"system": system, "protocol": protocol.name, "users": int(len(protocol.users)),
                "users_with_relevant": int((per_user["n_relevant"] > 0).sum()),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": summary, **(meta or {})}
    write_json(d / f"{system}.json", rec_meta)
    df = pl.DataFrame({"user_id": protocol.users, **{k: v for k, v in per_user.items()}}).with_columns(
        system=pl.lit(system))
    path = d / "per_user.parquet"
    if path.exists():
        old = pl.read_parquet(path).filter(pl.col("system") != system)
        df = pl.concat([old, df], how="diagonal_relaxed")
    df.write_parquet(path)
    return rec_meta


def load_summaries(cfg: Config, protocol: str) -> dict[str, dict]:
    d = cfg.artifacts / "runs" / protocol
    if not d.exists():
        return {}
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def print_table(rows: dict[str, dict], cols: list[str]) -> None:
    print(f"{'system':28s}" + "".join(f"{c:>14s}" for c in cols))
    for name, r in rows.items():
        m = r.get("metrics", r)
        print(f"{name:28s}" + "".join(f"{m.get(c, float('nan')):14.4f}" for c in cols))
