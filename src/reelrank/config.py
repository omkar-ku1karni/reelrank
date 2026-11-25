"""Configuration loading, paths, seeding and run bookkeeping shared by every stage."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "kuairec.yaml"
TZ_OFFSET_S = 8 * 3600  # KuaiRec `time` is Beijing time (UTC+8); calendar days use that clock.
EPOCH_DAY = dt.date(1970, 1, 1)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: dict
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def p(self, key: str) -> Path:
        """Absolute path for a `paths:` entry (relative entries resolve against the repo root)."""
        val = Path(self.raw["paths"][key])
        return val if val.is_absolute() else ROOT / val

    @property
    def processed(self) -> Path:
        return self.p("processed")

    @property
    def artifacts(self) -> Path:
        return self.p("artifacts")

    @property
    def reports(self) -> Path:
        return self.p("reports")

    def day(self, key: str) -> int:
        """Protocol date -> integer day index (days since 1970-01-01 on the UTC+8 calendar)."""
        return date_to_day(self.raw["protocol"][key])

    def windows(self) -> dict[str, tuple[int, int, str]]:
        """name -> (first day, last day, model tag) for every temporal evaluation window."""
        return {n: (date_to_day(w["start"]), date_to_day(w["end"]), w["model"])
                for n, w in self.raw["protocol"]["windows"].items()}

    def model_cutoffs(self) -> dict[str, int]:
        """model tag -> training cutoff day (the model sees only days strictly before it)."""
        return {tag: start for start, _, tag in self.windows().values()}

    def protocol_model(self, protocol: str) -> str:
        if protocol == "dense":
            return self.raw["protocol"]["dense_model"]
        return self.windows()[protocol][2]

    def protocol_cutoff(self, protocol: str) -> int:
        return self.model_cutoffs()[self.protocol_model(protocol)]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self.raw, sort_keys=True).encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("REELRANK_CONFIG") or DEFAULT_CONFIG)
    if not path.is_absolute():
        path = ROOT / path
    raw = yaml.safe_load(path.read_text())
    if "extends" in raw:
        parent = load_config(path.parent / raw.pop("extends"))
        raw = _deep_merge(parent.raw, raw)
    return Config(raw=raw, path=path)


def cli(description: str, extra: list[tuple[tuple, dict]] | None = None) -> tuple[Config, argparse.Namespace]:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=None, help="YAML config (default configs/kuairec.yaml or $REELRANK_CONFIG)")
    for args, kwargs in extra or []:
        ap.add_argument(*args, **kwargs)
    ns = ap.parse_args()
    cfg = load_config(ns.config)
    seed_everything(cfg.seed)
    return cfg, ns


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def date_to_day(s: str | dt.date) -> int:
    d = dt.date.fromisoformat(s) if isinstance(s, str) else s
    return (d - EPOCH_DAY).days


def day_to_date(day: int) -> str:
    return (EPOCH_DAY + dt.timedelta(days=int(day))).isoformat()


def ts_to_day(ts: np.ndarray) -> np.ndarray:
    return np.floor((np.asarray(ts, dtype=np.float64) + TZ_OFFSET_S) / 86400.0).astype(np.int32)


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, default=_json_default) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
