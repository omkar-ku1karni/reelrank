"""Shared fixtures.

`fixture_cfg` builds the synthetic KuaiRec-shaped dataset into a temp dir and runs ingest + features
in-process (DuckDB / Polars only, a few seconds). Tests that need trained artifacts use the smoke run in
data/fixture (created by `make smoke`, which CI runs before the tests).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def fixture_cfg(tmp_path_factory):
    from make_fixtures import main as make_fixtures

    from reelrank.config import load_config
    from reelrank.data.ingest import ingest
    from reelrank.features.build import build

    d = tmp_path_factory.mktemp("kuairec_fixture")
    make_fixtures(d / "raw")
    cfg_path = d / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "extends": str(ROOT / "configs" / "smoke.yaml"),
        "paths": {"raw": str(d / "raw"), "processed": str(d / "processed"), "artifacts": str(d / "artifacts"),
                  "reports": str(d / "reports"), "mlflow": str(d / "artifacts" / "mlflow.db")}}))
    cfg = load_config(cfg_path)
    ingest(cfg)
    build(cfg)
    return cfg


@pytest.fixture(scope="session")
def smoke_artifacts():
    art = ROOT / "data" / "fixture" / "artifacts"
    if not (art / "manifest.json").exists():
        subprocess.run(["make", "smoke"], cwd=ROOT, check=True)
    return art
