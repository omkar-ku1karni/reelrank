"""Thin local-MLflow wrapper (SQLite backend under artifacts/). Tracking never blocks the pipeline."""

from __future__ import annotations

import contextlib
import os
import warnings

from reelrank.config import Config


@contextlib.contextmanager
def mlflow_run(cfg: Config, experiment: str, run_name: str, params: dict | None = None):
    """Yield a `log(metrics: dict, step=None)` callable; a no-op if MLflow is unavailable or disabled."""
    if os.environ.get("REELRANK_NO_MLFLOW") == "1":
        yield lambda *a, **k: None
        return
    try:
        import logging

        logging.getLogger("mlflow").setLevel(logging.ERROR)
        logging.getLogger("alembic").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", module="mlflow")
        import mlflow

        db = cfg.p("mlflow")
        db.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        mlflow.set_experiment(f"reelrank/{cfg.name}/{experiment}")
        mlflow.start_run(run_name=run_name)
    except Exception as exc:  # pragma: no cover - tracking is best effort
        print(f"[mlflow disabled: {exc.__class__.__name__}: {exc}]")
        yield lambda *a, **k: None
        return
    try:
        if params:
            mlflow.log_params({k: (str(v)[:500]) for k, v in params.items()})
        mlflow.set_tag("config_hash", cfg.hash())

        def log(metrics: dict, step: int | None = None) -> None:
            clean = {k.replace("@", "_at_"): float(v) for k, v in metrics.items()
                     if isinstance(v, (int, float)) and v == v}
            mlflow.log_metrics(clean, step=step)

        yield log
    finally:
        mlflow.end_run()
