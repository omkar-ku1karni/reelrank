SHELL := /bin/bash
# My machine keeps caches on an external drive; elsewhere the tools' defaults are used.
ifneq ($(wildcard /Volumes/Crucial_1TB/cache),)
export UV_CACHE_DIR ?= /Volumes/Crucial_1TB/cache/uv
export PLAYWRIGHT_BROWSERS_PATH ?= /Volumes/Crucial_1TB/cache/ms-playwright
endif
export MLFLOW_DISABLE_AGENT_HINT := 1
export PYTHONWARNINGS := ignore
CONFIG ?= configs/kuairec.yaml
PY := uv run python
RUN := $(PY) -m
C := --config $(CONFIG)

# Every stage below is its own process. PyTorch, FAISS and LightGBM each bundle an OpenMP runtime and
# crash or deadlock when two of them run in one process on macOS (see README, "Process isolation").

.PHONY: help setup download ingest features eda baselines two-tower-select two-tower retrieval candidates \
        ranker-data ranker-lgb ranker-nn funnel index export simulate report pipeline all smoke fixtures test \
        test-unit test-torch test-faiss lint api demo mlflow-ui screenshots demo-gif clean

help:              ## list targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup:             ## install pinned dependencies into .venv
	uv sync

download:          ## fetch KuaiRec 2.0 from Zenodo (~432 MB) and verify its MD5
	$(RUN) reelrank.data.ingest download $(C)

ingest:            ## raw CSV -> typed, de-duplicated Parquet + QA report (DuckDB)
	$(RUN) reelrank.data.ingest ingest $(C)

features:          ## leakage-safe daily user/item snapshots + evaluation targets (Polars/NumPy)
	$(RUN) reelrank.features.build $(C)

eda:               ## exploratory figures and statistics
	$(RUN) reelrank.evaluation.eda $(C)

baselines:         ## random/popularity/trending/category/item-kNN/ALS (+ ALS tuning on validation)
	$(RUN) reelrank.retrieval.baselines $(C)

two-tower-select:  ## train every two-tower variant on the validation cutoff, then select on val recall@100 [torch]
	$(RUN) reelrank.retrieval.two_tower $(C) --tags R1 --variants all
	$(RUN) reelrank.retrieval.evaluate $(C)

two-tower:         ## retrain all variants for test (R2) and the selected one for R0 / gap cutoffs [torch]
	$(RUN) reelrank.retrieval.two_tower $(C) --tags R2 --variants all
	$(RUN) reelrank.retrieval.two_tower $(C) --tags R0 RG --variants selected

retrieval:         ## retrieval metrics, cold start, cohorts, exposure deciles (NumPy exact search)
	$(RUN) reelrank.retrieval.evaluate $(C)

candidates:        ## candidate union (two-tower + ALS + trending + category) and pre-ranking recall
	$(RUN) reelrank.retrieval.candidates $(C)

ranker-data:       ## ranker feature tables for observed windows and candidate pools
	$(RUN) reelrank.ranking.dataset $(C)

ranker-lgb:        ## LightGBM LambdaMART / regression rankers + ablations [lightgbm]
	$(RUN) reelrank.ranking.lgb_ranker $(C)

ranker-nn:         ## multitask neural ranker (watch / complete / skip heads) [torch]
	$(RUN) reelrank.ranking.neural $(C)

funnel:            ## end-to-end ladder, rerank sweep + Pareto operating point, bootstrap CIs
	$(RUN) reelrank.evaluation.funnel $(C)

index:             ## exact vs HNSW / IVF vector-search benchmark [faiss]
	$(RUN) reelrank.retrieval.faiss_bench $(C)

export:            ## serving bundle + hash-pinned artifacts/manifest.json (checks NumPy/PyTorch parity)
	$(RUN) reelrank.serving.export $(C)

simulate:          ## replay test users through the engine: latency, fallbacks, drift (offline simulation)
	$(RUN) reelrank.serving.simulate $(C)

report:            ## research report + figures from persisted metrics
	$(RUN) reelrank.evaluation.report $(C)

pipeline: ingest features eda baselines two-tower-select two-tower retrieval candidates ranker-data ranker-lgb \
          ranker-nn funnel index export simulate report  ## every stage after download, in order

all: setup download pipeline  ## one-command rebuild on the real data (~2 h on an M4 CPU)

fixtures:          ## write the synthetic KuaiRec-shaped fixture to data/fixture/raw
	$(PY) scripts/make_fixtures.py

smoke:             ## run the full pipeline on the synthetic fixture (~2 min)
	$(PY) scripts/make_fixtures.py
	$(MAKE) pipeline CONFIG=configs/smoke.yaml REELRANK_NO_MLFLOW=1

test: test-unit test-torch test-faiss  ## all tests (three processes, one per native runtime)

test-unit:         ## metrics, features, reranker, engine, API, contracts [lightgbm side]
	uv run pytest -q tests --ignore=tests/torch_tests --ignore=tests/faiss_tests

test-torch:        ## two-tower losses, masking, NumPy parity [torch side]
	uv run pytest -q tests/torch_tests

test-faiss:        ## FAISS exact index agrees with NumPy [faiss side]
	uv run pytest -q tests/faiss_tests

lint:              ## ruff
	uv run ruff check src app tests scripts

api:               ## FastAPI on http://localhost:8000 (docs at /docs)
	uv run uvicorn reelrank.serving.api:app --port 8000

demo:              ## Streamlit app on http://localhost:8501
	uv run streamlit run app/Home.py

mlflow-ui:         ## local MLflow UI on http://localhost:5000
	uv run mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000

screenshots:       ## capture the six screens (app must be running on :8501; needs `uv run playwright install chromium`)
	$(PY) scripts/screenshots.py http://localhost:8501

demo-gif:          ## record reports/demo.gif from the running app
	$(PY) scripts/demo_gif.py http://localhost:8501

clean:             ## remove derived data (keeps the downloaded raw files)
	rm -rf data/processed artifacts/runs artifacts/two_tower artifacts/als artifacts/ranker artifacts/serving \
	       artifacts/index artifacts/itemknn data/fixture
