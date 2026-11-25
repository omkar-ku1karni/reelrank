# ReelRank runtime: the full pipeline, the FastAPI service and the Streamlit app in one CPU image.
# Linux resolves PyTorch from the CPU-only index (see pyproject.toml), so the image carries no CUDA libraries.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 MLFLOW_DISABLE_AGENT_HINT=1
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8000 8501
# default: serve the API over artifacts built by `make all` (mounted or built in the container)
CMD ["uv", "run", "--no-dev", "uvicorn", "reelrank.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
