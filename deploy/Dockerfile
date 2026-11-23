# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[kafka]"

FROM python:3.12-slim
# libgomp: LightGBM's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 app \
    && mkdir -p /mlflow /work/data && chown app:app /mlflow /work/data
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels
ENV PYTHONUNBUFFERED=1 MLFLOW_DISABLE_AGENT_HINT=1
USER app
WORKDIR /work
EXPOSE 8080
CMD ["uvicorn", "fraudwatch.serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
