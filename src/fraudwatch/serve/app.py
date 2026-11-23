"""HTTP scoring API.

    uvicorn fraudwatch.serve.app:app --port 8080

Configuration (environment):
    FW_STORE=redis|memory          online feature store (default memory)
    FW_REDIS_URL=redis://...       when FW_STORE=redis
    FW_MLFLOW_URI=...              load champion/challenger by alias from the MLflow registry
    FW_MODEL_DIR=...               ... or load a bundle directory directly (no registry)
    FW_CHALLENGER_DIR=...          optional shadow model directory
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from fraudwatch import __version__
from fraudwatch.features.engine import FeatureEngine, FraudLabel, Transaction
from fraudwatch.features.store import InMemoryStore, RedisStore, StateStore
from fraudwatch.model.bundle import ModelBundle
from fraudwatch.serve.scorer import Scorer


class TransactionIn(BaseModel):
    tx_id: int
    customer_id: int
    terminal_id: int
    amount: float = Field(gt=0, lt=1_000_000)
    ts: int | None = Field(default=None, description="unix seconds; defaults to now")


class LabelIn(BaseModel):
    tx_id: int
    tx_ts: int
    customer_id: int
    terminal_id: int


class CustomerProfileIn(BaseModel):
    customer_id: int
    home_x: float
    home_y: float


class TerminalProfileIn(BaseModel):
    terminal_id: int
    x: float | None = None
    y: float | None = None
    category: str
    is_online: bool


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.decisions = Counter("fw_decisions_total", "Decisions", ["decision"], registry=r)
        self.latency = Histogram(
            "fw_latency_seconds",
            "Scoring latency by stage",
            ["stage"],
            registry=r,
            buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
        )
        self.score = Histogram(
            "fw_score",
            "Champion score distribution",
            registry=r,
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95),
        )
        self.shadow_disagree = Counter(
            "fw_shadow_disagreements_total",
            "Champion vs challenger decision mismatches",
            ["champion", "challenger"],
            registry=r,
        )
        self.labels = Counter("fw_labels_total", "Fraud reports received", registry=r)
        self.model = Gauge("fw_model_info", "Loaded models", ["role", "version"], registry=r)


def make_store() -> StateStore:
    if os.environ.get("FW_STORE", "memory") == "redis":
        import redis

        return RedisStore(
            redis.Redis.from_url(os.environ.get("FW_REDIS_URL", "redis://localhost:6379/0"))
        )
    return InMemoryStore()


def load_models() -> tuple[ModelBundle, ModelBundle | None]:
    uri = os.environ.get("FW_MLFLOW_URI")
    if uri:
        from fraudwatch.model.registry import Registry

        reg = Registry(uri)
        champion = reg.load("champion")
        challenger = reg.load("challenger") if reg.alias_version("challenger") else None
        return champion, challenger
    model_dir = os.environ.get("FW_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("set FW_MLFLOW_URI or FW_MODEL_DIR")
    challenger_dir = os.environ.get("FW_CHALLENGER_DIR")
    return ModelBundle.load(model_dir), ModelBundle.load(challenger_dir) if challenger_dir else None


def create_app(scorer: Scorer | None = None) -> FastAPI:
    if scorer is None:
        champion, challenger = load_models()
        scorer = Scorer(FeatureEngine(make_store()), champion, challenger)
    metrics = Metrics()
    app = FastAPI(title="fraudwatch", version=__version__)
    app.state.scorer = scorer

    def publish_model_info() -> None:
        metrics.model.clear()
        metrics.model.labels("champion", scorer.champion.version).set(1)
        if scorer.challenger:
            metrics.model.labels("challenger", scorer.challenger.version).set(1)

    publish_model_info()

    @app.post("/v1/score")
    def score(tx: TransactionIn) -> dict[str, Any]:
        # sync endpoint: FastAPI runs it in a worker thread, so blocking Redis calls don't stall
        # the event loop
        result = scorer.score(
            Transaction(
                tx.tx_id, tx.ts or int(time.time()), tx.customer_id, tx.terminal_id, tx.amount
            )
        )
        metrics.decisions.labels(result.decision).inc()
        metrics.score.observe(result.score)
        for stage, ms in (
            ("features", result.features_ms),
            ("model", result.model_ms),
            ("total", result.total_ms),
        ):
            metrics.latency.labels(stage).observe(ms / 1000)
        if result.shadow_decision and result.shadow_decision != result.decision:
            metrics.shadow_disagree.labels(result.decision, result.shadow_decision).inc()
        return result.as_dict()

    @app.post("/v1/labels", status_code=202)
    def label(lab: LabelIn) -> dict[str, str]:
        scorer.label(FraudLabel(lab.tx_id, lab.tx_ts, lab.customer_id, lab.terminal_id))
        metrics.labels.inc()
        return {"status": "accepted"}

    @app.put("/v1/profiles/customers", status_code=204)
    def customers(profiles: list[CustomerProfileIn]) -> None:
        for p in profiles:
            scorer.engine.put_customer_profile(p.customer_id, p.home_x, p.home_y)

    @app.put("/v1/profiles/terminals", status_code=204)
    def terminals(profiles: list[TerminalProfileIn]) -> None:
        nan = float("nan")
        for p in profiles:
            if p.category not in {
                "grocery",
                "restaurant",
                "fuel",
                "retail",
                "electronics",
                "travel",
                "online",
            }:
                raise HTTPException(422, f"unknown category {p.category!r}")
            scorer.engine.put_terminal_profile(
                p.terminal_id,
                p.x if p.x is not None else nan,
                p.y if p.y is not None else nan,
                p.category,
                p.is_online,
            )

    @app.get("/v1/model")
    def model() -> dict[str, Any]:
        return {
            "champion": scorer.champion.describe(),
            "challenger": scorer.challenger.describe() if scorer.challenger else None,
        }

    @app.post("/admin/reload")
    def reload() -> dict[str, Any]:
        """Re-resolve registry aliases: promotion/rollback without a redeploy."""
        champion, challenger = load_models()
        scorer.swap(champion, challenger)
        publish_model_info()
        return {
            "champion": champion.version,
            "challenger": challenger.version if challenger else None,
        }

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


def __getattr__(name: str) -> Any:
    if name == "app":  # `uvicorn fraudwatch.serve.app:app` builds from environment
        return create_app()
    raise AttributeError(name)
