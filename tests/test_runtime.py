"""Registry, API, stream worker and monitoring."""

from __future__ import annotations

import math
from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudwatch.features.batch import load_profiles
from fraudwatch.features.engine import FeatureEngine, Transaction
from fraudwatch.features.store import InMemoryStore, RedisStore
from fraudwatch.model.bundle import ModelBundle
from fraudwatch.model.registry import Registry
from fraudwatch.model.train import TrainResult
from fraudwatch.monitor.drift import drift_report, feature_psi, psi
from fraudwatch.monitor.promotion import promotion_gate
from fraudwatch.serve.app import create_app
from fraudwatch.serve.scorer import Scorer
from fraudwatch.simulate import SimData
from fraudwatch.stream.worker import DECISIONS, LABELS, TRANSACTIONS, InMemoryBus, run


def _scorer(
    sim: SimData,
    champion: ModelBundle,
    challenger: ModelBundle | None = None,
    store: object | None = None,
) -> Scorer:
    engine = FeatureEngine(store or InMemoryStore())  # type: ignore[arg-type]
    load_profiles(engine, sim.customers, sim.terminals)
    return Scorer(engine, champion, challenger)


# --- registry ------------------------------------------------------------------------------


def test_registry_aliases_promotion_and_rollback(trained: TrainResult, tmp_path: Path) -> None:
    reg = Registry(f"sqlite:///{tmp_path / 'mlflow.db'}")
    v1 = reg.log_and_register(
        trained.bundle,
        params={"a": 1},
        metrics={"pr_auc": 0.5},
        model_card="# card",
        alias="champion",
    )
    v2 = reg.log_and_register(
        trained.bundle, params={"a": 2}, metrics={"pr_auc": 0.6}, alias="challenger"
    )
    assert (reg.alias_version("champion"), reg.alias_version("challenger")) == (v1, v2)
    loaded = reg.load("champion")
    assert loaded.metadata["registry_version"] == v1
    assert loaded.review_threshold == pytest.approx(trained.bundle.review_threshold)
    assert reg.promote(v2) == v1
    assert reg.alias_version("champion") == v2
    assert reg.alias_version("previous") == v1
    assert reg.alias_version("challenger") is None  # consumed by the promotion
    with pytest.raises(LookupError):
        reg.load("challenger")


# --- scorer + API --------------------------------------------------------------------------


def test_api_scoring_labels_shadow_and_metrics(small_sim: SimData, trained: TrainResult) -> None:
    scorer = _scorer(small_sim, trained.bundle, challenger=trained.bundle)
    client = TestClient(create_app(scorer))
    tx = small_sim.transactions.iloc[5000]
    payload = {
        "tx_id": int(tx.tx_id),
        "ts": int(tx.ts),
        "customer_id": int(tx.customer_id),
        "terminal_id": int(tx.terminal_id),
        "amount": float(tx.amount),
    }
    body = client.post("/v1/score", json=payload).json()
    assert 0 <= body["score"] <= 1
    assert body["decision"] in {"approve", "review", "decline"}
    assert body["shadow_score"] == pytest.approx(body["score"])  # same model in shadow
    assert body["model_version"] == "t1"
    assert body["total_ms"] < 50
    assert client.post("/v1/score", json={**payload, "amount": -5}).status_code == 422

    lab = {
        "tx_id": int(tx.tx_id),
        "tx_ts": int(tx.ts),
        "customer_id": int(tx.customer_id),
        "terminal_id": int(tx.terminal_id),
    }
    assert client.post("/v1/labels", json=lab).status_code == 202
    again = client.post("/v1/score", json={**payload, "tx_id": 10**9, "ts": int(tx.ts) + 60})
    assert again.status_code == 200

    assert (
        client.put(
            "/v1/profiles/customers", json=[{"customer_id": 99999, "home_x": 1.0, "home_y": 2.0}]
        ).status_code
        == 204
    )
    assert (
        client.put(
            "/v1/profiles/terminals",
            json=[{"terminal_id": 99999, "category": "online", "is_online": True}],
        ).status_code
        == 204
    )
    assert (
        client.put(
            "/v1/profiles/terminals",
            json=[{"terminal_id": 1, "category": "casino", "is_online": True}],
        ).status_code
        == 422
    )

    metrics = client.get("/metrics").text
    assert "fw_decisions_total" in metrics
    assert 'fw_model_info{role="challenger",version="t1"} 1.0' in metrics
    assert client.get("/v1/model").json()["champion"]["version"] == "t1"
    assert client.get("/healthz").json() == {"status": "ok"}


def test_api_reload_from_model_dirs(
    small_sim: SimData,
    trained: TrainResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d = trained.bundle.save(tmp_path / "champion")
    monkeypatch.delenv("FW_MLFLOW_URI", raising=False)
    monkeypatch.setenv("FW_MODEL_DIR", str(d))
    client = TestClient(create_app())  # builds store + models from the environment
    assert client.post("/admin/reload").json() == {"champion": "t1", "challenger": None}
    monkeypatch.delenv("FW_MODEL_DIR")
    with pytest.raises(RuntimeError, match="FW_MODEL_DIR"):
        create_app()


# --- stream worker -------------------------------------------------------------------------


def test_worker_scores_labels_and_survives_redelivery(
    small_sim: SimData, trained: TrainResult
) -> None:
    tx = small_sim.transactions.head(300)
    bus = InMemoryBus()
    for r in tx.itertuples():
        bus.send(
            TRANSACTIONS,
            str(r.customer_id),
            {
                "tx_id": int(r.tx_id),
                "ts": int(r.ts),
                "customer_id": int(r.customer_id),
                "terminal_id": int(r.terminal_id),
                "amount": float(r.amount),
            },
        )
    first = tx.iloc[0]
    bus.send(
        LABELS,
        "0",
        {
            "tx_id": int(first.tx_id),
            "tx_ts": int(first.ts),
            "customer_id": int(first.customer_id),
            "terminal_id": int(first.terminal_id),
        },
    )
    store = RedisStore(fakeredis.FakeRedis())
    scorer = _scorer(small_sim, trained.bundle, store=store)

    # process half, "crash" before committing, then let Kafka redeliver
    for _ in range(150):
        msg = bus.poll(0)
        assert msg is not None
        v = msg.value
        scorer.score(
            Transaction(v["tx_id"], v["ts"], v["customer_id"], v["terminal_id"], v["amount"])
        )
    bus.crash_and_redeliver()
    processed = run(bus, scorer, commit_every=50, idle_polls=2)
    assert processed == len(tx) + 1
    decisions = [m for m in bus.outbox if m.topic == DECISIONS]
    assert len(decisions) == len(tx)
    assert bus.commits >= 1
    assert not bus.uncommitted

    # despite double processing of the first 150, state matches a clean single pass
    clean = _scorer(small_sim, trained.bundle)
    for r in tx.itertuples():
        clean.score(
            Transaction(
                int(r.tx_id), int(r.ts), int(r.customer_id), int(r.terminal_id), float(r.amount)
            )
        )
    c = int(tx.customer_id.iloc[0])
    assert store.hget(f"c:{c}:stats")["n"] == clean.engine.store.hget(f"c:{c}:stats")["n"]


# --- monitoring ----------------------------------------------------------------------------


def test_psi_properties() -> None:
    assert psi(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == pytest.approx(0.0)
    assert psi(np.array([0.5, 0.5]), np.array([0.9, 0.1])) > 0.25
    ref = {"edges": [0.0, 1.0], "expected": [1 / 3, 1 / 3, 1 / 3], "missing_rate": [0.0]}
    same = np.array([-1.0, 0.5, 2.0] * 100)
    assert feature_psi(ref, same) == pytest.approx(0.0, abs=1e-3)
    shifted = np.array([5.0] * 300)
    assert feature_psi(ref, shifted) > 1
    with_missing = np.append(same, [math.nan] * 300)
    assert feature_psi(ref, with_missing) > 0.25
    empty_ref = {"edges": [], "expected": [], "missing_rate": [1.0]}
    assert feature_psi(empty_ref, np.array([math.nan] * 10)) == pytest.approx(0.0, abs=1e-3)


def test_drift_report_flags_shift_and_performance(
    trained: TrainResult,
    small_features: pd.DataFrame,
    small_sim: SimData,
) -> None:
    b = trained.bundle
    y = small_sim.transactions.is_fraud.to_numpy()
    stable = drift_report(b, small_features, labels=y)
    assert stable.score_psi < 0.25
    assert stable.pr_auc is not None
    shifted = small_features.copy()
    shifted["amount"] *= 20
    shifted["log_amount"] = np.log1p(shifted["amount"])
    # adversarial labels: the "fraud" is exactly what the model considers safest
    scores = b.predict(shifted[b.features].to_numpy(dtype=float))
    inverted = (scores <= np.quantile(scores, 0.01)).astype(int)
    rep = drift_report(b, shifted, labels=inverted)
    assert rep.retrain_recommended
    assert any("amount" in a for a in rep.alerts)
    assert any("PR-AUC" in a for a in rep.alerts)
    assert "⚠️" in rep.to_markdown()


def test_promotion_gate() -> None:
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(14), 300)
    y = (rng.random(len(days)) < 0.05).astype(int)
    weak = y * 0.3 + rng.random(len(y))
    strong = y * 2.0 + rng.random(len(y))
    assert promotion_gate(days, y, weak, strong).promote
    no = promotion_gate(days, y, strong, weak)
    assert not no.promote
    assert "lower bound" in no.reason
    empty = promotion_gate(days, np.zeros(len(y), int), weak, strong)
    assert not empty.promote
    assert "no reported fraud" in empty.reason
