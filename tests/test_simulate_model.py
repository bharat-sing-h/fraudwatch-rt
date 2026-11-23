from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudwatch.features.engine import FEATURES
from fraudwatch.model.bundle import APPROVE, DECLINE, REVIEW, ModelBundle
from fraudwatch.model.metrics import (
    cost_optimal_threshold,
    day_block_bootstrap,
    operating_point,
    pr_auc,
    precision_threshold,
    summary,
)
from fraudwatch.model.train import TrainResult, known_labels, model_card
from fraudwatch.simulate import DAY, EPOCH, SimConfig, SimData, day_index, simulate

# --- simulator ----------------------------------------------------------------------------


def test_simulator_shape_and_determinism(small_sim: SimData) -> None:
    tx = small_sim.transactions
    assert tx.tx_id.is_unique
    assert tx.ts.is_monotonic_increasing
    assert 0.003 < tx.is_fraud.mean() < 0.05
    assert {"account_takeover", "card_testing", "compromised_terminal", "impossible_travel"} <= set(
        tx.scenario
    )
    fraud = tx[tx.is_fraud == 1]
    assert (fraud.label_ts > fraud.ts).all()
    assert (tx[tx.is_fraud == 0].label_ts == -1).all()
    assert list(small_sim.labels.label_ts) == sorted(small_sim.labels.label_ts)
    again = simulate(small_sim.config)
    pd.testing.assert_frame_equal(again.transactions, tx)


def test_drift_pattern_only_after_drift_day() -> None:
    sim = simulate(
        SimConfig(
            n_customers=300, n_terminals=500, n_online_terminals=30, n_days=12, drift_day=8, seed=1
        )
    )
    night = sim.transactions[sim.transactions.scenario == "night_online"]
    assert len(night) > 0
    assert day_index(night.ts).min() >= 8
    hours = (night.ts % DAY) // 3600
    assert hours.max() < 5


# --- metrics ------------------------------------------------------------------------------


def test_metrics_helpers() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4])
    amount = np.array([10, 10, 100, 50, 10, 30.0])
    assert pr_auc(y, s) == pytest.approx(1.0)
    assert math.isnan(pr_auc(np.zeros(3), np.ones(3)))
    op = operating_point(y, s, amount, threshold=0.5, review_cost=5)
    assert (op.precision, op.recall, op.alert_rate) == (
        1.0,
        pytest.approx(2 / 3),
        pytest.approx(1 / 3),
    )
    assert op.fraud_amount_caught == pytest.approx(150 / 180)
    assert op.net_savings == pytest.approx(150 - 10)
    assert precision_threshold(y, s, 1.0) == pytest.approx(0.4)
    assert precision_threshold(np.zeros(3, int), s[:3], 0.9) == math.inf
    assert 0.3 <= cost_optimal_threshold(y, s, amount) <= 0.9
    m = summary(y, s)
    assert m["roc_auc"] == 1.0
    assert m["fraud_rate"] == 0.5


def test_day_block_bootstrap_paired() -> None:
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(10), 200)
    y = (rng.random(2000) < 0.05).astype(int)
    good = y * 0.8 + rng.random(2000) * 0.3
    noise = rng.random(2000)
    cis = day_block_bootstrap(days, y, {"good": good, "noise": noise}, candidate="good", n=100)
    diff = cis["good - noise"]
    assert diff.low > 0
    assert cis["good"].low <= cis["good"].estimate <= cis["good"].high


# --- training -----------------------------------------------------------------------------


def test_known_labels_respect_report_time(small_sim: SimData) -> None:
    tx = small_sim.transactions
    as_of = EPOCH + 5 * DAY
    known = known_labels(tx, as_of)
    assert known.sum() < tx[tx.ts < as_of].is_fraud.sum() + 1
    assert not known[tx.label_ts.to_numpy() > as_of].any()


def test_training_beats_baselines(trained: TrainResult) -> None:
    m = trained.metrics
    assert m["lightgbm"]["pr_auc"] > m["rules"]["pr_auc"]
    assert m["lightgbm"]["roc_auc"] > 0.85
    assert trained.cis["lightgbm - rules"].estimate > 0
    b = trained.bundle
    assert b.review_threshold <= b.decline_threshold
    assert b.metadata["train_positive_known"] <= b.metadata["train_positive_true"]
    assert set(b.reference) == set(FEATURES) | {"__score__"}
    card = model_card(trained)
    assert "| lightgbm |" in card
    assert "Recall by fraud scenario" in card


def test_bundle_roundtrip_and_decisions(
    trained: TrainResult, small_features: pd.DataFrame, tmp_path: Path
) -> None:
    b = trained.bundle
    loaded = ModelBundle.load(b.save(tmp_path / "bundle"))
    X = small_features[FEATURES].to_numpy()[:500]
    np.testing.assert_allclose(loaded.predict(X), b.predict(X))
    row = dict(zip(FEATURES, X[0], strict=True))
    assert loaded.predict_one(row) == pytest.approx(b.predict(X[:1])[0])
    assert loaded.decide(0.0) == APPROVE
    assert loaded.decide(b.review_threshold) == REVIEW
    assert loaded.decide(max(b.decline_threshold, b.review_threshold) + 1e-9) in {REVIEW, DECLINE}
    assert loaded.describe()["version"] == "t1"


def test_bundle_rejects_unknown_features(trained: TrainResult, tmp_path: Path) -> None:
    import json

    d = trained.bundle.save(tmp_path / "b")
    meta = json.loads((d / "bundle.json").read_text())
    meta["features"] = [*meta["features"], "not_a_feature"]
    (d / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not_a_feature"):
        ModelBundle.load(d)
