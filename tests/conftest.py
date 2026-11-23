from __future__ import annotations

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd

from fraudwatch.features.batch import build_features
from fraudwatch.model.train import TrainConfig, TrainResult, train
from fraudwatch.simulate import SimConfig, SimData, simulate

SMALL = SimConfig(
    n_customers=400,
    n_terminals=800,
    n_online_terminals=40,
    n_days=21,
    seed=3,
    compromised_terminals_per_day=1,
    takeovers_per_day=2,
    card_tests_per_day=1,
    clones_per_day=1,
)


@pytest.fixture(scope="session")
def small_sim() -> SimData:
    return simulate(SMALL)


SMALL_TRAIN = TrainConfig(
    train_end_day=11,
    gap_days=2,
    val_end_day=16,
    test_end_day=21,
    num_boost_round=300,
    early_stopping_rounds=30,
)


@pytest.fixture(scope="session")
def small_features(small_sim: SimData) -> pd.DataFrame:
    return build_features(small_sim.transactions, small_sim.customers, small_sim.terminals)


@pytest.fixture(scope="session")
def trained(small_sim: SimData, small_features: pd.DataFrame) -> TrainResult:
    return train(small_features, small_sim.transactions, SMALL_TRAIN, version="t1")
