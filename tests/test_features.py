from __future__ import annotations

import math

import fakeredis
import numpy as np
import pandas as pd
import pytest

from fraudwatch.features.batch import build_features, load_profiles
from fraudwatch.features.engine import (
    DAY,
    FEATURES,
    HOUR,
    FeatureEngine,
    FraudLabel,
    Transaction,
)
from fraudwatch.features.store import InMemoryStore, RedisStore
from fraudwatch.simulate import EPOCH, SimData

T0 = EPOCH + 10 * DAY + 12 * HOUR  # a weekday at noon


@pytest.fixture(params=["memory", "redis"])
def engine(request: pytest.FixtureRequest) -> FeatureEngine:
    store = InMemoryStore() if request.param == "memory" else RedisStore(fakeredis.FakeRedis())
    eng = FeatureEngine(store)
    eng.put_customer_profile(1, home_x=10.0, home_y=10.0)
    eng.put_terminal_profile(100, x=12.0, y=10.0, category="grocery", is_online=False)
    eng.put_terminal_profile(101, x=90.0, y=10.0, category="electronics", is_online=False)
    eng.put_terminal_profile(200, x=math.nan, y=math.nan, category="online", is_online=True)
    return eng


def tx(i: int, ts: int, amount: float, terminal: int = 100) -> Transaction:
    return Transaction(tx_id=i, ts=ts, customer_id=1, terminal_id=terminal, amount=amount)


def test_first_transaction_has_no_history(engine: FeatureEngine) -> None:
    f = engine.process_transaction(tx(1, T0, 20.0))
    assert f["c_cnt_24h"] == 0
    assert math.isnan(f["c_avg_7d"])
    assert math.isnan(f["amount_z"])
    assert math.isnan(f["secs_since_last"])
    assert f["dist_home"] == pytest.approx(2.0)
    assert f["hour"] == 12
    assert f["is_night"] == 0
    assert set(f) == set(FEATURES)


def test_velocity_and_amount_profile(engine: FeatureEngine) -> None:
    for i, amount in enumerate([20, 22, 18, 21, 19, 20]):
        engine.process_transaction(tx(i, T0 - (6 - i) * DAY, float(amount)))
    # card testing burst: 3 small charges in 5 minutes online
    for k in range(3):
        engine.process_transaction(tx(10 + k, T0 + k * 100, 1.5, terminal=200))
    f = engine.process_transaction(tx(20, T0 + 400, 400.0, terminal=200))
    assert f["c_cnt_10m"] == 3
    assert f["c_small_cnt_1h"] == 3
    assert f["c_cnt_7d"] == 9  # six daily purchases + three test charges, all within 7 days
    assert f["amount_z"] > 3
    assert f["secs_since_last"] == 200  # last test charge at +200s, purchase at +400s
    assert math.isnan(f["dist_home"])  # online: no geography


def test_impossible_travel(engine: FeatureEngine) -> None:
    engine.process_transaction(tx(1, T0, 30.0, terminal=100))
    f = engine.process_transaction(tx(2, T0 + HOUR // 2, 30.0, terminal=101))
    assert f["dist_last"] == pytest.approx(78.0)
    assert f["speed_kmh"] == pytest.approx(156.0)  # 78 km in 30 minutes


def test_terminal_risk_only_counts_reported_fraud(engine: FeatureEngine) -> None:
    engine.process_transaction(tx(1, T0, 10.0))
    before = engine.process_transaction(tx(2, T0 + HOUR, 10.0), update=False)
    assert before["t_fraud_28d"] == 0  # tx 1 is fraud, but nobody has reported it yet
    engine.process_label(FraudLabel(tx_id=1, tx_ts=T0, customer_id=1, terminal_id=100))
    after = engine.process_transaction(tx(2, T0 + HOUR, 10.0))
    assert after["t_fraud_28d"] == 1
    assert after["t_fraud_rate_28d"] == pytest.approx(1.0)
    assert after["c_fraud_known_14d"] == 1


def test_windows_expire(engine: FeatureEngine) -> None:
    engine.process_transaction(tx(1, T0, 10.0))
    f = engine.process_transaction(tx(2, T0 + 8 * DAY, 10.0))
    assert f["c_cnt_7d"] == 0
    assert f["t_cnt_28d"] == 1  # terminal window is 28 days
    assert math.isnan(f["amount_z"])  # long-run profile has 1 purchase: too few for a z-score


def test_in_memory_store_out_of_order_and_trim() -> None:
    s = InMemoryStore()
    s.window_add("k", 30, "c", 3.0)
    s.window_add("k", 10, "a", 1.0)
    s.window_add("k", 20, "b", 2.0)
    assert s.window_get("k", 0) == [(10, 1.0), (20, 2.0), (30, 3.0)]
    assert s.window_count("k", 15) == 2
    s.window_trim("k", 20)
    assert s.window_get("k", 0) == [(20, 2.0), (30, 3.0)]
    assert s.window_get("missing", 0) == []
    assert s.window_count("missing", 0) == 0
    s.window_trim("missing", 5)
    assert s.hget("nope") == {}


def _online_replay(sim: SimData, store: RedisStore) -> pd.DataFrame:
    """Simulate production: events arrive one at a time, labels interleaved by arrival time."""
    engine = FeatureEngine(store)
    load_profiles(engine, sim.customers, sim.terminals)
    events: list[tuple[int, int, object]] = []
    for r in sim.transactions.itertuples(index=False):
        events.append(
            (
                int(r.ts),
                1,
                Transaction(
                    int(r.tx_id), int(r.ts), int(r.customer_id), int(r.terminal_id), float(r.amount)
                ),
            )
        )
        if r.is_fraud:
            events.append(
                (
                    int(r.label_ts),
                    0,
                    FraudLabel(int(r.tx_id), int(r.ts), int(r.customer_id), int(r.terminal_id)),
                )
            )
    rows = {}
    for _, _, ev in sorted(events, key=lambda e: (e[0], e[1])):
        if isinstance(ev, FraudLabel):
            engine.process_label(ev)
        else:
            assert isinstance(ev, Transaction)
            f = engine.process_transaction(ev)
            rows[ev.tx_id] = [f[c] for c in FEATURES]
    return pd.DataFrame.from_dict(rows, orient="index", columns=FEATURES).sort_index()


def test_online_redis_matches_offline_batch(small_sim: SimData) -> None:
    """The guarantee that matters most: training features == serving features."""
    sub = small_sim.transactions[small_sim.transactions.ts < EPOCH + 6 * DAY]
    sim = SimData(small_sim.customers, small_sim.terminals, sub)
    batch = build_features(sub, sim.customers, sim.terminals).set_index("tx_id").sort_index()
    online = _online_replay(sim, RedisStore(fakeredis.FakeRedis()))
    assert len(batch) == len(online) > 1000
    np.testing.assert_allclose(
        online.to_numpy(), batch[FEATURES].to_numpy(), rtol=1e-9, equal_nan=True
    )


def test_point_in_time_no_future_leakage(small_sim: SimData) -> None:
    """Features of past transactions must not change when the future is removed."""
    tx = small_sim.transactions
    cutoff = EPOCH + 12 * DAY
    full = build_features(tx, small_sim.customers, small_sim.terminals).set_index("tx_id")
    past = tx[tx.ts < cutoff].copy()
    # in the truncated world, reports that arrive after the cutoff don't exist yet
    late = past.label_ts >= cutoff
    past.loc[late, "is_fraud"] = 0
    trunc = build_features(past, small_sim.customers, small_sim.terminals).set_index("tx_id")
    np.testing.assert_allclose(
        trunc[FEATURES].to_numpy(), full.loc[trunc.index, FEATURES].to_numpy(), equal_nan=True
    )
