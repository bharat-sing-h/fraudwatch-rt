"""Offline features = replaying history through the online engine (point-in-time correct)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraudwatch.features.engine import FEATURES, FeatureEngine, FraudLabel, Transaction
from fraudwatch.features.store import InMemoryStore, StateStore


def load_profiles(engine: FeatureEngine, customers: pd.DataFrame, terminals: pd.DataFrame) -> None:
    for c in customers.itertuples(index=False):
        engine.put_customer_profile(int(c.customer_id), float(c.home_x), float(c.home_y))
    for t in terminals.itertuples(index=False):
        engine.put_terminal_profile(
            int(t.terminal_id),
            float(t.x) if not math.isnan(t.x) else math.nan,
            float(t.y) if not math.isnan(t.y) else math.nan,
            str(t.category),
            bool(t.is_online),
        )


def build_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    terminals: pd.DataFrame,
    *,
    store: StateStore | None = None,
) -> pd.DataFrame:
    """Return one feature row per transaction, as the online system would have computed it.

    Transactions and label events are merged into one timeline. A label at time T is applied
    before any transaction at or after T, so features only ever see fraud that had already
    been reported.
    """
    engine = FeatureEngine(store or InMemoryStore())
    load_profiles(engine, customers, terminals)

    tx = transactions.sort_values(["ts", "tx_id"], kind="stable")
    fraud = tx[tx.is_fraud == 1] if "is_fraud" in tx else tx.iloc[:0]
    labels = fraud.sort_values("label_ts", kind="stable") if len(fraud) else fraud

    lab_ts = labels.label_ts.to_numpy() if len(labels) else np.array([], dtype=np.int64)
    lab_rows = list(
        labels[["tx_id", "ts", "customer_id", "terminal_id"]].itertuples(index=False, name=None)
    )
    rows: np.ndarray = np.empty((len(tx), len(FEATURES)), dtype=np.float64)
    li = 0
    cols = tx[["tx_id", "ts", "customer_id", "terminal_id", "amount"]].itertuples(
        index=False, name=None
    )
    for i, (tx_id, ts, cust, term, amount) in enumerate(cols):
        while li < len(lab_ts) and lab_ts[li] <= ts:
            l_id, l_ts, l_c, l_t = lab_rows[li]
            engine.process_label(FraudLabel(int(l_id), int(l_ts), int(l_c), int(l_t)))
            li += 1
        f = engine.process_transaction(
            Transaction(int(tx_id), int(ts), int(cust), int(term), float(amount))
        )
        rows[i] = [f[name] for name in FEATURES]

    out = pd.DataFrame(rows, columns=FEATURES, index=tx.index)
    out.insert(0, "tx_id", tx.tx_id.to_numpy())
    return out.sort_index()
