"""Streaming feature engine - the single source of truth for features.

``FeatureEngine.process_transaction`` computes a transaction's features from state *before* the
transaction, then folds the transaction into state. ``process_label`` folds in a fraud report
when it arrives. Serving calls these per event; training (:mod:`fraudwatch.features.batch`)
*replays history through the same methods* in time order.

Consequences:

* **No train/serve skew**: there is no second implementation of any feature to drift apart.
* **Point-in-time correctness by construction**: when a historical transaction is replayed, the
  state contains only what was known at that moment. In particular the terminal-risk features
  count only fraud reports whose *label* time is earlier, never labels from the future, which is
  the classic leak in fraud models built with "terminal fraud rate" computed in one batch query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fraudwatch.features.store import StateStore

MIN = 60
HOUR = 3_600
DAY = 86_400
CATEGORIES = ["grocery", "restaurant", "fuel", "retail", "electronics", "travel", "online"]

FEATURES: list[str] = [
    "amount",
    "log_amount",
    "hour",
    "is_night",
    "is_weekend",
    "is_online",
    "category",
    "c_cnt_10m",
    "c_cnt_1h",
    "c_cnt_24h",
    "c_cnt_7d",
    "c_small_cnt_1h",
    "c_sum_24h",
    "c_avg_7d",
    # NB: no raw "number of past transactions" feature. It grows with the calendar, so it has a
    # different distribution at serving time than in any training window (PSI 0.3 -> 0.9 within
    # weeks in the drift experiment) and teaches the model about *time*, not fraud.
    "amount_ratio_7d",
    "amount_z",
    "secs_since_last",
    "dist_home",
    "dist_last",
    "speed_kmh",
    "t_cnt_7d",
    "t_cnt_28d",
    "t_fraud_28d",
    "t_fraud_rate_28d",
    "c_fraud_known_14d",
]


@dataclass(frozen=True)
class Transaction:
    tx_id: int
    ts: int
    customer_id: int
    terminal_id: int
    amount: float


@dataclass(frozen=True)
class FraudLabel:
    tx_id: int
    tx_ts: int
    customer_id: int
    terminal_id: int


class FeatureEngine:
    def __init__(self, store: StateStore) -> None:
        self.store = store

    # --- static profiles (loaded once from the customer / merchant registry) ---------------
    def put_customer_profile(self, customer_id: int, home_x: float, home_y: float) -> None:
        self.store.hset(f"cp:{customer_id}", {"home_x": home_x, "home_y": home_y})

    def put_terminal_profile(
        self, terminal_id: int, x: float, y: float, category: str, is_online: bool
    ) -> None:
        self.store.hset(
            f"tp:{terminal_id}",
            {
                "x": x,
                "y": y,
                "category": float(CATEGORIES.index(category)),
                "is_online": float(is_online),
            },
        )

    # --- events ------------------------------------------------------------------------------
    def process_label(self, label: FraudLabel) -> None:
        member = str(label.tx_id)
        self.store.window_add(f"t:{label.terminal_id}:fraud", label.tx_ts, member, 1.0)
        self.store.window_add(f"c:{label.customer_id}:fraud", label.tx_ts, member, 1.0)

    def process_transaction(self, tx: Transaction, *, update: bool = True) -> dict[str, float]:
        """Features from prior state, then (once per ``tx_id``) fold ``tx`` into state.

        Kafka gives at-least-once delivery; a redelivered transaction must not be counted twice
        in velocity windows or the amount profile, so state updates are deduplicated by id.
        """
        features = self.compute(tx)
        if update and self.store.mark_seen(f"seen:{tx.tx_id}", 2 * DAY):
            self._update(tx)
        return features

    def compute(self, tx: Transaction) -> dict[str, float]:
        s, nan = self.store, math.nan
        ts, amount = tx.ts, tx.amount
        term = s.hget(f"tp:{tx.terminal_id}")
        is_online = bool(term.get("is_online", 1.0))

        f: dict[str, float] = {
            "amount": amount,
            "log_amount": math.log1p(max(amount, 0.0)),
            "hour": float((ts % DAY) // HOUR),
            "is_night": float((ts % DAY) < 5 * HOUR),
            # 2025-01-01 was a Wednesday; epoch day 0 (1970-01-01) a Thursday
            "is_weekend": float(((ts // DAY) + 3) % 7 >= 5),
            "is_online": float(is_online),
            "category": term.get("category", float(CATEGORIES.index("online"))),
        }

        # customer velocity / spend windows (one read, many features)
        week = s.window_get(f"c:{tx.customer_id}:tx", ts - 7 * DAY)
        f["c_cnt_10m"] = float(sum(1 for t, _ in week if t >= ts - 10 * MIN))
        f["c_cnt_1h"] = float(sum(1 for t, _ in week if t >= ts - HOUR))
        f["c_small_cnt_1h"] = float(sum(1 for t, a in week if t >= ts - HOUR and a < 5))
        day = [a for t, a in week if t >= ts - DAY]
        f["c_cnt_24h"] = float(len(day))
        f["c_sum_24h"] = float(sum(day))
        f["c_cnt_7d"] = float(len(week))
        avg7 = sum(a for _, a in week) / len(week) if week else nan
        f["c_avg_7d"] = avg7
        f["amount_ratio_7d"] = amount / avg7 if week and avg7 > 0 else nan

        # long-run amount profile (Welford running mean/variance)
        st = s.hget(f"c:{tx.customer_id}:stats")
        n = st.get("n", 0.0)
        if n >= 5:
            std = math.sqrt(st["m2"] / (n - 1))
            f["amount_z"] = (amount - st["mean"]) / std if std > 0 else 0.0
        else:
            f["amount_z"] = nan
        f["secs_since_last"] = float(ts - st["last_ts"]) if "last_ts" in st else nan

        # geography (physical terminals only)
        f["dist_home"] = f["dist_last"] = f["speed_kmh"] = nan
        if not is_online and "x" in term:
            prof = s.hget(f"cp:{tx.customer_id}")
            if prof:
                f["dist_home"] = math.hypot(term["x"] - prof["home_x"], term["y"] - prof["home_y"])
            if "last_x" in st:
                d = math.hypot(term["x"] - st["last_x"], term["y"] - st["last_y"])
                f["dist_last"] = d
                hours = max((ts - st["last_loc_ts"]) / HOUR, 1 / 60)
                f["speed_kmh"] = d / hours

        # terminal risk, from fraud reports already received (delayed labels)
        t_cnt_28 = s.window_count(f"t:{tx.terminal_id}:tx", ts - 28 * DAY)
        t_fraud_28 = s.window_count(f"t:{tx.terminal_id}:fraud", ts - 28 * DAY)
        f["t_cnt_7d"] = float(s.window_count(f"t:{tx.terminal_id}:tx", ts - 7 * DAY))
        f["t_cnt_28d"] = float(t_cnt_28)
        f["t_fraud_28d"] = float(t_fraud_28)
        f["t_fraud_rate_28d"] = t_fraud_28 / t_cnt_28 if t_cnt_28 else nan
        f["c_fraud_known_14d"] = float(s.window_count(f"c:{tx.customer_id}:fraud", ts - 14 * DAY))
        return f

    def _update(self, tx: Transaction) -> None:
        s = self.store
        member = str(tx.tx_id)
        ckey, tkey = f"c:{tx.customer_id}:tx", f"t:{tx.terminal_id}:tx"
        s.window_add(ckey, tx.ts, member, tx.amount)
        s.window_add(tkey, tx.ts, member, 1.0)
        s.window_trim(ckey, tx.ts - 7 * DAY)
        s.window_trim(tkey, tx.ts - 28 * DAY)

        st = s.hget(f"c:{tx.customer_id}:stats")
        n = st.get("n", 0.0) + 1
        mean = st.get("mean", 0.0)
        delta = tx.amount - mean
        mean += delta / n
        update = {
            "n": n,
            "mean": mean,
            "m2": st.get("m2", 0.0) + delta * (tx.amount - mean),
            "last_ts": float(tx.ts),
        }
        term = s.hget(f"tp:{tx.terminal_id}")
        if term and not term.get("is_online", 1.0):
            update |= {"last_x": term["x"], "last_y": term["y"], "last_loc_ts": float(tx.ts)}
        s.hset(f"c:{tx.customer_id}:stats", update)
