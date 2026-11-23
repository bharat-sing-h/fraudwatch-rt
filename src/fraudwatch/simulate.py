"""Card-transaction simulator with realistic fraud patterns and delayed labels.

Public fraud datasets are either anonymised PCA components (no feature engineering possible) or
lack the entity/time structure a real-time system needs. This simulator, in the spirit of the
"Reproducible Machine Learning for Credit Card Fraud Detection" handbook (Le Borgne et al.),
produces customers, terminals and timestamped transactions, then injects fraud scenarios that
each leave a *different* footprint:

==================  =======================================  ==================================
scenario            what happens                              signal that can catch it
==================  =======================================  ==================================
compromised_term    a skimmed terminal: ~half its next 28     terminal fraud rate, from labels
                    days of transactions are fraudulent       that arrive *later* (delayed)
account_takeover    a stolen account spends 3-6x its usual    amount vs. the customer's history
                    amounts for two weeks
card_testing        burst of tiny online charges, then one    short-window velocity
                    big purchase
impossible_travel   card used far from home shortly after a   distance / implied speed
                    genuine purchase
night_online        (drift) online fraud at night with        none in pre-drift training data
                    normal-looking amounts, from day N         -> tests drift detection + retrain
==================  =======================================  ==================================

Labels arrive with a delay (chargebacks take days), so a production model can only learn from
fraud that has *already been reported*. Every downstream component respects that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DAY = 86_400
EPOCH = 1_735_689_600  # 2025-01-01T00:00:00Z
CATEGORIES = ["grocery", "restaurant", "fuel", "retail", "electronics", "travel", "online"]


@dataclass
class SimConfig:
    n_customers: int = 3_000
    n_terminals: int = 6_000
    n_online_terminals: int = 300
    n_days: int = 90
    seed: int = 7
    terminal_radius_km: float = 6.0
    # fraud intensity (new incidents per day)
    compromised_terminals_per_day: int = 2
    takeovers_per_day: int = 3
    card_tests_per_day: int = 2
    clones_per_day: int = 2
    # concept drift: a new pattern appears from this day on (None = never)
    drift_day: int | None = None
    night_online_per_day: int = 6
    label_delay_mean_days: float = 4.0


@dataclass
class SimData:
    customers: pd.DataFrame
    terminals: pd.DataFrame
    transactions: pd.DataFrame
    config: SimConfig = field(default_factory=SimConfig)

    @property
    def labels(self) -> pd.DataFrame:
        """Label events as a fraud team receives them: only frauds, at ``label_ts``."""
        frauds = self.transactions[self.transactions.is_fraud == 1]
        return frauds[["tx_id", "label_ts", "customer_id", "terminal_id", "ts"]].sort_values(
            "label_ts", kind="stable"
        )


def _customers(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_customers
    mean_amount = rng.uniform(5, 100, n)
    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "home_x": rng.uniform(0, 100, n),
            "home_y": rng.uniform(0, 100, n),
            "mean_amount": mean_amount,
            "std_amount": mean_amount / 2,
            "tx_per_day": rng.uniform(0.2, 4.0, n),
        }
    )


def _terminals(cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    n_phys, n_online = cfg.n_terminals, cfg.n_online_terminals
    phys_cats = rng.choice(CATEGORIES[:-1], n_phys, p=[0.3, 0.25, 0.15, 0.15, 0.1, 0.05])
    return pd.DataFrame(
        {
            "terminal_id": np.arange(n_phys + n_online),
            "x": np.concatenate([rng.uniform(0, 100, n_phys), np.full(n_online, np.nan)]),
            "y": np.concatenate([rng.uniform(0, 100, n_phys), np.full(n_online, np.nan)]),
            "category": np.concatenate([phys_cats, np.full(n_online, "online")]),
            "is_online": np.concatenate([np.zeros(n_phys, int), np.ones(n_online, int)]),
        }
    )


def _legit(
    cfg: SimConfig, rng: np.random.Generator, cust: pd.DataFrame, term: pd.DataFrame
) -> pd.DataFrame:
    phys = term[term.is_online == 0]
    online_ids = term.terminal_id[term.is_online == 1].to_numpy()
    px, py, pid = phys.x.to_numpy(), phys.y.to_numpy(), phys.terminal_id.to_numpy()
    rows: dict[str, list[np.ndarray]] = {
        k: [] for k in ("ts", "customer_id", "terminal_id", "amount")
    }
    for c in cust.itertuples(index=False):
        near = pid[(px - c.home_x) ** 2 + (py - c.home_y) ** 2 <= cfg.terminal_radius_km**2]
        counts = rng.poisson(c.tx_per_day, cfg.n_days)
        n = int(counts.sum())
        if n == 0:
            continue
        days: np.ndarray = np.repeat(np.arange(cfg.n_days), counts)
        # spending clusters around early afternoon; night purchases are rare for genuine users
        secs = np.clip(rng.normal(DAY * 0.55, DAY * 0.16, n), 0, DAY - 1).astype(np.int64)
        amount = rng.normal(c.mean_amount, c.std_amount, n)
        bad = amount <= 0
        amount[bad] = rng.uniform(0, 2 * c.mean_amount, bad.sum())
        use_online = (rng.random(n) < 0.2) | (len(near) == 0)
        terminals = np.where(
            use_online,
            rng.choice(online_ids, n),
            rng.choice(near, n) if len(near) else rng.choice(online_ids, n),
        )
        rows["ts"].append(EPOCH + days * DAY + secs)
        rows["customer_id"].append(np.full(n, c.customer_id))
        rows["terminal_id"].append(terminals)
        rows["amount"].append(np.round(amount, 2))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in rows.items()})
    df["is_fraud"] = 0
    df["scenario"] = "legit"
    return df


def _inject(
    cfg: SimConfig,
    rng: np.random.Generator,
    tx: pd.DataFrame,
    cust: pd.DataFrame,
    term: pd.DataFrame,
) -> pd.DataFrame:
    day_of = ((tx.ts.to_numpy() - EPOCH) // DAY).astype(int)
    phys_ids = term.terminal_id[term.is_online == 0].to_numpy()
    online_ids = term.terminal_id[term.is_online == 1].to_numpy()
    big_ticket = term.terminal_id[term.category.isin(["electronics", "online"])].to_numpy()
    extra: list[dict[str, object]] = []

    for day in range(cfg.n_days):
        window = (day_of >= day) & (day_of < day + 28)
        # 1) compromised terminals: half of the next 28 days' transactions are fraud
        for t in rng.choice(phys_ids, cfg.compromised_terminals_per_day, replace=False):
            idx = np.flatnonzero(window & (tx.terminal_id.to_numpy() == t))
            hit = idx[rng.random(len(idx)) < 0.5]
            tx.loc[hit, ["is_fraud", "scenario"]] = [1, "compromised_terminal"]

        # 2) account takeover: a third of the next 14 days' purchases inflated 3-6x
        two_weeks = (day_of >= day) & (day_of < day + 14)
        for c in rng.choice(cust.customer_id.to_numpy(), cfg.takeovers_per_day, replace=False):
            idx = np.flatnonzero(two_weeks & (tx.customer_id.to_numpy() == c))
            hit = idx[rng.random(len(idx)) < 1 / 3]
            tx.loc[hit, "amount"] = np.round(tx.loc[hit, "amount"] * rng.uniform(3, 6, len(hit)), 2)
            tx.loc[hit, ["is_fraud", "scenario"]] = [1, "account_takeover"]

        # 3) card testing: tiny online charges within minutes, then one big purchase
        for c in rng.choice(cust.customer_id.to_numpy(), cfg.card_tests_per_day, replace=False):
            start = EPOCH + day * DAY + int(rng.integers(0, DAY - 7200))
            merchant = rng.choice(online_ids)
            for k in range(int(rng.integers(3, 9))):
                extra.append(
                    {
                        "ts": start + k * int(rng.integers(20, 90)),
                        "customer_id": c,
                        "terminal_id": merchant,
                        "amount": round(rng.uniform(0.5, 3), 2),
                        "scenario": "card_testing",
                    }
                )
            extra.append(
                {
                    "ts": start + int(rng.integers(900, 3600)),
                    "customer_id": c,
                    "terminal_id": rng.choice(big_ticket),
                    "amount": round(rng.uniform(200, 900), 2),
                    "scenario": "card_testing",
                }
            )

        # 4) cloned card: used far from home shortly after a genuine purchase
        today = np.flatnonzero((day_of == day) & (tx.is_fraud.to_numpy() == 0))
        if len(today):
            for i in rng.choice(today, min(cfg.clones_per_day, len(today)), replace=False):
                c = int(tx.customer_id.iat[i])
                hx, hy = cust.home_x.iat[c], cust.home_y.iat[c]
                far = phys_ids[
                    np.hypot(term.x.to_numpy()[phys_ids] - hx, term.y.to_numpy()[phys_ids] - hy)
                    > 45
                ]
                for k in range(int(rng.integers(1, 4))):
                    extra.append(
                        {
                            "ts": int(tx.ts.iat[i]) + int(rng.integers(300, 7200)) + k,
                            "customer_id": c,
                            "terminal_id": rng.choice(far),
                            "amount": round(rng.uniform(40, 400), 2),
                            "scenario": "impossible_travel",
                        }
                    )

        # 5) drift: night-time online fraud with ordinary amounts
        if cfg.drift_day is not None and day >= cfg.drift_day:
            for c in rng.choice(
                cust.customer_id.to_numpy(), cfg.night_online_per_day, replace=False
            ):
                mean, std = cust.mean_amount.iat[c], cust.std_amount.iat[c]
                for _ in range(int(rng.integers(2, 5))):
                    extra.append(
                        {
                            "ts": EPOCH + day * DAY + int(rng.integers(0, 5 * 3600)),
                            "customer_id": c,
                            "terminal_id": rng.choice(online_ids),
                            "amount": round(max(1.0, rng.normal(mean, std)), 2),
                            "scenario": "night_online",
                        }
                    )

    if extra:
        injected = pd.DataFrame(extra)
        injected["is_fraud"] = 1
        tx = pd.concat([tx, injected], ignore_index=True)
    return tx


def simulate(cfg: SimConfig | None = None) -> SimData:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    customers = _customers(cfg, rng)
    terminals = _terminals(cfg, rng)
    tx = _legit(cfg, rng, customers, terminals)
    tx = _inject(cfg, rng, tx, customers, terminals)
    tx = tx[tx.ts < EPOCH + cfg.n_days * DAY].sort_values(["ts", "customer_id"], kind="stable")
    tx = tx.reset_index(drop=True)
    tx.insert(0, "tx_id", np.arange(len(tx)))
    delay = (rng.exponential(cfg.label_delay_mean_days, len(tx)) * DAY).astype(np.int64) + 3600
    tx["label_ts"] = np.where(tx.is_fraud == 1, tx.ts + delay, -1)
    tx["customer_id"] = tx.customer_id.astype(np.int64)
    tx["terminal_id"] = tx.terminal_id.astype(np.int64)
    return SimData(customers, terminals, tx, cfg)


def day_index(ts: np.ndarray | pd.Series) -> np.ndarray:
    out: np.ndarray = ((np.asarray(ts) - EPOCH) // DAY).astype(int)
    return out
