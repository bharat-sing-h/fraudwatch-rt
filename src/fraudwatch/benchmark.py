"""Latency benchmark for the scoring path.

Warms the online store with several weeks of history (so velocity windows and profiles look
like production), then measures per-transaction latency for:

* ``in-process``: features + model, as the Kafka worker runs it;
* ``http``: the full FastAPI round trip over a keep-alive connection.

Both run against the chosen store: in-memory, or a real Redis server.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudwatch.features.batch import load_profiles
from fraudwatch.features.engine import FeatureEngine, FraudLabel, Transaction
from fraudwatch.features.store import InMemoryStore, RedisStore, StateStore
from fraudwatch.model.bundle import ModelBundle
from fraudwatch.serve.scorer import Scorer
from fraudwatch.simulate import SimData, day_index


@dataclass
class LatencyStats:
    name: str
    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    throughput_per_s: float

    def row(self) -> str:
        return (
            f"| {self.name} | {self.n:,} | {self.p50_ms:.2f} | {self.p95_ms:.2f} | "
            f"{self.p99_ms:.2f} | {self.max_ms:.1f} | {self.throughput_per_s:,.0f} |"
        )


def _stats(name: str, samples_ms: list[float], wall_s: float) -> LatencyStats:
    a = np.asarray(samples_ms)
    q: np.ndarray = np.percentile(a, [50, 95, 99])
    return LatencyStats(
        name, len(a), float(q[0]), float(q[1]), float(q[2]), float(a.max()), len(a) / wall_s
    )


def _tx(r: tuple[int, int, int, int, float]) -> Transaction:
    return Transaction(int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4]))


def warm_scorer(
    sim: SimData, bundle: ModelBundle, store: StateStore, warm_days: int
) -> tuple[Scorer, pd.DataFrame]:
    engine = FeatureEngine(store)
    load_profiles(engine, sim.customers, sim.terminals)
    scorer = Scorer(engine, bundle)
    tx = sim.transactions
    days = day_index(tx.ts)
    warm = tx[days < warm_days]
    for row in warm[["tx_id", "ts", "customer_id", "terminal_id", "amount"]].itertuples(
        index=False, name=None
    ):
        engine.process_transaction(_tx(row))
    for lab in warm[warm.is_fraud == 1].itertuples(index=False):
        engine.process_label(
            FraudLabel(int(lab.tx_id), int(lab.ts), int(lab.customer_id), int(lab.terminal_id))
        )
    return scorer, tx[days >= warm_days]


def bench_inprocess(scorer: Scorer, tx: pd.DataFrame, n: int) -> LatencyStats:
    rows = list(
        tx[["tx_id", "ts", "customer_id", "terminal_id", "amount"]]
        .head(n)
        .itertuples(index=False, name=None)
    )
    samples = []
    start = time.perf_counter()
    for r in rows:
        t = time.perf_counter()
        scorer.score(_tx(r))
        samples.append((time.perf_counter() - t) * 1000)
    return _stats("in-process (features + model)", samples, time.perf_counter() - start)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def bench_http(scorer: Scorer, tx: pd.DataFrame, n: int) -> LatencyStats:
    import httpx
    import uvicorn

    from fraudwatch.serve.app import create_app

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(scorer), host="127.0.0.1", port=port, log_level="warning", access_log=False
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    rows = tx[["tx_id", "ts", "customer_id", "terminal_id", "amount"]].head(n)
    payloads = [
        {
            "tx_id": int(r.tx_id),
            "ts": int(r.ts),
            "customer_id": int(r.customer_id),
            "terminal_id": int(r.terminal_id),
            "amount": float(r.amount),
        }
        for r in rows.itertuples(index=False)
    ]
    samples = []
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as client:
            for p in payloads[:200]:  # warm-up: connection, JIT-free but caches/pools
                client.post("/v1/score", json={**p, "tx_id": -p["tx_id"] - 1})
            start = time.perf_counter()
            for p in payloads:
                t = time.perf_counter()
                client.post("/v1/score", json=p).raise_for_status()
                samples.append((time.perf_counter() - t) * 1000)
            wall = time.perf_counter() - start
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    return _stats("HTTP round trip (FastAPI)", samples, wall)


def run(
    sim: SimData,
    bundle: ModelBundle,
    *,
    redis_url: str | None = None,
    warm_days: int = 28,
    n: int = 5_000,
) -> list[LatencyStats]:
    results = []
    stores: list[tuple[str, StateStore]] = [("memory", InMemoryStore())]
    if redis_url:
        import redis

        client = redis.Redis.from_url(redis_url)
        client.flushdb()
        stores.append(("redis", RedisStore(client)))
    for label, store in stores:
        scorer, rest = warm_scorer(sim, bundle, store, warm_days)
        a = bench_inprocess(scorer, rest, n)
        b = bench_http(scorer, rest.iloc[n:], n)
        for s in (a, b):
            s.name = f"{s.name}, {label} store"
            results.append(s)
    return results


def to_markdown(results: list[LatencyStats]) -> str:
    lines = [
        "| Path | Requests | p50 ms | p95 ms | p99 ms | max ms | req/s (1 client) |",
        "|---|---|---|---|---|---|---|",
    ]
    return "\n".join(lines + [r.row() for r in results]) + "\n"
