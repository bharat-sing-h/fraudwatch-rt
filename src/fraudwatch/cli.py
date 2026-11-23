"""``fw`` command-line interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from fraudwatch import __version__

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Real-time fraud detection: simulate, train, serve, monitor, retrain.",
)
console = Console()

DataDir = Annotated[Path, typer.Option("--data", help="Directory with simulated data.")]
MlflowUri = Annotated[
    str, typer.Option("--mlflow", envvar="FW_MLFLOW_URI", help="MLflow tracking/registry URI.")
]


def _load_sim(data: Path):  # type: ignore[no-untyped-def]
    import pandas as pd

    from fraudwatch.simulate import SimData

    return SimData(
        pd.read_parquet(data / "customers.parquet"),
        pd.read_parquet(data / "terminals.parquet"),
        pd.read_parquet(data / "transactions.parquet"),
    )


@app.callback(invoke_without_command=True)
def _main(version: Annotated[bool, typer.Option("--version")] = False) -> None:
    if version:
        console.print(f"fraudwatch {__version__}")
        raise typer.Exit()


@app.command()
def simulate(
    out: DataDir = Path("data"),
    days: int = 90,
    customers: int = 3000,
    drift_day: Annotated[int | None, typer.Option(help="Day a new fraud pattern starts.")] = None,
    seed: int = 7,
) -> None:
    """Generate customers, terminals and labelled transactions."""
    from fraudwatch.simulate import SimConfig
    from fraudwatch.simulate import simulate as run_sim

    sim = run_sim(SimConfig(n_days=days, n_customers=customers, drift_day=drift_day, seed=seed))
    out.mkdir(parents=True, exist_ok=True)
    sim.customers.to_parquet(out / "customers.parquet")
    sim.terminals.to_parquet(out / "terminals.parquet")
    sim.transactions.to_parquet(out / "transactions.parquet")
    tx = sim.transactions
    console.print(f"{len(tx):,} transactions, fraud rate {tx.is_fraud.mean():.2%} -> {out}")


@app.command()
def features(data: DataDir = Path("data")) -> None:
    """Point-in-time features for every transaction (replay through the online engine)."""
    from fraudwatch.features.batch import build_features

    sim = _load_sim(data)
    feats = build_features(sim.transactions, sim.customers, sim.terminals)
    feats.to_parquet(data / "features.parquet")
    console.print(f"{len(feats):,} feature rows -> {data / 'features.parquet'}")


@app.command()
def train(
    data: DataDir = Path("data"),
    mlflow_uri: MlflowUri = "sqlite:///mlflow.db",
    alias: Annotated[str | None, typer.Option(help="champion | challenger")] = None,
    train_end: int = 49,
    val_end: int = 70,
    test_end: int = 90,
    version: Annotated[str | None, typer.Option()] = None,
    out: Annotated[Path | None, typer.Option(help="Also save the bundle directory here.")] = None,
) -> None:
    """Train LightGBM (+ baselines), log to MLflow, register, optionally set an alias."""
    import pandas as pd

    from fraudwatch.model.registry import Registry
    from fraudwatch.model.train import TrainConfig, model_card
    from fraudwatch.model.train import train as run_train

    sim = _load_sim(data)
    feats = pd.read_parquet(data / "features.parquet")
    cfg = TrainConfig(train_end_day=train_end, val_end_day=val_end, test_end_day=test_end)
    result = run_train(feats, sim.transactions, cfg, version=version)
    card = model_card(result)
    console.print(card)
    reg = Registry(mlflow_uri)
    registered = reg.log_and_register(
        result.bundle,
        params=cfg.model_dump(exclude={"lgb_params"}),
        metrics={f"test_{k}": v for k, v in result.metrics["lightgbm"].items()},
        model_card=card,
        alias=alias,
    )
    if out:
        result.bundle.save(out)
        (out / "model_card.md").write_text(card)
    console.print(f"registered fraudwatch v{registered}" + (f" as {alias}" if alias else ""))


@app.command()
def promote(version: str, mlflow_uri: MlflowUri = "sqlite:///mlflow.db") -> None:
    """Make VERSION the champion (previous champion becomes 'previous')."""
    from fraudwatch.model.registry import Registry

    previous = Registry(mlflow_uri).promote(version)
    console.print(f"champion: v{version} (was v{previous})")


@app.command()
def rollback(mlflow_uri: MlflowUri = "sqlite:///mlflow.db") -> None:
    """Point 'champion' back at the 'previous' version."""
    from fraudwatch.model.registry import Registry

    reg = Registry(mlflow_uri)
    previous = reg.alias_version("previous")
    if previous is None:
        console.print("[red]no 'previous' version to roll back to[/red]")
        raise typer.Exit(1)
    reg.promote(previous)
    console.print(f"rolled back: champion is v{previous}")


@app.command()
def monitor(
    data: DataDir = Path("data"),
    mlflow_uri: MlflowUri = "sqlite:///mlflow.db",
    start_day: int = 70,
    end_day: int = 90,
    alias: str = "champion",
) -> None:
    """Drift + performance report for the model behind ALIAS on days [start, end)."""
    import pandas as pd

    from fraudwatch.model.registry import Registry
    from fraudwatch.monitor.drift import drift_report
    from fraudwatch.simulate import day_index

    sim = _load_sim(data)
    feats = pd.read_parquet(data / "features.parquet")
    bundle = Registry(mlflow_uri).load(alias)
    days = day_index(sim.transactions.ts)
    m = (days >= start_day) & (days < end_day)
    rep = drift_report(bundle, feats[m], labels=sim.transactions.is_fraud.to_numpy()[m])
    console.print(rep.to_markdown())
    raise typer.Exit(1 if rep.retrain_recommended else 0)


@app.command()
def experiment(
    out: Annotated[Path, typer.Option()] = Path("reports"),
    mlflow_uri: MlflowUri = "sqlite:///mlflow-experiment.db",
    customers: int = 3000,
) -> None:
    """Full lifecycle: drift appears -> detected -> retrain -> shadow -> promotion gate."""
    from fraudwatch.experiment import ExperimentConfig
    from fraudwatch.experiment import run as run_experiment

    console.print(run_experiment(out, mlflow_uri, ExperimentConfig(n_customers=customers)))


@app.command()
def benchmark(
    data: DataDir = Path("data"),
    model_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)] = Path(
        "models/champion"
    ),
    redis_url: Annotated[str | None, typer.Option(envvar="FW_REDIS_URL")] = None,
    n: int = 5000,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure scoring latency (in-process and HTTP; memory and optionally Redis store)."""
    from fraudwatch import benchmark as bench
    from fraudwatch.model.bundle import ModelBundle

    md = bench.to_markdown(
        bench.run(_load_sim(data), ModelBundle.load(model_dir), redis_url=redis_url, n=n)
    )
    console.print(md)
    if out:
        out.write_text(md)


@app.command("load-profiles")
def load_profiles(
    data: DataDir = Path("data"),
    redis_url: Annotated[str, typer.Option(envvar="FW_REDIS_URL")] = "redis://localhost:6379/0",
) -> None:
    """Load customer and terminal profiles into the Redis online store."""
    import redis

    from fraudwatch.features.batch import load_profiles as load
    from fraudwatch.features.engine import FeatureEngine
    from fraudwatch.features.store import RedisStore

    sim = _load_sim(data)
    load(FeatureEngine(RedisStore(redis.Redis.from_url(redis_url))), sim.customers, sim.terminals)
    console.print(f"loaded {len(sim.customers):,} customers, {len(sim.terminals):,} terminals")


@app.command()
def replay(
    data: DataDir = Path("data"),
    bootstrap: Annotated[str, typer.Option(envvar="FW_KAFKA")] = "localhost:9092",
    start_day: int = 70,
    speedup: Annotated[float, typer.Option(help="0 = as fast as possible")] = 0.0,
    limit: int | None = None,
) -> None:  # pragma: no cover - needs a broker
    """Publish transactions (and their fraud reports, at report time) to Kafka."""
    import time

    from confluent_kafka import Producer

    from fraudwatch.simulate import day_index

    sim = _load_sim(data)
    tx = sim.transactions[day_index(sim.transactions.ts) >= start_day]
    if limit:
        tx = tx.head(limit)
    events = [
        (
            int(r.ts),
            "transactions",
            int(r.customer_id),
            {
                "tx_id": int(r.tx_id),
                "ts": int(r.ts),
                "customer_id": int(r.customer_id),
                "terminal_id": int(r.terminal_id),
                "amount": float(r.amount),
            },
        )
        for r in tx.itertuples()
    ]
    events += [
        (
            int(r.label_ts),
            "labels",
            int(r.customer_id),
            {
                "tx_id": int(r.tx_id),
                "tx_ts": int(r.ts),
                "customer_id": int(r.customer_id),
                "terminal_id": int(r.terminal_id),
            },
        )
        for r in tx[tx.is_fraud == 1].itertuples()
    ]
    events.sort(key=lambda e: e[0])
    producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 10})
    t0_sim, t0_wall = events[0][0], time.time()
    for ts, topic, key, value in events:
        if speedup:
            time.sleep(max(0.0, (ts - t0_sim) / speedup - (time.time() - t0_wall)))
        producer.produce(topic, key=str(key), value=json.dumps(value).encode())
        producer.poll(0)
    producer.flush()
    console.print(f"published {len(events):,} events")


@app.command()
def stream(
    bootstrap: Annotated[str, typer.Option(envvar="FW_KAFKA")] = "localhost:9092",
) -> None:  # pragma: no cover - needs a broker
    """Run the Kafka scoring worker (models + store configured like the API)."""
    from fraudwatch.features.engine import FeatureEngine
    from fraudwatch.serve.app import load_models, make_store
    from fraudwatch.serve.scorer import Scorer
    from fraudwatch.stream.worker import KafkaBus, run

    champion, challenger = load_models()
    run(KafkaBus(bootstrap), Scorer(FeatureEngine(make_store()), champion, challenger))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the scoring API."""
    import uvicorn

    uvicorn.run("fraudwatch.serve.app:app", host=host, port=port)
