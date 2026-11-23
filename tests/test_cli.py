"""End-to-end CLI workflow on a small simulation (the same steps as the README)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fraudwatch.cli import app

runner = CliRunner(env={"COLUMNS": "200"})


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fw")
    data = root / "data"
    res = runner.invoke(
        app, ["simulate", "--data", str(data), "--days", "30", "--customers", "500"]
    )
    assert res.exit_code == 0, res.output
    res = runner.invoke(app, ["features", "--data", str(data)])
    assert res.exit_code == 0, res.output
    return root


def test_train_register_promote_rollback_monitor(workspace: Path) -> None:
    data, uri = workspace / "data", f"sqlite:///{workspace / 'mlflow.db'}"
    common = [
        "--data",
        str(data),
        "--mlflow",
        uri,
        "--train-end",
        "14",
        "--val-end",
        "22",
        "--test-end",
        "30",
    ]
    res = runner.invoke(
        app,
        [
            "train",
            *common,
            "--alias",
            "champion",
            "--version",
            "a",
            "--out",
            str(workspace / "models" / "champion"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "| lightgbm |" in res.output
    assert (workspace / "models" / "champion" / "model_card.md").exists()
    assert runner.invoke(app, ["train", *common, "--version", "b"]).exit_code == 0

    assert "champion: v2 (was v1)" in runner.invoke(app, ["promote", "2", "--mlflow", uri]).output
    assert "champion is v1" in runner.invoke(app, ["rollback", "--mlflow", uri]).output

    mon = runner.invoke(
        app,
        ["monitor", "--data", str(data), "--mlflow", uri, "--start-day", "22", "--end-day", "30"],
    )
    assert mon.exit_code in (0, 1)  # 1 = retrain recommended
    assert "Score PSI" in mon.output


def test_benchmark_command(workspace: Path) -> None:
    out = workspace / "bench.md"
    res = runner.invoke(
        app,
        [
            "benchmark",
            "--data",
            str(workspace / "data"),
            "--model-dir",
            str(workspace / "models" / "champion"),
            "--n",
            "200",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert "in-process" in text
    assert "HTTP round trip" in text


def test_rollback_without_previous(tmp_path: Path) -> None:
    res = runner.invoke(app, ["rollback", "--mlflow", f"sqlite:///{tmp_path / 'x.db'}"])
    assert res.exit_code == 1


def test_experiment_small(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        [
            "experiment",
            "--out",
            str(tmp_path / "reports"),
            "--mlflow",
            f"sqlite:///{tmp_path / 'exp.db'}",
            "--customers",
            "400",
        ],
    )
    assert res.exit_code == 0, res.output
    report = (tmp_path / "reports" / "drift_experiment.md").read_text()
    assert "Weekly monitoring" in report
    assert "Gate:" in report
    assert (tmp_path / "reports" / "monitoring.csv").exists()


def test_version() -> None:
    assert "fraudwatch" in runner.invoke(app, ["--version"]).output
