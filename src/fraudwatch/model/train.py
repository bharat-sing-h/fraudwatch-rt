"""Train and evaluate fraud models on a strict time split.

    |---- train ----|- gap -|-- validation --|---- test ----|
    0          train_end          val_end            test_end   (days)

* **Time-based splits only.** Random splits leak future fraud campaigns (the same compromised
  terminal, the same stolen card) into training and make results look far better than they are.
* **Label maturity.** The model is trained at the start of validation. At that moment fraud that
  hasn't been reported yet looks legitimate, so training labels are the labels *known then*
  (``label_ts <= as_of``), not hindsight ground truth. The gap lets most reports arrive.
* **Baselines.** Hand-written rules and logistic regression are reported next to LightGBM.
  Without them, "PR-AUC 0.8" means nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudwatch.features.engine import FEATURES
from fraudwatch.model.bundle import ModelBundle, reference_distribution
from fraudwatch.model.metrics import (
    CI,
    cost_optimal_threshold,
    day_block_bootstrap,
    operating_point,
    precision_threshold,
    summary,
)
from fraudwatch.simulate import DAY, EPOCH, day_index


class TrainConfig(BaseModel):
    train_start_day: int = 0
    train_end_day: int = 49
    gap_days: int = 7
    val_end_day: int = 70
    test_end_day: int = 90
    review_cost: float = 5.0  # analyst cost per alert, in $
    decline_precision: float = 0.9  # auto-decline only where we're this sure
    seed: int = 42
    lgb_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "average_precision",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100


@dataclass
class TrainResult:
    bundle: ModelBundle
    metrics: dict[str, dict[str, float]]  # model -> test metrics
    cis: dict[str, CI]
    scenario_recall: dict[str, float]
    importance: list[tuple[str, float]]
    test_scores: dict[str, np.ndarray] = field(default_factory=dict)


def rules_score(X: pd.DataFrame) -> np.ndarray:
    """What a fraud team writes before ML: one point per triggered rule."""
    rules = [
        X.amount_z > 3,
        X.c_small_cnt_1h >= 2,
        X.speed_kmh > 120,
        X.t_fraud_rate_28d > 0.1,
        X.c_fraud_known_14d >= 1,
    ]
    score: np.ndarray = np.mean([r.fillna(False).to_numpy() for r in rules], axis=0)
    return score


def known_labels(tx: pd.DataFrame, as_of_ts: int) -> np.ndarray:
    """Labels as a training job running at ``as_of_ts`` would see them."""
    reported = (tx.is_fraud.to_numpy() == 1) & (tx.label_ts.to_numpy() <= as_of_ts)
    out: np.ndarray = reported.astype(int)
    return out


def split_masks(days: np.ndarray, cfg: TrainConfig) -> dict[str, np.ndarray]:
    val_start = cfg.train_end_day + cfg.gap_days
    return {
        "train": (days >= cfg.train_start_day) & (days < cfg.train_end_day),
        "val": (days >= val_start) & (days < cfg.val_end_day),
        "test": (days >= cfg.val_end_day) & (days < cfg.test_end_day),
    }


def train(
    features: pd.DataFrame,
    tx: pd.DataFrame,
    cfg: TrainConfig | None = None,
    *,
    version: str | None = None,
) -> TrainResult:
    cfg = cfg or TrainConfig()
    df = features.merge(
        tx[["tx_id", "ts", "amount", "is_fraud", "label_ts", "scenario"]].rename(
            columns={"amount": "tx_amount"}
        ),
        on="tx_id",
        validate="1:1",
    )
    days = day_index(df.ts)
    masks = split_masks(days, cfg)
    as_of = EPOCH + (cfg.train_end_day + cfg.gap_days) * DAY
    y_train = known_labels(df[masks["train"]], as_of)
    X = df[FEATURES]
    Xtr, Xva, Xte = X[masks["train"]], X[masks["val"]], X[masks["test"]]
    # validation (early stopping, thresholds) also only sees fraud reported by the end of the
    # validation window; the test period is scored with full hindsight, i.e. evaluated later,
    # once labels are in
    y_val = known_labels(df[masks["val"]], EPOCH + cfg.val_end_day * DAY)
    y_te = df.is_fraud.to_numpy()[masks["test"]]

    started = time.time()
    dtrain = lgb.Dataset(Xtr, y_train, categorical_feature=["category"], free_raw_data=False)
    dval = lgb.Dataset(Xva, y_val, reference=dtrain, categorical_feature=["category"])
    booster = lgb.train(
        {**cfg.lgb_params, "seed": cfg.seed},
        dtrain,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    train_secs = time.time() - started

    logreg = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=2000),
    )
    logreg.fit(Xtr, y_train)

    def score_all(Xs: pd.DataFrame) -> dict[str, np.ndarray]:
        return {
            "rules": rules_score(Xs),
            "logistic_regression": logreg.predict_proba(Xs)[:, 1],
            "lightgbm": booster.predict(Xs, num_iteration=booster.best_iteration),
        }

    val_scores, test_scores = score_all(Xva), score_all(Xte)
    amt_val = df.tx_amount.to_numpy()[masks["val"]]
    amt_te = df.tx_amount.to_numpy()[masks["test"]]

    metrics: dict[str, dict[str, float]] = {}
    thresholds: dict[str, float] = {}
    for name in val_scores:
        t = cost_optimal_threshold(y_val, val_scores[name], amt_val, cfg.review_cost)
        thresholds[name] = t
        op = operating_point(y_te, test_scores[name], amt_te, t, cfg.review_cost)
        metrics[name] = {
            **summary(y_te, test_scores[name]),
            **{f"op_{k}": v for k, v in op.as_dict().items()},
        }

    cis = day_block_bootstrap(days[masks["test"]], y_te, test_scores, candidate="lightgbm")
    decline_t = precision_threshold(y_val, val_scores["lightgbm"], cfg.decline_precision)

    alert = test_scores["lightgbm"] >= thresholds["lightgbm"]
    scen = df.scenario.to_numpy()[masks["test"]]
    scenario_recall = {
        s: float(alert[(scen == s)].mean()) for s in sorted(set(scen)) if s != "legit"
    }
    gain = booster.feature_importance("gain")
    importance = sorted(
        zip(FEATURES, (gain / gain.sum()).tolist(), strict=True), key=lambda kv: -kv[1]
    )

    reference = {name: reference_distribution(Xva[name].to_numpy(dtype=float)) for name in FEATURES}
    reference["__score__"] = reference_distribution(val_scores["lightgbm"])
    bundle = ModelBundle(
        booster=booster,
        features=list(FEATURES),
        review_threshold=thresholds["lightgbm"],
        decline_threshold=max(decline_t, thresholds["lightgbm"]),
        metadata={
            "version": version or time.strftime("%Y%m%d-%H%M%S"),
            "train_days": [cfg.train_start_day, cfg.train_end_day],
            "val_days": [cfg.train_end_day + cfg.gap_days, cfg.val_end_day],
            "test_days": [cfg.val_end_day, cfg.test_end_day],
            "best_iteration": int(booster.best_iteration),
            "train_rows": int(masks["train"].sum()),
            "train_positive_known": int(y_train.sum()),
            "train_positive_true": int(df.is_fraud.to_numpy()[masks["train"]].sum()),
            "train_seconds": round(train_secs, 1),
            "test_metrics": metrics["lightgbm"],
        },
        reference=reference,
    )
    return TrainResult(bundle, metrics, cis, scenario_recall, importance, test_scores)


def model_card(result: TrainResult) -> str:
    """Markdown summary written next to every trained model."""
    m = result.metrics
    lines = [
        f"# Model card: fraudwatch `{result.bundle.version}`",
        "",
        "Evaluated on the held-out **test period** (strict time split; threshold chosen on "
        "validation to maximise fraud $ caught minus $5 per analyst review).",
        "",
        "| Model | PR-AUC [95% CI] | ROC-AUC | Recall | Precision | Alert rate | Fraud $ caught |"
        " Net savings |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, mm in m.items():
        ci = result.cis[name]
        lines.append(
            f"| {name} | {mm['pr_auc']:.3f} [{ci.low:.3f}, {ci.high:.3f}] | {mm['roc_auc']:.3f} | "
            f"{mm['op_recall']:.1%} | {mm['op_precision']:.1%} | {mm['op_alert_rate']:.2%} | "
            f"{mm['op_fraud_amount_caught']:.1%} | ${mm['op_net_savings']:,.0f} |"
        )
    lines += ["", "Paired PR-AUC differences (day-block bootstrap, 95% CI):", ""]
    for key, ci in result.cis.items():
        if " - " in key:
            lines.append(f"- **{key}**: {ci.estimate:+.3f} [{ci.low:+.3f}, {ci.high:+.3f}]")
    lines += ["", "Recall by fraud scenario (LightGBM at its operating threshold):", ""]
    lines += [f"- {s}: {r:.1%}" for s, r in result.scenario_recall.items()]
    lines += ["", "Top features by gain:", ""]
    lines += [f"- `{f}`: {g:.1%}" for f, g in result.importance[:8]]
    meta = result.bundle.metadata
    lines += [
        "",
        f"Training rows: {meta['train_rows']:,}; frauds known at training time: "
        f"{meta['train_positive_known']:,} of {meta['train_positive_true']:,} "
        f"(the rest weren't reported yet). Best iteration: {meta['best_iteration']}.",
        f"Thresholds: review ≥ {result.bundle.review_threshold:.4f}, "
        f"decline ≥ {result.bundle.decline_threshold:.4f}.",
    ]
    return "\n".join(lines) + "\n"
