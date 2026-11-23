"""Fraud-specific evaluation.

ROC-AUC flatters models when positives are ~1% of traffic, so the headline metric is PR-AUC
(average precision). Operating metrics are computed at a threshold chosen on *validation* data
to minimise expected cost, then reported on the untouched *test* period:

    cost = (fraud $ missed) + review_cost x (alerts raised)

Confidence intervals use a **day-block bootstrap**: days are resampled rather than individual
transactions, because transactions on the same day share fraud campaigns and are not
independent. Resampling rows would make the intervals too narrow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    precision: float
    recall: float
    alert_rate: float
    fraud_amount_caught: float  # share of fraud $ blocked or reviewed
    net_savings: float  # fraud $ caught - review cost

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if y.any() else float("nan")


def summary(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": pr_auc(y, score),
        "roc_auc": float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan"),
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "fraud_rate": float(y.mean()),
    }


def operating_point(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, threshold: float, review_cost: float = 5.0
) -> OperatingPoint:
    alert = score >= threshold
    tp = float((alert & (y == 1)).sum())
    fraud_amount = float(amount[y == 1].sum())
    caught = float(amount[alert & (y == 1)].sum())
    return OperatingPoint(
        threshold=float(threshold),
        precision=float(tp / alert.sum()) if alert.any() else float("nan"),
        recall=float(tp / y.sum()) if y.any() else float("nan"),
        alert_rate=float(alert.mean()),
        fraud_amount_caught=caught / fraud_amount if fraud_amount else float("nan"),
        net_savings=caught - review_cost * float(alert.sum()),
    )


def cost_optimal_threshold(
    y: np.ndarray, score: np.ndarray, amount: np.ndarray, review_cost: float = 5.0
) -> float:
    """Threshold maximising (fraud $ caught - review cost), over a grid of score quantiles."""
    candidates = np.unique(np.quantile(score, np.linspace(0.80, 0.9999, 400)))
    savings = [operating_point(y, score, amount, t, review_cost).net_savings for t in candidates]
    return float(candidates[int(np.argmax(savings))])


def precision_threshold(y: np.ndarray, score: np.ndarray, target: float = 0.9) -> float:
    """Lowest threshold whose precision reaches ``target`` (for automatic declines)."""
    order = np.argsort(-score)
    tp: np.ndarray = np.cumsum(y[order])
    precision = tp / np.arange(1, len(y) + 1)
    ok = np.flatnonzero(precision >= target)
    if not len(ok):
        return float("inf")
    # the deepest cut that still meets the target
    return float(score[order][ok.max()])


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float
    high: float


def day_block_bootstrap(
    days: np.ndarray,
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    candidate: str,
    n: int = 300,
    seed: int = 0,
) -> dict[str, CI]:
    """PR-AUC CI per model, plus the paired difference ``candidate - other`` for each other
    model. Pairing (same resampled days for both models) removes day-to-day difficulty from
    the comparison, so the difference CI is much tighter than comparing two separate CIs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(days)
    idx_by_day = [np.flatnonzero(days == d) for d in uniq]
    samples: dict[str, list[float]] = {k: [] for k in scores}
    for _ in range(n):
        pick = np.concatenate([idx_by_day[i] for i in rng.integers(0, len(uniq), len(uniq))])
        for k, s in scores.items():
            samples[k].append(pr_auc(y[pick], s[pick]))
    out = {k: CI(pr_auc(y, s), *np.percentile(samples[k], [2.5, 97.5])) for k, s in scores.items()}
    for k in scores:
        if k == candidate:
            continue
        diff = np.array(samples[candidate]) - np.array(samples[k])
        point = out[candidate].estimate - out[k].estimate
        out[f"{candidate} - {k}"] = CI(point, *np.percentile(diff, [2.5, 97.5]))
    return out
