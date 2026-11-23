"""Production monitoring: input drift (PSI) and label-based performance.

Two complementary signals:

* **Population Stability Index** per feature and for the score, against the reference
  distributions stored in the model bundle. Available immediately, needs no labels, but fraud is
  ~1% of traffic: a new fraud pattern can barely move the overall input distribution.
* **Performance on reported labels** (PR-AUC on recent traffic whose fraud reports have come in).
  Lags by the reporting delay, but measures what actually matters.

Rule of thumb for PSI: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudwatch.model.bundle import ModelBundle
from fraudwatch.model.metrics import pr_auc

EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(reference: dict[str, list[float]], values: np.ndarray) -> float:
    """PSI over the reference's quantile bins plus a separate 'missing' bin."""
    edges = np.asarray(reference["edges"])
    missing_ref = reference["missing_rate"][0]
    finite = values[np.isfinite(values)]
    missing_cur = 1 - len(finite) / len(values) if len(values) else 0.0
    if len(edges) == 0 and not reference["expected"]:
        return psi(
            np.array([missing_ref, 1 - missing_ref]), np.array([missing_cur, 1 - missing_cur])
        )
    counts = np.bincount(np.searchsorted(edges, finite, side="right"), minlength=len(edges) + 1)
    actual = counts / max(counts.sum(), 1) * (1 - missing_cur)
    expected = np.asarray(reference["expected"]) * (1 - missing_ref)
    return psi(np.append(expected, missing_ref), np.append(actual, missing_cur))


@dataclass
class DriftReport:
    n: int
    feature_psi: dict[str, float]
    score_psi: float
    pr_auc: float | None = None
    reference_pr_auc: float | None = None
    alerts: list[str] = field(default_factory=list)

    @property
    def retrain_recommended(self) -> bool:
        return bool(self.alerts)

    def top_drifting(self, k: int = 5) -> list[tuple[str, float]]:
        return sorted(self.feature_psi.items(), key=lambda kv: -kv[1])[:k]

    def to_markdown(self) -> str:
        lines = [
            f"**Window size:** {self.n:,} transactions  ",
            f"**Score PSI:** {self.score_psi:.3f}  ",
        ]
        if self.pr_auc is not None:
            lines.append(
                f"**PR-AUC on reported labels:** {self.pr_auc:.3f} "
                f"(reference {self.reference_pr_auc:.3f})  "
            )
        lines.append(
            "**Most drifted features:** "
            + ", ".join(f"`{k}` {v:.3f}" for k, v in self.top_drifting())
        )
        lines.append("")
        lines += [f"- ⚠️ {a}" for a in self.alerts] or ["- ✅ no alerts"]
        return "\n".join(lines) + "\n"


def drift_report(
    bundle: ModelBundle,
    features: pd.DataFrame,
    *,
    labels: np.ndarray | None = None,
    psi_alert: float = 0.25,
    max_pr_auc_drop: float = 0.10,
) -> DriftReport:
    X = features[bundle.features].to_numpy(dtype=float)
    scores = bundle.predict(X)
    fpsi = {
        name: feature_psi(bundle.reference[name], X[:, i])
        for i, name in enumerate(bundle.features)
        if name in bundle.reference
    }
    spsi = feature_psi(bundle.reference["__score__"], scores)
    report = DriftReport(n=len(features), feature_psi=fpsi, score_psi=spsi)
    for name, value in report.top_drifting(len(fpsi)):
        if value > psi_alert:
            report.alerts.append(f"feature `{name}` PSI {value:.2f} > {psi_alert}")
    if spsi > psi_alert:
        report.alerts.append(f"score PSI {spsi:.2f} > {psi_alert}")
    if labels is not None and labels.any():
        report.pr_auc = pr_auc(labels, scores)
        ref = bundle.metadata.get("test_metrics", {}).get("pr_auc")
        report.reference_pr_auc = ref
        if ref is not None and report.pr_auc < ref - max_pr_auc_drop:
            report.alerts.append(
                f"PR-AUC {report.pr_auc:.3f} dropped more than {max_pr_auc_drop:.2f} below the "
                f"model's test PR-AUC {ref:.3f}"
            )
    return report
