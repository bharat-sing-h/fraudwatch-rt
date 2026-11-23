"""Promotion gate: should the challenger replace the champion?

Both models are scored on the *same* shadow-period transactions, and the PR-AUC difference is
bootstrapped by day (paired). The challenger is promoted only if the lower bound of the 95% CI
for ``challenger - champion`` is above ``min_gain``, so a lucky week is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudwatch.model.metrics import CI, day_block_bootstrap


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    champion: CI
    challenger: CI
    difference: CI
    reason: str


def promotion_gate(
    days: np.ndarray,
    y: np.ndarray,
    champion_scores: np.ndarray,
    challenger_scores: np.ndarray,
    *,
    min_gain: float = 0.0,
    n_boot: int = 500,
    seed: int = 0,
) -> GateDecision:
    if not y.any():
        nan = CI(float("nan"), float("nan"), float("nan"))
        return GateDecision(False, nan, nan, nan, "no reported fraud in the shadow period yet")
    cis = day_block_bootstrap(
        days,
        y,
        {"challenger": challenger_scores, "champion": champion_scores},
        candidate="challenger",
        n=n_boot,
        seed=seed,
    )
    diff = cis["challenger - champion"]
    promote = diff.low > min_gain
    reason = (
        f"challenger better by {diff.estimate:+.3f} PR-AUC (95% CI [{diff.low:+.3f}, "
        f"{diff.high:+.3f}])" + ("" if promote else f"; lower bound not above {min_gain:+.3f}")
    )
    return GateDecision(promote, cis["champion"], cis["challenger"], diff, reason)
