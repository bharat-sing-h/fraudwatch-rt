"""Everything the scorer needs, versioned together.

A model is not just weights: the feature list, decision thresholds and the reference
distributions used for drift monitoring must travel with it. Otherwise a new model can be
deployed with old thresholds, or drift can be measured against the wrong baseline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from fraudwatch.features.engine import FEATURES

APPROVE, REVIEW, DECLINE = "approve", "review", "decline"


@dataclass
class ModelBundle:
    booster: lgb.Booster
    features: list[str]
    review_threshold: float
    decline_threshold: float
    metadata: dict[str, Any] = field(default_factory=dict)
    # drift reference: per-feature bin edges and expected proportions (+ the score)
    reference: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.asarray(self.booster.predict(X, num_threads=1), dtype=np.float64)
        return out

    def predict_one(self, features: dict[str, float]) -> float:
        row = np.array([[features[name] for name in self.features]], dtype=np.float64)
        return float(self.predict(row)[0])

    def decide(self, score: float) -> str:
        if score >= self.decline_threshold:
            return DECLINE
        if score >= self.review_threshold:
            return REVIEW
        return APPROVE

    @property
    def version(self) -> str:
        return str(self.metadata.get("version", "unversioned"))

    def save(self, directory: str | Path) -> Path:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(out / "model.txt"))
        meta = {
            "features": self.features,
            "review_threshold": self.review_threshold,
            "decline_threshold": self.decline_threshold
            if math.isfinite(self.decline_threshold)
            else None,
            "metadata": self.metadata,
            "reference": self.reference,
        }
        (out / "bundle.json").write_text(json.dumps(meta, indent=2, default=float))
        return out

    @classmethod
    def load(cls, directory: str | Path) -> ModelBundle:
        d = Path(directory)
        meta = json.loads((d / "bundle.json").read_text())
        if meta["features"] != FEATURES:
            missing = set(meta["features"]) - set(FEATURES)
            if missing:
                raise ValueError(f"model needs features the engine doesn't produce: {missing}")
        decline = meta["decline_threshold"]
        return cls(
            booster=lgb.Booster(model_file=str(d / "model.txt")),
            features=meta["features"],
            review_threshold=float(meta["review_threshold"]),
            decline_threshold=float(decline) if decline is not None else float("inf"),
            metadata=meta["metadata"],
            reference=meta["reference"],
        )

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "review_threshold": self.review_threshold,
            "decline_threshold": self.decline_threshold,
            **{k: v for k, v in self.metadata.items() if k != "version"},
        }


def reference_distribution(values: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    """Quantile bin edges + proportions, the baseline for PSI drift checks."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"edges": [], "expected": [], "missing_rate": [1.0]}
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, finite, side="right"), minlength=len(edges) + 1)
    return {
        "edges": edges.tolist(),
        "expected": (counts / counts.sum()).tolist(),
        "missing_rate": [float(1 - len(finite) / len(values))],
    }
