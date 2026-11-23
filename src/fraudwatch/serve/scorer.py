"""Scoring core shared by the HTTP API and the Kafka worker.

Per transaction: features (online store) -> champion score -> decision, plus a **shadow**
score from the challenger. The shadow is computed on the same features and logged, but never
changes the decision. That lets a new model be judged on live traffic, including agreement
with the champion and the alert volume it *would* generate, before it can affect a customer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass

from fraudwatch.features.engine import FeatureEngine, FraudLabel, Transaction
from fraudwatch.model.bundle import ModelBundle


@dataclass(frozen=True)
class ScoreResult:
    tx_id: int
    score: float
    decision: str
    model_version: str
    shadow_score: float | None
    shadow_decision: str | None
    shadow_version: str | None
    features_ms: float
    model_ms: float
    total_ms: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Scorer:
    def __init__(
        self, engine: FeatureEngine, champion: ModelBundle, challenger: ModelBundle | None = None
    ) -> None:
        self.engine = engine
        self._lock = threading.Lock()
        self._champion = champion
        self._challenger = challenger

    @property
    def champion(self) -> ModelBundle:
        return self._champion

    @property
    def challenger(self) -> ModelBundle | None:
        return self._challenger

    def swap(self, champion: ModelBundle, challenger: ModelBundle | None) -> None:
        """Hot-swap models (promotion / rollback) without dropping in-flight requests."""
        with self._lock:
            self._champion, self._challenger = champion, challenger

    def score(self, tx: Transaction) -> ScoreResult:
        start = time.perf_counter()
        champion, challenger = self._champion, self._challenger  # consistent snapshot
        features = self.engine.process_transaction(tx)
        t_feat = time.perf_counter()
        score = champion.predict_one(features)
        shadow = challenger.predict_one(features) if challenger else None
        t_model = time.perf_counter()
        return ScoreResult(
            tx_id=tx.tx_id,
            score=score,
            decision=champion.decide(score),
            model_version=champion.version,
            shadow_score=shadow,
            shadow_decision=challenger.decide(shadow)
            if challenger and shadow is not None
            else None,
            shadow_version=challenger.version if challenger else None,
            features_ms=(t_feat - start) * 1000,
            model_ms=(t_model - t_feat) * 1000,
            total_ms=(t_model - start) * 1000,
        )

    def label(self, label: FraudLabel) -> None:
        self.engine.process_label(label)
