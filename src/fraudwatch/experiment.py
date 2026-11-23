"""End-to-end lifecycle experiment: drift -> detection -> retrain -> shadow -> promotion.

Timeline (days of simulated traffic):

    0 ........ 49 | gap | 56 .... 70 | 70 .. 80 | 80 ........ 105 | 105 ........... 120
    train v1         val v1      test v1   new fraud      retrain v2      v2 in shadow,
                                           pattern        (train 49-91,   gate on its outcome
                                           appears        val 98-105)

Everything is recorded in an MLflow registry: v1 becomes ``champion``, v2 is registered as
``challenger`` and promoted only if the promotion gate passes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraudwatch.features.batch import build_features
from fraudwatch.model.registry import Registry
from fraudwatch.model.train import TrainConfig, model_card, train
from fraudwatch.monitor.drift import drift_report
from fraudwatch.monitor.promotion import promotion_gate
from fraudwatch.simulate import SimConfig, day_index, simulate


@dataclass
class ExperimentConfig:
    n_days: int = 120
    drift_day: int = 80
    retrain_day: int = 105
    n_customers: int = 3_000
    seed: int = 7


def run(out_dir: str | Path, tracking_uri: str, cfg: ExperimentConfig | None = None) -> str:
    cfg = cfg or ExperimentConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sim = simulate(
        SimConfig(
            n_days=cfg.n_days, drift_day=cfg.drift_day, n_customers=cfg.n_customers, seed=cfg.seed
        )
    )
    tx = sim.transactions
    feats = build_features(tx, sim.customers, sim.terminals)
    feature_secs = time.time() - t0
    days = day_index(tx.ts)
    reg = Registry(tracking_uri)

    # v1: trained before the drift
    cfg1 = TrainConfig(train_end_day=49, val_end_day=70, test_end_day=80)
    r1 = train(feats, tx, cfg1, version="v1")
    reg.log_and_register(
        r1.bundle,
        params=cfg1.model_dump(exclude={"lgb_params"}),
        metrics={f"test_{k}": v for k, v in r1.metrics["lightgbm"].items()},
        model_card=model_card(r1),
        alias="champion",
    )
    (out / "model_card_v1.md").write_text(model_card(r1))

    # weekly monitoring of the champion
    rows = []
    # full weeks only: a partial week has a skewed weekday mix, which alone moves the
    # weekend/hour features' PSI and would raise false drift alarms
    for start in range(70, cfg.n_days - 6, 7):
        m = (days >= start) & (days < start + 7)
        rep = drift_report(r1.bundle, feats[m], labels=tx.is_fraud.to_numpy()[m])
        night = m & (tx.scenario.to_numpy() == "night_online")
        rows.append(
            {
                "days": f"{start}–{start + 6}",
                "fraud_rate": tx.is_fraud.to_numpy()[m].mean(),
                "new_pattern_share": night.sum() / max(tx.is_fraud.to_numpy()[m].sum(), 1),
                "pr_auc": rep.pr_auc,
                "max_feature_psi": max(rep.feature_psi.values()),
                "score_psi": rep.score_psi,
                "alerts": len(rep.alerts),
            }
        )
    monitor = pd.DataFrame(rows)

    # v2: retrained at retrain_day on data that includes the new pattern
    end = cfg.retrain_day
    # train window must reach into the drift period, or the new model can't learn the pattern
    cfg2 = TrainConfig(
        train_start_day=end - 56,
        train_end_day=end - 14,
        gap_days=7,
        val_end_day=end,
        test_end_day=cfg.n_days,
    )
    r2 = train(feats, tx, cfg2, version="v2")
    v2 = reg.log_and_register(
        r2.bundle,
        params=cfg2.model_dump(exclude={"lgb_params"}),
        metrics={f"test_{k}": v for k, v in r2.metrics["lightgbm"].items()},
        model_card=model_card(r2),
        alias="challenger",
    )
    (out / "model_card_v2.md").write_text(model_card(r2))

    # shadow period: both models score the same traffic; gate on the outcome
    shadow = days >= end
    X = feats[r1.bundle.features].to_numpy(dtype=float)[shadow]
    y = tx.is_fraud.to_numpy()[shadow]
    s1, s2 = r1.bundle.predict(X), r2.bundle.predict(X)
    gate = promotion_gate(days[shadow], y, s1, s2)
    if gate.promote:
        reg.promote(v2)
    scen = tx.scenario.to_numpy()[shadow]
    recall_rows = []
    for s in sorted(set(scen) - {"legit"}):
        mask = scen == s
        recall_rows.append(
            {
                "scenario": s,
                "v1_recall": float((s1[mask] >= r1.bundle.review_threshold).mean()),
                "v2_recall": float((s2[mask] >= r2.bundle.review_threshold).mean()),
            }
        )
    alerts_v1 = float((s1 >= r1.bundle.review_threshold).mean())
    alerts_v2 = float((s2 >= r2.bundle.review_threshold).mean())

    md = [
        "# Drift experiment: detect, retrain, shadow, promote",
        "",
        f"{len(tx):,} simulated transactions over {cfg.n_days} days; a new fraud pattern "
        f"(night-time online fraud with normal-looking amounts) starts on day {cfg.drift_day}. "
        f"Feature replay: {feature_secs:.0f}s for the whole history.",
        "",
        "## 1. Weekly monitoring of the champion (v1)",
        "",
        "| Days | Fraud rate | New pattern share of fraud | PR-AUC | Max feature PSI | Score PSI |",
        "|---|---|---|---|---|---|",
    ]
    for r in monitor.itertuples():
        md.append(
            f"| {r.days} | {r.fraud_rate:.2%} | {r.new_pattern_share:.0%} | "
            f"{r.pr_auc:.3f} | {r.max_feature_psi:.3f} | {r.score_psi:.3f} |"
        )
    pre = float(monitor.pr_auc.iloc[0])
    post = float(monitor.pr_auc.iloc[2:5].mean())
    max_psi, max_spsi = monitor.max_feature_psi.max(), monitor.score_psi.max()
    md += [
        "",
        f"PR-AUC: {pre:.3f} in the week before the drift, {post:.3f} on average in "
        f"weeks 3–5 after it. Largest feature PSI over the whole period: {max_psi:.3f}; "
        f"largest score PSI: {max_spsi:.3f}.",
    ]
    if max_psi < 0.1 and max_spsi < 0.1 and pre - post > 0.05:
        md.append(
            "**Input-drift monitors stayed green (all PSI < 0.1) while performance fell**:"
            " the new fraud is a small share of all traffic, so it barely moves input "
            "distributions. Monitoring performance on *reported labels* is what catches it."
        )
    md += [
        "",
        f"## 2. Retrain at day {end}, then shadow until day {cfg.n_days}",
        "",
        "Champion and challenger scored the same shadow-period traffic. Paired day-block "
        "bootstrap, 95% CI:",
        "",
        "| Model | PR-AUC [95% CI] | Alert rate |",
        "|---|---|---|",
        f"| v1 (champion) | {gate.champion.estimate:.3f} [{gate.champion.low:.3f}, "
        f"{gate.champion.high:.3f}] | {alerts_v1:.2%} |",
        f"| v2 (challenger) | {gate.challenger.estimate:.3f} [{gate.challenger.low:.3f}, "
        f"{gate.challenger.high:.3f}] | {alerts_v2:.2%} |",
        "",
        f"**Gate:** {'PROMOTE' if gate.promote else 'KEEP CHAMPION'}. {gate.reason}.",
        "",
        "| Scenario | v1 recall | v2 recall |",
        "|---|---|---|",
    ]
    md += [
        f"| {r['scenario']} | {r['v1_recall']:.1%} | {r['v2_recall']:.1%} |" for r in recall_rows
    ]
    md += [
        "",
        f"Registry after the gate: champion = v{reg.alias_version('champion')}, "
        f"previous = v{reg.alias_version('previous')}.",
        "",
        f"Total runtime: {time.time() - t0:.0f}s on CPU.",
    ]
    report = "\n".join(md) + "\n"
    (out / "drift_experiment.md").write_text(report)
    monitor.to_csv(out / "monitoring.csv", index=False)
    return report
