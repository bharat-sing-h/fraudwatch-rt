# Drift experiment: detect, retrain, shadow, promote

762,979 simulated transactions over 120 days; a new fraud pattern (night-time online fraud with normal-looking amounts) starts on day 80. Feature replay: 29s for the whole history.

## 1. Weekly monitoring of the champion (v1)

| Days | Fraud rate | New pattern share of fraud | PR-AUC | Max feature PSI | Score PSI |
|---|---|---|---|---|---|
| 70–76 | 1.10% | 0% | 0.708 | 0.001 | 0.001 |
| 77–83 | 1.21% | 14% | 0.609 | 0.004 | 0.001 |
| 84–90 | 1.33% | 22% | 0.534 | 0.007 | 0.000 |
| 91–97 | 1.39% | 21% | 0.561 | 0.012 | 0.002 |
| 98–104 | 1.38% | 19% | 0.593 | 0.019 | 0.003 |
| 105–111 | 1.52% | 20% | 0.596 | 0.030 | 0.008 |
| 112–118 | 1.52% | 19% | 0.571 | 0.038 | 0.010 |

PR-AUC: 0.708 in the week before the drift, 0.563 on average in weeks 3–5 after it. Largest feature PSI over the whole period: 0.038; largest score PSI: 0.010.
**Input-drift monitors stayed green (all PSI < 0.1) while performance fell**: the new fraud is a small share of all traffic, so it barely moves input distributions. Monitoring performance on *reported labels* is what catches it.

## 2. Retrain at day 105, then shadow until day 120

Champion and challenger scored the same shadow-period traffic. Paired day-block bootstrap, 95% CI:

| Model | PR-AUC [95% CI] | Alert rate |
|---|---|---|
| v1 (champion) | 0.586 [0.562, 0.611] | 2.27% |
| v2 (challenger) | 0.760 [0.738, 0.783] | 2.89% |

**Gate:** PROMOTE. challenger better by +0.174 PR-AUC (95% CI [+0.161, +0.190]).

| Scenario | v1 recall | v2 recall |
|---|---|---|
| account_takeover | 84.1% | 85.3% |
| card_testing | 92.6% | 94.2% |
| compromised_terminal | 57.2% | 67.2% |
| impossible_travel | 100.0% | 100.0% |
| night_online | 3.9% | 96.8% |

Registry after the gate: champion = v2, previous = v1.

Total runtime: 81s on CPU.
