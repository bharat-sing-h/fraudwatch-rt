# Model card: fraudwatch `v1`

Evaluated on the held-out **test period** (strict time split; threshold chosen on validation to maximise fraud $ caught minus $5 per analyst review).

| Model | PR-AUC [95% CI] | ROC-AUC | Recall | Precision | Alert rate | Fraud $ caught | Net savings |
|---|---|---|---|---|---|---|---|
| rules | 0.198 [0.179, 0.217] | 0.854 | 30.1% | 53.5% | 0.61% | 54.6% | $105,545 |
| logistic_regression | 0.570 [0.547, 0.592] | 0.930 | 63.0% | 39.6% | 1.72% | 85.6% | $160,730 |
| lightgbm | 0.698 [0.672, 0.724] | 0.956 | 79.5% | 38.2% | 2.25% | 91.3% | $168,690 |

Paired PR-AUC differences (day-block bootstrap, 95% CI):

- **lightgbm - rules**: +0.500 [+0.473, +0.521]
- **lightgbm - logistic_regression**: +0.129 [+0.112, +0.142]

Recall by fraud scenario (LightGBM at its operating threshold):

- account_takeover: 82.3%
- card_testing: 94.4%
- compromised_terminal: 65.2%
- impossible_travel: 100.0%

Top features by gain:

- `amount`: 25.7%
- `c_small_cnt_1h`: 10.3%
- `dist_home`: 9.2%
- `amount_ratio_7d`: 8.6%
- `c_fraud_known_14d`: 8.3%
- `t_fraud_rate_28d`: 7.9%
- `amount_z`: 7.7%
- `c_avg_7d`: 3.7%

Training rows: 310,269; frauds known at training time: 2,882 of 2,928 (the rest weren't reported yet). Best iteration: 265.
Thresholds: review ≥ 0.0875, decline ≥ 0.9959.
