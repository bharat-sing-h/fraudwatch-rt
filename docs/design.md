# Design notes

## 1. One feature implementation (why not a separate batch pipeline)

The most common way real-time ML systems fail quietly is **training/serving skew**: features are written twice, once in SQL or Spark for training and once in the service for real-time scoring, and the two drift apart. Fraud makes this worse because the strongest features are stateful (velocity windows, running amount statistics, terminal risk).

Here there is exactly one implementation, `FeatureEngine`. Serving calls it per event. Training data is produced by **replaying** history through the same methods in time order. Two tests enforce this:

- **Parity:** events processed one at a time through a Redis-backed engine produce *identical* features to the offline replay.
- **No leakage:** removing all data after a cutoff, including fraud reports not yet received, leaves every earlier feature unchanged.

Trade-off: replay is sequential. The 760k-transaction history takes about 30 s on one core, which is fine at this scale. At larger scale, the replay partitions by customer (state is per-card, except terminal risk), or the engine's window logic is reused inside a stream processor such as Flink or Bytewax. A feature store like Feast manages the storage side, but you still write the streaming transformations yourself, which is what this engine does.

## 2. Delayed labels

Fraud is confirmed by chargebacks days or weeks later. Ignoring this inflates offline results and produces leaky features.

- **Terminal risk** counts only fraud whose *report* arrived before the transaction being scored. Computing "fraud rate of this terminal" with one batch query over all labels is the classic leak, because the model then sees the future.
- **Training labels** are the labels known at training time (`label_ts <= as_of`). In the default run, 2,882 of 2,928 training frauds were known. The other 46 look legitimate to the model, exactly as they would in production.
- **Splits** are strictly by time, with a gap between train and validation so most reports have arrived.

## 3. Evaluation

- **PR-AUC** is the headline metric (ROC-AUC flatters at 1% prevalence), plus operating metrics at a threshold chosen on *validation* to maximise *fraud $ caught − $5 × alerts*. A model that catches many small frauds but misses large ones loses here, as it should.
- **Baselines** (rules, logistic regression) are always reported. LightGBM's gain over logistic regression (+0.13 PR-AUC) is the real value of the extra complexity.
- **Day-block bootstrap:** fraud comes in campaigns, so transactions on the same day aren't independent. Resampling days instead of rows gives honest intervals. Model comparisons are **paired** (same resampled days for both models).

## 4. A drift bug found by the drift experiment

The first experiment run showed feature PSI at 0.31 *before* the new fraud started, rising to 0.9 within weeks. The cause was `c_history_n`, a customer's total transaction count. It grows with the calendar, so its serving-time distribution never matches any training window, and the model was learning *time* rather than behaviour. It was removed, and weekly PSI dropped to 0.001–0.038. Monotonic counters (account age, lifetime totals) should be bucketed, capped or normalised by tenure.

## 5. Monitoring needs labels

The experiment's central finding: when the new fraud appeared, **feature PSI and score PSI stayed below 0.04 while PR-AUC fell from 0.71 to 0.56.** New fraud is a small share of traffic and barely moves input distributions. Input-drift monitoring catches pipeline breakage and population shifts; only performance on reported labels catches new attack patterns. Both are implemented. The label-based check lags by the reporting delay, which is why the challenger is trained as soon as the drop is confirmed and judged in shadow.

## 6. Registry, shadow and promotion

- **Bundles** version together the model, feature list, thresholds and drift reference distributions, so a new model can't ship with old thresholds or be monitored against the wrong baseline.
- **Aliases** (`champion`, `challenger`, `previous`) decouple "which version" from deployments. `POST /admin/reload` and `fw rollback` are one alias move.
- **Shadow scoring:** the challenger scores every live transaction on the same features but never decides. Disagreements are a Prometheus counter.
- **Promotion gate:** promote only if the lower bound of the 95% paired-bootstrap CI for *challenger − champion* PR-AUC is above 0. One lucky week isn't enough.

## 7. Streaming semantics

- Topics are keyed by `customer_id`, so each card's events are processed in order by one consumer, and per-card state has a single writer.
- Terminal windows are Redis sorted sets: concurrent writes from different partitions are independent `ZADD`s, so no locks or read-modify-write races.
- **At-least-once:** decisions are produced and flushed *before* offsets are committed. Redelivered transactions are deduplicated by `tx_id` (`SET NX`), so velocity counts and running statistics aren't double-counted. A test crashes the consumer mid-batch and checks the state matches a clean single pass.

## <a name="latency"></a>8. Latency

Measured p99 is 2.9 ms in-process and 5.7 ms over HTTP with Redis, against a typical 50 ms budget. Model inference takes about 20 µs (LightGBM, one thread, a single row). Almost all the remaining time is about a dozen sequential Redis round trips per transaction. Next steps, in order of payoff:

1. Pipeline all reads into one round trip (they're independent), then all writes into another.
2. Move the window maintenance into a Lua script, so updates are one server-side call.
3. Run more API workers; the model is CPU-light and the work is I/O-bound.

## 9. Known limitations

- The data is simulated. The fraud patterns are realistic in kind but simpler than real attacks, so absolute numbers shouldn't be compared with production systems.
- Retraining is triggered by CLI or cron. A production setup would run `fw monitor` on a schedule and kick off a training pipeline (Airflow, Prefect, Argo) when it exits non-zero.
- No feature-level explanation is returned to analysts. LightGBM SHAP values would be a cheap addition to `/v1/score`.
