| Path | Requests | p50 ms | p95 ms | p99 ms | max ms | req/s (1 client) |
|---|---|---|---|---|---|---|
| in-process (features + model), memory store | 5,000 | 0.07 | 0.12 | 0.15 | 1.0 | 12,651 |
| HTTP round trip (FastAPI), memory store | 5,000 | 1.52 | 2.35 | 3.19 | 14.5 | 618 |
| in-process (features + model), redis store | 5,000 | 1.54 | 2.33 | 2.93 | 5.4 | 597 |
| HTTP round trip (FastAPI), redis store | 5,000 | 3.79 | 4.85 | 5.74 | 11.2 | 258 |
