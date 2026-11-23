"""State stores for the feature engine.

The engine only needs two primitives, so the same feature code runs against an in-process dict
(training replay, tests) or Redis (production):

* **time windows**: ``(timestamp, member, value)`` entries per key, queried by time
  (Redis sorted sets: ``ZADD``/``ZRANGEBYSCORE``/``ZCOUNT``/``ZREMRANGEBYSCORE``);
* **hashes**: small per-entity records such as running amount statistics (Redis ``HSET``).

Windows are append-only and trimmed by time, so concurrent writers for the same terminal (cards
on different Kafka partitions) never need read-modify-write locking. Per-card hashes have a
single writer because the stream is partitioned by card.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from typing import Any, Protocol


class StateStore(Protocol):
    def window_add(self, key: str, ts: int, member: str, value: float) -> None: ...

    def window_get(self, key: str, since: int) -> list[tuple[int, float]]: ...

    def window_count(self, key: str, since: int) -> int: ...

    def window_trim(self, key: str, before: int) -> None: ...

    def hget(self, key: str) -> dict[str, float]: ...

    def hset(self, key: str, mapping: dict[str, float]) -> None: ...

    def mark_seen(self, key: str, ttl_s: int) -> bool:
        """Atomically record ``key``; True if it was not seen before (dedupe redeliveries)."""
        ...


class InMemoryStore:
    def __init__(self) -> None:
        self._windows: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
        self._hashes: dict[str, dict[str, float]] = {}
        self._seen: set[str] = set()

    def window_add(self, key: str, ts: int, member: str, value: float) -> None:
        entries = self._windows[key]
        if not entries or entries[-1][0] <= ts:
            entries.append((ts, member, value))  # fast path: events mostly arrive in order
        else:
            bisect.insort(entries, (ts, member, value))

    def _start(self, entries: list[tuple[int, str, float]], since: int) -> int:
        return bisect.bisect_left(entries, (since, "", float("-inf")))

    def window_get(self, key: str, since: int) -> list[tuple[int, float]]:
        entries = self._windows.get(key)
        if not entries:
            return []
        return [(ts, v) for ts, _, v in entries[self._start(entries, since) :]]

    def window_count(self, key: str, since: int) -> int:
        entries = self._windows.get(key)
        return len(entries) - self._start(entries, since) if entries else 0

    def window_trim(self, key: str, before: int) -> None:
        entries = self._windows.get(key)
        if entries:
            del entries[: self._start(entries, before)]

    def hget(self, key: str) -> dict[str, float]:
        return dict(self._hashes.get(key, {}))

    def hset(self, key: str, mapping: dict[str, float]) -> None:
        self._hashes.setdefault(key, {}).update(mapping)

    def mark_seen(self, key: str, ttl_s: int) -> bool:
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class RedisStore:
    """Production store. ``client`` is a ``redis.Redis`` (or ``fakeredis`` in tests)."""

    def __init__(self, client: Any, ttl_s: int = 35 * 86_400) -> None:
        self.r = client
        self.ttl_s = ttl_s  # idle entities expire on their own

    def window_add(self, key: str, ts: int, member: str, value: float) -> None:
        pipe = self.r.pipeline(transaction=False)
        pipe.zadd(key, {f"{member}|{value!r}": ts})
        pipe.expire(key, self.ttl_s)
        pipe.execute()

    def window_get(self, key: str, since: int) -> list[tuple[int, float]]:
        raw = self.r.zrangebyscore(key, since, "+inf", withscores=True)
        out = []
        for member, score in raw:
            text = member.decode() if isinstance(member, bytes) else member
            out.append((int(score), float(text.rsplit("|", 1)[1])))
        return out

    def window_count(self, key: str, since: int) -> int:
        return int(self.r.zcount(key, since, "+inf"))

    def window_trim(self, key: str, before: int) -> None:
        self.r.zremrangebyscore(key, "-inf", f"({before}")

    def hget(self, key: str) -> dict[str, float]:
        raw = self.r.hgetall(key)
        return {(k.decode() if isinstance(k, bytes) else k): float(v) for k, v in raw.items()}

    def hset(self, key: str, mapping: dict[str, float]) -> None:
        if mapping:
            pipe = self.r.pipeline(transaction=False)
            pipe.hset(key, mapping={k: repr(float(v)) for k, v in mapping.items()})
            pipe.expire(key, self.ttl_s)
            pipe.execute()

    def mark_seen(self, key: str, ttl_s: int) -> bool:
        return bool(self.r.set(key, 1, nx=True, ex=ttl_s))
