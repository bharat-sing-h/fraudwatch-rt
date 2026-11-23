"""Kafka stream processor: ``transactions`` -> score -> ``decisions``; ``labels`` -> state.

Delivery semantics: at-least-once. Offsets are committed only after the decision has been
produced and flushed; a crash before commit means redelivery, and redelivered transactions
don't double-count in feature state (deduplicated by ``tx_id`` in the engine). Topics are keyed
by ``customer_id``, so each card's events are processed in order by a single consumer.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from fraudwatch.features.engine import FraudLabel, Transaction
from fraudwatch.serve.scorer import Scorer

log = logging.getLogger(__name__)

TRANSACTIONS, LABELS, DECISIONS = "transactions", "labels", "decisions"


@dataclass(frozen=True)
class Message:
    topic: str
    key: str
    value: dict[str, Any]


class Bus(Protocol):
    def poll(self, timeout_s: float) -> Message | None: ...

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None: ...

    def flush(self) -> None: ...

    def commit(self) -> None: ...


class InMemoryBus:
    """Test double with Kafka-like semantics (uncommitted messages can be redelivered)."""

    def __init__(self) -> None:
        self.inbox: deque[Message] = deque()
        self.outbox: list[Message] = []
        self.uncommitted: list[Message] = []
        self.commits = 0

    def send(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self.inbox.append(Message(topic, key, value))

    def poll(self, timeout_s: float) -> Message | None:
        if not self.inbox:
            return None
        msg = self.inbox.popleft()
        self.uncommitted.append(msg)
        return msg

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self.outbox.append(Message(topic, key, value))

    def flush(self) -> None:
        pass

    def commit(self) -> None:
        self.uncommitted.clear()
        self.commits += 1

    def crash_and_redeliver(self) -> None:
        """Simulate a consumer crash: everything uncommitted comes back."""
        self.inbox.extendleft(reversed(self.uncommitted))
        self.uncommitted.clear()


class KafkaBus:  # pragma: no cover - exercised against a real broker in docker-compose
    def __init__(self, bootstrap: str, group_id: str = "fraudwatch-scorer") -> None:
        from confluent_kafka import Consumer, Producer

        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap,
                "group.id": group_id,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
        self.consumer.subscribe([TRANSACTIONS, LABELS])
        self.producer = Producer(
            {"bootstrap.servers": bootstrap, "linger.ms": 5, "enable.idempotence": True}
        )

    def poll(self, timeout_s: float) -> Message | None:
        msg = self.consumer.poll(timeout_s)
        if msg is None:
            return None
        if msg.error():
            log.warning("kafka error: %s", msg.error())
            return None
        key = msg.key().decode() if msg.key() else ""
        return Message(msg.topic(), key, json.loads(msg.value()))

    def produce(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self.producer.produce(topic, key=key, value=json.dumps(value).encode())
        self.producer.poll(0)

    def flush(self) -> None:
        self.producer.flush(10)

    def commit(self) -> None:
        self.consumer.commit(asynchronous=False)


def run(
    bus: Bus,
    scorer: Scorer,
    *,
    commit_every: int = 100,
    idle_polls: int | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Process messages until ``idle_polls`` consecutive empty polls (None = forever)."""
    processed = pending = idle = 0
    while idle_polls is None or idle < idle_polls:
        msg = bus.poll(0.5)
        if msg is None:
            idle += 1
            if pending:
                bus.flush()
                bus.commit()
                pending = 0
            continue
        idle = 0
        v = msg.value
        if msg.topic == TRANSACTIONS:
            result = scorer.score(
                Transaction(
                    int(v["tx_id"]),
                    int(v["ts"]),
                    int(v["customer_id"]),
                    int(v["terminal_id"]),
                    float(v["amount"]),
                )
            )
            decision = result.as_dict()
            bus.produce(DECISIONS, msg.key, decision)
            if on_decision:
                on_decision(decision)
        elif msg.topic == LABELS:
            scorer.label(
                FraudLabel(
                    int(v["tx_id"]), int(v["tx_ts"]), int(v["customer_id"]), int(v["terminal_id"])
                )
            )
        processed += 1
        pending += 1
        if pending >= commit_every:
            bus.flush()  # decisions durable *before* offsets move: at-least-once
            bus.commit()
            pending = 0
    return processed
