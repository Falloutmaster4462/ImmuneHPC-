"""
utils/events.py — Internal Event Bus

Lightweight publish/subscribe bus used by all modules to communicate
without tight coupling. Modules emit events; the controller and supervisor
route responses.
"""

from __future__ import annotations
import time
import uuid
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class EventType(Enum):
    # Monitor events
    TELEMETRY_COLLECTED = "telemetry.collected"
    HEALTH_DEGRADED = "health.degraded"
    HEALTH_RECOVERED = "health.recovered"

    # Anomaly events
    ANOMALY_DETECTED = "anomaly.detected"
    ANOMALY_CLEARED = "anomaly.cleared"

    # Quarantine events
    NODE_QUARANTINED = "node.quarantined"
    NODE_RELEASED = "node.released"

    # Healing events
    HEAL_STARTED = "heal.started"
    HEAL_SUCCESS = "heal.success"
    HEAL_FAILED = "heal.failed"
    REIMAGE_REQUESTED = "reimage.requested"

    # Optimisation events
    OPTIMISATION_APPLIED = "optimisation.applied"
    OPTIMISATION_FAILED = "optimisation.failed"

    # Defense events
    INTRUSION_DETECTED = "defense.intrusion"
    NODE_UNTRUSTED = "defense.untrusted"
    MITIGATION_APPLIED = "defense.mitigated"

    # ASL events
    PATCH_GENERATED = "asl.patch_generated"
    PATCH_VALIDATED = "asl.patch_validated"
    PATCH_REJECTED = "asl.patch_rejected"
    PATCH_DEPLOYED = "asl.patch_deployed"
    PATCH_ROLLED_BACK = "asl.patch_rolled_back"

    # System events
    LOOP_TICK = "system.loop_tick"
    SHUTDOWN = "system.shutdown"


@dataclass
class Event:
    type: EventType
    source: str
    payload: Any = None
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[{self.type.value}] from={self.source} id={self.event_id}"


Handler = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[Handler]] = defaultdict(list)
        self._wildcard: List[Handler] = []
        self._lock = threading.Lock()
        self._history: List[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe to every event type (useful for logging/metrics)."""
        with self._lock:
            self._wildcard.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        with self._lock:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h != handler
            ]

    def emit(self, event: Event) -> None:
        with self._lock:
            handlers = list(self._handlers[event.type]) + list(self._wildcard)
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                print(f"[EventBus] Handler error for {event}: {exc}")

    def emit_simple(self, event_type: EventType, source: str, payload: Any = None) -> Event:
        event = Event(type=event_type, source=source, payload=payload)
        self.emit(event)
        return event

    def recent(self, event_type: Optional[EventType] = None, n: int = 10) -> List[Event]:
        with self._lock:
            history = list(self._history)
        if event_type:
            history = [e for e in history if e.type == event_type]
        return history[-n:]


# Module-level singleton — import and use directly
bus = EventBus()
