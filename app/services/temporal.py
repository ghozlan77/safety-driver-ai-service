"""Temporal state tracking across sequential frames / sessions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CounterState:
    """Tracks consecutive positive frames and elapsed active duration."""

    consecutive: int = 0
    active_since: Optional[float] = None
    last_value: Optional[float] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def update(self, positive: bool, now: Optional[float] = None) -> None:
        ts = now if now is not None else time.monotonic()
        if positive:
            if self.consecutive == 0:
                self.active_since = ts
            self.consecutive += 1
        else:
            self.consecutive = 0
            self.active_since = None

    def duration_seconds(self, now: Optional[float] = None) -> float:
        if self.active_since is None:
            return 0.0
        ts = now if now is not None else time.monotonic()
        return max(0.0, ts - self.active_since)

    def is_triggered(
        self,
        min_frames: int,
        min_duration_seconds: float = 0.0,
        now: Optional[float] = None,
    ) -> bool:
        if self.consecutive < min_frames:
            return False
        if min_duration_seconds <= 0:
            return True
        return self.duration_seconds(now) >= min_duration_seconds


@dataclass
class SessionState:
    """Per-client temporal buffers for all detectors."""

    session_id: str
    created_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    frame_index: int = 0
    drowsiness: CounterState = field(default_factory=CounterState)
    yawning: CounterState = field(default_factory=CounterState)
    phone_usage: CounterState = field(default_factory=CounterState)
    distraction: CounterState = field(default_factory=CounterState)
    abnormal_movement: CounterState = field(default_factory=CounterState)
    previous_landmarks: Optional[Any] = None
    previous_timestamp: Optional[float] = None


class SessionStore:
    """Thread-safe in-memory session store with TTL eviction."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: Optional[str] = None) -> SessionState:
        with self._lock:
            self._evict_expired()
            sid = session_id or str(uuid.uuid4())
            state = self._sessions.get(sid)
            if state is None:
                state = SessionState(session_id=sid)
                self._sessions[sid] = state
                logger.debug("Created session %s", sid)
            state.last_seen = time.monotonic()
            return state

    def touch(self, session: SessionState) -> None:
        with self._lock:
            session.last_seen = time.monotonic()
            session.frame_index += 1

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict_expired(self) -> None:
        ttl = self._settings.session_ttl_seconds
        now = time.monotonic()
        expired = [
            sid
            for sid, state in self._sessions.items()
            if (now - state.last_seen) > ttl
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("Evicted expired session %s", sid)


# Process-wide store used by API dependencies
session_store = SessionStore()
