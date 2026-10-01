from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class AppState:
    gesture: str = "—"
    confidence: float = 0.0
    status: str = "STARTING"
    inference_mode: str = "LOCAL"
    hand_detected: bool = False
    fps: float = 0.0
    latency_ms: float = 0.0
    speaking: bool = False
    jpeg_frame: bytes | None = None


class SharedState:
    def __init__(self) -> None:
        self._state = AppState()
        self._lock = threading.Lock()

    def update(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self._state, key, value)

    def snapshot(self) -> AppState:
        with self._lock:
            return AppState(**vars(self._state))