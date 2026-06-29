from __future__ import annotations

from collections import deque
from typing import Any


class RecordingTrackState:
    """Stores the current playback frame data and trail history for a recorded tracker."""

    def __init__(self, name: str, color: tuple[int, int, int], history_length: int | None = 80):
        self.name = name
        self.color = color
        if history_length is None or history_length <= 0:
            self.history: deque[dict[str, float]] = deque()
        else:
            self.history: deque[dict[str, float]] = deque(maxlen=history_length)
        self.data: dict[str, Any] = {}

    def update(self, tracker_data: dict[str, Any]) -> None:
        self.data = tracker_data
        if tracker_data.get("tracking"):
            self.history.append({
                "x": tracker_data.get("x", 0.0),
                "y": tracker_data.get("y", 0.0),
                "z": tracker_data.get("z", 0.0),
            })
