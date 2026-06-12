"""
mock_tracker.py
---------------
Simulates HTC Vive tracker data at ~50 Hz.
"""

import math
import time

from .openvr_tracker import BaseTracker
# ──────────────────────────────────────────────
#  MOCK IMPLEMENTATIONS
# ──────────────────────────────────────────────

class MockUltimateTracker(BaseTracker):
    """
    Simulates an HTC Vive Ultimate Tracker.
    Moves in a slow, figure-8 / Lissajous pattern in the X-Z plane,
    with gentle Y drift, to exercise the visualizer realistically.
    """

    def __init__(self):
        self._start = time.perf_counter()
        self.name = "Vive Ultimate Tracker"

    def get_data(self) -> dict:
        t = time.perf_counter() - self._start
        return {
            "name":      self.name,
            "connected": True,
            "tracking":  True,
            "x":  0.6 * math.sin(0.4 * t),
            "y":  1.0 + 0.08 * math.sin(0.15 * t),
            "z":  0.4 * math.sin(0.8 * t),
        }

    def shutdown(self) -> None:
        pass


class MockTracker30(BaseTracker):
    """
    Simulates an HTC Vive Tracker 3.0.
    Uses a circular orbit offset from the origin.
    """

    def __init__(self):
        self._start = time.perf_counter()
        self.name = "Vive Tracker 3.0"

    def get_data(self) -> dict:
        t = time.perf_counter() - self._start
        angle = 0.55 * t
        radius = 0.35 + 0.1 * math.sin(0.25 * t)
        return {
            "name":      self.name,
            "connected": True,
            "tracking":  True,
            "x":  radius * math.cos(angle) + 0.1,
            "y":  0.95 + 0.05 * math.sin(0.3 * t),
            "z":  radius * math.sin(angle) - 0.05,
        }

    def shutdown(self) -> None:
        pass
