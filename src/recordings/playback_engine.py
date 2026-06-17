"""
playback_engine.py
-----------------

Handles playback of recorded tracker data using original timestamps.
"""

import numpy as np

class PlaybackEngine:
    def __init__(self, recording_data: dict[str, dict[str, np.ndarray]], loop: bool = False):
        self.data = recording_data
        self.tracker_names = list(recording_data.keys())
        self.loop = loop

        self._timestamps: dict[str, np.ndarray] = {}
        self.duration = 0.0

        for name, info in recording_data.items():
            raw_timestamps = np.array(info["time"], dtype=np.float64)
            if raw_timestamps.size == 0:
                self._timestamps[name] = raw_timestamps
                continue

            relative_timestamps = raw_timestamps - raw_timestamps[0]
            self._timestamps[name] = relative_timestamps
            if relative_timestamps.size > 0:
                self.duration = max(self.duration, float(relative_timestamps[-1]))

        self.current_time = 0.0
        self.is_playing = True
        self.speed = 1.0

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def stop(self) -> None:
        self.is_playing = False
        self.current_time = 0.0

    def restart(self) -> None:
        self.current_time = 0.0
        self.is_playing = True

    def seek(self, seconds: float) -> None:
        self.current_time = max(0.0, min(self.duration, self.current_time + seconds))

    def set_time(self, seconds: float) -> None:
        self.current_time = max(0.0, min(self.duration, seconds))

    def set_speed(self, new_speed: float) -> None:
        if new_speed > 0:
            self.speed = new_speed

    def update(self, delta_seconds: float) -> None:
        if not self.is_playing:
            return

        self.current_time += delta_seconds * self.speed

        if self.current_time >= self.duration:
            if self.loop and self.duration > 0.0:
                self.current_time %= self.duration
            else:
                self.current_time = self.duration
                self.is_playing = False

    def get_current_frame_data(self) -> dict[str, dict[str, np.ndarray]]:
        frame_snapshots: dict[str, dict[str, np.ndarray]] = {}

        for name in self.tracker_names:
            positions = np.asarray(self.data[name]["pos"])
            rotations = np.asarray(self.data[name]["rot"])
            timestamps = self._timestamps.get(name, np.array([], dtype=np.float64))

            if timestamps.size == 0 or positions.size == 0 or rotations.size == 0:
                frame_snapshots[name] = {
                    "name": name,
                    "tracking": False,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }
                continue

            index = int(np.searchsorted(timestamps, self.current_time, side="right") - 1)
            index = max(0, min(index, len(timestamps) - 1))
            position = positions[index]
            rotation = rotations[index]

            frame_snapshots[name] = {
                "name": name,
                "tracking": True,
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "qx": float(rotation[0]),
                "qy": float(rotation[1]),
                "qz": float(rotation[2]),
                "qw": float(rotation[3]),
            }

        return frame_snapshots
