"""
recorder.py
-----------------

Handles recording tracker motion data and exporting it to NPZ files
"""

from __future__ import annotations

import time 
import numpy as np
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..visualization import TrackerRenderState

class TrackerRecorder:
    """
    Records position and rotation data from active trackers and saves the captured motion as a compressed NPZ file.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """ Initialize the recorder.
            Parameters:
            output_dir: str
                Directory where recording will be stored
        """
        self.output_dir = output_dir
        self.is_recording = False
        self._buffer: dict[TrackerRenderState, list] = {}

    def toggle_recording(self, render_states: list[TrackerRenderState]) -> None:
    def toggle_recording(self, render_states: list[TrackerRenderState]) -> None:
        """Start recording if idle, otherwise stop the current recording"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording(render_states)        """Begin a new recording session and create an empty buffer for each tracker.
        
            Parameters:
            render_states: list[TrackerRenderState]
                Active tracker render state
        """
        self.is_recording = True
        self._buffer = {rs: [] for rs in render_states}
        print("Recording started")

    def sample(self, render_states: list[TrackerRenderState]) -> None:
        """Capture a snapshot of each tracked device and store it in the recording buffer"""
        if not self.is_recording:
            return

        for rs in render_states:
            if rs.data.get("tracking") and rs in self._buffer:
                self._buffer[rs].append({
                    "time": time.time(),
                    "pos": [rs.data.get("x", 0.0), rs.data.get("y", 0.0), rs.data.get("z", 0.0)],
                    "rot": [rs.data.get("qx", 0.0), rs.data.get("qy", 0.0), rs.data.get("qz", 0.0), rs.data.get("qw", 1.0)]
                })

    def stop_recording(self) -> None:
        """End the current recording session and export the data."""
        if not self.is_recording:
            return

        self.is_recording = False
        print("Recording finished")
        self._save()

    def _save(self) -> None:
        """ Save all recorded tracker data into a compressed NPZ file.
            Each tracker contributes:
                -timestamps
                -positions
                -rotation(quaternion)
        """
        if not self._buffer:
            print("There's no data to save")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        save_dict = {}
        has_valid_data = False

        for idx, (rs, frames) in enumerate(self._buffer.items()):
            # Skip trackers with no recorded samples
            if not frames:
                continue

            has_valid_data = True
            tracker_name = rs.data.get("name", f"Tracker_{idx}").replace(" ", "_")

            # Convert recorded samples to numpy arrays
            tracker_name = f"{rs.data.get('name', 'Tracker')}_{idx}".replace(" ", "_")

            # Convert recorded samples to numpy arrays
            timestamps = np.array([f["time"] for f in frames], dtype = np.float64)
            positions = np.array([f["pos"] for f in frames], dtype = np.float32)
            rotations = np.array([f["rot"] for f in frames], dtype = np.float32)

            # Store tracker data using descriptive keys
            save_dict[f"{tracker_name}_timestamps"] = timestamps
            save_dict[f"{tracker_name}_position"] = positions
            save_dict[f"{tracker_name}_rotation"] = rotations            np.savez_compressed(filename, **save_dict)
            print(f"Gesture exported to: {filename}")
        else:
            print("Recording canceled. There weren't valid frames to export")

        self._buffer.clear()