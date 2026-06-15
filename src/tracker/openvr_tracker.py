"""
openvr_tracker.py
-----------------

    OpenVRSession  — manage openvr init/shutdown (a single global init).
    OpenVRTracker  — implements BaseTracker for a single device index (controller/tracker).

The viewer calls a tracker.get_data() every frame. 
Each OpenVRTracker reads the pose of its device_index directly from the shared VRSystem.
"""

from __future__ import annotations

import sys
import openvr
import numpy as np
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────
#  BASE INTERFACE  (contract for all data sources)
# ──────────────────────────────────────────────

class BaseTracker(ABC):
    """Abstract base class. OpenVR reader must implement this."""

    @abstractmethod
    def get_data(self) -> dict:
        """
        Returns a dict with the current tracker state.

        Schema:
            {
                "name":      str,   # display name
                "connected": bool,  # device found by runtime
                "tracking":  bool,  # pose is valid
                "x":         float, # metres, world space
                "y":         float,
                "z":         float,
                "qx":        float, # rotation as quaternion
                "qy":        float,
                "qz":        float,
                "qw":        float,
            }
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release any resources held by the tracker source."""

# ──────────────────────────────────────────────────────────────────────────────
#  OPENVR TRACKER  (One instance per tracked device))
# ──────────────────────────────────────────────────────────────────────────────

class OpenVRTracker(BaseTracker):
    """
    Represents a single physical tracker device connected to SteamVR.
    """

    def __init__(
        self,
        vr_system: openvr.IVRSystem,
        device_index: int,
        name: str,
        origin: np.ndarray | None = None,
    ):
        self._vr          = vr_system
        self._device_index = device_index
        self.name         = name
        self.origin       = origin

    def get_data(self) -> dict:
        """
        Reads the current pose of the assigned device index and returns the standard schema.
        Called every frame by the viewer (~90 Hz).
        """
        # Capturar las poses de todos los dispositivos en un solo batch
        poses = self._vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount,
        )

        pose = poses[self._device_index]

        # Si el tracking no es válido, reportar pero no crashear
        if not pose.bPoseIsValid:
            return {
                "name":      self.name,
                "connected": True,   # el dispositivo existe en SteamVR
                "tracking":  False,  # pero la pose no es confiable
                # Position
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                # Rotation (quaternion)
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
                # Battery
                "battery": self._read_battery(),
            }

        # Extraer traslación de la matriz 3×4
        matrix = pose.mDeviceToAbsoluteTracking

        matrix_np = np.array([list(row) for row in matrix], dtype=np.float64)

        # Homogeneous matrix
        homogeneous_matrix = np.append(matrix_np, [[0, 0, 0, 1]], axis=0)
        
        # Reset origin if not set
        if self.origin is not None:
            final_matrix = self.origin @ homogeneous_matrix
        else:
            final_matrix = homogeneous_matrix

        rotation_matrix = final_matrix[:3,:3]
        position = final_matrix[:3, 3]

        ROT = R.from_matrix(rotation_matrix)
        quat = ROT.as_quat()  # (x, y, z, w)

        return {
            "name":      self.name,
            "connected": True,
            "tracking":  True,
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "qx": quat[0],
            "qy": quat[1],
            "qz": quat[2],
            "qw": quat[3],
            "battery": self._read_battery(),
        }

    def _read_battery(self) -> str:
        """Returns battery percentage as a string, or 'N/A' if not available."""
        try:
            value = self._vr.getFloatTrackedDeviceProperty(
                self._device_index,
                openvr.Prop_DeviceBatteryPercentage_Float,
            )
            return f"{int(value * 100)}%"
        except Exception:
            return "N/A"

    def reset_origin(self) -> None:
        """Sets the current position as the new origin"""
        poses = self._vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        pose = poses[self._device_index]

        if pose.bPoseIsValid:
            matrix = pose.mDeviceToAbsoluteTracking
            matrix_np = np.array([list(row) for row in matrix], dtype=np.float64)

            print(f"Origin reset: {self.name} at position ({matrix_np[0,3]:.2f}, {matrix_np[1,3]:.2f}, {matrix_np[2,3]:.2f})\n with rotation matrix:\n{matrix_np[:3,:3]}")
            current_homogeneous = np.append(matrix_np, [[0, 0, 0, 1]], axis=0)
            self.origin = np.linalg.inv(current_homogeneous)



    def shutdown(self) -> None:
        """Nothing"""
        pass


# ──────────────────────────────────────────────────────────────────────────────
#  OPENVR SESSION  (manages the global lifecycle of the openvr connection)
# ──────────────────────────────────────────────────────────────────────────────

class OpenVRSession:
    """
    Context manager that initializes SteamVR once and exposes detected trackers as OpenVRTracker objects.
    """

    def __init__(self):
        self._vr_system: openvr.IVRSystem | None = None

    # ── Context ──────────────────────────────────────────────────────────────

    def __enter__(self) -> "OpenVRSession":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Conection ──────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """
        Initializes SteamVR in Background mode.
        """
        try:
            self._vr_system = openvr.init(openvr.VRApplication_Background)
            print("=" * 50)
            print("Successfully connected to SteamVR!")
            print("=" * 50)
        except openvr.OTFError as e:
            print("Critical Error. Make sure SteamVR is open.")
            sys.exit(1)

    def disconnect(self) -> None:
        """Close the connection to SteamVR. Call when the viewer is closed."""
        openvr.shutdown()
        print("\n" + "=" * 50)
        print("Conection closed cleanly.")
        print("=" * 50)

    # ── Tracker Detection ─────────────────────────────────────────────────

    def get_trackers(self) -> list[OpenVRTracker]:
        """
        Scans every device slot and returns an OpenVRTracker for each detected tracker/controller.
        """
        if self._vr_system is None:
            raise RuntimeError("Call connect() before get_trackers().")

        trackers: list[OpenVRTracker] = []
        name_counts: dict[str, int] = {}  # for IDs with duplicate model names

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = self._vr_system.getTrackedDeviceClass(i)

            if device_class not in (
                openvr.TrackedDeviceClass_Controller,
                openvr.TrackedDeviceClass_GenericTracker,
            ):
                continue

            # Read the model name
            raw_model = self._get_string_property(
                i, openvr.Prop_ModelNumber_String
            )

            display_name = raw_model

            # Add a sufix if the model is duplicated (1), (2)...
            count = name_counts.get(display_name, 0) + 1
            name_counts[display_name] = count
            if count > 1:
                display_name = f"{display_name} ({count})"

            tracker = OpenVRTracker(self._vr_system, i, display_name)
            trackers.append(tracker)
            print(f"  Tracker → ID {i:2d}  |  {display_name}  |  model: {raw_model}  |  battery: {tracker._read_battery()}")

        if not trackers:
            print("No trackers found.")

        return trackers

    def _get_string_property(self, device_index: int, prop: int) -> str:
        """Reads a string property from the VR system, with error handling."""
        try:
            return self._vr_system.getStringTrackedDeviceProperty(device_index, prop)
        except Exception:
            return ""
