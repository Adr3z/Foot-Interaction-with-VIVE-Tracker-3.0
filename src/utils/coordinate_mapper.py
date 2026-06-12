"""
coordinate_mapper.py
--------------------
Converts VR world-space coordinates (metres) to Pygame screen pixels.

    +X  →  right
    +Y  →  up        (not shown on the X-Z map)
    +Z  →  backward (toward user)

Map convention (Cartesian display):
    Horizontal axis  →  X  (right = +X)
    Vertical   axis  →  Z  (up on screen = -Z, i.e. away from user)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class CoordinateMapper:
    """
    Maps world coordinates to screen pixels for a single rectangular viewport.

    Parameters
    ----------
    viewport_x, viewport_y : int
        Top-left corner of the drawable viewport in window pixels.
    viewport_w, viewport_h : int
        Pixel dimensions of the viewport.
    world_range_x : float
        Half-width of the world region to display, in metres.
        E.g. 2.0 means the map shows X ∈ [-2, +2].
    world_range_z : float
        Half-height of the world region to display, in metres.
    """

    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    world_range_x: float = 0.250
    world_range_z: float = 0.250

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def origin_px(self) -> tuple[int, int]:
        """Screen pixel position of world origin (0, 0)."""
        return (
            self.viewport_x + self.viewport_w // 2,
            self.viewport_y + self.viewport_h // 2,
        )

    @property
    def scale_x(self) -> float:
        """Pixels per metre along the X axis."""
        return self.viewport_w / (2.0 * self.world_range_x)

    @property
    def scale_z(self) -> float:
        """Pixels per metre along the Z axis."""
        return self.viewport_h / (2.0 * self.world_range_z)

    # ── public API ──────────────────────────────────────────────────────────

    def world_to_screen(self, x: float, z: float) -> tuple[int, int]:
        """
        Convert world (x, z) in metres to screen (px, py) in pixels.

        Z is negated so that +Z (toward the user) maps downward on screen,
        matching the standard Cartesian orientation where +Z-away is "up".
        """
        ox, oy = self.origin_px
        px = int(ox + x * self.scale_x)
        py = int(oy + z * self.scale_z)   
        return px, py

    def screen_to_world(self, px: int, py: int) -> tuple[float, float]:
        """Inverse mapping — useful for future interaction (e.g. click-to-place)."""
        ox, oy = self.origin_px
        x = (px - ox) / self.scale_x
        z = (py - oy) / self.scale_z
        return x, z

    def update_viewport(
        self,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
    ) -> None:
        """Call when the window is resized."""
        self.viewport_x = viewport_x
        self.viewport_y = viewport_y
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h

    def world_distance_to_pixels(self, metres: float) -> int:
        """Convert a world-space distance to an average pixel count."""
        avg_scale = (self.scale_x + self.scale_z) / 2.0
        return max(1, int(metres * avg_scale))

    def grid_spacing_pixels(self, grid_step_m: float = 0.025) -> tuple[float, float]:
        """Returns (px_per_step_x, px_per_step_z) for a given grid step in metres."""
        return self.scale_x * grid_step_m, self.scale_z * grid_step_m
