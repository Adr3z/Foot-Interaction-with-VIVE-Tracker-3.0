"""
coordinate_mapper.py
--------------------
Converts VR world-space coordinates (metres) to Pygame screen pixels.

Coordinate conventions
    +X  →  right
    +Y  →  up
    +Z  →  backward (toward user)

Projections rendered:
    XZ plane - top-down view (horiz=X, vert=Z)
    XY plane - front-on view (horiz=X, vert=Y)
    ZY plane - right-side view (horiz=Z, vert=Y)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

Plane = Literal["XZ", "XY", "ZY"]


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
    world_range_h : float
        Half-width of the world region shown in the horizontal screen axis.
    world_range_v : float
        Half-height of the world region shown on the vertical screen axis.
    plane : Plane
        Which plane of the world to project onto the screen.
    """

    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    world_range_h: float = 0.300
    world_range_v: float = 0.300
    plane: Plane = "XZ"

    # ── derived properties ──────────────────────────────────────────────────

    @property
    def origin_px(self) -> tuple[int, int]:
        """Screen pixel position of world origin (0, 0)."""
        return (
            self.viewport_x + self.viewport_w // 2,
            self.viewport_y + self.viewport_h // 2,
        )

    @property
    def scale_h(self) -> float:
        """Pixels per metre along the horizontal axis."""
        return self.viewport_w / (2.0 * self.world_range_h)

    @property
    def scale_v(self) -> float:
        """Pixels per metre along the vertical axis."""
        return self.scale_h

    # ── public API ──────────────────────────────────────────────────────────

    def world_to_screen(self, x: float, y: float, z: float) -> tuple[int, int]:
        """
        Convert world (x, y, z) in metres to screen (px, py) in pixels according to this mapper's plane.

        Screen Y is inverted for XY and ZY planes so that +Y maps upward.
        """
        ox, oy = self.origin_px
        
        if self.plane == "XZ":
            # horiz = X, vert = Z 
            px = int(ox + x* self.scale_h)
            py = int(oy +z * self.scale_v)
        elif self.plane == "XY":
            # horiz = X, vert = Y (invert Y for screen)
            px = int(ox + x * self.scale_h)
            py = int(oy - y * self.scale_v)
        elif self.plane == "ZY":
            # horiz = Z, vert = Y
            px = int(ox + z * self.scale_h)
            py = int(oy - y * self.scale_v)
        else:
            raise ValueError(f"Invalid plane: {self.plane}")

        return px, py

    def screen_to_world(self, px: int, py: int) -> tuple[float, float]:
        """ Mapping for the two axes of this plane. Returns (horiz, vert) in metres. """
        ox, oy = self.origin_px
        h = (px - ox) / self.scale_h
        v = (py - oy) / self.scale_v

        if self.plane in ("XY", "ZY"):
            v= -v
        return h, v

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
        avg_scale = (self.scale_h + self.scale_v) / 2.0
        return max(1, int(metres * avg_scale))

    def grid_spacing_pixels(self, grid_step_m: float = 0.025) -> tuple[float, float]:
        """Returns (px_per_step_x, px_per_step_z) for a given grid step in metres."""
        return self.scale_h * grid_step_m, self.scale_v * grid_step_m
