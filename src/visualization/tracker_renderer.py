"""
tracker_renderer.py
----------------

"""

from __future__ import annotations
import math
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R
import pygame

from ..utils.coordinate_mapper import CoordinateMapper

from enum import Enum, auto

class TrackerRenderer:
    """
    Tracker Drawing logic
    """

    AIRCRAFT_VERTICES = np.array([
        [ 0.000, 0.0, -0.020],  # nariz
        [-0.012, 0.0,  0.006],  # ala izquierda
        [ 0.000, 0.0,  0.012],  # cola
        [ 0.012, 0.0,  0.006],  # ala derecha
    ])

    AIRCRAFT_VERTICES_2D = np.array([
        [  0, -20],   # nariz
        [-12,   6],   # ala izquierda
        [  0,  12],   # cola
        [ 12,   6],   # ala derecha
    ])

    SHADOW_COLOR = (180, 180, 180)
    OUTLINE_COLOR = (255, 255, 255)
    TRAIL_RADIUS = 4
    TRAIL_MIN_ALPHA = 40

    @classmethod
    def draw_trail(
        cls,
        screen: pygame.Surface,
        history: deque[dict[str, float]],
        color: tuple[int, int, int],
        mapper: CoordinateMapper,
        clip_rect: pygame.Rect,
        title_bar_height: int,
    ) -> None:
        """Draw a fading trail of recent tracker positions in this projection."""
        if not history:
            return

        active_rect = pygame.Rect(
            clip_rect.x,
            clip_rect.y + title_bar_height,
            clip_rect.w,
            clip_rect.h - title_bar_height,
        )

        max_range = max(mapper.world_range_h, mapper.world_range_v)
        count = len(history)
        for index, sample in enumerate(history):
            x = sample.get("x", 0.0)
            y = sample.get("y", 0.0)
            z = sample.get("z", 0.0)
            px, py = mapper.world_to_screen(x, y, z)

            if not active_rect.collidepoint(px, py):
                continue

            distance = math.sqrt(x * x + y * y + z * z)
            distance_fraction = min(1.0, distance / max_range)
            age_fraction = index / max(1, count - 1)

            fade_factor = age_fraction * (1.0 - distance_fraction)
            alpha = int(cls.TRAIL_MIN_ALPHA + (255 - cls.TRAIL_MIN_ALPHA) * fade_factor)
            alpha = max(cls.TRAIL_MIN_ALPHA, min(255, alpha))

            trail_color = (*color, alpha)
            point_surf = pygame.Surface(
                (cls.TRAIL_RADIUS * 2 + 2, cls.TRAIL_RADIUS * 2 + 2),
                pygame.SRCALPHA,
            )
            pygame.draw.circle(
                point_surf,
                trail_color,
                (cls.TRAIL_RADIUS + 1, cls.TRAIL_RADIUS + 1),
                cls.TRAIL_RADIUS,
            )
            screen.blit(
                point_surf,
                (px - cls.TRAIL_RADIUS - 1, py - cls.TRAIL_RADIUS - 1),
            )

    @classmethod
    def draw(
        cls,
        screen: pygame.Surface,
        tracker_data: dict,
        color: tuple[int, int, int],
        mapper: CoordinateMapper,
        clip_rect: pygame.Rect,
        title_bar_height: int,
        orientation_mode: OrientationMode,
    ) -> None:

        x = tracker_data.get("x", 0.0)
        y = tracker_data.get("y", 0.0)
        z = tracker_data.get("z", 0.0)

        px, py = mapper.world_to_screen(x, y, z)

        active_rect = pygame.Rect(
            clip_rect.x,
            clip_rect.y + title_bar_height,
            clip_rect.w,
            clip_rect.h - title_bar_height,
        )

        if not active_rect.collidepoint(px, py):
            return

        if orientation_mode == OrientationMode.QUATERNION:
            cls._draw_quaternion(screen, tracker_data, color, mapper)
        else:
            cls._draw_euler(screen, tracker_data, color, mapper)

        return

    @classmethod
    def _draw_quaternion(
                        cls, screen: pygame.Surface,
                        tracker_data: dict,
                        color: tuple[int, int, int],
                        mapper: CoordinateMapper,
    ) -> None:
        x = tracker_data.get("x", 0.0)
        y = tracker_data.get("y", 0.0)
        z = tracker_data.get("z", 0.0)

        qx = tracker_data.get("qx", 0.0)
        qy = tracker_data.get("qy", 0.0)
        qz = tracker_data.get("qz", 0.0)
        qw = tracker_data.get("qw", 1.0)

        try:
            rotation = R.from_quat([qx, qy, qz, qw])
        except Exception:
            rotation = R.identity()

        vertices_rotated = rotation.apply(cls.AIRCRAFT_VERTICES)

        vertices_world = vertices_rotated + np.array([
            x,
            y,
            z,
        ])

        screen_points = [
            mapper.world_to_screen(v[0], v[1], v[2])
            for v in vertices_world
        ]

        shadow_offset = 2

        shadow_points = [
            (p[0] + shadow_offset, p[1] + shadow_offset)
            for p in screen_points
        ]

        pygame.draw.polygon(
            screen,
            cls.SHADOW_COLOR,
            shadow_points,
        )

        pygame.draw.polygon(
            screen,
            color,
            screen_points,
        )

        # pygame.draw.polygon(
        #     screen,
        #     cls.OUTLINE_COLOR,
        #     screen_points,
        #     2,
        # )

    @classmethod
    def _draw_euler(
        cls,
        screen: pygame.Surface,
        tracker_data: dict,
        color: tuple[int, int, int],
        mapper: CoordinateMapper,
    ) -> None:

        x = tracker_data.get("x", 0.0)
        y = tracker_data.get("y", 0.0)
        z = tracker_data.get("z", 0.0)

        px, py = mapper.world_to_screen(x, y, z)

        qx = tracker_data.get("qx", 0.0)
        qy = tracker_data.get("qy", 0.0)
        qz = tracker_data.get("qz", 0.0)
        qw = tracker_data.get("qw", 1.0)

        # Roll (X), Pitch (Y), Yaw (Z)
        try:
            rot = R.from_quat([qx, qy, qz, qw])
            roll, pitch, yaw = rot.as_euler("xyz", degrees=False)
        except Exception:
            roll = pitch = yaw = 0.0

        # Select angle to each view

        if mapper.plane == "XZ":
            angle = -yaw

        elif mapper.plane == "XY":
            angle = roll

        elif mapper.plane == "ZY":
            angle = pitch

        else:
            angle = 0.0

        # marker
        vertices = cls.AIRCRAFT_VERTICES_2D

        c = np.cos(angle)
        s = np.sin(angle)

        rot2d = np.array([
            [ c, -s],
            [ s,  c],
        ])

        vertices_rot = vertices @ rot2d.T

        # Trasladar al centro del tracker
        screen_points = [
            (
                int(px + v[0]),
                int(py + v[1]),
            )
            for v in vertices_rot
        ]

        # Shadow
        shadow_offset = 2

        shadow_points = [
            (
                p[0] + shadow_offset,
                p[1] + shadow_offset,
            )
            for p in screen_points
        ]

        pygame.draw.polygon(
            screen,
            cls.SHADOW_COLOR,
            shadow_points,
        )

        # Body shape
        pygame.draw.polygon(
            screen,
            color,
            screen_points,
        )

        # Contorno
        # pygame.draw.polygon(
        #     screen,
        #     cls.OUTLINE_COLOR,
        #     screen_points,
        #     2,
        # )

class OrientationMode(Enum):
    QUATERNION = auto()
    EULER = auto()