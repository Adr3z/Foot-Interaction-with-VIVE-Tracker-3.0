"""
pygame_viewer.py
----------------
Renders three projections side by side:

    ┌─────────────┬─────────────┬─────────────┬───────────┐
    │   XZ plane  │   XY plane  │   ZY plane  │   Panel   │
    │  (top-down) │   (front)   │   (side)    │  (stats)  │
    └─────────────┴─────────────┴─────────────┴───────────┘

    • Static elements (grid, axes, labels, panel chrome) are pre-rendered
      onto a cached Surface and blitted once per frame.
    • Only tracker markers and live text are redrawn each frame.
    • Fonts are instantiated once at startup.
"""

from __future__ import annotations

import pygame
from collections import deque
from typing import Sequence

from ..tracker.openvr_tracker import BaseTracker
from ..utils.coordinate_mapper import CoordinateMapper
from .tracker_renderer import TrackerRenderer, OrientationMode


# ──────────────────────────────────────────────────────────────────────────────
#  PLANE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
# For each entry: plane_id, title, horiz_pos_label, horiz_neg_label, vert_pos_label, vert_neg_label
PLANE_DEFS: list[tuple[Plane, str, str, str, str, str]] = [
    ("XZ", "Top-Down View (XZ)", "X+", "X-", "Z-", "Z+"),
    ("XY", "Front View (XY)", "X+", "X-", "Y+", "Y-" ),
    ("ZY", "Side View (ZY)", "Z+", "Z-", "Y+", "Y-"),
]


# ──────────────────────────────────────────────────────────────────────────────
#  THEME
# ──────────────────────────────────────────────────────────────────────────────

class Theme:
    BG              = (245, 245, 245)
    GRID            = (210, 210, 210)
    AXES            = ( 80,  80,  80)
    TEXT            = ( 40,  40,  40)
    TEXT_DIM        = (130, 130, 130)
    PANEL_BG        = (232, 235, 240)
    PANEL_BORDER    = (180, 185, 195)
    DIVIDER         = (195, 200, 210)
    VIEW_BORDER     = (160, 165, 175)
    VIEW_TITLE_BG   = (220, 223, 230)

    TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
        "VIVE Ultimate Tracker 1": (230, 159,   0),
        "VIVE Tracker 3.0 MV": (  0, 114, 178),
        "VIVE Ultimate Tracker 2": (130,   0, 204),
    }
    TRACKER_DEFAULT = (100, 100, 100)

    TRACKER_RADIUS   = 10   # px, filled circle 
    TRACKER_OUTLINE  = 2    # px, white ring
    
    FONT_FAMILY = None      # None = pygame default monospace


# ──────────────────────────────────────────────────────────────────────────────
#  TRACKER RENDER STATE  (holds latest data + colour, no rendering logic)
# ──────────────────────────────────────────────────────────────────────────────

class TrackerRenderState:
    """Holds latest tracker data + per-plane screen positions"""

    __slots__ = ("source", "color", "data", "history",)

    def __init__(self, source: BaseTracker, color: tuple[int, int, int], history_length: int = 80):
        self.source     = source
        self.color      = color
        self.data: dict = {}
        self.history: deque[dict[str, float]] = deque(maxlen=history_length)

    def update(self) -> None:
        self.data = self.source.get_data()
        if self.data.get("tracking"):
            self.history.append({
                "x": self.data.get("x", 0.0),
                "y": self.data.get("y", 0.0),
                "z": self.data.get("z", 0.0),
            })


# ──────────────────────────────────────────────────────────────────────────────
#  STATIC SURFACE BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_view_surface(
    viewport_w: int,
    viewport_h: int,
    mapper: CoordinateMapper,
    font_axis: pygame.font.Font,
    font_title : pygame.font.Font,
    plane_def: tuple[Plane, str, str, str, str, str],
    grid_step_m: float = 0.05,
) -> pygame.Surface:
    """
    Pre-render one view: fill, grid, axes, metric labels, title bar.
    """

    plane_id, title, h_pos, h_neg, v_pos, v_neg = plane_def

    surf = pygame.Surface((viewport_w, viewport_h))
    surf.fill(Theme.BG)

    cx = viewport_w // 2
    cy = viewport_h // 2

    step_px_h, step_px_v = mapper.grid_spacing_pixels(grid_step_m)

    # ── grid ───────────────────────────────────────────────────────────────
    x = cx % step_px_h
    while x <= viewport_w:
        pygame.draw.line(surf, Theme.GRID, (int(x), 0), (int(x), viewport_h), 1)
        x += step_px_h

    y = cy % step_px_v
    while y <= viewport_h:
        pygame.draw.line(surf, Theme.GRID, (0, int(y)), (viewport_w, int(y)), 1)
        y += step_px_v

    # ── axes ───────────────────────────────────────────────────────────────
    pygame.draw.line(surf, Theme.AXES, (0, cy), (viewport_w, cy), 2)   # Z axis
    pygame.draw.line(surf, Theme.AXES, (cx, 0), (cx, viewport_h), 2)   # X axis

    # ── metric tick labels ─────────────────────────────────────────────────
    world_range_h = mapper.world_range_h
    world_range_v = mapper.world_range_v

    # X ticks (horizontal)
    m = grid_step_m
    while m <= world_range_h:
        for sign in (-1, 1):
            val = sign * m
            px = cx + int(val * mapper.scale_h)
            if 0 <= px <= viewport_w:
                pygame.draw.line(surf, Theme.AXES, (px, cy - 5), (px, cy + 5), 1)
                label = f"{abs(val) * 100:.0f} cm" if abs(val) >= 0.01 else "0 cm"
                lbl = font_axis.render(label, True, Theme.TEXT_DIM)
                surf.blit(lbl, (px - lbl.get_width() // 2, cy + 8))
        m += grid_step_m

    # Z ticks (vertical)
    max_v_m = (viewport_h / 2) / mapper.scale_v
    m = grid_step_m
    while m <= world_range_v:
        for sign in (-1, 1):
            val = sign * m
            py = cy + int(val * mapper.scale_v)
            if 0 <= py <= viewport_h:
                pygame.draw.line(surf, Theme.AXES, (cx - 5, py), (cx + 5, py), 1)
                label = f"{abs(val) * 100:.0f} cm" if abs(val) >= 0.01 else "0 cm"
                lbl = font_axis.render(label, True, Theme.TEXT_DIM)
                surf.blit(lbl, (cx + 8, py - lbl.get_height() // 2))
        m += grid_step_m

    # ── axis direction labels ──────────────────────────────────────────────
    margin = 10
    for text, pos in [
        (h_pos, (viewport_w - margin - font_axis.size(h_pos)[0], cy + 6)),
        (h_neg, (margin, cy + 6)),
        (v_pos, (cx + 6, margin + _TITLE_BAR_H + 4)),
        (v_neg, (cx + 6, viewport_h - margin - font_axis.get_height())),
    ]:
        lbl = font_axis.render(text, True, Theme.AXES)
        surf.blit(lbl, pos)

    # ── title bar ───────────────────────────────────────────────────────────
    title_bar = pygame.Rect(0, 0, viewport_w, _TITLE_BAR_H)
    pygame.draw.rect(surf, Theme.VIEW_TITLE_BG, title_bar)
    pygame.draw.line(surf, Theme.VIEW_BORDER, (0, _TITLE_BAR_H -1), (viewport_w, _TITLE_BAR_H -1), 1)

    t_surf = font_title.render(title, True, Theme.TEXT)
    surf.blit(t_surf, (
        (viewport_w - t_surf.get_width()) // 2,
        (_TITLE_BAR_H - t_surf.get_height()) // 2,
    ))

    # ── outer border ───────────────────────────────────────────────────────────
    pygame.draw.rect(surf, Theme.VIEW_BORDER, pygame.Rect(0, 0, viewport_w, viewport_h), 1)

    return surf.convert()

_TITLE_BAR_H = 24 # px reserved for the plane title


def _build_panel_surface(
    panel_w: int,
    win_h: int,
    font_title: pygame.font.Font,
) -> pygame.Surface:
    """
    Pre-render the right-side panel chrome (background, border, title).
    Dynamic text is drawn on top each frame.
    """
    surf = pygame.Surface((panel_w, win_h))
    surf.fill(Theme.PANEL_BG)
    pygame.draw.line(surf, Theme.PANEL_BORDER, (0, 0), (0, win_h), 2)

    title = font_title.render("Tracker Status", True, Theme.TEXT)
    surf.blit(title, (16, 16))

    line_y = 16 + title.get_height() + 10
    pygame.draw.line(surf, Theme.DIVIDER, (16, line_y), (panel_w - 16, line_y), 1)

    return surf.convert()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN VIEWER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class PygameViewer:
    """
    Pygame-based VR tracker visualizer.
    """

    TARGET_FPS      = 90
    WORLD_RANGE   = 0.250 # metres visible left/right of origin 
    PANEL_RATIO     = 0.22  # fraction of window width for the right panel
    TRAIL_LENGTH = 80

    def __init__(
        self,
        tracker_sources: Sequence[BaseTracker],
        window_size: tuple[int, int] = (1000, 1000),
    ):
        self._sources      = tracker_sources
        self._window_size  = window_size
        self._running      = False

        # These are initialized in _init_pygame()
        self._screen:       pygame.Surface | None = None
        self._clock:        pygame.time.Clock | None = None
        self._render_states: list[TrackerRenderState] = []

        # One mapper per plane
        self._mappers: dict[Plane, CoordinateMapper] = {}

        self.orientation_mode = OrientationMode.QUATERNION

        # Cached static surfaces
        self._view_surfs: dict[Plane, pygame.Surface] = {}
        self._view_rects: dict[Plane, pygame.Rect] = {}
        self._panel_surf:  pygame.Surface | None = None
        self._panel_rect: pygame.Rect     | None = None

        # Fonts (created once)
        self._font_title:  pygame.font.Font | None = None
        self._font_body:   pygame.font.Font | None = None
        self._font_value:  pygame.font.Font | None = None
        self._font_axis:   pygame.font.Font | None = None
        self._font_fps:    pygame.font.Font | None = None


    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self) -> None:
        self._init_pygame()
        self._running = True
        try:
            while self._running:
                self._handle_events()
                self.update()
                self.render()
                self._clock.tick(self.TARGET_FPS)
        finally:
            self._shutdown()

    def update(self) -> None:
        """Poll all tracker sources and compute screen positions for each plane."""
        for rs in self._render_states:
            rs.update()

    def render(self) -> None:
        """Blit static surfaces, then draw dynamic elements on top."""
        screen = self._screen

        screen.fill((210, 214, 220))

        # 1. Each view
        for plane_id, plane_def in zip(
            [p[0] for p in PLANE_DEFS], PLANE_DEFS
        ):
            rect = self._view_rects[plane_id]
            screen.blit(self._view_surfs[plane_id], rect.topleft)

            for rs in self._render_states:
                if rs.data.get("tracking"):
                    mapper = self._mappers[plane_id]

                    TrackerRenderer.draw_trail(
                        screen=self._screen,
                        history=rs.history,
                        color=rs.color,
                        mapper=mapper,
                        clip_rect=rect,
                        title_bar_height=_TITLE_BAR_H,
                    )

                    TrackerRenderer.draw(
                        screen=self._screen,
                        tracker_data=rs.data,
                        color=rs.color,
                        mapper=mapper,
                        clip_rect=rect,
                        title_bar_height=_TITLE_BAR_H,
                        orientation_mode=self.orientation_mode,
                    )

        # 2. Panel background 
        screen.blit(self._panel_surf, self._panel_rect.topleft)

        # 3. Panel text
        self._draw_panel_text()

        # 5. FPS overlay (top-left of first view)
        first_rect = self._view_rects[PLANE_DEFS[0][0]]
        fps = self._clock.get_fps()
        fps_surf = self._font_fps.render(f"FPS {fps:.0f}", True, Theme.TEXT_DIM)
        screen.blit(fps_surf, (first_rect.x + 8, first_rect.y + _TITLE_BAR_H + 4))

        pygame.display.flip()

    # ── private: initialization ──────────────────────────────────────────────

    def _init_pygame(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode(
            self._window_size, pygame.RESIZABLE
        )
        pygame.display.set_caption("VR Tracker Visualizer")
        self._clock = pygame.time.Clock()

        self._init_fonts()
        self._compute_layout(*self._window_size)
        self._build_render_states()
        self._rebuild_static_surfaces()

    def _init_fonts(self) -> None:
        fam = Theme.FONT_FAMILY
        self._font_title  = pygame.font.SysFont(fam, 30)
        self._font_body   = pygame.font.SysFont(fam, 25)
        self._font_value  = pygame.font.SysFont(fam, 21)
        self._font_axis   = pygame.font.SysFont(fam, 16)
        self._font_fps    = pygame.font.SysFont(fam, 16)

    def _compute_layout(self, win_w: int, win_h: int) -> None:
        GAP = 12
        panel_w = max(240, int(win_w * self.PANEL_RATIO))
        views_w = win_w - panel_w - (GAP * 4)
        view_w  = views_w // 3

        view_widths = [view_w, view_w, views_w -2 * view_w]

        x = GAP
        for i, (plane_id, *_) in enumerate(PLANE_DEFS):
            self._view_rects[plane_id] = pygame.Rect(x, GAP, view_w, win_h - (GAP * 2))
            x += view_w + GAP

        self._panel_rect = pygame.Rect(x, 0, panel_w, win_h)

    def _build_render_states(self) -> None:
        self._render_states = []
        for src in self._sources:
            data  = src.get_data()
            name  = data.get("name", "Unknown")
            color = Theme.TRACKER_COLORS.get(name, Theme.TRACKER_DEFAULT)
            self._render_states.append(
                TrackerRenderState(src, color, history_length=self.TRAIL_LENGTH)
            )

    def _rebuild_static_surfaces(self) -> None:
        for plane_def in PLANE_DEFS:
            plane_id = plane_def[0]
            rect = self._view_rects[plane_id]

            mapper = CoordinateMapper(
                viewport_x = rect.x,
                viewport_y = rect.y,
                viewport_w = rect.w,
                viewport_h = rect.h,
                world_range_h = self.WORLD_RANGE,
                world_range_v = self.WORLD_RANGE,
                plane = plane_id,
            )
            self._mappers[plane_id] = mapper

            self._view_surfs[plane_id] = _build_view_surface(
                rect.w, rect.h,
                mapper,
                self._font_axis,
                self._font_title,
                plane_def,
                grid_step_m = 0.05
            )

        self._panel_surf = _build_panel_surface(
            self._panel_rect.w,
            self._panel_rect.h,
            self._font_title
        )

    # ── private: event handling ──────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.key == pygame.K_c:
                    print("Reset origin requested")
                    for rs in self._render_states:
                        rs.source.reset_origin()
                if event.key == pygame.K_o:
                    if self.orientation_mode == OrientationMode.QUATERNION:
                        self.orientation_mode = OrientationMode.EULER
                    else:
                        self.orientation_mode = OrientationMode.QUATERNION
                    print(f"Orientation Mode: " 
                            f"{self.orientation_mode.name}")
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._compute_layout(event.w, event.h)
                self._rebuild_static_surfaces()

    # ── private: drawing helpers ─────────────────────────────────────────────

    def _draw_panel_text(self) -> None:
        pr   = self._panel_rect
        surf = self._screen
        pad  = 18

        # Starting Y below the pre-rendered title + divider
        y = 16 + self._font_title.get_height() + 20

        for i, rs in enumerate(self._render_states):
            data = rs.data
            if not data:
                continue

            name       = data.get("name", "Unknown")
            connected  = data.get("connected", False)
            tracking   = data.get("tracking", False)

            # Tracker name (coloured)
            name_surf = self._font_body.render(name, True, rs.color)
            surf.blit(name_surf, (pr.x + pad, pr.y + y))
            y += name_surf.get_height() + 6

            # Color label
            # color_name = _color_label(rs.color)
            # self._blit_kv("Color",     color_name,
            #               pr.x + pad, pr.y + y, surf)
            # y += self._font_value.get_height() + 3

            # Connected
            conn_str = "Yes" if connected else "No"
            self._blit_kv("Connected", conn_str,
                        pr.x + pad, pr.y + y, surf)
            y += self._font_value.get_height() + 3

            # Tracking
            track_str = "Active" if tracking else "Lost"
            self._blit_kv("Tracking",  track_str,
                        pr.x + pad, pr.y + y, surf)
            y += self._font_value.get_height() + 10

            # Position
            title = self._font_value.render("Position:", True, (75, 75, 75))
            surf.blit(title, (pr.x + pad, pr.y + y))
            y += title.get_height() + 3

            # X / Y / Z
            for axis in ("x", "y", "z"):
                val = data.get(axis, 0.0)
                self._blit_kv(axis.upper(), f"{val:+.3f} m",
                            pr.x + pad, pr.y + y, surf)
                y += self._font_value.get_height() + 3

            # Quaternion
            y += 8
            title = self._font_value.render("Rotation (quat):", True, (75, 75, 75))
            surf.blit(title, (pr.x + pad, pr.y + y))
            y += title.get_height() + 3

            # Quaternion
            for axis in ("qx", "qy", "qz", "qw"):
                val = data.get(axis, 0.0)
                self._blit_kv(axis.upper(), f"{val:+.3f}",
                            pr.x + pad, pr.y + y, surf)
                y += self._font_value.get_height() + 3

            y += 8

            # Divider between trackers
            if i < len(self._render_states) :
                pygame.draw.line(
                    surf, Theme.DIVIDER,
                    (pr.x + pad, pr.y + y),
                    (pr.x + pr.w - pad, pr.y + y), 1
                )
                y += 16

    def _blit_kv(
        self,
        key: str,
        value: str,
        x: int,
        y: int,
        surf: pygame.Surface,
    ) -> None:
        """Render a key: value pair with distinct styling."""
        k_surf = self._font_value.render(f"{key}:", True, Theme.TEXT_DIM)
        v_surf = self._font_value.render(value,     True, Theme.TEXT)
        surf.blit(k_surf, (x, y))
        surf.blit(v_surf, (x + 88, y))

    # ── private: shutdown ────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        for rs in self._render_states:
            rs.source.shutdown()
        pygame.quit()
