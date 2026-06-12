"""
pygame_viewer.py
----------------
    • Static elements (grid, axes, labels, panel chrome) are pre-rendered
      onto a cached Surface and blitted once per frame.
    • Only tracker markers and live text are redrawn each frame.
    • Fonts are instantiated once at startup.
"""

from __future__ import annotations

import pygame
from typing import Sequence

from ..tracker.openvr_tracker import BaseTracker
from ..utils.coordinate_mapper import CoordinateMapper


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
    AXIS_LABEL_BG   = (245, 245, 245)

    TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
        "VIVE Ultimate Tracker 1": (230, 159,   0),
        "VIVE Tracker 3.0 MV":        (  0, 114, 178),
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
    """Lightweight container updated each frame from a BaseTracker."""

    __slots__ = ("source", "color", "data", "screen_pos")

    def __init__(self, source: BaseTracker, color: tuple[int, int, int]):
        self.source     = source
        self.color      = color
        self.data: dict = {}
        self.screen_pos: tuple[int, int] = (0, 0)

    def update(self) -> None:
        self.data = self.source.get_data()


# ──────────────────────────────────────────────────────────────────────────────
#  STATIC SURFACE BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def _build_map_surface(
    viewport_w: int,
    viewport_h: int,
    mapper: CoordinateMapper,
    font_axis: pygame.font.Font,
    grid_step_m: float = 0.05,
) -> pygame.Surface:
    """
    Pre-render the map background: fill, grid, axes, metric labels.
    Returns a converted Surface ready for fast blitting.
    """
    surf = pygame.Surface((viewport_w, viewport_h))
    surf.fill(Theme.BG)

    cx = viewport_w // 2
    cy = viewport_h // 2

    step_px_x, step_px_z = mapper.grid_spacing_pixels(grid_step_m)

    # ── grid ───────────────────────────────────────────────────────────────
    x = cx % step_px_x
    while x <= viewport_w:
        pygame.draw.line(surf, Theme.GRID, (int(x), 0), (int(x), viewport_h), 1)
        x += step_px_x

    y = cy % step_px_z
    while y <= viewport_h:
        pygame.draw.line(surf, Theme.GRID, (0, int(y)), (viewport_w, int(y)), 1)
        y += step_px_z

    # ── axes ───────────────────────────────────────────────────────────────
    pygame.draw.line(surf, Theme.AXES, (0, cy), (viewport_w, cy), 2)   # Z axis
    pygame.draw.line(surf, Theme.AXES, (cx, 0), (cx, viewport_h), 2)   # X axis

    # ── metric tick labels ─────────────────────────────────────────────────
    world_range_x = mapper.world_range_x
    world_range_z = mapper.world_range_z

    # X ticks (horizontal)
    m = grid_step_m
    while m <= world_range_x:
        for sign in (-1, 1):
            val = sign * m
            px = cx + int(val * mapper.scale_x)
            if 0 <= px <= viewport_w:
                pygame.draw.line(surf, Theme.AXES, (px, cy - 5), (px, cy + 5), 1)
                label = f"{abs(val) * 100:.0f} cm" if abs(val) >= 0.01 else "0 cm"
                lbl = font_axis.render(label, True, Theme.TEXT_DIM)
                surf.blit(lbl, (px - lbl.get_width() // 2, cy + 8))
        m += grid_step_m

    # Z ticks (vertical)
    m = grid_step_m
    while m <= world_range_z:
        for sign in (-1, 1):
            val = sign * m
            py = cy + int(val * mapper.scale_z)
            if 0 <= py <= viewport_h:
                pygame.draw.line(surf, Theme.AXES, (cx - 5, py), (cx + 5, py), 1)
                label = f"{abs(val) * 100:.0f} cm" if abs(val) >= 0.01 else "0 cm"
                lbl = font_axis.render(label, True, Theme.TEXT_DIM)
                surf.blit(lbl, (cx + 8, py - lbl.get_height() // 2))
        m += grid_step_m

    # ── axis direction labels ──────────────────────────────────────────────
    margin = 10
    for text, pos in [
        ("X+", (viewport_w - margin - font_axis.size("X+")[0], cy + 6)),
        ("X−", (margin, cy + 6)),
        ("Z-", (cx + 6, margin)),
        ("Z+", (cx + 6, viewport_h - margin - font_axis.get_height())),
    ]:
        lbl = font_axis.render(text, True, Theme.AXES)
        surf.blit(lbl, pos)

    return surf.convert()


def _build_panel_surface(
    panel_x: int,
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
    WORLD_RANGE_X   = 0.250 # metres visible left/right of origin 
    WORLD_RANGE_Z   = 0.250 # metres visible above/below origin 
    PANEL_RATIO     = 0.25  # fraction of window width for the right panel

    def __init__(
        self,
        tracker_sources: Sequence[BaseTracker],
        window_size: tuple[int, int] = (1280, 720),
    ):
        self._sources      = tracker_sources
        self._window_size  = window_size
        self._running      = False

        # These are initialized in _init_pygame()
        self._screen:       pygame.Surface | None = None
        self._clock:        pygame.time.Clock | None = None
        self._mapper:       CoordinateMapper | None = None
        self._render_states: list[TrackerRenderState] = []

        # Cached static surfaces
        self._map_surf:    pygame.Surface | None = None
        self._panel_surf:  pygame.Surface | None = None

        # Fonts (created once)
        self._font_title:  pygame.font.Font | None = None
        self._font_body:   pygame.font.Font | None = None
        self._font_value:  pygame.font.Font | None = None
        self._font_axis:   pygame.font.Font | None = None
        self._font_fps:    pygame.font.Font | None = None

        # Layout rects computed in _compute_layout()
        self._map_rect:   pygame.Rect | None = None
        self._panel_rect: pygame.Rect | None = None

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
        """Poll all tracker sources and compute screen positions."""
        for rs in self._render_states:
            rs.update()
            if rs.data.get("tracking"):
                rs.screen_pos = self._mapper.world_to_screen(
                    rs.data["x"], rs.data["z"]
                )

    def render(self) -> None:
        """Blit static surfaces, then draw dynamic elements on top."""
        screen = self._screen

        # 1. Map background (pre-rendered)
        screen.blit(self._map_surf, self._map_rect.topleft)

        # 2. Tracker markers
        for rs in self._render_states:
            if rs.data.get("tracking"):
                self._draw_tracker_marker(rs)

        # 3. Panel background (pre-rendered)
        screen.blit(self._panel_surf, self._panel_rect.topleft)

        # 4. Live panel text
        self._draw_panel_text()

        # 5. FPS overlay (top-left of map)
        fps = self._clock.get_fps()
        fps_surf = self._font_fps.render(f"FPS {fps:.0f}", True, Theme.TEXT_DIM)
        screen.blit(fps_surf, (self._map_rect.x + 8, self._map_rect.y + 6))

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
        panel_w = max(240, int(win_w * self.PANEL_RATIO))
        map_w   = win_w - panel_w
        self._map_rect   = pygame.Rect(0, 0, map_w, win_h)
        self._panel_rect = pygame.Rect(map_w, 0, panel_w, win_h)

    def _build_render_states(self) -> None:
        self._render_states = []
        for src in self._sources:
            data  = src.get_data()
            name  = data.get("name", "Unknown")
            color = Theme.TRACKER_COLORS.get(name, Theme.TRACKER_DEFAULT)
            self._render_states.append(TrackerRenderState(src, color))

    def _rebuild_static_surfaces(self) -> None:
        mr = self._map_rect
        pr = self._panel_rect

        self._mapper = CoordinateMapper(
            viewport_x=mr.x, viewport_y=mr.y,
            viewport_w=mr.w, viewport_h=mr.h,
            world_range_x=self.WORLD_RANGE_X,
            world_range_z=self.WORLD_RANGE_Z,
        )

        self._map_surf = _build_map_surface(
            mr.w, mr.h, self._mapper, self._font_axis,
            grid_step_m=0.05,
        )
        self._panel_surf = _build_panel_surface(
            pr.x, pr.w, pr.h, self._font_title
        )

    # ── private: event handling ──────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.key == pygame.K_r:
                    print("Reset origin requested")
                    for rs in self._render_states:
                        rs.source.reset_origin()
            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._compute_layout(event.w, event.h)
                self._rebuild_static_surfaces()

    # ── private: drawing helpers ─────────────────────────────────────────────

    def _draw_tracker_marker(self, rs: TrackerRenderState) -> None:
        px, py = rs.screen_pos
        r  = Theme.TRACKER_RADIUS

        # Clip to map area
        if not self._map_rect.collidepoint(px, py):
            return

        # Shadow / depth ring
        pygame.draw.circle(self._screen, (200, 200, 200), (px + 2, py + 2), r)

        # Filled circle
        pygame.draw.circle(self._screen, rs.color, (px, py), r)

        # White outline
        pygame.draw.circle(self._screen, (255, 255, 255), (px, py), r,
                        Theme.TRACKER_OUTLINE)

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
            tittle = self._font_value.render("Position:", True, (75, 75, 75))
            surf.blit(tittle, (pr.x + pad, pr.y + y))
            y += tittle.get_height() + 3

            # X / Y / Z
            for axis in ("x", "y", "z"):
                val = data.get(axis, 0.0)
                self._blit_kv(axis.upper(), f"{val:+.3f} m",
                            pr.x + pad, pr.y + y, surf)
                y += self._font_value.get_height() + 3

            # Quaternion
            y += 8
            tittle = self._font_value.render("Rotation (quat):", True, (75, 75, 75))
            surf.blit(tittle, (pr.x + pad, pr.y + y))
            y += tittle.get_height() + 3

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


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _color_label(color: tuple[int, int, int]) -> str:
    """Return a human-readable colour name for known tracker colours."""
    _map = {
        (230, 159,   0): "Orange",
        (  0, 114, 178): "Blue",
    }
    return _map.get(color, f"RGB{color}")
