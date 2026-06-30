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

Runtime gesture classification keybindings:
    G       toggle classification on / off
    W / S   window size  ± 15 frames
    E / Q   step size    ±  5 frames
    Z / X   debounce     ±  1 vote
"""

from __future__ import annotations

import os
import datetime
import pygame
from collections import deque
from typing import Sequence

from .theme import Theme
from ..tracker import BaseTracker
from ..utils import CoordinateMapper
from .tracker_renderer import TrackerRenderer, OrientationMode, TrackerRenderState
from ..recordings import TrackerRecorder
from .view_utils import (PLANE_DEFS, TITLE_BAR_H, build_view_surface, build_plane_views, render_trackers_on_views, toggle_orientation)
from ml import RealtimeGestureClassifier


# Resolved at import time so it works from any working directory
_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ml", "models", "svm_model_final.pkl")
)

# Per-gesture accent colours (mirror tracker palette so 1→orange, 2→blue, 3→purple)
_GESTURE_COLORS = {
    1: (230, 159,   0),
    2: ( 40,  90, 230),
    3: (130,   0, 204),
}
_GESTURE_NAMES = {1: "One", 2: "Two", 3: "Three"}


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

    return surf.convert()


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN VIEWER CLASS
# ──────────────────────────────────────────────────────────────────────────────

class PygameViewer:
    """
    Pygame-based VR tracker visualizer with real-time gesture classification.
    """

    TARGET_FPS    = 90
    WORLD_RANGE   = 0.250  # metres visible left/right of origin
    PANEL_RATIO   = 0.22   # fraction of window width for the right panel
    TRAIL_LENGTH  = 80

    # ── gesture classifier defaults (adjustable at runtime via W/S, E/Q, Z/X) ──
    PRED_WINDOW   = 105   # rolling buffer size in frames   (W / S)
    PRED_STEP     = 20   # frames between prediction calls (E / Q)
    PRED_DEBOUNCE = 4   # consecutive matches to confirm  (Z / X)
    PRED_COOLDOWN = 2   # prediction steps blocked after a confirmed gesture
    PRED_ALERT_FRAMES = 110  # how many frames the alert overlay lasts

    def __init__(
        self,
        tracker_sources: Sequence[BaseTracker],
        window_size: tuple[int, int] = (1000, 1000),
    ):
        self._sources      = tracker_sources
        self._window_size  = window_size
        self._running      = False

        # These are initialized in _init_pygame()
        self._screen:        pygame.Surface | None = None
        self._clock:         pygame.time.Clock | None = None
        self._render_states: list[TrackerRenderState] = []

        # One mapper per plane
        self._mappers: dict[Plane, CoordinateMapper] = {}

        self.orientation_mode = OrientationMode.QUATERNION

        # Cached static surfaces
        self._view_surfs: dict[Plane, pygame.Surface] = {}
        self._view_rects: dict[Plane, pygame.Rect] = {}
        self._panel_surf:  pygame.Surface | None = None
        self._panel_rect:  pygame.Rect | None = None

        # Runtime options
        self.show_trail = True

        # Recording
        self._recorder = TrackerRecorder()

        # Gesture classification
        self._classification_enabled: bool = True
        self._classifier: RealtimeGestureClassifier | None = None
        self._current_gesture: int | None = None
        self._gesture_history: deque[tuple[str, int]] = deque(maxlen=10)
        self._alert_timer: int = 0
        self._alert_label: int | None = None

        # Fonts (created once)
        self._font_title:   pygame.font.Font | None = None
        self._font_body:    pygame.font.Font | None = None
        self._font_value:   pygame.font.Font | None = None
        self._font_axis:    pygame.font.Font | None = None
        self._font_fps:     pygame.font.Font | None = None
        self._font_large:   pygame.font.Font | None = None
        self._font_section: pygame.font.Font | None = None
        self._font_gesture: pygame.font.Font | None = None
        self._font_alert:   pygame.font.Font | None = None


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
        """Poll all tracker sources, compute screen positions, and run classification."""
        for rs in self._render_states:
            rs.update()

        self._recorder.sample(self._render_states)

        # Feed the first actively-tracked source into the gesture classifier
        if self._classifier is not None and self._classification_enabled:
            for rs in self._render_states:
                if rs.data.get("tracking"):
                    confirmed = self._classifier.add_sample(
                        rs.data.get("x", 0.0),
                        rs.data.get("y", 0.0),
                        rs.data.get("z", 0.0),
                    )
                    if confirmed is not None:
                        self._current_gesture = confirmed
                        ts = datetime.datetime.now().strftime("%H:%M:%S")
                        self._gesture_history.appendleft((ts, confirmed))
                        self._alert_timer = self.PRED_ALERT_FRAMES
                        self._alert_label = confirmed
                    break  # classify only the first tracking source

    def render(self) -> None:
        """Blit static surfaces, then draw dynamic elements on top."""
        screen = self._screen

        screen.fill((210, 214, 220))

        # 1. Each view
        render_trackers_on_views(
            screen=self._screen,
            tracker_states=self._render_states,
            view_rects=self._view_rects,
            view_surfs=self._view_surfs,
            mappers=self._mappers,
            show_trail=self.show_trail,
            orientation_mode=self.orientation_mode,
        )

        # 2. Panel background
        screen.blit(self._panel_surf, self._panel_rect.topleft)

        # 3. Panel text
        self._draw_panel_text()

        # 4. Gesture alert overlay (fades out over PRED_ALERT_FRAMES)
        self._render_alert()

        # 5. FPS overlay (top-left of first view)
        first_rect = self._view_rects[PLANE_DEFS[0][0]]
        fps = self._clock.get_fps()
        fps_surf = self._font_fps.render(f"FPS {fps:.0f}", True, Theme.TEXT_DIM)
        screen.blit(fps_surf, (first_rect.x + 8, first_rect.y + TITLE_BAR_H + 4))

        pygame.display.flip()

    # ── initialization ───────────────────────────────────────────────────────

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
        self._load_classifier()

    def _init_fonts(self) -> None:
        fam = Theme.FONT_FAMILY
        self._font_title   = pygame.font.SysFont(fam, 30)
        self._font_body    = pygame.font.SysFont(fam, 18)
        self._font_value   = pygame.font.SysFont(fam, 21)
        self._font_axis    = pygame.font.SysFont(fam, 16)
        self._font_fps     = pygame.font.SysFont(fam, 16)
        self._font_large   = pygame.font.SysFont(fam, 16)
        self._font_section = pygame.font.SysFont(fam, 21)
        self._font_gesture = pygame.font.SysFont(fam, 18)
        self._font_alert   = pygame.font.SysFont(fam, 34, bold=True)

    def _compute_layout(self, win_w: int, win_h: int) -> None:
        GAP = 12
        panel_w = max(240, int(win_w * self.PANEL_RATIO))
        views_w = win_w - panel_w - (GAP * 4)
        view_w  = views_w // 3

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
        self._mappers, self._view_surfs = build_plane_views(
            view_rects=self._view_rects,
            world_range=self.WORLD_RANGE,
            font_axis=self._font_axis,
            font_title=self._font_title,
        )

        self._panel_surf = _build_panel_surface(
            self._panel_rect.w,
            self._panel_rect.h,
            self._font_title,
        )

    def _load_classifier(self) -> None:
        try:
            self._classifier = RealtimeGestureClassifier(
                _MODEL_PATH,
                window_size=self.PRED_WINDOW,
                step_size=self.PRED_STEP,
                debounce_count=self.PRED_DEBOUNCE,
                cooldown_frames=self.PRED_COOLDOWN,
            )
            print(f"Gesture classifier loaded  (window={self.PRED_WINDOW}  step={self.PRED_STEP}  debounce={self.PRED_DEBOUNCE}  cooldown={self.PRED_COOLDOWN})")
        except FileNotFoundError:
            print(f"[Warning] No SVM model at {_MODEL_PATH}. Gesture classification disabled.")
            self._classifier = None
        except Exception as exc:
            print(f"[Warning] Could not load gesture classifier: {exc}. Gesture classification disabled.")
            self._classifier = None
    # ── event handling ───────────────────────────────────────────────────────

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False

                elif event.key == pygame.K_1:
                    self.show_trail = not self.show_trail
                    print(f"Trail {'enabled' if self.show_trail else 'disabled'}")

                elif event.key == pygame.K_c:
                    print("Reset origin requested")
                    for rs in self._render_states:
                        rs.source.reset_origin()

                elif event.key == pygame.K_o:
                    self.orientation_mode = toggle_orientation(self.orientation_mode)
                    print(f"Orientation Mode: {self.orientation_mode.name}")

                elif event.key == pygame.K_r:
                    self._recorder.toggle_recording(self._render_states)

                # ── classifier config ──────────────────────────────────────
                elif event.key == pygame.K_g and self._classifier:
                    self._classification_enabled = not self._classification_enabled
                    state = "enabled" if self._classification_enabled else "disabled"
                    print(f"Classification {state}")

                elif event.key == pygame.K_w and self._classifier:
                    self._classifier.resize_window(self._classifier.window_size + 15)
                    print(f"Window size → {self._classifier.window_size}")

                elif event.key == pygame.K_s and self._classifier:
                    self._classifier.resize_window(self._classifier.window_size - 15)
                    print(f"Window size → {self._classifier.window_size}")

                elif event.key == pygame.K_e and self._classifier:
                    self._classifier.step_size = max(1, self._classifier.step_size + 5)
                    print(f"Step size   → {self._classifier.step_size}")

                elif event.key == pygame.K_q and self._classifier:
                    self._classifier.step_size = max(1, self._classifier.step_size - 5)
                    print(f"Step size   → {self._classifier.step_size}")

                elif event.key == pygame.K_z and self._classifier:
                    self._classifier.debounce_count = min(20, self._classifier.debounce_count + 1)
                    print(f"Debounce    → {self._classifier.debounce_count}")

                elif event.key == pygame.K_x and self._classifier:
                    self._classifier.debounce_count = max(1, self._classifier.debounce_count - 1)
                    print(f"Debounce    → {self._classifier.debounce_count}")

            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._compute_layout(event.w, event.h)
                self._rebuild_static_surfaces()

    # ── drawing: panel ───────────────────────────────────────────────────────

    def _draw_card_background(self, x: int, y: int, w: int, h: int) -> None:
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self._screen, Theme.CARD_BG, rect, border_radius=8)
        pygame.draw.rect(self._screen, Theme.CARD_BORDER, rect, width=1, border_radius=8)

    def _blit_row(self, key: str, value: str, x: int, y: int, width: int, val_color=Theme.TEXT) -> None:
        k_surf = self._font_body.render(key, True, Theme.TEXT_DIM)
        v_surf = self._font_body.render(value, True, val_color)
        self._screen.blit(k_surf, (x, y))
        self._screen.blit(v_surf, (x + width - v_surf.get_width(), y))
    def _draw_panel_text(self) -> None:
        pr     = self._panel_rect
        pad    = 18
        x      = pr.x + pad
        card_w = pr.w - (pad * 2)
        y      = 8

        for rs in self._render_states:
            if not rs.data:
                continue
            y = self._draw_section_tracker_info(x, y, card_w, rs)
            y = self._draw_section_position(x, y, card_w, rs.data)
            y = self._draw_section_orientation(x, y, card_w, rs.data)

        y = self._draw_section_classifier(x, y, card_w)
        if y + 120 <= pr.h:
            self._draw_section_gesture_history(x, y, card_w)
        if y + 120 <= pr.h:
            self._draw_section_gesture_history(x, y, card_w)
            if y + 120 <= pr.h:
                self._draw_section_gesture_history(x, y, card_w)

    def _draw_section_tracker_info(self, x: int, y: int, card_w: int, rs) -> int:
        surf     = self._screen
        data     = rs.data
        col_w    = (card_w - 8) // 2
        card_h   = 92
        card_pad = 14
        inner_w  = col_w - (card_pad * 2)

        info_title = self._font_section.render("Tracker Info", True, Theme.TEXT)
        sys_title  = self._font_section.render("System Status", True, Theme.TEXT)
        surf.blit(info_title, (x, y))
        surf.blit(sys_title,  (x + col_w + 8, y))
        y += max(info_title.get_height(), sys_title.get_height()) + 12

        # Tracker info card
        self._draw_card_background(x, y, col_w, card_h)
        self._blit_row("Model",      data.get("name", "Unknown"),                              x + card_pad, y + 14, inner_w, val_color=rs.color)
        self._blit_row("Connection", "Connected" if data.get("connected") else "Disconnected", x + card_pad, y + 40, inner_w, val_color=Theme.GREEN_CONN)
        self._blit_row("Tracking",   "Active"    if data.get("tracking")  else "Lost",         x + card_pad, y + 67, inner_w, val_color=Theme.BLUE_ACTIVE)

        # System status card
        sys_x       = x + col_w + 8
        trail_color = Theme.GREEN_CONN if self.show_trail          else Theme.TEXT_DIM
        rec_color   = (230, 40, 40)   if self._recorder.is_recording else Theme.TEXT_DIM
        orient_lbl  = "Quaternion"    if self.orientation_mode == OrientationMode.QUATERNION else "Euler"
        self._draw_card_background(sys_x, y, col_w, card_h)
        self._blit_row("Trail Status",     "Enabled"   if self.show_trail             else "Disabled", sys_x + card_pad, y + 12, inner_w, val_color=trail_color)
        self._blit_row("Recording Status", "RECORDING" if self._recorder.is_recording else "Off",      sys_x + card_pad, y + 40, inner_w, val_color=rec_color)
        self._blit_row("Orientation",      orient_lbl,                                                 sys_x + card_pad, y + 68, inner_w, val_color=Theme.BLUE_ACTIVE)

        return y + card_h + 10

    def _draw_section_position(self, x: int, y: int, card_w: int, data: dict) -> int:
        surf      = self._screen
        pos_title = self._font_section.render("Position Data (cm)", True, Theme.TEXT)
        surf.blit(pos_title, (x, y))
        y += pos_title.get_height() + 12

        box_w  = (card_w - 16) // 3
        coords = [("X", data.get("x", 0.0)), ("Y", data.get("y", 0.0)), ("Z", data.get("z", 0.0))]
        for idx, (axis, val) in enumerate(coords):
            box_x   = x + idx * (box_w + 8)
            ax_lbl  = self._font_fps.render(axis, True, Theme.TEXT_DIM)
            val_lbl = self._font_large.render(f"{val * 100:,.1f}", True, Theme.TEXT)
            self._draw_card_background(box_x, y, box_w, 50)
            surf.blit(ax_lbl,  (box_x + (box_w - ax_lbl.get_width())  // 2, y + 8))
            surf.blit(val_lbl, (box_x + (box_w - val_lbl.get_width()) // 2, y + 30))

        return y + 50 + 10

    def _draw_section_orientation(self, x: int, y: int, card_w: int, data: dict) -> int:
        surf      = self._screen
        rot_title = self._font_section.render("Orientation (Quat)", True, Theme.TEXT)
        surf.blit(rot_title, (x, y))
        y += rot_title.get_height() + 12

        box_w     = (card_w - 24) // 4
        quat_axes = [
            ("QX", data.get("qx", 0.0)), ("QY", data.get("qy", 0.0)),
            ("QZ", data.get("qz", 0.0)), ("QW", data.get("qw", 0.0)),
        ]
        for idx, (q_ax, q_val) in enumerate(quat_axes):
            box_x   = x + idx * (box_w + 8)
            ax_lbl  = self._font_fps.render(q_ax, True, Theme.TEXT_DIM)
            val_lbl = self._font_large.render(f"{q_val: .3f}", True, Theme.TEXT)
            self._draw_card_background(box_x, y, box_w, 50)
            surf.blit(ax_lbl,  (box_x + (box_w - ax_lbl.get_width())  // 2, y + 8))
            surf.blit(val_lbl, (box_x + (box_w - val_lbl.get_width()) // 2, y + 30))

        return y + 50 + 10

    def _draw_section_classifier(self, x: int, y: int, card_w: int) -> int:
        surf       = self._screen
        cfg_w      = (card_w - 8) // 2
        gest_w     = card_w - cfg_w - 8
        card_h     = 92
        cfg_title  = self._font_section.render("Classifier Config", True, Theme.TEXT)
        gest_title = self._font_section.render("Gesture Detection", True, Theme.TEXT)
        surf.blit(cfg_title,  (x, y))
        surf.blit(gest_title, (x + cfg_w + 8, y))
        y += max(cfg_title.get_height(), gest_title.get_height()) + 12

        # Classifier config card
        cfg_x = x
        self._draw_card_background(cfg_x, y, cfg_w, card_h)
        if self._classifier is None:
            self._blit_row("Classifier", "No model", cfg_x + 14, y + 12, cfg_w - 28, val_color=Theme.TEXT_DIM)
        else:
            self._blit_row("Window  [W/S]",  f"{self._classifier.window_size} fr",       cfg_x + 14, y + 12, cfg_w - 28)
            self._blit_row("Step    [E/Q]",  f"{self._classifier.step_size} fr",         cfg_x + 14, y + 38, cfg_w - 28)
            self._blit_row("Debounce [Z/X]", f"{self._classifier.debounce_count} votes", cfg_x + 14, y + 64, cfg_w - 28)

        # Gesture detection card
        gest_x = x + cfg_w + 8
        self._draw_card_background(gest_x, y, gest_w, card_h)
        if self._classifier is None:
            self._blit_row("Gesture", "No model", gest_x + 14, y + 12, gest_w - 28, val_color=Theme.TEXT_DIM)
        else:
            cl_color = Theme.GREEN_CONN if self._classification_enabled else (230, 40, 40)
            self._blit_row("Classify", "On  [G]" if self._classification_enabled else "Off [G]", gest_x + 14, y + 12, gest_w - 28, val_color=cl_color)
            if self._current_gesture is not None:
                g_color = _GESTURE_COLORS.get(self._current_gesture, Theme.TEXT)
                g_name  = _GESTURE_NAMES.get(self._current_gesture, f"Gesture {self._current_gesture}")
                self._blit_row("Current", g_name, gest_x + 14, y + 38, gest_w - 28, val_color=g_color)
            else:
                self._blit_row("Current", "—", gest_x + 14, y + 38, gest_w - 28, val_color=Theme.TEXT_DIM)
            raw     = self._classifier.raw_label
            raw_str = _GESTURE_NAMES.get(raw, "—") if raw is not None else "—"
            self._blit_row("Raw", raw_str, gest_x + 14, y + 64, gest_w - 28, val_color=Theme.TEXT_DIM)

        return y + card_h + 10

    def _draw_section_gesture_history(self, x: int, y: int, card_w: int) -> None:
        surf    = self._screen
        history = list(self._gesture_history)[:5]
        row_h   = 24

        hist_title = self._font_section.render("Gesture History", True, Theme.TEXT)
        surf.blit(hist_title, (x, y))
        y += hist_title.get_height() + 12

        card_h = max(36, len(history) * row_h + 8)
        self._draw_card_background(x, y, card_w, card_h)

        if not history:
            none_lbl = self._font_body.render("No gestures yet", True, Theme.TEXT_DIM)
            surf.blit(none_lbl, (x + 14, y + 8))
        else:
            for row_i, (ts, label) in enumerate(history):
                ry      = y + 6 + row_i * row_h
                ts_surf = self._font_fps.render(ts, True, Theme.TEXT_DIM)
                g_color = _GESTURE_COLORS.get(label, Theme.TEXT)
                g_surf  = self._font_fps.render(_GESTURE_NAMES.get(label, str(label)), True, g_color)
                surf.blit(ts_surf, (x + 14, ry))
                surf.blit(g_surf,  (x + card_w - 14 - g_surf.get_width(), ry))

    # ── drawing: gesture alert overlay ───────────────────────────────────────

    def _render_alert(self) -> None:
        """Draw a fading colored banner at the bottom of the views area on gesture confirm."""
        if self._alert_timer <= 0 or self._alert_label is None:
            return

        # alpha: full for first ~60 frames, then linear fade to 0
        fade_start = 60
        if self._alert_timer > fade_start:
            alpha = 255
        else:
            alpha = int(255 * self._alert_timer / fade_start)

        color  = _GESTURE_COLORS.get(self._alert_label, (180, 180, 180))
        name   = _GESTURE_NAMES.get(self._alert_label, f"Gesture {self._alert_label}")

        # Span the full views area
        first_rect = self._view_rects[PLANE_DEFS[0][0]]
        last_key   = PLANE_DEFS[-1][0]
        last_rect  = self._view_rects[last_key]
        banner_w   = last_rect.right - first_rect.left
        banner_h   = 52
        banner_x   = first_rect.left
        banner_y   = last_rect.bottom - banner_h

        banner = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        r, g, b = color
        banner.fill((r, g, b, min(200, alpha)))

        text = self._font_alert.render(f"  {name}  DETECTED  ", True, (255, 255, 255))
        text.set_alpha(alpha)
        banner.blit(
            text,
            ((banner_w - text.get_width()) // 2, (banner_h - text.get_height()) // 2),
        )

        self._screen.blit(banner, (banner_x, banner_y))
        self._alert_timer -= 1

    # ── shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        for rs in self._render_states:
            rs.source.shutdown()
        pygame.quit()
