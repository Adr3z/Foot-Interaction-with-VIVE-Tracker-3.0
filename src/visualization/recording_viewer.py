"""
recording_viewer.py
-------------------

Displays a recording file as three projected planes without the side info panel.
The viewer loads a .npz recording, plays it with original timing, and supports
pause/play, seek, restart, and file reload controls.
"""

from __future__ import annotations

import os

import numpy as np
import pygame
from collections import deque
from typing import Any

from .theme import Theme
from .tracker_renderer import TrackerRenderer, OrientationMode
from ..utils import CoordinateMapper
from ..recordings import FileManager, PlaybackEngine
from ..gesture_processing import PCAGestureProcessor
from .view_utils import (PLANE_DEFS, TITLE_BAR_H, build_view_surface, build_plane_views, render_trackers_on_views, toggle_orientation)

Plane = str

# ──────────────────────────────────────────────────────────────────────────────
#  RECORDING TRACK STATE
# ──────────────────────────────────────────────────────────────────────────────
class RecordingTrackState:
    """Stores the current playback frame data and trail history for a recorded tracker"""
    def __init__(self, name: str, color: tuple[int, int, int], history_length: int | None = 80):
        self.name = name
        self.color = color

        # Recent world-space positions used for trail rendering
        if history_length is None or history_length <= 0:
            self.history: deque[dict[str, float]] = deque()
        else:
            self.history: deque[dict[str, float]] = deque(maxlen=history_length)
        # Current frame data
        self.data: dict[str, Any] = {}

    def update(self, tracker_data: dict[str, Any]) -> None:
        """ Updata tracker state from the current playback frame"""
        self.data = tracker_data
        # Append position to the trail only while tracking is valid
        if tracker_data.get("tracking"):
            self.history.append({
                "x": tracker_data.get("x", 0.0),
                "y": tracker_data.get("y", 0.0),
                "z": tracker_data.get("z", 0.0),
            })

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN VIEWER CLASS
# ──────────────────────────────────────────────────────────────────────────────
class RecordingViewer:
    """ Playback viewer for recorded tracker sessions """
    TARGET_FPS = 90
    WORLD_RANGE = 0.250
    TRAIL_LENGTH = 120
    SEEK_STEP = 0.5
    CONTROL_BUTTON_HEIGHT = 36
    SIDE_PANEL_WIDTH = 220
    PCA_PANEL_WIDTH = 360
    PROGRESS_BAR_HEIGHT = 32
    CONTROL_BAR_HEIGHT = 64

    def __init__(
        self,
        recording_path: str | None = None,
        window_size: tuple[int, int] = (1400, 720),
        loop: bool = False,
    ):
        # Window and application state
        self._window_size = window_size
        self._running = False

        # Pygame objects
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        # Per-view rendering resources
        self._mappers: dict[Plane, CoordinateMapper] = {}
        self._view_surfs: dict[Plane, pygame.Surface] = {}
        self._view_rects: dict[Plane, pygame.Rect] = {}

        # Side panel and control bar layout
        self._pca_rect: pygame.Rect | None = None
        self._panel_rect: pygame.Rect | None = None
        self._progress_rect: pygame.Rect | None = None
        self._controls_rect: pygame.Rect | None = None

        # Clickable control buttons
        self._control_buttons: dict[str, pygame.Rect] = {}

        # PCA preview surface
        self._pca_surface: pygame.Surface | None = None

        # Playback control labels and associated actions
        self._control_labels: list[tuple[str, str]] = [
            ("Play/Pause", "toggle"),
            ("<< 0.5s", "rewind"),
            ("0.5s >>", "forward"),
            ("Restart", "restart"),
            ("Reset", "reset"),
            ("End", "end"),
            ("Toggle Trail", "trail"),
            ("Orientation", "orient"),
            ("Speed", "speed"),
            ("Load File", "load"),
        ]

        self._font_title: pygame.font.Font | None = None
        self._font_body: pygame.font.Font | None = None
        self._font_axis: pygame.font.Font | None = None
        self._font_overlay: pygame.font.Font | None = None
        self._orientation_mode = OrientationMode.QUATERNION
        self._show_trail = True
        self._recording_path = recording_path
        self._recording_data: dict[str, dict[str, Any]] = {}
        self._playback: PlaybackEngine | None = None
        self._track_states: list[RecordingTrackState] = []
        self._loop = loop

    # ── lifecycle ────────────────────────────────────────────────────────────
    def run(self) -> None:
        """ Start playback viewer main loop"""
        if not self._load_recording():
            return

        self._init_pygame()
        self._running = True

        try:
            while self._running:
                # Process input, update playback state and render frame
                self._handle_events()
                self.update()
                self.render()
                self._clock.tick(self.TARGET_FPS)
        finally:
            pygame.quit()

    def update(self) -> None:
        """ Advance playback and update tracker render states """
        delta_seconds = self._clock.get_time() / 1000.0
        if self._playback is not None:
            self._playback.update(delta_seconds)
            current_data = self._playback.get_current_frame_data()
            for state in self._track_states:
                state.update(current_data.get(state.name, {
                    "name": state.name,
                    "tracking": False,
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": 0.0,
                    "qw": 1.0,
                }))

    def render(self) -> None:
        """ Render the complete playback viewer frame """
        self._screen.fill(Theme.BG)

        # Draw tracker projections on all plane views
        render_trackers_on_views(
            screen=self._screen,
            tracker_states=self._track_states,
            view_rects=self._view_rects,
            view_surfs=self._view_surfs,
            mappers=self._mappers,
            show_trail=self._show_trail,
            orientation_mode=self._orientation_mode,
        )

        # Draw PCA preview surface between last view and info panel
        if self._pca_surface is not None and self._pca_rect is not None:
            self._screen.blit(self._pca_surface, self._pca_rect.topleft)
            pygame.draw.rect(self._screen, Theme.VIEW_BORDER, self._pca_rect, 1)

        # Draw information panel
        if self._panel_rect is not None:
            pygame.draw.rect(self._screen, Theme.PANEL_BG, self._panel_rect)
            pygame.draw.rect(self._screen, Theme.PANEL_BORDER, self._panel_rect, 1)
            self._draw_panel(self._panel_rect)

        # Draw progress bar
        if self._progress_rect is not None:
            pygame.draw.rect(self._screen, Theme.PANEL_BG, self._progress_rect)
            pygame.draw.rect(self._screen, Theme.PANEL_BORDER, self._progress_rect, 1)
            self._draw_progress_bar(self._progress_rect)

        # Draw control bar
        if self._controls_rect is not None:
            pygame.draw.rect(self._screen, Theme.PANEL_BG, self._controls_rect)
            pygame.draw.rect(self._screen, Theme.PANEL_BORDER, self._controls_rect, 1)
            self._draw_control_bar(self._controls_rect)
        pygame.display.flip()


    # ── recording management ─────────────────────────────────────────────────
    def _load_recording(self) -> bool:
        """ Load a recording file and initialize playback state """
        path = self._recording_path
        if not path:
            path = FileManager.select_recording_path()

        if not path or not os.path.exists(path):
            print("No record selected")
            return False

        try:
            self._recording_data = FileManager.load_recording_data(path)
        except Exception as exc:
            print(f"Error to load recording: {exc}")
            return False

        self._recording_path = path
        self._playback = PlaybackEngine(self._recording_data, loop=self._loop)
        self._track_states = []

        # Create render states for every recorded tracker
        for name in self._recording_data.keys():
            color = Theme.TRACKER_COLORS.get(name, Theme.TRACKER_DEFAULT)
            self._track_states.append(
                RecordingTrackState(name, color, history_length=None)
            )

        self._load_pca_surface()

        return True


    def _reload_recording(self) -> None:
        """ Reload the current recording from disk """
        if self._load_recording():
            self._track_states = []
            for name in self._recording_data.keys():
                color = Theme.TRACKER_COLORS.get(name, Theme.TRACKER_DEFAULT)
                self._track_states.append(
                    RecordingTrackState(name, color, history_length=None)
                )
            self._load_pca_surface()


    def _change_recording(self) -> None:
        """ Load a different recording selected by the user """
        path = FileManager.select_recording_path()
        if not path or not os.path.exists(path):
            print("Selection canceled")
            return

        try:
            new_data = FileManager.load_recording_data(path)
        except Exception as exc:
            print(f"Error in loading new file: {exc}")
            return

        self._recording_path = path
        self._recording_data = new_data
        self._playback = PlaybackEngine(self._recording_data, loop=self._loop)
        self._track_states = []

        for name in self._recording_data.keys():
            color = Theme.TRACKER_COLORS.get(name, Theme.TRACKER_DEFAULT)
            self._track_states.append(
                RecordingTrackState(name, color, history_length=None)
            )

        self._load_pca_surface()

    # ── initialization ───────────────────────────────────────────────────────
    def _init_pygame(self) -> None:
        pygame.init()
        self._screen = pygame.display.set_mode(self._window_size, pygame.RESIZABLE)
        pygame.display.set_caption("Recording Playback Viewer")
        self._clock = pygame.time.Clock()
        self._init_fonts()
        self._compute_layout(*self._window_size)
        self._rebuild_static_surfaces()

    def _init_fonts(self) -> None:
        fam = Theme.FONT_FAMILY
        self._font_title = pygame.font.SysFont(fam, 24)
        self._font_body = pygame.font.SysFont(fam, 18)
        self._font_axis = pygame.font.SysFont(fam, 16)
        self._font_overlay = pygame.font.SysFont(fam, 18)

    def _compute_layout(self, win_w: int, win_h: int) -> None:
        """ Recompute view, panel and control bar layout """
        GAP = 12

        # Available area excluding progress and bottom control bar
        content_h = win_h - self.CONTROL_BAR_HEIGHT - self.PROGRESS_BAR_HEIGHT - GAP *4

        # Divide available width among the three projection views plus PCA and panel columns
        views_w = win_w - self.SIDE_PANEL_WIDTH - self.PCA_PANEL_WIDTH - GAP * 5
        view_w = max(180, views_w // 3)
        view_h = content_h
        x = GAP

        # Create projection view rectangles
        for plane_id, *_ in PLANE_DEFS:
            self._view_rects[plane_id] = pygame.Rect(x, GAP, view_w, view_h)
            x += view_w + GAP

        # Create PCA preview panel
        self._pca_rect = pygame.Rect(x, GAP, self.PCA_PANEL_WIDTH, content_h)
        x += self.PCA_PANEL_WIDTH + GAP

        # Create side information panel
        self._panel_rect = pygame.Rect(x, GAP, self.SIDE_PANEL_WIDTH, content_h)
        # Create middle progress bar section
        self._progress_rect = pygame.Rect(GAP, GAP + content_h + GAP, win_w - GAP * 2, self.PROGRESS_BAR_HEIGHT)
        # Create bottom control bar
        self._controls_rect = pygame.Rect(GAP, GAP + content_h + GAP + self.PROGRESS_BAR_HEIGHT + GAP, win_w - GAP * 2, self.CONTROL_BAR_HEIGHT)


    def _rebuild_static_surfaces(self) -> None:
        self._mappers, self._view_surfs = build_plane_views(
            view_rects=self._view_rects,
            world_range=self.WORLD_RANGE,
            font_axis=self._font_axis,
            font_title=self._font_title,
        )
        self._load_pca_surface()

    def _load_pca_surface(self) -> None:
        """Generate or refresh the PCA comparison surface for the current recording."""
        self._pca_surface = None
        if self._recording_path is None or self._pca_rect is None:
            return

        try:
            surface = PCAGestureProcessor.generate_comparison_surface(
                self._recording_path,
                width=self._pca_rect.w,
                height=self._pca_rect.h,
                bg_color_hex="#1E1E24",
                text_color_hex="#E0E0E6",
            )
            if surface is None:
                return

            surface = surface.convert()
            if surface.get_size() != (self._pca_rect.w, self._pca_rect.h):
                bg = pygame.Surface((self._pca_rect.w, self._pca_rect.h))
                bg.fill(Theme.BG)
                bg.blit(surface, ((bg.get_width() - surface.get_width()) // 2,
                                   (bg.get_height() - surface.get_height()) // 2))
                self._pca_surface = bg.convert()
            else:
                self._pca_surface = surface
        except Exception as exc:
            print(f"Error generating PCA preview: {exc}")
            self._pca_surface = None


    # ── event handling ───────────────────────────────────────────────────────
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_SPACE:
                    self._toggle_playback()
                elif event.key == pygame.K_LEFT:
                    self._seek(-self.SEEK_STEP)
                elif event.key == pygame.K_RIGHT:
                    self._seek(self.SEEK_STEP)
                elif event.key == pygame.K_HOME:
                    self._reset_playback()
                elif event.key == pygame.K_END:
                    self._goto_end()
                elif event.key == pygame.K_s:
                    self._show_trail = not self._show_trail
                elif event.key == pygame.K_o:
                    self._orientation_mode = toggle_orientation(
                        self._orientation_mode
                    )
                elif event.key == pygame.K_l:
                    self._change_recording()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_mouse_click(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                self._window_size = (event.w, event.h)
                self._compute_layout(event.w, event.h)
                self._rebuild_static_surfaces()


    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        for action, rect in self._control_buttons.items():
            if rect.collidepoint(pos):
                if action == "toggle":
                    self._toggle_playback()
                elif action == "rewind":
                    self._seek(-self.SEEK_STEP)
                elif action == "forward":
                    self._seek(self.SEEK_STEP)
                elif action == "reset":
                    self._reload_recording()
                elif action == "end":
                    self._goto_end()
                elif action == "load":
                    self._change_recording()
                elif action == "trail":
                    self._show_trail = not self._show_trail
                elif action == "orient":
                    self._orientation_mode = toggle_orientation(
                        self._orientation_mode
                    )
                elif action == "speed":
                    self._cycle_playback_speed()
                elif action == "restart":
                    self._reset_playback()
                break


    # ── drawing helpers ──────────────────────────────────────────────────────
    def _draw_panel(self, panel_rect: pygame.Rect) -> None:
        """ Draw playback information and keyboard shortcuts """
        if self._playback is None:
            return

        status = "Playing" if self._playback.is_playing else "Paused"
        current = self._playback.current_time
        duration = self._playback.duration
        file_name = os.path.basename(self._recording_path or "")

        left_x = panel_rect.x + 16
        top_y = panel_rect.y + 16
        line_height = self._font_overlay.get_height() + 8

        lines = [
            "Playback Information", 
            "",
            f"File: {file_name}",
            f"Status: {status}",
            f"Time: {current:.2f}s / {duration:.2f}s",
            f"Speed: {self._playback.speed:.1f}x",
            f"Trail: {'On' if self._show_trail else 'Off'}",
            f"Orientation: {self._orientation_mode.name}",
            "",
            "Keyboard Controls:",
            "SPACE - Play / Pause",
            "LEFT - Rewind 0.5s",
            "RIGHT - Forward 0.5s", 
            "HOME - Restart",
            "END - Go to End",
            "S - Toggle Trail",
            "O - Toggle Orientation",
            "L - Load Recording",
            "Esc - Exit"
        ]

        for i, line in enumerate(lines):
            color = Theme.TEXT

            if line in ("Playback Information", "Keyboard Controls"):
                color = Theme.ACCENT if hasattr(Theme, "ACCENT") else Theme.TEXT

            surf = self._font_overlay.render(line, True, color)
            self._screen.blit(
                surf,
                (
                    left_x,
                    top_y + i * line_height,
                ),
            )


    def _draw_control_bar(self, bar_rect: pygame.Rect) -> None:
        """ Draw bottom playback control buttons """
        button_count = len(self._control_labels)

        spacing = 10
        margin = 14
        total_spacing = spacing * (button_count -1)
        available_width = bar_rect.w - margin *2 - total_spacing

        button_width = max(90, min(130, available_width // button_count))
        button_height = self.CONTROL_BUTTON_HEIGHT
        total_width = (button_width * button_count + total_spacing)

        x = bar_rect.x + (bar_rect.w - total_width) // 2
        y = bar_rect.y + (bar_rect.h - button_height) // 2
        self._control_buttons.clear()

        for label, key in self._control_labels:
            if key == "speed" and self._playback is not None:
                label = f"Speed {self._playback.speed:.1f}x"

            rect = pygame.Rect(x, y, button_width, button_height)
            pygame.draw.rect(self._screen, Theme.CARD_BG, rect, border_radius=8)
            pygame.draw.rect(self._screen, Theme.VIEW_BORDER, rect, 1, border_radius=8)
            text = self._font_body.render(label, True, Theme.TEXT)
            self._screen.blit(text, (rect.x + (rect.w - text.get_width()) // 2, rect.y + (rect.h - text.get_height()) // 2,), )

            self._control_buttons[key] = rect

            x += button_width + spacing

    # ── playback controls ────────────────────────────────────────────────────
    def _toggle_playback(self) -> None:
        if self._playback is None:
            return
        if self._playback.is_playing:
            self._playback.pause()
        else:
            self._playback.play()

    def _seek(self, seconds: float) -> None:
        if self._playback is None:
            return
        self._playback.seek(seconds)

    def _reset_playback(self) -> None:
        if self._playback is None:
            return
        self._playback.stop()
        for state in self._track_states:
            state.history.clear()

    def _goto_end(self) -> None:
        if self._playback is None:
            return
        self._playback.set_time(self._playback.duration)

    def _cycle_playback_speed(self) -> None:
        if self._playback is None:
            return

        speeds = [0.25, 0.5, 1.0, 1.5, 2.0]
        current = self._playback.speed
        try:
            next_index = (speeds.index(current) + 1) % len(speeds)
        except ValueError:
            next_index = 2
        self._playback.set_speed(speeds[next_index])

    def _get_playback_frame_progress(self) -> tuple[int, int]:
        if self._playback is None or not self._recording_data:
            return 0, 0

        longest_track = max(
            self._recording_data.values(),
            key=lambda info: len(info.get("time", [])),
        )
        timestamps = np.asarray(longest_track["time"], dtype=np.float64)
        if timestamps.size == 0:
            return 0, 0

        relative = timestamps - timestamps[0]
        position = min(self._playback.current_time, relative[-1])
        index = int(np.searchsorted(relative, position, side="right") - 1)
        index = max(0, min(index, len(timestamps) - 1))
        return index + 1, len(timestamps)

    def _draw_progress_bar(self, bar_rect: pygame.Rect) -> None:
        frame_index, total_frames = self._get_playback_frame_progress()
        if total_frames == 0:
            text = "No frames"
            progress = 0.0
        else:
            text = f"Frame {frame_index} / {total_frames}"
            progress = frame_index / total_frames

        label_surf = self._font_body.render(text, True, Theme.TEXT)
        label_x = bar_rect.x + 16
        label_y = bar_rect.y + (bar_rect.h - label_surf.get_height()) // 2
        self._screen.blit(label_surf, (label_x, label_y))

        progress_margin = 16
        progress_x = label_x + label_surf.get_width() + 24
        progress_w = bar_rect.w - (progress_x - bar_rect.x) - progress_margin
        progress_h = 16
        progress_y = bar_rect.y + (bar_rect.h - progress_h) // 2

        progress_back = pygame.Rect(progress_x, progress_y, progress_w, progress_h)
        pygame.draw.rect(self._screen, Theme.CARD_BG, progress_back, border_radius=8)
        pygame.draw.rect(self._screen, Theme.VIEW_BORDER, progress_back, 1, border_radius=8)

        fill_width = max(0, min(progress_w, int(progress_w * progress)))
        progress_fill = pygame.Rect(progress_x, progress_y, fill_width, progress_h)
        pygame.draw.rect(self._screen, Theme.ACCENT, progress_fill, border_radius=8)

    # ── shutdown ─────────────────────────────────────────────────────────────
    def _shutdown(self) -> None:
        pygame.quit()
