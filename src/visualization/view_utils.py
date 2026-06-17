"""
view_utils.py
"""

from __future__ import annotations

import pygame

from ..utils import CoordinateMapper
from .tracker_renderer import TrackerRenderer, OrientationMode
from .theme import Theme


# ──────────────────────────────────────────────────────────────────────────────
#  PLANE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────
# For each entry: plane_id, title, horiz_pos_label, horiz_neg_label, vert_pos_label, vert_neg_label
PLANE_DEFS: list[tuple[Plane, str, str, str, str, str]] = [
    ("XZ", "Top-Down View (XZ)", "X+", "X-", "Z-", "Z+"),
    ("XY", "Front View (XY)", "X+", "X-", "Y+", "Y-"),
    ("ZY", "Side View (ZY)", "Z+", "Z-", "Y+", "Y-"),
]


TITLE_BAR_H = 32

# ──────────────────────────────────────────────────────────────────────────────
#  STATIC SURFACE BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_view_surface(
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
    # canvas = pygame.Surface((viewport_w, viewport_h))
    # canvas.fill(Theme.BG)

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
        (v_pos, (cx + 6, margin + TITLE_BAR_H + 4)),
        (v_neg, (cx + 6, viewport_h - margin - font_axis.get_height())),
    ]:
        lbl = font_axis.render(text, True, Theme.AXES)
        surf.blit(lbl, pos)

    # ── title bar ───────────────────────────────────────────────────────────
    title_bar = pygame.Rect(0, 0, viewport_w, TITLE_BAR_H)
    pygame.draw.rect(surf, Theme.VIEW_TITLE_BG, title_bar)
    pygame.draw.line(surf, Theme.VIEW_BORDER, (0, TITLE_BAR_H -1), (viewport_w, TITLE_BAR_H -1), 1)

    t_surf = font_title.render(title, True, Theme.TEXT)
    surf.blit(t_surf, ((viewport_w - t_surf.get_width()) // 2, (TITLE_BAR_H - t_surf.get_height()) // 2))

    # ── outer border ───────────────────────────────────────────────────────────
    pygame.draw.rect(surf, Theme.VIEW_BORDER, pygame.Rect(0, 0, viewport_w, viewport_h), 1)

    return surf.convert()


def build_plane_views(
    view_rects: dict,
    world_range: float,
    font_axis: pygame.font.Font,
    font_title: pygame.font.Font,
) -> tuple[dict, dict]:
    """
    Creates mappers and cached static surfaces for all planes.
    """
    mappers = {}
    surfaces = {}

    for plane_def in PLANE_DEFS:
        plane_id = plane_def[0]
        rect = view_rects[plane_id]

        mapper = CoordinateMapper(
            viewport_x=rect.x,
            viewport_y=rect.y,
            viewport_w=rect.w,
            viewport_h=rect.h,
            world_range_h=world_range,
            world_range_v=world_range,
            plane=plane_id,
        )

        mappers[plane_id] = mapper

        surfaces[plane_id] = build_view_surface(
            rect.w,
            rect.h,
            mapper,
            font_axis,
            font_title,
            plane_def,
            grid_step_m=0.05,
        )

    return mappers, surfaces

# ──────────────────────────────────────────────────────────────────────────────
#  TRACKER RENDERING
# ──────────────────────────────────────────────────────────────────────────────
def render_trackers_on_views(
    screen: pygame.Surface,
    tracker_states,
    view_rects: dict,
    view_surfs: dict,
    mappers: dict,
    show_trail: bool,
    orientation_mode: OrientationMode,
):
    """
    Render all trackers on every projection view.

    Each view surface is first blitted to the screen, then the tracker
    trail and orientation marker are drawn on top using the corresponding
    coordinate mapper.
    """

    for plane_id, _ in [(p[0], p) for p in PLANE_DEFS]:
        # View rectangle and coordinate mapper for this projection
        rect = view_rects[plane_id]

        screen.blit(view_surfs[plane_id], rect.topleft)

        for state in tracker_states:
            if not state.data.get("tracking"):
                continue

            mapper = mappers[plane_id]

            if show_trail:
                TrackerRenderer.draw_trail(
                    screen=screen,
                    history=state.history,
                    color=state.color,
                    mapper=mapper,
                    clip_rect=rect,
                    title_bar_height=TITLE_BAR_H,
                )

            TrackerRenderer.draw(
                screen=screen,
                tracker_data=state.data,
                color=state.color,
                mapper=mapper,
                clip_rect=rect,
                title_bar_height=TITLE_BAR_H,
                orientation_mode=orientation_mode,
            )


def toggle_orientation(
    current_mode: OrientationMode,
) -> OrientationMode:
    """
    Switches between quaternion and Euler modes.
    """

    if current_mode == OrientationMode.QUATERNION:
        return OrientationMode.EULER

    return OrientationMode.QUATERNION