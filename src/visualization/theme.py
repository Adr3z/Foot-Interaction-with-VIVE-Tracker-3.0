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

    CARD_BG         = (255, 255, 255)
    CARD_BORDER     = (218, 222, 230)
    GREEN_CONN      = (20, 165, 90)
    BLUE_ACTIVE     = (40, 90, 230)

    TRACKER_COLORS: dict[str, tuple[int, int, int]] = {
        "VIVE Ultimate Tracker 1": (230, 159,   0),
        "VIVE Tracker 3.0 MV": (  0, 114, 178),
        "VIVE Ultimate Tracker 2": (130,   0, 204),
    }
    TRACKER_DEFAULT = ( 102, 0, 102)

    TRACKER_RADIUS   = 10   # px, filled circle 
    TRACKER_OUTLINE  = 2    # px, white ring
    
    FONT_FAMILY = None      # None = pygame default monospace
    ACCENT = ( 102, 0, 102)
