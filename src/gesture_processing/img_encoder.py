"""
img_encoder.py
--------------

Renders normalized 2D gesture coordinates into image representations.

    - to_numpy: produce a fixed-size grayscale image as a numpy array (ML input)
    - to_pygame_surface: produce a pygame surface for visualization in the viewer
"""

import numpy as np
import pygame
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GestureImageEncoder:

    @staticmethod
    def to_numpy(coords_2d: np.ndarray, size: int = 64) -> np.ndarray:
        """
        Renders normalized 2D coordinates into a square grayscale image.

        Coordinates are expected in [-1, 1] range (output of GesturePreprocessor).
        The resulting image encodes temporal progression via pixel intensity:
        earlier points are darker, later points are brighter.

        Args:
            coords_2d: np.ndarray of shape (n_frames, 2), normalized to [-1, 1]
            size: output image resolution in pixels (size x size)

        Returns:
            np.ndarray of shape (size, size), dtype float32, values in [0, 1]
        """
        image = np.zeros((size, size), dtype=np.float32)
        n_frames = len(coords_2d)

        for i, (x, y) in enumerate(coords_2d):
            # map [-1, 1] to [0, size - 1]
            col = int(np.clip((x + 1) / 2 * (size - 1), 0, size - 1))
            # invert y so that positive y maps to the top of the image
            row = int(np.clip((1 - (y + 1) / 2) * (size - 1), 0, size - 1))

            intensity = (i + 1) / n_frames  # temporal progression: 0 → 1
            image[row, col] = max(image[row, col], intensity)

        return image

    @staticmethod
    def to_pygame_surface(
        coords_2d: np.ndarray,
        width: int = 500,
        height: int = 500,
        bg_color_hex: str = "#1E1E24",
        text_color_hex: str = "#DCDFE6"
    ) -> pygame.Surface:
        """
        Renders normalized 2D coordinates into a pygame surface for the viewer.

        Uses the same visual style as PCAGestureProcessor._render_to_surface
        so both panels look consistent in the recording viewer.

        """
        time_index = np.arange(len(coords_2d))

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor(bg_color_hex)

        header_height = 0.09
        fig.patches.append(
            plt.Rectangle( (0, 1 - header_height), 1, header_height, transform=fig.transFigure, color=text_color_hex, zorder=10,)
        )

        fig.text(0.5, 1 - header_height / 2, "PCA Preprocessed", ha="center", va="center", fontsize=10, color="black", fontweight="bold", zorder=11,)

        ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
        ax.set_facecolor(bg_color_hex)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        # fix axes range to [-1, 1] since coordinates are normalized
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        ax.scatter( coords_2d[:, 0], coords_2d[:, 1], c=time_index, cmap="viridis", s=8,zorder=3,)

        ax.plot( coords_2d[:, 0], coords_2d[:, 1], color=text_color_hex, alpha=0.4, linewidth=1,zorder=2,)

        canvas = fig.canvas
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()
        size = canvas.get_width_height()

        pygame_surface = pygame.image.frombuffer(rgba_buffer, size, "RGBA")

        plt.close(fig)

        return pygame_surface