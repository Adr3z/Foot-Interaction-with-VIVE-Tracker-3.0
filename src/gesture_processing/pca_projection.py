"""
pca_projection.py
-----------------

Runs PCA on a recording to identify the principal plane and generate a static 2D image.

    - get_2d_coords: returns raw 2d projected coordinates
    - generate_comparision_surface: rendering onlym calls get_2d_coords internally
"""

import io
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pygame

from ..recordings import FileManager

class PCAGestureProcessor:

    @staticmethod
    def get_2d_coords(file_path: str) -> np.ndarray | None:
        """ Loads a recording and returns the raw 2d PCA projection as a numpy array"""
        recording_data = FileManager.load_recording_data(file_path)
        if not recording_data:
            print("Error: invalid data.")
            return None

        tracker_name = list(recording_data.keys())[0]
        positions = recording_data[tracker_name]["pos"]

        if len(positions) < 3:
            print("Error: insufficient data to run PCA")
            return None

        #  DIRECT PCA 
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(positions)

        return coords_2d


    @staticmethod
    def generate_comparison_surface(
        file_path: str, 
        width: int = 500, 
        height: int = 500, 
        bg_color_hex: str = "#1E1E24",
        text_color_hex: str = "#E0E0E6"
    ) -> pygame.Surface | None:
        """
        Loads a recording, runs PCA, and returns a Pygame surface ready to blit.
        """
        coords_2d = PCAGestureProcessor.get_2d_coords(file_path)
        if coords_2d is None:
            return None

        # temporal index to show the time progression
        time_index = np.arange(len(coords_2d))

        return PCAGestureProcessor._render_to_surface(
            coords_2d = coords_2d,
            time_index=time_index,
            width=width,
            height=height,
            bg_color_hex=bg_color_hex,
            text_color_hex=text_color_hex,
            title="PCA Projection"
        )


    @staticmethod
    def _render_to_surface(
        coords_2d: np.ndarray,
        time_index: np.ndarray,
        width: int,
        height: int,
        bg_color_hex: str,
        text_color_hex: str,
        title: str
    ) -> pygame.Surface:
        """ Renders a 2d coordinate array to a pygame surface using matplotlib.
        Shared by both raw and processed render data"""

        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor(bg_color_hex)

        header_height = 0.09

        fig.patches.append(
            plt.Rectangle( (0, 1 - header_height), 1, header_height, transform=fig.transFigure, color=text_color_hex, zorder=10,)
        )

        fig.text(0.5, 1 - header_height / 2, title, ha="center", va="center", fontsize=10, color="black", fontweight="bold", zorder=11,)

        ax = fig.add_axes([0.02, 0.02, 0.96, 0.86])
        ax.set_facecolor(bg_color_hex)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        ax.scatter( coords_2d[:, 0], coords_2d[:, 1], c=time_index, cmap="viridis", s=8, zorder=3,)

        ax.plot( coords_2d[:, 0], coords_2d[:, 1], color=text_color_hex, alpha=0.4, linewidth=1, zorder=2,)

        canvas = fig.canvas
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()
        size = canvas.get_width_height()

        pygame_surface = pygame.image.frombuffer(rgba_buffer, size, "RGBA")

        plt.close(fig)

        return pygame_surface