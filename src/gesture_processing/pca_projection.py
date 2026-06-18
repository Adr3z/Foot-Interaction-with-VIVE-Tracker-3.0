"""
pca_projection.py
-----------------

Runs PCA on a recording to identify the principal plane and generate a static 2D image.
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
    def generate_comparison_surface(
        file_path: str, 
        width: int = 500, 
        height: int = 500, 
        bg_color_hex: str = "#1E1E24",
        text_color_hex: str = "#E0E0E6"
    ) -> pygame.Surface | None:
        """
        Loads a recording, executes the two PCA approaches, and returns a Pygame surface ready to be drawn (blit) to the screen.
        """
        # load file
        recording_data = FileManager.load_recording_data(file_path)
        if not recording_data:
            print("Error: Invalid data.")
            return None

        # get the first tracker on file
        tracker_name = list(recording_data.keys())[0]
        positions = recording_data[tracker_name]["pos"]

        if len(positions) < 3:
            print("Error: insuficient data to run PCA.")
            return None

        # =========================================================================
        #  DIRECT PCA 
        # =========================================================================
        pca_direct = PCA(n_components=2)
        coords_2d = pca_direct.fit_transform(positions)

        # =========================================================================
        # MATPLOT RENDER
        # =========================================================================
        # Adjust dimensions for pygame 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor(bg_color_hex)

        # temporal index to show the time progression
        time_index = np.arange(len(positions))

        fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        fig.patch.set_facecolor(bg_color_hex)

        ax.set_facecolor(bg_color_hex)
        
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=time_index, cmap="viridis", s=8, zorder=3)
        ax.plot(coords_2d[:, 0], coords_2d[:, 1], color=text_color_hex, alpha=0.4, linewidth=1, zorder=2)

        ax.axis('off')
        
        ax.set_title("PCA Projection", color=text_color_hex, fontsize=11, pad=10)
        ax.tick_params(colors=text_color_hex, labelsize=8)
        
        for spine in ax.spines.values():
            spine.set_color(text_color_hex)
            spine.set_alpha(0.3)

        plt.tight_layout()

        # =========================================================================
        # MATPLOT TO PYGAME SURFACE CONVERSION
        # =========================================================================
        canvas = fig.canvas
        canvas.draw()
        rgba_buffer = canvas.buffer_rgba()
        size = canvas.get_width_height()

        pygame_surface = pygame.image.frombuffer(rgba_buffer, size, "RGBA")
        
        plt.close(fig)

        return pygame_surface