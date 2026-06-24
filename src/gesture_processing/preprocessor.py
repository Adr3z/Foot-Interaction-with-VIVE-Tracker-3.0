""" 
preprocessor.py
-----------------
Normalizes raw 2D PCA Coordinates to make gesture representations consistent across recordings.

    - Fix PCA component direction
    - Center coordinates around the origin
    - Normalize scale to [-1, 1]

"""

import numpy as np

class GesturePreprocessor:

    @staticmethod
    def _fix_sign_flip(coords: np.ndarray) -> np.ndarray:
        """ Stabilize PCA sign ambiguity on each component independently. 
            If the mean of a component is negativa, flip it"""

        for i in range(coords.shape[1]):
            if np.mean(coords[:, i]) < 0:
                coords[:, i] *= -1

        return coords    


    @staticmethod
    def _center(coords: np.ndarray) -> np.ndarray:
        """ Translates coordinates so their centroid is at the origin"""
        coords -= coords.mean(axis=0)
        return coords

    
    @staticmethod
    def _normalize_scale(coords: np.ndarray) -> np.ndarray:
        """ Scales coordinates uniformly to fit within [-1, 1] on both axes. """
        max_abs = np.abs(coords).max()
        if max_abs > 0:
            coords /= max_abs

        return coords


    @staticmethod
    def normalize(coords_2d: np.ndarray) -> np.ndarray:
        """ Applies the full normalization pipeline to raw 2d pca coordinates. Returns np.ndarray normalized"""

        coords = coords_2d.copy()
        coords = GesturePreprocessor._fix_sign_flip(coords)
        coords = GesturePreprocessor._center(coords)
        coords = GesturePreprocessor._normalize_scale(coords)
        return coords

