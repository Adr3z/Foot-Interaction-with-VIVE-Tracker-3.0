"""
preprocessor.py
-----------------
Normalizes raw 2D PCA coordinates and turns them into stable ML-ready features.

    - Fix PCA component direction
    - Center coordinates around the origin
    - Normalize scale to [-1, 1]
    - Convert normalized coordinates to image features for training
    - Flatten and scale image features for k-NN training
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class GesturePreprocessor:

    @staticmethod
    def _fix_sign_flip(coords: np.ndarray) -> np.ndarray:
        """Stabilize PCA sign ambiguity on each component independently."""
        for i in range(coords.shape[1]):
            if np.mean(coords[:, i]) < 0:
                coords[:, i] *= -1

        return coords

    @staticmethod
    def _center(coords: np.ndarray) -> np.ndarray:
        """Translate coordinates so their centroid is at the origin."""
        coords -= coords.mean(axis=0)
        return coords

    @staticmethod
    def _normalize_scale(coords: np.ndarray) -> np.ndarray:
        """Scale coordinates uniformly to fit within [-1, 1] on both axes."""
        max_abs = np.abs(coords).max()
        if max_abs > 0:
            coords /= max_abs

        return coords

    @staticmethod
    def normalize(coords_2d: np.ndarray) -> np.ndarray:
        """Apply the full normalization pipeline to raw 2D PCA coordinates."""
        coords = coords_2d.copy()
        coords = GesturePreprocessor._fix_sign_flip(coords)
        coords = GesturePreprocessor._center(coords)
        coords = GesturePreprocessor._normalize_scale(coords)
        return coords

    @staticmethod
    def encode_to_image(coords_2d: np.ndarray, size: int = 64) -> np.ndarray:
        """Render normalized 2D coordinates into a square grayscale image."""
        coords = np.asarray(coords_2d, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords_2d must have shape (n_frames, 2)")
        if size <= 0:
            raise ValueError("size must be positive")

        image = np.zeros((size, size), dtype=np.float32)
        n_frames = len(coords)
        if n_frames == 0:
            return image

        for i, (x, y) in enumerate(coords):
            col = int(np.clip((x + 1) / 2 * (size - 1), 0, size - 1))
            row = int(np.clip((1 - (y + 1) / 2) * (size - 1), 0, size - 1))
            intensity = (i + 1) / n_frames
            image[row, col] = max(image[row, col], intensity)

        return image

    @staticmethod
    def preprocess_to_image(coords_2d: np.ndarray, size: int = 64) -> np.ndarray:
        """Normalize coordinates and convert them to an image feature map."""
        normalized = GesturePreprocessor.normalize(coords_2d)
        return GesturePreprocessor.encode_to_image(normalized, size=size)

    @staticmethod
    def augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand the dataset with horizontal, vertical, and combined flips."""
        X_aug = [
            X,
            X[:, :, ::-1],
            X[:, ::-1, :],
            X[:, ::-1, ::-1],
        ]
        y_aug = [y, y, y, y]
        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

    @staticmethod
    def flatten_and_scale(
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Flatten images to 1D vectors and fit a scaler on the training split only."""
        X_train_flat = X_train.reshape(len(X_train), -1)
        X_test_flat = X_test.reshape(len(X_test), -1)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        return X_train_scaled, X_test_scaled, scaler

