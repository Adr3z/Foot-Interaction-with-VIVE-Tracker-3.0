"""
realtime_classifier.py
----------------------
Classifies gestures in real-time from a rolling buffer of 3D tracker positions.

Pipeline per prediction step:
    N * [x,y,z] in buffer  →  PCA (2D)  →  normalize  →  encode image  →  flatten  →  SVM
"""

from __future__ import annotations

import pickle
from collections import deque

import numpy as np
from sklearn.decomposition import PCA

from src.gesture_processing import GesturePreprocessor


class RealtimeGestureClassifier:
    """
    Emits confirmed gesture labels (1, 2, or 3) from a live stream of 3D positions.

    Configurable at runtime:
        window_size     rolling buffer length in frames
        step_size       frames skipped between prediction attempts
        debounce_count  consecutive matching predictions required to confirm
    """

    def __init__(
        self,
        model_path: str,
        window_size: int = 90,
        step_size: int = 15,
        debounce_count: int = 3,
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.debounce_count = debounce_count

        self._buffer: deque[list[float]] = deque(maxlen=window_size)
        self._step_counter: int = 0
        self._pending_label: int | None = None
        self._pending_count: int = 0
        self.raw_label: int | None = None  # latest unconfirmed prediction

        self._model = None
        self._scaler = None
        self._image_size: int = 32
        self._load_model(model_path)

    # ── public ───────────────────────────────────────────────────────────────

    def add_sample(self, x: float, y: float, z: float) -> int | None:
        """
        Ingest a 3D tracker position sample.

        Returns a confirmed gesture label when debounce fires, None otherwise.
        """
        self._buffer.append([x, y, z])
        self._step_counter += 1

        if self._step_counter < self.step_size:
            return None
        self._step_counter = 0

        if len(self._buffer) < max(3, self.window_size // 2):
            return None

        self.raw_label = self._predict()
        return self._debounce(self.raw_label)

    def resize_window(self, new_size: int) -> None:
        """Resize the rolling buffer without discarding current history."""
        new_size = max(10, new_size)
        old_data = list(self._buffer)
        self.window_size = new_size
        self._buffer = deque(old_data[-new_size:], maxlen=new_size)
        self._step_counter = 0

    def reset(self) -> None:
        self._buffer.clear()
        self._step_counter = 0
        self._pending_label = None
        self._pending_count = 0
        self.raw_label = None

    # ── private ──────────────────────────────────────────────────────────────

    def _load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        self._model = saved["model"]
        self._scaler = saved["scaler"]
        # Infer image size from scaler feature count (size² = n_features)
        self._image_size = int(round(self._scaler.n_features_in_ ** 0.5))

    def _predict(self) -> int:
        positions = np.array(self._buffer, dtype=np.float64)
        coords_2d = PCA(n_components=2).fit_transform(positions)
        image = GesturePreprocessor.preprocess_to_image(coords_2d, size=self._image_size)
        X_scaled = self._scaler.transform(image.reshape(1, -1).astype(np.float32))
        return int(self._model.predict(X_scaled)[0])

    def _debounce(self, label: int) -> int | None:
        if label == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = label
            self._pending_count = 1

        if self._pending_count >= self.debounce_count:
            return label
        return None
