"""
pipeline.py
-----------

Preprocessing pipeline for gesture classification.

Walks data/raw/{1,2,3}, applies the full preprocessing chain to each
recording, and saves the resulting dataset to data/processed/dataset.npz.

Output dataset:
    X: np.ndarray of shape (n_samples, size, size), float32, values in [0, 1]
    y: np.ndarray of shape (n_samples,), int, values in {1, 2, 3}

Usage:
    python -m ml.pipeline
    python -m ml.pipeline --size 64
    python -m ml.pipeline --size 32 --raw_dir data/raw --out_dir data/processed
"""

import argparse
import os
import numpy as np

from src.gesture_processing.pca_projection import PCAGestureProcessor
from src.gesture_processing.preprocessor import GesturePreprocessor
from src.gesture_processing.img_encoder import GestureImageEncoder


CLASSES = [1, 2, 3]


def process_recording(file_path: str, size: int) -> np.ndarray | None:
    """
    Runs the full preprocessing pipeline on a single recording file.

    Steps:
        1. PCA projection → raw 2D coords
        2. Normalization  → centered, scale-stabilized, flip-corrected coords
        3. Image encoding → (size x size) float32 numpy array

    Args:
        file_path: path to a .npz recording file
        size: output image resolution in pixels

    Returns:
        np.ndarray of shape (size, size), or None if the file cannot be processed
    """
    coords_raw = PCAGestureProcessor.get_2d_coords(file_path)
    if coords_raw is None:
        return None

    coords_normalized = GesturePreprocessor.normalize(coords_raw)
    image = GestureImageEncoder.to_numpy(coords_normalized, size=size)

    return image


def build_dataset(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    size: int = 64,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Iterates over all class folders in raw_dir, processes each recording,
    and saves the resulting dataset to out_dir/dataset.npz.

    Args:
        raw_dir: root directory containing subfolders 1/, 2/, 3/
        out_dir: directory where dataset.npz will be saved
        size: image resolution passed to GestureImageEncoder.to_numpy
        verbose: whether to print progress

    Returns:
        X: np.ndarray of shape (n_samples, size, size)
        y: np.ndarray of shape (n_samples,)
    """
    X, y = [], []
    skipped = 0

    for label in CLASSES:
        class_dir = os.path.join(raw_dir, str(label))

        if not os.path.isdir(class_dir):
            print(f"Warning: class folder not found: {class_dir}")
            continue

        files = sorted([
            f for f in os.listdir(class_dir)
            if f.endswith(".npz")
        ])

        if verbose:
            print(f"\nClass {label}: {len(files)} recordings found")

        for fname in files:
            file_path = os.path.join(class_dir, fname)
            image = process_recording(file_path, size=size)

            if image is None:
                if verbose:
                    print(f"  [skip] {fname}")
                skipped += 1
                continue

            X.append(image)
            y.append(label)

            if verbose:
                print(f"  [ok]   {fname}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset.npz")
    np.savez(out_path, X=X, y=y)

    if verbose:
        print(f"\nDataset saved to: {out_path}")
        print(f"  X shape : {X.shape}")
        print(f"  y shape : {y.shape}")
        print(f"  Classes : {dict(zip(*np.unique(y, return_counts=True)))}")
        if skipped:
            print(f"  Skipped : {skipped} files")

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build gesture classification dataset.")
    parser.add_argument("--size",    type=int, default=64,             help="Image resolution (default: 64)")
    parser.add_argument("--raw_dir", type=str, default="data/raw",     help="Raw recordings root directory")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for dataset.npz")
    args = parser.parse_args()

    build_dataset(raw_dir=args.raw_dir, out_dir=args.out_dir, size=args.size)