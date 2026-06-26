"""
trainer.py
----------

Trains a k-NN classifier on the preprocessed gesture image dataset.

Pipeline:
    1. Load data/processed/dataset.npz
    2. Stratified train/test split (80/20)
    3. Augment train set only (horizontal, vertical, and both flips)
    4. Flatten images and normalize features
    5. Train k-NN classifier
    6. Evaluate on test set and report results
    7. Save trained model to ml/models/knn_model.pkl

Usage:
    python -m ml.trainer
    python -m ml.trainer --dataset data/processed/dataset.npz
    python -m ml.trainer --dataset data/processed/dataset.npz --k 5 --test_size 0.2
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.gesture_processing import GesturePreprocessor


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
CLASS_NAMES = {1: "One", 2: "Two", 3: "Three"}


# ── preprocessing ─────────────────────────────────────────────────────────────

def flatten_and_scale(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Delegate feature flattening and scaling to the shared gesture preprocessor."""
    return GesturePreprocessor.flatten_and_scale(X_train, X_test)


# ── evaluation ────────────────────────────────────────────────────────────────

def print_results(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Prints accuracy, classification report, and confusion matrix."""
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)], zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    out_dir: str
) -> None:
    """Saves a confusion matrix plot to out_dir/confusion_matrix.png."""
    cm = confusion_matrix(y_test, y_pred)
    labels = [CLASS_NAMES[c] for c in sorted(CLASS_NAMES)]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix - k-NN")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "confusion_matrix_knn.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nConfusion matrix saved to: {out_path}")


# ── model persistence ─────────────────────────────────────────────────────────

def save_model(
    model: KNeighborsClassifier,
    scaler: StandardScaler,
    out_dir: str
) -> None:
    """Saves the trained model and scaler as a single pickle file."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "knn_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Model saved to: {out_path}")


# ── main training function ────────────────────────────────────────────────────

def train(
    dataset_path: str = "data/processed/dataset.npz",
    k: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> dict:
    """
    Runs the full training pipeline.

    Args:
        dataset_path: path to the dataset.npz file
        k: number of neighbors for k-NN
        test_size: fraction of data reserved for test (default: 0.2)
        random_state: random seed for reproducibility
        verbose: whether to print progress

    Returns:
        dict with keys: model, scaler, accuracy, y_test, y_pred
    """
    # ── load ─────────────────────────────────────────────────────────────────
    data = np.load(dataset_path)
    X, y = data["X"], data["y"]

    if verbose:
        print(f"Loaded dataset: {X.shape[0]} samples, image size {X.shape[1]}x{X.shape[2]}")
        for cls in sorted(np.unique(y)):
            print(f"  Class {cls}: {np.sum(y == cls)} samples")

    # ── stratified split ──────────────────────────────────────────────────────
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if verbose:
        print(f"\nSplit: {len(X_train)} train / {len(X_test)} test")

    # ── augment train only ────────────────────────────────────────────────────
    X_train_aug, y_train_aug = GesturePreprocessor.augment(X_train, y_train)

    if verbose:
        print(f"After augmentation: {len(X_train_aug)} train samples")

    # ── flatten and scale ─────────────────────────────────────────────────────
    X_train_scaled, X_test_scaled, scaler = flatten_and_scale(X_train_aug, X_test)

    # ── train ─────────────────────────────────────────────────────────────────
    model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    model.fit(X_train_scaled, y_train_aug)

    if verbose:
        print(f"\nTrained k-NN with k={k}")

    # ── evaluate ──────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_scaled)

    if verbose:
        print_results(y_test, y_pred)
        save_confusion_matrix(y_test, y_pred, out_dir="data/plots")

    # ── save model ────────────────────────────────────────────────────────────
    save_model(model, scaler, out_dir=MODELS_DIR)

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": accuracy_score(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train k-NN gesture classifier.")
    parser.add_argument("--dataset",    type=str,   default="data/processed/dataset.npz")
    parser.add_argument("--k",          type=int,   default=3,   help="Number of neighbors (default: 3)")
    parser.add_argument("--test_size",  type=float, default=0.2, help="Test set fraction (default: 0.2)")
    args = parser.parse_args()

    train(dataset_path=args.dataset, k=args.k, test_size=args.test_size)

