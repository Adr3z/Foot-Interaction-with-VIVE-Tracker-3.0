"""
trainer_svm.py
--------------

Retrains the SVM classifier on the full augmented dataset using the best
hyperparameters already found by trainer_svm_validation.py.

Pipeline:
    1. Load best hyperparameters from ml/models/svm_model.pkl (GridSearch result)
    2. Load data/processed/dataset_augmented.npz
    3. Flatten images and normalize features
    4. Retrain SVM with the same C and gamma on all augmented samples
    5. Save ml/models/svm_model.pkl with the new model

Usage:
    python -m ml.trainer_svm
    python -m ml.trainer_svm --dataset data/processed/dataset_augmented.npz
                            --source_model ml/models/svm_model.pkl
"""

import argparse
import os
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")


# ── model persistence ─────────────────────────────────────────────────────────

def load_model(model_path: str) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def save_model(model: SVC, scaler: StandardScaler, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "svm_model_final.pkl")
    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"Model saved to: {out_path}")


# ── main training function ────────────────────────────────────────────────────

def train(
    dataset_path: str = "data/processed/dataset_augmented.npz",
    source_model_path: str = DEFAULT_MODEL_PATH,
    verbose: bool = True,
) -> dict:
    """
    Retrains the validated SVM on the full augmented dataset.

    Args:
        dataset_path: path to the augmented dataset_augmented.npz file
        source_model_path: path to the pkl produced by trainer_svm_validation
        verbose: whether to print progress

    Returns:
        dict with keys: model, scaler
    """
    # ── load best hyperparameters from validation model ────────────────────────
    if not os.path.exists(source_model_path):
        raise FileNotFoundError(
            f"Validation model not found at {source_model_path}. "
            "Run trainer_svm_validation first."
        )
    saved = load_model(source_model_path)
    validated_svm: SVC = saved["model"]
    params = validated_svm.get_params()

    if verbose:
        print(f"Loaded hyperparameters from: {source_model_path}")
        print(f"  C={params['C']}, gamma={params['gamma']}, kernel={params['kernel']}")

    # ── load augmented dataset ─────────────────────────────────────────────────
    data = np.load(dataset_path)
    X, y = data["X"], data["y"]

    if verbose:
        print(f"\nLoaded dataset: {X.shape[0]} samples, image size {X.shape[1]}x{X.shape[2]}")
        for cls in sorted(np.unique(y)):
            print(f"  Class {cls}: {np.sum(y == cls)} samples")

    # ── flatten and scale ─────────────────────────────────────────────────────
    X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # ── retrain with validated hyperparameters ────────────────────────────────
    model = SVC(**params)
    model.fit(X_scaled, y)

    if verbose:
        print(f"\nRetrained SVM on {X.shape[0]} augmented samples.")

    # ── save model ────────────────────────────────────────────────────────────
    save_model(model, scaler, out_dir=MODELS_DIR)

    return {"model": model, "scaler": scaler}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain SVM with validated hyperparameters on the augmented dataset."
    )
    parser.add_argument("--dataset",      type=str, default="data/processed/dataset_augmented.npz")
    parser.add_argument("--source_model", type=str, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    train(dataset_path=args.dataset, source_model_path=args.source_model)
