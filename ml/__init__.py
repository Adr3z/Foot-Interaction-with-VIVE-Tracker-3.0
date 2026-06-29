def _load_pipeline():
    from .dataset_builder import build_dataset, process_recording, CLASSES
    return build_dataset, process_recording, CLASSES


def _load_preview():
    from .preview import preview_dataset
    return preview_dataset


def train_knn(*args, **kwargs):
    from .trainer_knn_validation import train
    return train(*args, **kwargs)


def train_svm_validation(*args, **kwargs):
    from .trainer_svm_validation import train
    return train(*args, **kwargs)


def train_svm_final(*args, **kwargs):
    from .trainer_svm import train
    return train(*args, **kwargs)


def run_ml_pipeline(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    size: int = 64,
    verbose: bool = True,
    generate_preview: bool = True,
):
    """Builds the base dataset from raw recordings."""
    build_dataset, _, _ = _load_pipeline()
    return build_dataset(
        raw_dir=raw_dir,
        out_dir=out_dir,
        size=size,
        verbose=verbose,
        generate_preview=generate_preview,
    )


def run_augmented_pipeline(
    input_path: str = "data/processed/dataset.npz",
    output_path: str = "data/processed/dataset_augmented.npz",
):
    """Builds the augmented dataset from the base dataset."""
    from .dataset_augmented import create_augmented_dataset
    return create_augmented_dataset(input_path=input_path, output_path=output_path)


from .realtime_classifier import RealtimeGestureClassifier

__all__ = [
    "RealtimeGestureClassifier",
    "run_ml_pipeline",
    "run_augmented_pipeline",
    "train_knn",
    "train_svm_validation",
    "train_svm_final",
    "_load_pipeline",
    "_load_preview",
]