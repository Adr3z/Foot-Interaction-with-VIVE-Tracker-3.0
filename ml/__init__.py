def _load_pipeline():
    from .dataset_builder import build_dataset, process_recording, CLASSES
    return build_dataset, process_recording, CLASSES


def _load_preview():
    from .preview import preview_dataset
    return preview_dataset


def train_knn(*args, **kwargs):
    from .trainer_knn import train
    return train(*args, **kwargs)


def train_svm(*args, **kwargs):
    from .trainer_svm import train
    return train(*args, **kwargs)


def run_ml_pipeline(
    raw_dir: str = "data/raw",
    out_dir: str = "data/processed",
    size: int = 64,
    verbose: bool = True,
    generate_preview: bool = True,
):
    """Central entry point for ML dataset creation and optional preview generation."""
    build_dataset, _, _ = _load_pipeline()
    return build_dataset(
        raw_dir=raw_dir,
        out_dir=out_dir,
        size=size,
        verbose=verbose,
        generate_preview=generate_preview,
    )


__all__ = ["run_ml_pipeline", "train_knn", "train_svm", "_load_pipeline", "_load_preview"]