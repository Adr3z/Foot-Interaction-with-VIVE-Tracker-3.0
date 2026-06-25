from .preprocessor import GesturePreprocessor

__all__ = ["GesturePreprocessor", "PCAGestureProcessor", "GestureImageEncoder"]


def __getattr__(name):
    if name == "PCAGestureProcessor":
        from .pca_projection import PCAGestureProcessor
        return PCAGestureProcessor
    if name == "GestureImageEncoder":
        from .img_encoder import GestureImageEncoder
        return GestureImageEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")