"""P0.3 — PyTorch tensors and the training-loop skeleton."""

from .data import (
    FEATURE_COLUMNS,
    ReturnsDataset,
    build_dataset,
    chronological_split,
    standardise,
)
from .loop import TrainConfig, evaluate, seed_everything, train
from .model import TinyMLP, count_parameters

__all__ = [
    "FEATURE_COLUMNS",
    "ReturnsDataset",
    "TinyMLP",
    "TrainConfig",
    "build_dataset",
    "chronological_split",
    "count_parameters",
    "evaluate",
    "seed_everything",
    "standardise",
    "train",
]
