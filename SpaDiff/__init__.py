"""SpaDiff: topology-aware conditional score diffusion."""

__version__ = "0.1.0"

from .config import SpaDiffConfig
from .spadiff import SpaDiff
from .spatial import Neiber, spatial_reconstruction
from .topology import build_simplicial_operators, to_torch_operators
from .train import TrainingResult, train_spadiff

__all__ = [
    "Neiber",
    "SpaDiff",
    "SpaDiffConfig",
    "TrainingResult",
    "build_simplicial_operators",
    "spatial_reconstruction",
    "to_torch_operators",
    "train_spadiff",
]
