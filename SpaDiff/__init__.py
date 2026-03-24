"""SpaDiff"""

__version__ = "0.2.0"

from .SpaDiff.config import SpaDiffConfig
from .SpaDiff.spadiff import SpaDiff
from .SpaDiff.spatial import Neiber, spatial_reconstruction
from .SpaDiff.topology import build_simplicial_operators, to_torch_operators
from .SpaDiff.train import train_spadiff

__all__ = [
    "Neiber",
    "spatial_reconstruction",
    "SpaDiff",
    "SpaDiffConfig",
    "build_simplicial_operators",
    "to_torch_operators",
    "train_spadiff",
]
