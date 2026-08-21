"""SpaDiff: topology-aware conditional score diffusion."""

from ._version import __version__

from .alignment import (
    SerialAlignmentResult,
    SliceAffineTransform,
    align_serial_slices,
)
from .config import SpaDiffConfig
from .denoising import (
    inverse_pca_expression,
    row_normalize_adjacency,
    smooth_generated_expression,
    write_denoised_expression,
)
from .pipeline import SpaDiffPipeline
from .multiomics import (
    build_spatially_regularized_connectivity,
    robust_atac_lsi,
)
from .spadiff import SpaDiff
from .spatial import (
    Neiber,
    SpatialTopologyResult,
    build_spatial_topology,
    spatial_reconstruction,
)
from .topology import build_simplicial_operators, to_torch_operators
from .train import TrainingResult, train_spadiff

__all__ = [
    "Neiber",
    "SerialAlignmentResult",
    "SliceAffineTransform",
    "SpaDiff",
    "SpaDiffConfig",
    "SpaDiffPipeline",
    "SpatialTopologyResult",
    "TrainingResult",
    "align_serial_slices",
    "build_simplicial_operators",
    "build_spatial_topology",
    "build_spatially_regularized_connectivity",
    "inverse_pca_expression",
    "row_normalize_adjacency",
    "robust_atac_lsi",
    "smooth_generated_expression",
    "spatial_reconstruction",
    "to_torch_operators",
    "train_spadiff",
    "write_denoised_expression",
]
