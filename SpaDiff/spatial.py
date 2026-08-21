"""Spatial graph and topology utilities for aligned tissue coordinates."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence

import numpy as np

from .topology import build_simplicial_operators, to_torch_operators


TopologyMode = Literal["slice_aware", "global_knn"]


@dataclass(frozen=True)
class SpatialTopologyResult:
    """Spatial graph and higher-order operators produced by one build call."""

    coordinates: np.ndarray
    adjacency: Any
    operators: dict[int, Any]
    simplex_counts: dict[int, int]
    mode: TopologyMode


def _coordinates_from_adata(adata, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {spatial_key!r}")
    coordinates = np.asarray(adata.obsm[spatial_key])
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have shape [n_obs, dimensions]"
        )
    if coordinates.shape[0] < 2:
        raise ValueError("at least two spots are required to build a spatial graph")
    if not np.isfinite(coordinates).all():
        raise ValueError("spatial coordinates must contain only finite values")
    return coordinates


def _symmetrize_adjacency(adjacency):
    matrix = adjacency.tocsr()
    matrix = matrix.maximum(matrix.T.tocsr()).tocsr()
    matrix = matrix.tolil()
    matrix.setdiag(0)
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _build_global_knn_adjacency(
    coordinates: np.ndarray,
    *,
    n_neighbors: int,
):
    """Reproduce the tutorial's all-spots kNN construction."""

    from sklearn.neighbors import NearestNeighbors

    if isinstance(n_neighbors, bool) or not isinstance(
        n_neighbors, (int, np.integer)
    ):
        raise TypeError("n_neighbors must be an integer")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    k = min(int(n_neighbors), coordinates.shape[0])
    estimator = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
        coordinates
    )
    adjacency = estimator.kneighbors_graph(coordinates, mode="connectivity")
    return _symmetrize_adjacency(adjacency)


def _validate_slice_order(slice_ids: np.ndarray, slice_order: Optional[Sequence]):
    observed_order = list(dict.fromkeys(slice_ids.tolist()))
    if slice_order is None:
        return observed_order

    requested_order = list(slice_order)
    if not requested_order:
        raise ValueError("slice_order must not be empty")
    if len(set(requested_order)) != len(requested_order):
        raise ValueError("slice_order must not contain duplicate labels")
    observed = set(observed_order)
    requested = set(requested_order)
    if observed != requested:
        missing = [value for value in observed_order if value not in requested]
        unknown = [value for value in requested_order if value not in observed]
        raise ValueError(
            "slice_order must contain every observed slice exactly once; "
            f"missing={missing}, unknown={unknown}"
        )
    return requested_order


def _build_slice_aware_adjacency(
    adata,
    coordinates: np.ndarray,
    *,
    batch_key: str,
    slice_order: Optional[Sequence],
    k_intra: int,
    k_inter: int,
):
    """Build the legacy within-slice and consecutive-slice graph exactly."""

    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    for name, value, allow_zero in (
        ("k_intra", k_intra, False),
        ("k_inter", k_inter, True),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        if value < int(not allow_zero):
            requirement = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {requirement}")

    if batch_key not in adata.obs:
        raise KeyError(f"adata.obs does not contain {batch_key!r}")
    batch_values = adata.obs[batch_key]
    if hasattr(batch_values, "isna") and bool(batch_values.isna().any()):
        raise ValueError(f"adata.obs[{batch_key!r}] contains missing labels")
    slice_ids = np.asarray(batch_values.values)
    ordered_slices = _validate_slice_order(slice_ids, slice_order)

    n = coordinates.shape[0]
    adjacency = sp.lil_matrix((n, n), dtype=np.float32)
    for slice_id in ordered_slices:
        index = np.flatnonzero(slice_ids == slice_id)
        k = min(int(k_intra), len(index))
        if k <= 1:
            continue
        estimator = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
            coordinates[index]
        )
        graph = estimator.kneighbors_graph(
            coordinates[index], mode="connectivity"
        )
        graph.setdiag(0)
        graph.eliminate_zeros()
        adjacency[np.ix_(index, index)] = graph

    if k_inter > 0:
        for left_id, right_id in zip(ordered_slices[:-1], ordered_slices[1:]):
            left = np.flatnonzero(slice_ids == left_id)
            right = np.flatnonzero(slice_ids == right_id)
            if not len(left) or not len(right):
                continue
            k_lr = min(int(k_inter), len(right))
            k_rl = min(int(k_inter), len(left))
            lr = (
                NearestNeighbors(n_neighbors=k_lr)
                .fit(coordinates[right])
                .kneighbors_graph(coordinates[left], mode="connectivity")
            )
            rl = (
                NearestNeighbors(n_neighbors=k_rl)
                .fit(coordinates[left])
                .kneighbors_graph(coordinates[right], mode="connectivity")
            )
            adjacency[np.ix_(left, right)] = lr
            adjacency[np.ix_(right, left)] = rl
    return _symmetrize_adjacency(adjacency)


def build_spatial_topology(
    adata,
    *,
    mode: TopologyMode = "slice_aware",
    spatial_key: str = "spatial",
    batch_key: str = "batch_name",
    slice_order: Optional[Sequence] = None,
    k_intra: int = 6,
    k_inter: int = 2,
    n_neighbors: int = 10,
    max_order: int = 2,
    device=None,
    verbose: bool = True,
) -> SpatialTopologyResult:
    """Build a spatial graph and its node-level simplicial operators.

    ``slice_aware`` preserves the legacy multi-slice construction: kNN edges
    are built within slices and between each consecutive pair in
    ``slice_order``. ``global_knn`` builds one kNN graph across all coordinates.
    Both modes return a symmetric CSR graph with an empty diagonal.
    """

    normalized_mode = str(mode).lower()
    if normalized_mode not in {"slice_aware", "global_knn"}:
        raise ValueError("mode must be 'slice_aware' or 'global_knn'")
    coordinates = _coordinates_from_adata(adata, spatial_key)
    if normalized_mode == "slice_aware":
        adjacency = _build_slice_aware_adjacency(
            adata,
            coordinates,
            batch_key=batch_key,
            slice_order=slice_order,
            k_intra=k_intra,
            k_inter=k_inter,
        )
    else:
        adjacency = _build_global_knn_adjacency(
            coordinates, n_neighbors=n_neighbors
        )

    scipy_operators, simplex_counts = build_simplicial_operators(
        adjacency,
        max_order=max_order,
        verbose=verbose,
        return_counts=True,
    )
    operators = to_torch_operators(scipy_operators, device=device)
    return SpatialTopologyResult(
        coordinates=coordinates,
        adjacency=adjacency,
        operators=operators,
        simplex_counts=simplex_counts,
        mode=normalized_mode,
    )


def spatial_reconstruction(
    adata,
    alpha: float = 0.0,
    n_neighbors: int = 10,
    n_pcs: int = 15,
    use_highly_variable: Optional[bool] = None,
):
    """Smooth expression over a single all-spots spatial kNN graph."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    coordinates = _coordinates_from_adata(adata, "spatial")
    adjacency = _build_global_knn_adjacency(
        coordinates, n_neighbors=n_neighbors
    )

    if alpha > 0.0:
        import scanpy as sc
        from scipy.sparse import csr_matrix, issparse
        from sklearn.metrics.pairwise import cosine_distances

        sc.pp.pca(
            adata,
            n_comps=n_pcs,
            use_highly_variable=use_highly_variable,
        )
        distances = np.exp(2.0 - cosine_distances(adata.obsm["X_pca"])) - 1.0
        connections = adjacency.T.toarray() * distances
        values = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        denominator = np.sum(connections, axis=0, keepdims=True)
        normalized = np.divide(
            connections,
            denominator,
            out=np.zeros_like(connections),
            where=denominator > 0,
        )
        adata.X = csr_matrix(alpha * np.matmul(normalized, values) + values)
        del adata.obsm["X_pca"]
    return adata, adjacency


def Neiber(
    adata,
    k_intra: int = 6,
    k_inter: int = 2,
    slice_order: Optional[Sequence] = None,
    *,
    spatial_key: str = "spatial",
    batch_key: str = "batch_name",
):
    """Compatibility wrapper for the legacy multi-slice neighbor builder."""

    warnings.warn(
        "Neiber is deprecated; use build_spatial_topology(mode='slice_aware')",
        DeprecationWarning,
        stacklevel=2,
    )
    coordinates = _coordinates_from_adata(adata, spatial_key)
    adjacency = _build_slice_aware_adjacency(
        adata,
        coordinates,
        batch_key=batch_key,
        slice_order=slice_order,
        k_intra=k_intra,
        k_inter=k_inter,
    )
    return coordinates, adjacency
