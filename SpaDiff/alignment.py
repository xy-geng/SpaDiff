from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SliceAffineTransform:
    """Global affine map from one slice into the reference coordinate system."""

    source_index: int
    reference_index: int
    matrix: np.ndarray
    translation: np.ndarray
    residual_rms: float

    def apply(self, coordinates) -> np.ndarray:
        values = np.asarray(coordinates, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError("coordinates must have shape [spots, at least 2]")
        return values[:, :2] @ self.matrix + self.translation


@dataclass(frozen=True)
class SerialAlignmentResult:
    """Aligned slice copies and diagnostics from :func:`align_serial_slices`."""

    slices: tuple[Any, ...]
    selected_genes: tuple[str, ...]
    candidate_genes: tuple[str, ...]
    moran_variance: np.ndarray
    transforms: tuple[SliceAffineTransform, ...]
    spatial_weight_sources: tuple[str, ...]
    reference_index: int
    aligned_key: str


def _validate_grid_shape(grid_shape) -> tuple[int, int]:
    if isinstance(grid_shape, (int, np.integer)):
        grid_shape = (int(grid_shape), int(grid_shape))
    if len(grid_shape) != 2:
        raise ValueError("grid_shape must be an integer or a length-2 sequence")
    shape = tuple(int(value) for value in grid_shape)
    if any(value <= 0 for value in shape):
        raise ValueError("grid_shape entries must be positive")
    return shape


def _coordinates(adata, spatial_key: str) -> np.ndarray:
    if not hasattr(adata, "obsm") or spatial_key not in adata.obsm:
        raise KeyError(f"slice does not contain obsm[{spatial_key!r}]")
    values = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != adata.n_obs or values.shape[1] < 2:
        raise ValueError(
            f"obsm[{spatial_key!r}] must have shape [n_obs, at least 2]"
        )
    values = values[:, :2]
    if not np.isfinite(values).all():
        raise ValueError("spatial coordinates must contain only finite values")
    return values


def _common_gene_indices(slices) -> tuple[list, list[np.ndarray]]:
    names_per_slice = []
    for adata in slices:
        if not hasattr(adata, "var_names"):
            raise AttributeError("every slice must provide var_names")
        names = list(adata.var_names)
        if len(names) != len(set(names)):
            raise ValueError("gene names must be unique before serial alignment")
        names_per_slice.append(names)

    common = set(names_per_slice[0])
    for names in names_per_slice[1:]:
        common.intersection_update(names)
    ordered = [name for name in names_per_slice[0] if name in common]
    if not ordered:
        raise ValueError("serial slices do not share any genes")

    indices = []
    for names in names_per_slice:
        lookup = {name: index for index, name in enumerate(names)}
        indices.append(np.asarray([lookup[name] for name in ordered], dtype=np.int64))
    return ordered, indices


def _expression_matrix(adata, gene_indices, layer: Optional[str]):
    import scipy.sparse as sp

    if layer is None:
        matrix = adata.X
    else:
        if not hasattr(adata, "layers") or layer not in adata.layers:
            raise KeyError(f"slice does not contain layers[{layer!r}]")
        matrix = adata.layers[layer]
    if matrix.shape[0] != adata.n_obs:
        raise ValueError("expression matrix row count does not match n_obs")
    if sp.issparse(matrix):
        selected = matrix.tocsr()[:, gene_indices].astype(np.float64)
        if selected.data.size and (
            not np.isfinite(selected.data).all() or np.min(selected.data) < 0.0
        ):
            raise ValueError(
                "alignment expression must be finite and non-negative; use a count "
                "or log-normalized layer rather than z-scored expression"
            )
        return selected

    selected = np.asarray(matrix, dtype=np.float64)[:, gene_indices]
    if not np.isfinite(selected).all() or np.min(selected) < 0.0:
        raise ValueError(
            "alignment expression must be finite and non-negative; use a count "
            "or log-normalized layer rather than z-scored expression"
        )
    return selected


def _coordinate_knn_weights(coordinates: np.ndarray, n_neighbors: int):
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    n = coordinates.shape[0]
    k = min(int(n_neighbors), n - 1)
    if k <= 0:
        raise ValueError("at least two spots are required to build spatial weights")
    estimator = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    distances, indices = estimator.kneighbors(coordinates)
    rows = []
    columns = []
    weights = []
    for row in range(n):
        keep = indices[row] != row
        local_indices = indices[row][keep][:k]
        local_distances = distances[row][keep][:k]
        positive = local_distances[local_distances > 0]
        bandwidth = float(np.median(positive)) if positive.size else 1.0
        bandwidth = max(bandwidth, 1e-12)
        local_weights = np.exp(-np.square(local_distances / bandwidth))
        rows.extend([row] * len(local_indices))
        columns.extend(local_indices.tolist())
        weights.extend(local_weights.tolist())
    matrix = sp.csr_matrix(
        (np.asarray(weights), (rows, columns)), shape=(n, n), dtype=np.float64
    )
    matrix = matrix.maximum(matrix.T).tolil()
    matrix.setdiag(0)
    matrix = matrix.tocsr()
    matrix.eliminate_zeros()
    return matrix


def _validate_spatial_weights(matrix, n_obs: int):
    import scipy.sparse as sp

    values = sp.csr_matrix(matrix, dtype=np.float64)
    if values.shape != (n_obs, n_obs):
        raise ValueError(f"spatial weights must have shape [{n_obs}, {n_obs}]")
    if values.data.size and (
        not np.isfinite(values.data).all() or np.min(values.data) < 0.0
    ):
        raise ValueError("spatial weights must be finite and non-negative")
    values = values.maximum(values.T).tolil()
    values.setdiag(0)
    values = values.tocsr()
    values.eliminate_zeros()
    return values


def _resolve_spatial_weights(
    adata,
    coordinates: np.ndarray,
    explicit,
    *,
    spatial_weight_key: Optional[str],
    n_neighbors: int,
):
    if explicit is not None:
        return _validate_spatial_weights(explicit, adata.n_obs), "explicit"
    if (
        spatial_weight_key is not None
        and hasattr(adata, "obsp")
        and spatial_weight_key in adata.obsp
    ):
        return (
            _validate_spatial_weights(adata.obsp[spatial_weight_key], adata.n_obs),
            f"obsp[{spatial_weight_key!r}]",
        )
    return _coordinate_knn_weights(coordinates, n_neighbors), "coordinate_knn_fallback"


def _grid_regions(coordinates: np.ndarray, grid_shape: tuple[int, int]):
    bins = []
    for axis, count in enumerate(grid_shape):
        lower = float(coordinates[:, axis].min())
        upper = float(coordinates[:, axis].max())
        if np.isclose(lower, upper):
            bins.append(np.zeros(coordinates.shape[0], dtype=np.int64))
            continue
        edges = np.linspace(lower, upper, count + 1)
        index = np.searchsorted(edges, coordinates[:, axis], side="right") - 1
        bins.append(np.clip(index, 0, count - 1))
    labels = bins[0] * grid_shape[1] + bins[1]
    return [np.flatnonzero(labels == value) for value in np.unique(labels)]


def _moran_by_gene(expression, weights) -> np.ndarray:
    import scipy.sparse as sp

    values = expression.toarray() if sp.issparse(expression) else np.asarray(expression)
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    weight_sum = float(weights.sum())
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    if n < 2 or weight_sum <= 0.0:
        return result
    centered = values - values.mean(axis=0, keepdims=True)
    denominator = np.square(centered).sum(axis=0)
    numerator = np.sum(centered * (weights @ centered), axis=0)
    valid = denominator > 1e-12
    result[valid] = (n / weight_sum) * numerator[valid] / denominator[valid]
    return result


def _moran_variance(
    expressions,
    coordinates,
    weights,
    *,
    grid_shape: tuple[int, int],
    min_region_spots: int,
    n_neighbors: int,
) -> np.ndarray:
    rows = []
    for expression, coords, matrix in zip(expressions, coordinates, weights):
        for indices in _grid_regions(coords, grid_shape):
            if len(indices) < min_region_spots:
                continue
            local_weights = matrix[indices][:, indices].tocsr()
            if local_weights.nnz == 0:
                local_weights = _coordinate_knn_weights(
                    coords[indices], min(n_neighbors, len(indices) - 1)
                )
            rows.append(_moran_by_gene(expression[indices], local_weights))
    if len(rows) < 2:
        raise ValueError(
            "fewer than two valid spatial subregions were available for Moran's I"
        )
    scores = np.vstack(rows)
    finite = np.isfinite(scores)
    counts = finite.sum(axis=0)
    means = np.divide(
        np.nansum(scores, axis=0),
        counts,
        out=np.zeros(scores.shape[1], dtype=np.float64),
        where=counts > 0,
    )
    centered = np.where(finite, scores - means, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        counts,
        out=np.full(scores.shape[1], -np.inf, dtype=np.float64),
        where=counts >= 2,
    )
    return variance


def _expression_weighted_centroids(expression, coordinates: np.ndarray):
    totals = np.asarray(expression.sum(axis=0)).ravel()
    weighted = np.asarray(expression.T @ coordinates)
    centroids = np.divide(
        weighted,
        totals[:, None],
        out=np.full((expression.shape[1], 2), np.nan, dtype=np.float64),
        where=totals[:, None] > 1e-12,
    )
    return centroids


def _fit_affine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_index: int,
    reference_index: int,
) -> SliceAffineTransform:
    design = np.column_stack((source, np.ones(source.shape[0], dtype=np.float64)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise ValueError(
            f"anchor centroids for slice {source_index} do not span a 2D affine map"
        )
    matrix = coefficients[:2]
    translation = coefficients[2]
    residual = design @ coefficients - target
    rms = float(np.sqrt(np.mean(np.square(residual))))
    return SliceAffineTransform(
        source_index=source_index,
        reference_index=reference_index,
        matrix=matrix,
        translation=translation,
        residual_rms=rms,
    )


def align_serial_slices(
    slices: Sequence,
    *,
    reference_index: int = 0,
    spatial_key: str = "spatial",
    aligned_key: str = "spatial_aligned",
    expression_layer: Optional[str] = None,
    spatial_weights: Optional[Sequence] = None,
    spatial_weight_key: Optional[str] = "spagcn_connectivities",
    grid_shape=(4, 4),
    n_anchor_genes: int = 100,
    spatial_neighbors: int = 6,
    min_region_spots: int = 4,
    copy: bool = True,
) -> SerialAlignmentResult:
    """Align serial slices using the manuscript's Moran-centroid linear map.

    Each slice is partitioned by evenly spaced x/y grid lines. Gene-wise Moran's
    I is evaluated within every sufficiently populated subregion, and the genes
    with the largest variance across all subregions and slices are retained as
    spatial anchors. Expression-weighted gene centroids are then mapped into the
    reference slice by one global affine layer with no nonlinear activation.

    For the strict paper workflow, provide SpaGCN-derived weights either through
    ``spatial_weights`` or ``adata.obsp[spatial_weight_key]``. When neither is
    available, the function emits a warning and uses coordinate-kNN weights so
    that the preprocessing step remains usable without a SpaGCN dependency.
    The returned transforms map every target slice into the fixed reference
    coordinate system and never apply local nonlinear warping.
    """

    slices = tuple(slices)
    if len(slices) < 2:
        raise ValueError("align_serial_slices requires at least two slices")
    if not 0 <= int(reference_index) < len(slices):
        raise IndexError("reference_index is outside the slice sequence")
    if n_anchor_genes <= 0 or spatial_neighbors <= 0 or min_region_spots < 2:
        raise ValueError(
            "n_anchor_genes and spatial_neighbors must be positive and "
            "min_region_spots must be at least 2"
        )
    grid_shape = _validate_grid_shape(grid_shape)
    if spatial_weights is not None and len(spatial_weights) != len(slices):
        raise ValueError("spatial_weights must contain one matrix per slice")

    coords = [_coordinates(adata, spatial_key) for adata in slices]
    common_genes, gene_indices = _common_gene_indices(slices)
    expressions = [
        _expression_matrix(adata, indices, expression_layer)
        for adata, indices in zip(slices, gene_indices)
    ]

    resolved_weights = []
    sources = []
    for index, (adata, values) in enumerate(zip(slices, coords)):
        explicit = None if spatial_weights is None else spatial_weights[index]
        matrix, source = _resolve_spatial_weights(
            adata,
            values,
            explicit,
            spatial_weight_key=spatial_weight_key,
            n_neighbors=spatial_neighbors,
        )
        resolved_weights.append(matrix)
        sources.append(source)
    if any(source == "coordinate_knn_fallback" for source in sources):
        warnings.warn(
            "SpaGCN-derived spatial weights were not available for every slice; "
            "coordinate-kNN weights are being used as a documented fallback",
            RuntimeWarning,
            stacklevel=2,
        )

    moran_variance = _moran_variance(
        expressions,
        coords,
        resolved_weights,
        grid_shape=grid_shape,
        min_region_spots=min_region_spots,
        n_neighbors=spatial_neighbors,
    )
    centroids = [
        _expression_weighted_centroids(expression, values)
        for expression, values in zip(expressions, coords)
    ]
    valid_centroids = np.logical_and.reduce(
        [np.isfinite(values).all(axis=1) for values in centroids]
    )
    ranked = np.argsort(-moran_variance, kind="stable")
    selected = [
        index
        for index in ranked
        if np.isfinite(moran_variance[index]) and valid_centroids[index]
    ][:n_anchor_genes]
    if len(selected) < 3:
        raise ValueError("fewer than three valid anchor genes were available")
    if len(selected) < n_anchor_genes:
        warnings.warn(
            f"only {len(selected)} of {n_anchor_genes} requested anchor genes had "
            "valid Moran statistics and expression-weighted centroids",
            RuntimeWarning,
            stacklevel=2,
        )
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_genes = tuple(str(common_genes[index]) for index in selected)

    reference = int(reference_index)
    target_centroids = centroids[reference][selected_array]
    transforms = []
    aligned_coordinates = []
    for index, values in enumerate(coords):
        if index == reference:
            transform = SliceAffineTransform(
                source_index=index,
                reference_index=reference,
                matrix=np.eye(2, dtype=np.float64),
                translation=np.zeros(2, dtype=np.float64),
                residual_rms=0.0,
            )
        else:
            transform = _fit_affine(
                centroids[index][selected_array],
                target_centroids,
                source_index=index,
                reference_index=reference,
            )
        transforms.append(transform)
        aligned_coordinates.append(transform.apply(values))

    outputs = tuple(adata.copy() if copy else adata for adata in slices)
    for index, (adata, aligned, transform) in enumerate(
        zip(outputs, aligned_coordinates, transforms)
    ):
        adata.obsm[aligned_key] = aligned
        if not hasattr(adata, "uns"):
            adata.uns = {}
        adata.uns["spadiff_serial_alignment"] = {
            "slice_index": int(index),
            "reference_index": reference,
            "spatial_key": spatial_key,
            "aligned_key": aligned_key,
            "expression_layer": expression_layer,
            "selected_genes": np.asarray(selected_genes, dtype=str),
            "matrix": transform.matrix.copy(),
            "translation": transform.translation.copy(),
            "residual_rms": transform.residual_rms,
            "spatial_weight_source": sources[index],
        }

    return SerialAlignmentResult(
        slices=outputs,
        selected_genes=selected_genes,
        candidate_genes=tuple(str(name) for name in common_genes),
        moran_variance=moran_variance,
        transforms=tuple(transforms),
        spatial_weight_sources=tuple(sources),
        reference_index=reference,
        aligned_key=aligned_key,
    )
