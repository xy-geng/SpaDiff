"""Preprocessing and graph helpers used by the case4 multi-omics workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np


def standardize_columns(values, eps: float = 1e-6) -> np.ndarray:
    """Standardize latent dimensions across spots."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [spots, features]")
    mean = array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, ddof=0, keepdims=True)
    standardized = np.divide(
        array - mean,
        scale,
        out=np.zeros_like(array),
        where=scale > eps,
    )
    return np.asarray(standardized, dtype=np.float32)


def robust_atac_lsi(
    adata,
    *,
    n_components: int = 50,
    min_cells: int = 10,
    scale_factor: float = 1e4,
    drop_depth_component: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Compute standardized TF-IDF/LSI features from raw ATAC counts."""

    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if min_cells <= 0:
        raise ValueError("min_cells must be positive")

    matrix = (
        adata.X.tocsr().astype(np.float64, copy=True)
        if sp.issparse(adata.X)
        else sp.csr_matrix(np.asarray(adata.X, dtype=np.float64))
    )
    if matrix.shape[0] < 3:
        raise ValueError("at least three spots are required for LSI")
    if matrix.data.size and np.min(matrix.data) < 0:
        raise ValueError("ATAC counts must be non-negative")

    detected = np.asarray(matrix.getnnz(axis=0)).ravel()
    selected = detected >= min_cells
    if selected.sum() < 3:
        raise ValueError("too few ATAC peaks remain after min_cells filtering")
    matrix = matrix[:, selected]

    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(library_size <= 0):
        count = int(np.sum(library_size <= 0))
        raise ValueError(f"{count} ATAC spots have zero counts after peak filtering")

    term_frequency = sp.diags(1.0 / library_size) @ matrix
    document_frequency = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    inverse_document_frequency = np.log1p(
        matrix.shape[0] / np.maximum(document_frequency, 1.0)
    )
    tfidf = (term_frequency @ sp.diags(inverse_document_frequency)).tocsr()
    tfidf.data = np.log1p(tfidf.data * scale_factor)

    maximum_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    requested = n_components + int(drop_depth_component)
    fitted_components = min(requested, maximum_components)
    if fitted_components < 2:
        raise ValueError("ATAC matrix is too small for the requested LSI")

    decomposition = TruncatedSVD(
        n_components=fitted_components,
        algorithm="randomized",
        random_state=random_state,
    )
    values = decomposition.fit_transform(tfidf)

    log_depth = np.log1p(library_size)
    correlations = np.zeros(values.shape[1], dtype=np.float64)
    if np.std(log_depth) > 0:
        for index in range(values.shape[1]):
            if np.std(values[:, index]) > 0:
                correlations[index] = np.corrcoef(values[:, index], log_depth)[0, 1]
    correlations = np.nan_to_num(correlations)

    depth_component: Optional[int] = None
    kept = list(range(values.shape[1]))
    if drop_depth_component and len(kept) > 1:
        depth_component = int(np.argmax(np.abs(correlations)))
        kept.remove(depth_component)
    kept = kept[:n_components]
    values = standardize_columns(values[:, kept])

    diagnostics = {
        "n_input_peaks": int(adata.n_vars),
        "n_selected_peaks": int(selected.sum()),
        "n_components": int(values.shape[1]),
        "depth_component": depth_component,
        "depth_correlations": correlations,
        "kept_components": np.asarray(kept, dtype=np.int64),
        "explained_variance_ratio": decomposition.explained_variance_ratio_,
    }
    return values, diagnostics


def build_spatially_regularized_connectivity(
    adata,
    spatial_adjacency,
    *,
    use_rep: str = "spadiff",
    n_neighbors: int = 10,
    latent_weight: float = 0.4,
    spatial_weight: float = 0.6,
    random_state: int = 42,
    neighbors_key: str = "spadiff",
    output_key: str = "spadiff_spatial_connectivities",
):
    """Mix a latent-neighbor graph with a binary spatial graph."""

    import scanpy as sc
    import scipy.sparse as sp

    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {use_rep!r}")
    if adata.n_obs < 2:
        raise ValueError("at least two observations are required")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if latent_weight < 0.0 or spatial_weight < 0.0:
        raise ValueError("connectivity weights must be non-negative")
    if latent_weight + spatial_weight <= 0.0:
        raise ValueError("at least one connectivity weight must be positive")
    if not neighbors_key or not output_key:
        raise ValueError("connectivity keys must be non-empty strings")

    spatial = (
        spatial_adjacency.tocsr().astype(np.float32, copy=True)
        if sp.issparse(spatial_adjacency)
        else sp.csr_matrix(np.asarray(spatial_adjacency, dtype=np.float32))
    )
    expected_shape = (adata.n_obs, adata.n_obs)
    if spatial.shape != expected_shape:
        raise ValueError(
            f"spatial_adjacency must have shape {expected_shape}, got {spatial.shape}"
        )

    sc.pp.neighbors(
        adata,
        n_neighbors=min(int(n_neighbors), adata.n_obs - 1),
        use_rep=use_rep,
        metric="euclidean",
        random_state=random_state,
        key_added=neighbors_key,
    )
    latent = adata.obsp[f"{neighbors_key}_connectivities"].tocsr().astype(
        np.float32
    )
    if latent.nnz:
        maximum = float(latent.data.max())
        if maximum > 0.0:
            latent = latent / maximum

    spatial = spatial.tolil()
    spatial.setdiag(0)
    spatial = spatial.tocsr()
    spatial.eliminate_zeros()
    if spatial.nnz:
        spatial.data[:] = 1.0

    combined = (latent_weight * latent + spatial_weight * spatial).tocsr()
    combined = combined.maximum(combined.T).tocsr()
    combined = combined.tolil()
    combined.setdiag(0)
    combined = combined.tocsr()
    combined.eliminate_zeros()
    combined.sort_indices()
    adata.obsp[output_key] = combined
    return combined
