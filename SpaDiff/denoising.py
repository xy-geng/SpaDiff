from __future__ import annotations

from typing import Optional

import numpy as np


def row_normalize_adjacency(adjacency, *, remove_self_loops: bool = True):
    """Return a sparse row-stochastic spatial adjacency matrix."""

    import scipy.sparse as sp

    matrix = sp.csr_matrix(adjacency, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or np.min(matrix.data) < 0.0
    ):
        raise ValueError("adjacency weights must be finite and non-negative")
    if remove_self_loops:
        matrix = matrix.tolil()
        matrix.setdiag(0)
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=row_sum > 0.0,
    )
    return (sp.diags(inverse) @ matrix).tocsr()


def smooth_generated_expression(
    expression,
    adjacency,
    *,
    alpha: float = 1.0,
    clip_nonnegative: bool = True,
):
    """Apply the manuscript's post-generation ``X + alpha * A X`` smoothing.

    ``A`` is always converted to a sparse row-normalized operator. This function
    is intentionally independent of the existing training-time
    :func:`SpaDiff.spatial.spatial_reconstruction` helper.
    """

    import scipy.sparse as sp

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    operator = row_normalize_adjacency(adjacency)
    sparse_input = sp.issparse(expression)
    values = (
        expression.tocsr().astype(np.float64)
        if sparse_input
        else np.asarray(expression, dtype=np.float64)
    )
    if values.ndim != 2 or values.shape[0] != operator.shape[0]:
        raise ValueError("expression and adjacency must contain the same spots")
    data = values.data if sparse_input else values
    if not np.isfinite(data).all():
        raise ValueError("expression must contain only finite values")
    if clip_nonnegative:
        if sparse_input:
            values = values.copy()
            values.data = np.maximum(values.data, 0.0)
            values.eliminate_zeros()
        else:
            values = np.maximum(values, 0.0)
    smoothed = values + float(alpha) * (operator @ values)
    if clip_nonnegative:
        if sp.issparse(smoothed):
            smoothed.data = np.maximum(smoothed.data, 0.0)
            smoothed.eliminate_zeros()
        else:
            smoothed = np.maximum(smoothed, 0.0)
    return smoothed


def inverse_pca_expression(
    latent,
    loadings,
    *,
    feature_mean=None,
    inverse_scale=None,
    inverse_offset=None,
    clip_nonnegative: bool = True,
) -> np.ndarray:
    """Map denoised PCA coordinates back into the modeled gene feature space."""

    scores = np.asarray(latent, dtype=np.float64)
    components = np.asarray(loadings, dtype=np.float64)
    if scores.ndim != 2 or components.ndim != 2:
        raise ValueError("latent and loadings must both be two-dimensional")
    if components.shape[1] < scores.shape[1]:
        raise ValueError("PCA loadings contain fewer components than latent values")
    reconstructed = scores @ components[:, : scores.shape[1]].T
    n_genes = reconstructed.shape[1]

    if feature_mean is not None:
        mean = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
        if mean.shape != (n_genes,):
            raise ValueError("feature_mean must contain one value per modeled gene")
        reconstructed = reconstructed + mean
    if inverse_scale is not None:
        scale = np.asarray(inverse_scale, dtype=np.float64).reshape(-1)
        if scale.shape != (n_genes,):
            raise ValueError("inverse_scale must contain one value per modeled gene")
        reconstructed = reconstructed * scale
    if inverse_offset is not None:
        offset = np.asarray(inverse_offset, dtype=np.float64).reshape(-1)
        if offset.shape != (n_genes,):
            raise ValueError("inverse_offset must contain one value per modeled gene")
        reconstructed = reconstructed + offset
    if not np.isfinite(reconstructed).all():
        raise FloatingPointError("inverse PCA reconstruction produced non-finite values")
    if clip_nonnegative:
        reconstructed = np.maximum(reconstructed, 0.0)
    return reconstructed


def _feature_mean_from_adata(adata) -> np.ndarray:
    import scipy.sparse as sp

    matrix = adata.X
    if sp.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel().astype(np.float64)
    return np.asarray(matrix, dtype=np.float64).mean(axis=0)


def write_denoised_expression(
    adata,
    *,
    latent_key: str = "X_spadiff",
    loadings_key: str = "PCs",
    layer_key: str = "spadiff_denoised",
    feature_mean=None,
    inverse_scale=None,
    inverse_offset=None,
    adjacency=None,
    smoothing_alpha: float = 1.0,
    clip_nonnegative: bool = True,
    sparse_output: bool = False,
):
    """Inverse-transform SpaDiff PCA features and write a denoised gene layer.

    The default feature mean is calculated from ``adata.X``, which is correct
    when Scanpy PCA was run on the current, unscaled expression matrix. If PCA
    was fitted after gene-wise scaling, pass the corresponding inverse scale and
    offset explicitly. When ``adjacency`` is supplied, spatial smoothing is
    performed only after inverse PCA and non-negative truncation.
    """

    import scipy.sparse as sp

    if latent_key not in adata.obsm:
        raise KeyError(f"adata.obsm does not contain {latent_key!r}")
    if not hasattr(adata, "varm") or loadings_key not in adata.varm:
        raise KeyError(f"adata.varm does not contain {loadings_key!r}")
    mean = _feature_mean_from_adata(adata) if feature_mean is None else feature_mean
    expression = inverse_pca_expression(
        adata.obsm[latent_key],
        adata.varm[loadings_key],
        feature_mean=mean,
        inverse_scale=inverse_scale,
        inverse_offset=inverse_offset,
        clip_nonnegative=clip_nonnegative,
    )
    if expression.shape != (adata.n_obs, adata.n_vars):
        raise ValueError(
            "inverse PCA result does not match the AnnData observation/variable axes"
        )
    if adjacency is not None and smoothing_alpha > 0.0:
        expression = smooth_generated_expression(
            expression,
            adjacency,
            alpha=smoothing_alpha,
            clip_nonnegative=clip_nonnegative,
        )
    output = np.asarray(expression, dtype=np.float32)
    adata.layers[layer_key] = sp.csr_matrix(output) if sparse_output else output
    if not hasattr(adata, "uns"):
        adata.uns = {}
    adata.uns["spadiff_denoising"] = {
        "latent_key": latent_key,
        "loadings_key": loadings_key,
        "layer_key": layer_key,
        "clip_nonnegative": bool(clip_nonnegative),
        "spatial_smoothing": adjacency is not None and smoothing_alpha > 0.0,
        "smoothing_alpha": float(smoothing_alpha),
        "row_normalized_adjacency": adjacency is not None,
    }
    return adata
