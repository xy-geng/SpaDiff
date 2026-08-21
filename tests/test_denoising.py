from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from SpaDiff.denoising import (
    inverse_pca_expression,
    row_normalize_adjacency,
    smooth_generated_expression,
    write_denoised_expression,
)


class TinyDenoisingData:
    def __init__(self):
        self.X = np.asarray([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        self.n_obs, self.n_vars = self.X.shape
        self.obsm = {"X_spadiff": np.asarray([[1.0, 0.5], [0.5, -1.0]])}
        self.varm = {
            "PCs": np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]])
        }
        self.layers = {}
        self.uns = {}


def test_sparse_row_normalization_and_generated_smoothing():
    adjacency = sp.csr_matrix([[0.0, 2.0], [1.0, 0.0]])
    normalized = row_normalize_adjacency(adjacency)
    np.testing.assert_allclose(np.asarray(normalized.sum(axis=1)).ravel(), 1.0)

    expression = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    smoothed = smooth_generated_expression(expression, adjacency, alpha=1.0)
    np.testing.assert_allclose(smoothed, [[4.0, 6.0], [4.0, 6.0]])


def test_inverse_pca_and_layer_write():
    adata = TinyDenoisingData()
    mean = adata.X.mean(axis=0)
    expected = inverse_pca_expression(
        adata.obsm["X_spadiff"], adata.varm["PCs"], feature_mean=mean
    )
    adjacency = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]])
    write_denoised_expression(
        adata,
        feature_mean=mean,
        adjacency=adjacency,
        smoothing_alpha=0.5,
    )
    assert "spadiff_denoised" in adata.layers
    np.testing.assert_allclose(
        adata.layers["spadiff_denoised"],
        expected + 0.5 * expected[::-1],
        rtol=1e-6,
    )
    assert adata.uns["spadiff_denoising"]["row_normalized_adjacency"]
