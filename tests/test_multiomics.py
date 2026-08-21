from __future__ import annotations

import sys
import types

import numpy as np
import scipy.sparse as sp

from SpaDiff.multiomics import (
    build_spatially_regularized_connectivity,
    robust_atac_lsi,
    standardize_columns,
)


class RawAtac:
    def __init__(self, matrix):
        self.X = matrix
        self.n_vars = matrix.shape[1]


class ConnectivityAdata:
    def __init__(self, n_obs):
        self.n_obs = n_obs
        self.obsm = {"spadiff": np.zeros((n_obs, 2), dtype=np.float32)}
        self.obsp = {}


def test_standardize_columns_handles_variable_and_constant_dimensions():
    values = np.asarray(
        [[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]], dtype=np.float32
    )
    standardized = standardize_columns(values)

    np.testing.assert_allclose(standardized[:, 0].mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(standardized[:, 0].std(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(standardized[:, 1], 0.0)


def test_robust_atac_lsi_returns_finite_requested_features():
    rng = np.random.default_rng(42)
    counts = rng.poisson(0.5, size=(16, 40)).astype(np.float32)
    counts[:, :8] += 1.0

    values, diagnostics = robust_atac_lsi(
        RawAtac(sp.csr_matrix(counts)),
        n_components=5,
        min_cells=2,
        random_state=42,
    )

    assert values.shape == (16, 5)
    assert np.isfinite(values).all()
    assert diagnostics["n_input_peaks"] == 40
    assert diagnostics["n_selected_peaks"] >= 5
    assert diagnostics["n_components"] == 5


def test_spatially_regularized_connectivity_mixes_expected_graphs(monkeypatch):
    adata = ConnectivityAdata(4)
    latent = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 0.5, 0.0, 0.0],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.25],
                [0.0, 0.0, 0.25, 0.0],
            ],
            dtype=np.float32,
        )
    )

    def fake_neighbors(target, **kwargs):
        target.obsp[f"{kwargs['key_added']}_connectivities"] = latent.copy()

    monkeypatch.setitem(
        sys.modules,
        "scanpy",
        types.SimpleNamespace(pp=types.SimpleNamespace(neighbors=fake_neighbors)),
    )
    spatial = sp.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    combined = build_spatially_regularized_connectivity(
        adata,
        spatial,
        latent_weight=0.4,
        spatial_weight=0.6,
    )
    expected = (0.4 * latent + 0.6 * spatial).maximum(
        (0.4 * latent + 0.6 * spatial).T
    )

    np.testing.assert_allclose(combined.toarray(), expected.toarray())
    assert adata.obsp["spadiff_spatial_connectivities"] is combined
