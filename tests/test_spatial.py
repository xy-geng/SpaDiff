from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

import SpaDiff as sd


SLICE_ORDER = ["slice_a", "slice_b", "slice_c"]


def legacy_slice_aware(adata, k_intra=3, k_inter=1):
    coordinates = np.asarray(adata.obsm["spatial"])
    slice_ids = np.asarray(adata.obs["batch_name"].values)
    adjacency = sp.lil_matrix((adata.n_obs, adata.n_obs), dtype=np.float32)
    for slice_id in SLICE_ORDER:
        index = np.flatnonzero(slice_ids == slice_id)
        k = min(k_intra, len(index))
        if k <= 1:
            continue
        graph = (
            NearestNeighbors(n_neighbors=k, metric="euclidean")
            .fit(coordinates[index])
            .kneighbors_graph(coordinates[index], mode="connectivity")
        )
        graph.setdiag(0)
        graph.eliminate_zeros()
        adjacency[np.ix_(index, index)] = graph
    for left_id, right_id in zip(SLICE_ORDER[:-1], SLICE_ORDER[1:]):
        left = np.flatnonzero(slice_ids == left_id)
        right = np.flatnonzero(slice_ids == right_id)
        lr = (
            NearestNeighbors(n_neighbors=min(k_inter, len(right)))
            .fit(coordinates[right])
            .kneighbors_graph(coordinates[left], mode="connectivity")
        )
        rl = (
            NearestNeighbors(n_neighbors=min(k_inter, len(left)))
            .fit(coordinates[left])
            .kneighbors_graph(coordinates[right], mode="connectivity")
        )
        adjacency[np.ix_(left, right)] = lr
        adjacency[np.ix_(right, left)] = rl
    adjacency = adjacency.tocsr().maximum(adjacency.T.tocsr()).tolil()
    adjacency.setdiag(0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency


def legacy_global_knn(coordinates, n_neighbors=3):
    estimator = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean"
    ).fit(coordinates)
    adjacency = estimator.kneighbors_graph(coordinates, mode="connectivity")
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.sort_indices()
    return adjacency


def assert_same_sparse(left, right):
    assert sp.isspmatrix_csr(left)
    assert sp.isspmatrix_csr(right)
    difference = (left - right).tocsr()
    difference.eliminate_zeros()
    assert difference.nnz == 0


@pytest.mark.parametrize("max_order, expected", [(0, {0}), (1, {1}), (2, {1, 2})])
def test_slice_aware_matches_legacy_and_builds_orders(
    multislice_adata, max_order, expected
):
    result = sd.build_spatial_topology(
        multislice_adata,
        mode="slice_aware",
        slice_order=SLICE_ORDER,
        k_intra=3,
        k_inter=1,
        max_order=max_order,
        verbose=False,
    )
    expected_adjacency = legacy_slice_aware(multislice_adata)
    assert_same_sparse(result.adjacency, expected_adjacency)
    assert set(result.operators) == expected
    assert result.simplex_counts[0] == multislice_adata.n_obs
    assert result.mode == "slice_aware"
    assert result.adjacency.diagonal().sum() == 0
    assert_same_sparse(result.adjacency, result.adjacency.T.tocsr())


def test_global_knn_matches_legacy(multislice_adata):
    result = sd.build_spatial_topology(
        multislice_adata,
        mode="global_knn",
        n_neighbors=3,
        max_order=2,
        verbose=False,
    )
    expected = legacy_global_knn(multislice_adata.obsm["spatial"])
    assert_same_sparse(result.adjacency, expected)
    assert result.mode == "global_knn"
    assert result.adjacency.diagonal().sum() == 0


def test_neiber_is_compatible_and_returns_symmetric_graph(multislice_adata):
    expected = legacy_slice_aware(multislice_adata)
    with pytest.deprecated_call(match="Neiber is deprecated"):
        coordinates, adjacency = sd.Neiber(
            multislice_adata,
            k_intra=3,
            k_inter=1,
            slice_order=SLICE_ORDER,
        )
    np.testing.assert_array_equal(coordinates, multislice_adata.obsm["spatial"])
    assert_same_sparse(adjacency, expected)


def test_spatial_validation(multislice_adata):
    with pytest.raises(ValueError, match="mode must be"):
        sd.build_spatial_topology(multislice_adata, mode="unknown", verbose=False)
    with pytest.raises(ValueError, match="every observed slice"):
        sd.build_spatial_topology(
            multislice_adata,
            slice_order=["slice_a", "slice_b"],
            verbose=False,
        )
    missing = multislice_adata.copy()
    missing.obs = missing.obs.drop(columns=["batch_name"])
    with pytest.raises(KeyError, match="batch_name"):
        sd.build_spatial_topology(missing, mode="slice_aware", verbose=False)


def test_small_slices_and_disabled_inter_slice_edges():
    from conftest import MiniAnnData

    adata = MiniAnnData([[0, 0], [1, 0], [2, 0]], ["a", "b", "b"])
    result = sd.build_spatial_topology(
        adata,
        slice_order=["a", "b"],
        k_intra=6,
        k_inter=0,
        max_order=1,
        verbose=False,
    )
    assert result.adjacency.shape == (3, 3)
    assert result.adjacency[0].nnz == 0
