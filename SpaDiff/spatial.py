"""Spatial graph utilities for already aligned coordinates."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .topology import build_simplicial_operators, to_torch_operators


def spatial_reconstruction(
    adata,
    alpha: float = 0.0,
    n_neighbors: int = 10,
    n_pcs: int = 15,
    use_highly_variable: Optional[bool] = None,
    # copy: bool = False,
    # device=None,
    # max_order: int = 2,
):
    """Build single-slice simplicial operators from processed coordinates.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    # adata = adata.copy() if copy else adata

    from sklearn.neighbors import NearestNeighbors
    coordinates = np.asarray(adata.obsm["spatial"])
    estimator = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(coordinates)
    adjacency = estimator.kneighbors_graph(coordinates)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency = adjacency.maximum(adjacency.T)
    # operators = to_torch_operators(build_simplicial_operators(adjacency, max_order=max_order), device=device)

    if alpha > 0.0:
        import scanpy as sc
        from scipy.sparse import csr_matrix, issparse
        from sklearn.metrics.pairwise import cosine_distances

        sc.pp.pca(adata,n_comps=n_pcs,use_highly_variable=use_highly_variable)
        distances = np.exp(2.0 - cosine_distances(adata.obsm["X_pca"])) - 1.0
        connections = adjacency.T.toarray() * distances
        values = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
        denominator = np.sum(connections, axis=0, keepdims=True)
        normalized = np.divide(connections,denominator,out=np.zeros_like(connections),where=denominator > 0)
        adata.X = csr_matrix(alpha * np.matmul(normalized, values) + values)
        del adata.obsm["X_pca"]
    return adata, adjacency


def Neiber(adata, k_intra=6, k_inter=2, slice_order: Optional[Sequence] = None):
    """Preserved multi-slice neighbor builder with explicit slice ordering."""
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    coord = np.asarray(adata.obsm["spatial"])
    slice_ids = np.asarray(adata.obs["batch_name"].values)
    n = coord.shape[0]
    adjacency = sp.lil_matrix((n, n), dtype=np.float32)
    if slice_order is None:

        slice_order = list(dict.fromkeys(slice_ids.tolist()))
    for slice_id in slice_order:
        index = np.flatnonzero(slice_ids == slice_id)
        k = min(k_intra, len(index))
        if k <= 1:
            continue
        estimator = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(
            coord[index]
        )
        graph = estimator.kneighbors_graph(coord[index], mode="connectivity")
        graph.setdiag(0)
        graph.eliminate_zeros()
        adjacency[np.ix_(index, index)] = graph
    for left_id, right_id in zip(slice_order[:-1], slice_order[1:]):
        left = np.flatnonzero(slice_ids == left_id)
        right = np.flatnonzero(slice_ids == right_id)
        if not len(left) or not len(right):
            continue
        k_lr = min(k_inter, len(right))
        k_rl = min(k_inter, len(left))
        lr = (
            NearestNeighbors(n_neighbors=k_lr)
            .fit(coord[right])
            .kneighbors_graph(coord[left], mode="connectivity")
        )
        rl = (
            NearestNeighbors(n_neighbors=k_rl)
            .fit(coord[left])
            .kneighbors_graph(coord[right], mode="connectivity")
        )
        adjacency[np.ix_(left, right)] = lr
        adjacency[np.ix_(right, left)] = rl
    return coord, adjacency.tocsr()
