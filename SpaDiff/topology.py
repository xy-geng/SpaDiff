"""Sparse simplicial operators"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


Simplex = Tuple[int, ...]


def _undirected_neighbors(adjacency) -> List[set[int]]:
    """Symmetrize an adjacency matrix and return neighbors without self-loops."""
    if hasattr(adjacency, "tocsr"):
        matrix = adjacency.tocsr().maximum(adjacency.T.tocsr()).tolil()
        matrix.setdiag(0)
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        return [
            set(matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]].tolist())
            for i in range(matrix.shape[0])
        ]

    dense = np.asarray(adjacency)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("adjacency must be square")
    dense = np.logical_or(dense != 0, dense.T != 0)
    np.fill_diagonal(dense, False)
    return [set(np.flatnonzero(dense[i]).tolist()) for i in range(dense.shape[0])]


def enumerate_simplices(adjacency, max_order: int = 2) -> Dict[int, List[Simplex]]:
    """Enumerate edges and clique-induced higher-order simplices.

    The manuscript experiments use orders 1 and 2. Higher orders are supported
    for small graphs via clique expansion, but should be used carefully because
    the number of cliques can grow rapidly.
    """
    if max_order < 1:
        raise ValueError("max_order must be at least 1")
    neighbors = _undirected_neighbors(adjacency)
    edges = [(u, v) for u, nbrs in enumerate(neighbors) for v in nbrs if u < v]
    result: Dict[int, List[Simplex]] = {1: edges}
    if max_order == 1:
        return result

    # Build larger cliques incrementally. This keeps the original triangle
    # intention while avoiding the unconditional _D[2] access in creat_L2.
    previous = edges
    for order in range(2, max_order + 1):
        current: List[Simplex] = []
        for simplex in previous:
            common = set.intersection(*(neighbors[v] for v in simplex))
            for vertex in common:
                if vertex > simplex[-1]:
                    current.append((*simplex, vertex))
        result[order] = current
        previous = current
    return result


def normalized_node_simplex_operator(
    num_nodes: int, simplices: Sequence[Simplex], order: int
):
    """Return A_p = D_v^-1/2 C_p D_s^-1 C_p^T D_v^-1/2 as CSR."""
    import scipy.sparse as sp

    simplex_size = order + 1
    if any(len(simplex) != simplex_size for simplex in simplices):
        raise ValueError(f"order-{order} simplices must contain {simplex_size} nodes")
    if not simplices:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)

    columns = np.repeat(np.arange(len(simplices)), simplex_size)
    rows = np.asarray([v for simplex in simplices for v in simplex], dtype=np.int64)
    incidence = sp.coo_matrix(
        (np.ones(rows.size, dtype=np.float32), (rows, columns)),
        shape=(num_nodes, len(simplices)),
    ).tocsr()
    degree = np.asarray(incidence.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt[nonzero] = degree[nonzero] ** -0.5
    d_inv = sp.diags(inv_sqrt)
    return (d_inv @ incidence @ incidence.T @ d_inv / simplex_size).tocsr()


def build_simplicial_operators(
    adjacency,
    max_order: int = 2,
    *,
    verbose: bool = True,
    return_counts: bool = False,
):
    """Build simplicial operators and report the number of simplices per order.

    """
    import scipy.sparse as sp

    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    neighbors = _undirected_neighbors(adjacency)
    num_nodes = len(neighbors)
    counts = {0: num_nodes}

    if max_order == 0:
        operators = {0: sp.identity(num_nodes, dtype=np.float32, format="csr")}
    else:
        simplices = enumerate_simplices(adjacency, max_order=max_order)
        counts.update(
            {order: len(simplices.get(order, [])) for order in range(1, max_order + 1)}
        )
        operators = {
            order: normalized_node_simplex_operator(
                num_nodes, simplices.get(order, []), order
            )
            for order in range(1, max_order + 1)
        }

    if verbose:
        names = {
            0: "vertices",
            1: "edges",
            2: "triangles",
            3: "tetrahedra",
            4: "4-simplices",
        }
        print("Simplicial complex statistics:")
        for order in range(max_order + 1):
            name = names.get(order, f"order-{order} simplices")
            print(f"  order {order} ({name}): {counts.get(order, 0):,}")

    if return_counts:
        return operators, counts
    return operators


def scipy_to_torch_sparse(matrix, device=None):
    """Convert scipy sparse matrix to coalesced torch COO."""
    import torch

    coo = matrix.tocoo()
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)), dtype=torch.long, device=device
    )
    values = torch.as_tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def to_torch_operators(operators, device=None):
    """Convert an order-keyed scipy operator mapping to torch sparse tensors."""
    return {
        order: scipy_to_torch_sparse(op, device=device)
        for order, op in operators.items()
    }
