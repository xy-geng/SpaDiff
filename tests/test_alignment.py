from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp

from SpaDiff.alignment import align_serial_slices


class TinySlice:
    def __init__(self, expression, coordinates, gene_names, weights):
        self.X = np.asarray(expression, dtype=np.float64)
        self.n_obs, self.n_vars = self.X.shape
        self.var_names = pd.Index(gene_names)
        self.obsm = {"spatial": np.asarray(coordinates, dtype=np.float64)}
        self.obsp = {"spagcn_connectivities": sp.csr_matrix(weights)}
        self.layers = {}
        self.uns = {}

    def copy(self):
        return deepcopy(self)


def test_align_serial_slices_recovers_global_affine_map():
    reference = np.asarray(
        [(x, y) for y in range(3) for x in range(3)], dtype=np.float64
    )
    matrix = np.asarray([[1.2, 0.15], [-0.1, 0.9]])
    translation = np.asarray([4.0, -3.0])
    moved = reference @ matrix + translation

    expression = np.full((9, 5), 0.01, dtype=np.float64)
    for gene, spot in enumerate((0, 2, 6, 8, 4)):
        expression[spot, gene] = 10.0
    weights = np.ones((9, 9), dtype=np.float64) - np.eye(9)
    genes = [f"gene_{index}" for index in range(expression.shape[1])]
    slices = (
        TinySlice(expression, reference, genes, weights),
        TinySlice(expression, moved, genes, weights),
    )

    result = align_serial_slices(
        slices,
        reference_index=0,
        grid_shape=(1, 1),
        n_anchor_genes=5,
        min_region_spots=4,
    )

    np.testing.assert_allclose(
        result.slices[1].obsm["spatial_aligned"], reference, atol=1e-10
    )
    assert len(result.selected_genes) == 5
    assert result.spatial_weight_sources == (
        "obsp['spagcn_connectivities']",
        "obsp['spagcn_connectivities']",
    )
    assert "spatial_aligned" not in slices[1].obsm


def test_align_serial_slices_can_write_in_place():
    coordinates = np.asarray(
        [(x, y) for y in range(2) for x in range(2)], dtype=np.float64
    )
    expression = np.eye(4, dtype=np.float64) + 0.01
    weights = np.ones((4, 4), dtype=np.float64) - np.eye(4)
    genes = ["a", "b", "c", "d"]
    slices = [
        TinySlice(expression, coordinates, genes, weights),
        TinySlice(expression, coordinates + 2.0, genes, weights),
    ]
    result = align_serial_slices(
        slices,
        grid_shape=1,
        n_anchor_genes=4,
        min_region_spots=4,
        copy=False,
    )
    assert result.slices[0] is slices[0]
    assert "spatial_aligned" in slices[1].obsm
