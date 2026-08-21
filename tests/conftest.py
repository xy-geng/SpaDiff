from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


class MiniAnnData:
    """Small AnnData-shaped fixture without Scanpy preprocessing overhead."""

    def __init__(self, coordinates, batches):
        self.obsm = {"spatial": np.asarray(coordinates, dtype=np.float64)}
        self.obs = pd.DataFrame({"batch_name": list(batches)})
        self.n_obs = len(self.obs)

    def copy(self):
        return deepcopy(self)


@pytest.fixture
def multislice_adata():
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.1, 0.1],
            [1.1, 0.1],
            [0.1, 1.1],
            [1.1, 1.1],
            [0.2, 0.2],
            [1.2, 0.2],
            [0.2, 1.2],
            [1.2, 1.2],
        ]
    )
    batches = ["slice_a"] * 4 + ["slice_b"] * 4 + ["slice_c"] * 4
    return MiniAnnData(coordinates, batches)
