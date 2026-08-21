import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
import torch

from utils import (
    ResourceMonitor,
    calculate_metrics,
    load_h5ad,
    update_summary,
)


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "SpaGCN"
SEED = 42
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing SpaGCN: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = int(
        N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    )

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    coords = np.asarray(data.obsm["spatial"])
    adjacency = spg.calculate_adj_matrix(
        x=coords[:, 0],
        y=coords[:, 1],
        x_pixel=coords[:, 0],
        y_pixel=coords[:, 1],
        beta=49,
        alpha=1,
        histology=False,
    )
    spg.prefilter_genes(data, min_cells=3)
    spg.prefilter_specialgenes(data)
    sc.pp.normalize_per_cell(data)
    sc.pp.log1p(data)
    length_scale = spg.search_l(
        0.5,
        adjacency,
        start=0.01,
        end=1000,
        tol=0.01,
        max_run=100,
    )
    resolution = spg.search_res(
        data,
        adjacency,
        length_scale,
        n_clusters,
        start=0.5,
        step=0.05,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=SEED,
        t_seed=SEED,
        n_seed=SEED,
    )
    model = spg.SpaGCN()
    model.set_l(length_scale)
    model.train(
        data,
        adjacency,
        init_spa=True,
        init="louvain",
        res=resolution,
        tol=5e-3,
        lr=0.05,
        max_epochs=200,
    )
    prediction, _ = model.predict()
    refinement_coordinates = (
        data.obs[["array_row", "array_col"]].to_numpy()
        if {"array_row", "array_col"}.issubset(data.obs.columns)
        else coords
    )
    refinement_adjacency = spg.calculate_adj_matrix(
        x=refinement_coordinates[:, 0],
        y=refinement_coordinates[:, 1],
        histology=False,
    )
    prediction = spg.refine(
        sample_id=data.obs_names.tolist(),
        pred=prediction.tolist(),
        dis=refinement_adjacency,
        shape="hexagon",
    )
    data.obs["refined_pred"] = prediction
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "SpaGCN",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"SpaGCN_{input_h5ad.stem}.h5ad"
    data.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "SpaGCN_runtime.csv")
    print(row, flush=True)
