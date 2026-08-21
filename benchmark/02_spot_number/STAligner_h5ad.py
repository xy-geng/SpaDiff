import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import STAligner
import torch

from utils import ResourceMonitor, calculate_metrics, load_h5ad, update_summary


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "STAligner"
SEED = 666
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing STAligner: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = int(
        N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    )
    device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    data.obs["batch_name"] = "single_slice"
    STAligner.Cal_Spatial_Net(data, rad_cutoff=150)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        n_top_genes=min(3000, data.n_vars),
    )
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    data = data[:, data.var["highly_variable"]].copy()
    adjacency = data.uns["adj"].toarray()
    data.uns["edgeList"] = np.nonzero(adjacency)
    result = STAligner.train_STAligner(
        data,
        iter_comb=None,
        verbose=True,
        knn_neigh=50,
        device=device,
        margin=1.0,
        random_seed=SEED,
    )
    result = STAligner.mclust_R(
        result,
        num_cluster=n_clusters,
        used_obsm="STAligner",
        random_seed=SEED,
    )
    prediction = result.obs["mclust"].to_numpy()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "STAligner",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    result.uns.pop("edgeList", None)
    output = OUTPUT_DIR / f"STAligner_{input_h5ad.stem}.h5ad"
    result.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "STAligner_runtime.csv")
    print(row, flush=True)
