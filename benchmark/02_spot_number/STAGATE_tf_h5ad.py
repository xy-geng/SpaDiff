import os
import time
from pathlib import Path

DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "STAGATE_TF"
SEED = 2023
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import scanpy as sc
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import STAGATE

from utils import ResourceMonitor, calculate_metrics, load_h5ad, update_summary

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing STAGATE_TF: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    STAGATE.Cal_Spatial_Net(data, rad_cutoff=150)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        n_top_genes=min(3000, data.n_vars),
    )
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    result = STAGATE.train_STAGATE(data, alpha=0)
    result = STAGATE.mclust_R(
        result,
        num_cluster=int(n_clusters),
        used_obsm="STAGATE",
    )
    prediction = result.obs["mclust"].to_numpy()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "STAGATE_TF",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"STAGATE_TF_{input_h5ad.stem}.h5ad"
    result.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "STAGATE_TF_runtime.csv")
    print(row, flush=True)
