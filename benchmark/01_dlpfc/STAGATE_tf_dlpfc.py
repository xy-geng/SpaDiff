import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import STAGATE

from utils import ResourceMonitor, calculate_ari_nmi


SLICES = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)
N_CLUSTERS = {
    sample_id: 5 if sample_id in SLICES[4:8] else 7
    for sample_id in SLICES
}

DATA_DIR = Path(r"D:\SpaDiff\0_data\1_DLPFC")
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "STAGATE_TF"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_ids = SLICES
summary_rows = []

for sample_id in sample_ids:
    sample_dir = DATA_DIR / sample_id
    adata = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
    )
    adata.var_names_make_unique()

    truth = pd.read_csv(
        sample_dir / "truth.txt",
        sep="\t",
        header=None,
        index_col=0,
    ).iloc[:, 0]
    adata.obs["ground_truth"] = truth.reindex(adata.obs_names).astype("category")

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=150)
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=3000,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata = STAGATE.train_STAGATE(adata, alpha=0)
    adata = STAGATE.mclust_R(
        adata,
        num_cluster=N_CLUSTERS[sample_id],
        used_obsm="STAGATE",
    )
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = monitor.stop()

    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        adata.obs["mclust"],
    )
    adata.uns["STAGATE_TF_ARI"] = ari
    adata.uns["STAGATE_TF_NMI"] = nmi

    output_path = OUTPUT_DIR / (
        f"STAGATE_TF_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output_path, compression="gzip")

    row = {
        "method": "STAGATE_TF",
        "slice_id": sample_id,
        "status": "completed",
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_truth_spots": n_truth_spots,
        "output_file": str(output_path),
        **resource_metrics,
    }

    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "STAGATE_TF_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)
