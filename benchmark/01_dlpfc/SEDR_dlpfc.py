import json
import time
from pathlib import Path

import pandas as pd
import scanpy as sc
import SEDR
import torch

from utils import ResourceMonitor, calculate_ari_nmi, gpu_metrics

DATA_DIR = Path(r"D:\SpaDiff\0_data\1_DLPFC")
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "SEDR"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLICES = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)
N_CLUSTERS = {
    sample_id: 5 if sample_id in SLICES[4:8] else 7
    for sample_id in SLICES
}

SEED = 2024
FORCE_CPU = False
device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"

sample_ids = SLICES
summary_rows = []

for sample_id in sample_ids:
    SEDR.fix_seed(SEED)
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

    adata = SEDR.adata_preprocess(
        adata,
        min_cells=5,
        pca_n_comps=30,
    )
    graph = SEDR.graph_construction(adata, 12)
    model = SEDR.Sedr(
        adata.obsm["X_pca"],
        graph,
        mode="clustering",
        device=device,
    )
    model.train_without_dec()
    features, _, _, _ = model.process()
    adata.obsm["SEDR"] = features
    adata = SEDR.mclust_R(
        adata,
        N_CLUSTERS[sample_id],
        use_rep="SEDR",
        key_added="SEDR",
        random_seed=SEED,
    )
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }

    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        adata.obs["SEDR"],
    )
    adata.uns["SEDR_ARI"] = ari
    adata.uns["SEDR_NMI"] = nmi
    adata.uns["SEDR_parameters"] = json.dumps(
        {
            "pca": 30,
            "graph_k": 12,
            "using_dec": False,
            "clustering": "SEDR.mclust_R",
        }
    )

    output_path = OUTPUT_DIR / (
        f"SEDR_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output_path, compression="gzip")

    row = {
        "method": "SEDR",
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
        OUTPUT_DIR / "SEDR_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)
