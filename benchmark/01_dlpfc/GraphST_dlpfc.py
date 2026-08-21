import json
import time
from pathlib import Path

import pandas as pd
import scanpy as sc
import torch
from GraphST.GraphST import GraphST
from GraphST.utils import clustering

from utils import ResourceMonitor, calculate_ari_nmi, gpu_metrics

DATA_DIR = Path(r"D:\SpaDiff\0_data\1_DLPFC")
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "GraphST"
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

FORCE_CPU = False
device = torch.device(
    "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
)
summary_rows = []

sample_ids = SLICES
for sample_id in sample_ids:
    sample_dir = DATA_DIR / sample_id
    adata = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True,
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

    adata = GraphST(adata, device=device).train()
    clustering(
        adata,
        n_clusters=N_CLUSTERS[sample_id],
        radius=50,
        method="mclust",
        key="emb",
        refinement=True,
    )
    prediction_column = "domain" if "domain" in adata.obs else "mclust"
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }

    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        adata.obs[prediction_column],
    )
    adata.uns["GraphST_ARI"] = ari
    adata.uns["GraphST_NMI"] = nmi
    adata.uns["GraphST_parameters"] = json.dumps(
        {
            "workflow": "GraphST.train + GraphST.utils.clustering",
            "n_clusters": N_CLUSTERS[sample_id],
            "radius": 50,
            "refinement": True,
        }
    )

    output_path = OUTPUT_DIR / (
        f"GraphST_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output_path, compression="gzip")

    row = {
        "method": "GraphST",
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
        OUTPUT_DIR / "GraphST_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)
