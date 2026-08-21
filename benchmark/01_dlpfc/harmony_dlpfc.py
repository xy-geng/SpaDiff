import time
from pathlib import Path

import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc

from utils import (
    ResourceMonitor,
    calculate_ari_nmi,
    mclust_labels,
)

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
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "Harmony"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2024
N_TOP_GENES = 3000
N_PCS = 30

sample_ids = SLICES
summary_rows = []

for sample_id in sample_ids:
    print(f"\nProcessing DLPFC slice {sample_id}", flush=True)

    sample_dir = DATA_DIR / sample_id
    data = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=False,
    )
    data.var_names_make_unique()

    truth = pd.read_csv(
        sample_dir / "truth.txt",
        sep="\t",
        header=None,
        index_col=0,
    ).iloc[:, 0]
    data.obs["ground_truth"] = truth.reindex(
        data.obs_names
    ).astype("category")

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        n_top_genes=min(N_TOP_GENES, data.n_vars),
    )
    data = data[:, data.var["highly_variable"]].copy()
    sc.pp.scale(data, max_value=10)
    sc.tl.pca(
        data,
        n_comps=min(
            N_PCS,
            data.n_obs - 1,
            data.n_vars - 1,
        ),
        random_state=SEED,
    )

    # A single slice contains one batch, so no cross-slice correction occurs.
    data.obs["harmony_batch"] = pd.Categorical(
        ["single_slice"] * data.n_obs
    )
    harmony_result = harmonypy.run_harmony(
        data.obsm["X_pca"],
        data.obs,
        "harmony_batch",
        random_state=SEED,
    )
    corrected = np.asarray(harmony_result.Z_corr)
    expected_shape = data.obsm["X_pca"].shape

    if corrected.shape == expected_shape:
        data.obsm["X_harmony"] = corrected
    elif corrected.T.shape == expected_shape:
        data.obsm["X_harmony"] = corrected.T
    else:
        raise ValueError(
            "Unexpected Harmony output shape for "
            f"{sample_id}: received {corrected.shape}, "
            f"expected {expected_shape} or its transpose."
        )

    prediction = mclust_labels(
        data.obsm["X_harmony"],
        N_CLUSTERS[sample_id],
        seed=SEED,
    )
    data.obs["mclust"] = pd.Categorical(prediction)
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = monitor.stop()
    ari, nmi, n_truth_spots = calculate_ari_nmi(
        data.obs["ground_truth"],
        prediction,
    )

    data.uns["Harmony_ARI"] = ari
    data.uns["Harmony_NMI"] = nmi
    data.uns["Harmony_n_clusters"] = N_CLUSTERS[sample_id]
    data.uns["Harmony_seed"] = SEED
    data.uns["Harmony_n_top_genes"] = N_TOP_GENES
    data.uns["Harmony_n_pcs"] = N_PCS
    data.uns["Harmony_mode"] = "independent_single_slice"

    output_path = OUTPUT_DIR / (
        f"Harmony_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    data.write_h5ad(output_path, compression="gzip")
    row = {
        "method": "Harmony",
        "slice_id": sample_id,
        "status": "completed",
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_clusters": N_CLUSTERS[sample_id],
        "n_spots": data.n_obs,
        "n_truth_spots": n_truth_spots,
        "output_file": str(output_path),
        **resource_metrics,
    }
    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "Harmony_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)

print(
    "\nHarmony summary: "
    f"{OUTPUT_DIR / 'Harmony_DLPFC_summary.csv'}",
    flush=True,
)
