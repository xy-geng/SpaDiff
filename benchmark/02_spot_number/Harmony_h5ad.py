import time
from pathlib import Path

import harmonypy
import numpy as np
import pandas as pd
import scanpy as sc

from utils import (
    ResourceMonitor,
    calculate_metrics,
    load_h5ad,
    mclust_labels,
    update_summary,
)


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "Harmony"
SEED = 2023
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing Harmony: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    data.obs["harmony_batch"] = pd.Categorical(
        ["single_slice"] * data.n_obs
    )
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        n_top_genes=min(3000, data.n_vars),
    )
    data = data[:, data.var["highly_variable"]].copy()
    sc.pp.scale(data, max_value=10)
    sc.pp.pca(
        data,
        n_comps=min(30, data.n_obs - 1, data.n_vars - 1),
        random_state=SEED,
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
            "Unexpected Harmony output shape: "
            f"received {corrected.shape}, expected {expected_shape} "
            "or its transpose."
        )

    prediction = mclust_labels(
        data.obsm["X_harmony"],
        int(n_clusters),
        seed=SEED,
    )
    data.obs["mclust"] = pd.Categorical(prediction)
    data.uns["Harmony_mode"] = "one_h5ad_as_one_slice"
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "Harmony",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"Harmony_{input_h5ad.stem}.h5ad"
    data.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "Harmony_runtime.csv")
    print(row, flush=True)
