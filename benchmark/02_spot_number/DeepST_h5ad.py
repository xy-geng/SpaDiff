import time
from pathlib import Path

import deepstkit as dt
import pandas as pd
import torch

from utils import ResourceMonitor, calculate_metrics, load_h5ad, update_summary


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "DeepST"
SEED = 2023
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing DeepST: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    use_gpu = torch.cuda.is_available() and not FORCE_CPU

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    dt.utils_func.seed_torch(seed=SEED)
    deepst = dt.main.run(
        save_path=str(OUTPUT_DIR / "model_files"),
        task="Identify_Domain",
        pre_epochs=500,
        epochs=500,
        use_gpu=use_gpu,
    )
    data = deepst._get_augment(
        data,
        spatial_type="BallTree",
        use_morphological=False,
    )
    graph_dict = deepst._get_graph(
        data.obsm["spatial"],
        distType="KDTree",
    )
    processed = deepst._data_process(data, pca_n_comps=200)
    embedding = deepst._fit(data=processed, graph_dict=graph_dict)
    data.obsm["DeepST_embed"] = embedding
    data = deepst._get_cluster_data(
        data,
        n_domains=int(n_clusters),
        priori=True,
    )
    prediction_column = (
        "domains" if "domains" in data.obs else "DeepST_domain"
    )
    prediction = data.obs[prediction_column].to_numpy()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "DeepST",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"DeepST_{input_h5ad.stem}.h5ad"
    data.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "DeepST_runtime.csv")
    print(row, flush=True)
