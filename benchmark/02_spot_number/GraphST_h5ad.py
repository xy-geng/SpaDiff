import time
from pathlib import Path

import pandas as pd
import torch
from GraphST.GraphST import GraphST
from GraphST.utils import clustering

from utils import ResourceMonitor, calculate_metrics, load_h5ad, update_summary


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "GraphST"
SEED = 2023
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing GraphST: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    device = torch.device(
        "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
    )

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    result = GraphST(data, device=device).train()
    clustering(
        result,
        n_clusters=int(n_clusters),
        radius=50,
        method="mclust",
        key="emb",
        refinement=True,
    )
    prediction_column = "domain" if "domain" in result.obs else "mclust"
    prediction = result.obs[prediction_column].to_numpy()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "GraphST",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"GraphST_{input_h5ad.stem}.h5ad"
    result.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "GraphST_runtime.csv")
    print(row, flush=True)
