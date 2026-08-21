import sys
import time
from pathlib import Path

SPAVAE_SOURCE_DIR = Path(r"D:\SpaDiff\baselines\SpaVAE\src\spaVAE")
sys.path.append(str(SPAVAE_SOURCE_DIR))

from preprocess import normalize
from spaVAE import SPAVAE

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler

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
OUTPUT_DIR = OUTPUT_ROOT / "SpaVAE"
SEED = 2023
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing SpaVAE: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = int(
        N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    )
    device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    if hasattr(data.X, "toarray"):
        counts = data.X.toarray().astype("float64")
    else:
        counts = np.asarray(data.X, dtype="float64")
    locations = (
        MinMaxScaler().fit_transform(np.asarray(data.obsm["spatial"]))
        * 20.0
    )
    train_data = sc.AnnData(counts, dtype="float64")
    train_data = normalize(
        train_data,
        size_factors=True,
        normalize_input=True,
        logtrans_input=True,
    )
    inducing_points = (
        np.mgrid[0:1.00001:1 / 6, 0:1.00001:1 / 6]
        .reshape(2, -1)
        .T
        * 20.0
    )
    model = SPAVAE(
        input_dim=train_data.n_vars,
        GP_dim=2,
        Normal_dim=8,
        encoder_layers=[128, 64],
        decoder_layers=[128],
        noise=0,
        encoder_dropout=0,
        decoder_dropout=0,
        fixed_inducing_points=True,
        initial_inducing_points=inducing_points,
        fixed_gp_params=False,
        kernel_scale=20.0,
        N_train=train_data.n_obs,
        KL_loss=0.025,
        dynamicVAE=True,
        init_beta=10,
        min_beta=4,
        max_beta=25,
        dtype=torch.float64,
        device=device,
    )
    model.train_model(
        pos=locations,
        ncounts=train_data.X,
        raw_counts=train_data.raw.X,
        size_factors=train_data.obs["size_factors"],
        lr=1e-3,
        weight_decay=1e-6,
        batch_size=512,
        num_samples=1,
        train_size=0.95,
        maxiter=5000,
        patience=200,
        save_model=False,
    )
    latent = model.batching_latent_samples(
        X=locations,
        Y=train_data.X,
        batch_size=512,
    )
    prediction = mclust_labels(latent, n_clusters, SEED)
    data.obsm["spaVAE"] = latent
    data.obs["mclust"] = prediction
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "SpaVAE",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"SpaVAE_{input_h5ad.stem}.h5ad"
    data.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "SpaVAE_runtime.csv")
    print(row, flush=True)
