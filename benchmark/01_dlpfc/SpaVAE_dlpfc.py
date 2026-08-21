import sys
import time
from pathlib import Path

SPAVAE_SOURCE_DIR = Path(r"D:\SpaDiff\SpaVAE\src\spaVAE")
sys.path.append(str(SPAVAE_SOURCE_DIR))

from preprocess import normalize
from spaVAE import SPAVAE

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import MinMaxScaler

from utils import ResourceMonitor, calculate_ari_nmi, gpu_metrics, mclust_labels


SLICES = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)

N_CLUSTERS = {
    sample_id: 5 if sample_id in SLICES[4:8] else 7 for sample_id in SLICES
}

DATA_DIR = Path(r"D:\SpaDiff\0_data\1_DLPFC")
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "SpaVAE"
FORCE_CPU = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_ids = SLICES
device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
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

    counts = adata.X.toarray().astype("float64")
    locations = MinMaxScaler().fit_transform(adata.obsm["spatial"]) * 20.0
    train_data = sc.AnnData(counts, dtype="float64")
    train_data = normalize(
        train_data,
        size_factors=True,
        normalize_input=True,
        logtrans_input=True,
    )
    inducing_points = (
        np.mgrid[0:1.00001:1 / 6, 0:1.00001:1 / 6].reshape(2, -1).T * 20.0
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
    prediction = mclust_labels(latent, N_CLUSTERS[sample_id])

    adata.obsm["spaVAE"] = latent
    adata.obs["mclust"] = pd.Categorical(prediction)
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }
    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        prediction,
    )
    adata.uns["SpaVAE_ARI"] = ari
    adata.uns["SpaVAE_NMI"] = nmi

    output_path = OUTPUT_DIR / (
        f"SpaVAE_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output_path, compression="gzip")

    row = {
        "method": "SpaVAE",
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
        OUTPUT_DIR / "SpaVAE_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)
