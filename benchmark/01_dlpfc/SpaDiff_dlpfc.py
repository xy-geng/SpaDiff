import fcntl
import gc
import os
import sys
import time
from pathlib import Path

SPADIFF_SOURCE_DIR = Path(r"D:\SpaDiff\SpaDiff")
sys.path.append(str(SPADIFF_SOURCE_DIR.parent))

import SpaDiff as sd
from SpaDiff.spatial import spatial_reconstruction
from SpaDiff.topology import (
    build_simplicial_operators,
    to_torch_operators,
)
from SpaDiff.utils import mclust_R, set_seed

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from utils import ResourceMonitor, calculate_ari_nmi, gpu_metrics


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
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "SpaDiff"
SEED = 42
EPOCHS = 500
N_NEIGHBORS = 8
N_TOP_GENES = 3000
N_PCS = 50
MAX_ORDER = 2
DSM_WEIGHTING = "variance"
DSM_WEIGHT = 1.0
BATCH_ALIGNMENT_WEIGHT = 0.0
BATCH_POSTERIOR_WEIGHT = 0.0
PRIOR_KL_WEIGHT = 1.0
HARMONIZE_STRENGTH = 0.10
ODE_STEPS = 200
FORCE_CPU = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_ids = SLICES
device = torch.device(
    "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"
)
simplex_orders = (
    (0,)
    if MAX_ORDER == 0
    else tuple(range(1, MAX_ORDER + 1))
)
for sample_id in sample_ids:
    print(f"\nProcessing DLPFC slice {sample_id}", flush=True)

    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    sample_dir = DATA_DIR / sample_id
    data = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    data.var_names_make_unique()
    data.layers["counts"] = data.X.copy()

    truth = pd.read_csv(
        sample_dir / "truth.txt",
        sep="\t",
        header=None,
        index_col=0,
    ).iloc[:, 0]
    data.obs["Truth"] = truth.reindex(data.obs_names)

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=min(N_TOP_GENES, data.n_vars),
        subset=True,
    )

    data, adjacency = spatial_reconstruction(
        data,
        alpha=1.0,
        n_neighbors=N_NEIGHBORS,
    )
    operators = to_torch_operators(
        build_simplicial_operators(
            adjacency,
            max_order=MAX_ORDER,
        ),
        device=device,
    )
    sc.tl.pca(
        data,
        n_comps=min(
            N_PCS,
            data.n_obs - 1,
            data.n_vars - 1,
        ),
        random_state=SEED,
    )
    features = torch.as_tensor(
        np.asarray(data.obsm["X_pca"]),
        dtype=torch.float32,
        device=device,
    )
    batch_ids = torch.zeros(
        data.n_obs,
        dtype=torch.long,
        device=device,
    )
    modality_ids = torch.zeros(
        data.n_obs,
        dtype=torch.long,
        device=device,
    )

    config = sd.SpaDiffConfig(
        data_dim=features.shape[1],
        condition_input_dim=features.shape[1],
        num_batches=1,
        num_modalities=1,
        num_scales=1000,
        topology_hidden_dim=128,
        topology_dim=64,
        propagation_steps=5,
        propagation_alpha=0.4,
        hidden_dim=128,
        dropout=0.1,
        topology_projection_dropout=0.0,
        learnable_propagation=False,
        topology_residual=True,
        topology_output_normalization="feature",
        simplex_orders=simplex_orders,
        dsm_weighting=DSM_WEIGHTING,
        dsm_weight=DSM_WEIGHT,
        batch_alignment_weight=BATCH_ALIGNMENT_WEIGHT,
        batch_posterior_weight=BATCH_POSTERIOR_WEIGHT,
        prior_kl_weight=PRIOR_KL_WEIGHT,
    )
    model = sd.SpaDiff(config).to(device)
    training = sd.train_spadiff(
        model,
        features,
        operators,
        batch_ids,
        modality_ids,
        epochs=EPOCHS,
        learning_rate=1e-3,
        weight_decay=1e-4,
        ema_decay=0.99,
        verbose_every=25,
    )

    if training.ema is not None:
        training.ema.store(model.parameters())
        training.ema.copy_to(model.parameters())
    try:
        model.eval()
        with torch.no_grad():
            topology_embedding = model.encode_condition(
                features,
                operators,
            )
            harmonized_embedding = model.harmonize(
                observed_features=features,
                operators=operators,
                reference_batch_ids=batch_ids,
                modality_ids=modality_ids,
                strength=HARMONIZE_STRENGTH,
                guidance_scale=1.0,
                ode_steps=ODE_STEPS,
            )
    finally:
        if training.ema is not None:
            training.ema.restore(model.parameters())

    data.obsm["spadiff"] = topology_embedding.detach().cpu().numpy()
    data.obsm["X_spadiff"] = (
        harmonized_embedding.detach().cpu().numpy()
    )
    labels = mclust_R(
        data,
        num_cluster=N_CLUSTERS[sample_id],
        used_obsm="X_spadiff",
        pca_num=20,
        random_seed=SEED,
    )
    data.obs["mclust"] = pd.Categorical(labels.astype(str))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }
    ari, nmi, n_truth_spots = calculate_ari_nmi(
        data.obs["Truth"],
        data.obs["mclust"],
    )

    data.uns["SpaDiff_ARI"] = ari
    data.uns["SpaDiff_NMI"] = nmi
    data.uns["SpaDiff_n_clusters"] = N_CLUSTERS[sample_id]
    data.uns["SpaDiff_seed"] = SEED
    data.uns["SpaDiff_epochs"] = EPOCHS
    data.uns["SpaDiff_max_order"] = MAX_ORDER
    data.uns["SpaDiff_dsm_weighting"] = DSM_WEIGHTING
    data.uns["SpaDiff_dsm_weight"] = DSM_WEIGHT
    data.uns["SpaDiff_batch_alignment_weight"] = (
        BATCH_ALIGNMENT_WEIGHT
    )
    data.uns["SpaDiff_batch_posterior_weight"] = (
        BATCH_POSTERIOR_WEIGHT
    )
    data.uns["SpaDiff_prior_kl_weight"] = PRIOR_KL_WEIGHT
    data.uns["SpaDiff_harmonize_strength"] = HARMONIZE_STRENGTH
    data.uns["SpaDiff_ode_steps"] = ODE_STEPS
    data.uns["SpaDiff_cluster_representation"] = "X_spadiff"
    data.uns["SpaDiff_best_loss"] = training.best_loss
    data.uns["SpaDiff_best_epoch"] = training.best_epoch

    output_path = OUTPUT_DIR / (
        f"SpaDiff_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    data.write_h5ad(output_path, compression="gzip")
    row = {
        "method": "SpaDiff",
        "slice_id": sample_id,
        "status": "completed",
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_clusters": N_CLUSTERS[sample_id],
        "n_spots": data.n_obs,
        "n_truth_spots": n_truth_spots,
        "best_loss": training.best_loss,
        "best_epoch": training.best_epoch,
        "output_file": str(output_path),
        **resource_metrics,
    }
    summary_path = OUTPUT_DIR / "SpaDiff_DLPFC_summary.csv"
    lock_path = OUTPUT_DIR / ".SpaDiff_DLPFC_summary.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if summary_path.is_file():
            summary = pd.read_csv(summary_path)
            if "slice_id" in summary.columns:
                summary = summary[
                    summary["slice_id"].astype(str) != sample_id
                ]
        else:
            summary = pd.DataFrame()
        summary = pd.concat(
            [summary, pd.DataFrame([row])],
            ignore_index=True,
            sort=False,
        )
        slice_order = {
            value: index for index, value in enumerate(SLICES)
        }
        summary["_slice_order"] = (
            summary["slice_id"].astype(str).map(slice_order)
        )
        summary = summary.sort_values("_slice_order").drop(
            columns="_slice_order"
        )
        temporary_path = summary_path.with_name(
            f".{summary_path.name}.{os.getpid()}.tmp"
        )
        summary.to_csv(temporary_path, index=False)
        os.replace(temporary_path, summary_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print(row, flush=True)

    del data, adjacency, operators, features
    del model, training, topology_embedding, harmonized_embedding
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

print(
    "\nSpaDiff summary: "
    f"{OUTPUT_DIR / 'SpaDiff_DLPFC_summary.csv'}",
    flush=True,
)
