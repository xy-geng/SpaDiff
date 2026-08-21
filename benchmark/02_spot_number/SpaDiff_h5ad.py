import sys
import time
from pathlib import Path

SPADIFF_SOURCE_DIR = Path(r"D:\SpaDiff\SpaDiff_improved")
sys.path.append(str(SPADIFF_SOURCE_DIR.parent))

import SpaDiff_improved as sd
from SpaDiff_improved.spatial import spatial_reconstruction
from SpaDiff_improved.topology import (
    build_simplicial_operators,
    to_torch_operators,
)
from SpaDiff_improved.utils import mclust_R, set_seed

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from utils import (
    ResourceMonitor,
    calculate_metrics,
    load_h5ad,
    update_summary,
)


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "SpaDiff"
SEED = 42
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device(
    "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"
)
simplex_orders = (
    (0,)
    if MAX_ORDER == 0
    else tuple(range(1, MAX_ORDER + 1))
)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing SpaDiff: {input_h5ad}", flush=True)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    data = load_h5ad(input_h5ad)
    data.obs["Truth"] = data.obs["ground_truth"]
    n_clusters = int(
        N_CLUSTERS_OVERRIDE or data.obs["Truth"].nunique()
    )

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    if "counts" not in data.layers:
        data.layers["counts"] = data.X.copy()
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
        num_cluster=n_clusters,
        used_obsm="X_spadiff",
        pca_num=20,
        random_seed=SEED,
    )
    data.obs["mclust"] = pd.Categorical(labels.astype(str))
    if device.type == "cuda":
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = monitor.stop()
    metrics = calculate_metrics(
        data.obs["Truth"],
        data.obs["mclust"],
    )

    data.uns["SpaDiff_ARI"] = metrics["ari"]
    data.uns["SpaDiff_NMI"] = metrics["nmi"]
    data.uns["SpaDiff_n_clusters"] = n_clusters
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
    data.uns["SpaDiff_mode"] = "one_h5ad_as_one_slice"

    output_path = OUTPUT_DIR / f"SpaDiff_{input_h5ad.stem}.h5ad"
    row = {
        "method": "SpaDiff",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "runtime_seconds": runtime_seconds,
        "n_clusters": n_clusters,
        "n_spots": data.n_obs,
        "n_truth_spots": metrics["n_truth_spots"],
        "best_loss": training.best_loss,
        "best_epoch": training.best_epoch,
        "output_file": str(output_path),
        **resource_metrics,
    }
    data.write_h5ad(output_path, compression="gzip")
    update_summary(row, OUTPUT_DIR / "SpaDiff_runtime.csv")
    print(row, flush=True)
