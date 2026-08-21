import sys
from pathlib import Path

SPADIFF_SOURCE_DIR = Path(r"D:\SpaDiff\SpaDiff")
sys.path.append(str(SPADIFF_SOURCE_DIR.parent))

import SpaDiff as sd
from SpaDiff.utils import set_seed

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


INPUT_ROOT = Path(r'D:\SpaDiff\0_data\donor3_151673_151676')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\3_batch')
LEVELS = ("low", "mid", "high")
BATCH_KEY = "batch"
SEED = 42
FORCE_CPU = False
EPOCHS = 500
BATCH_LOSS_WEIGHT = 0.5
PRIOR_KL_WEIGHT = 1.0
HARMONIZE_STRENGTH = 0.10

for level in LEVELS:
    input_h5ad = INPUT_ROOT / level / "simulated_data.h5ad"
    output_path = OUTPUT_ROOT / 'SpaDiff' / level / "integrated.h5ad"
    print(f"Processing SpaDiff: {level}", flush=True)
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"
    )
    data = ad.read_h5ad(input_h5ad)
    data.var_names_make_unique()
    data.obs_names_make_unique()

    obs_lookup = {str(column).lower(): column for column in data.obs.columns}
    for required in (BATCH_KEY, "truth"):
        if required not in obs_lookup:
            raise KeyError(f"{input_h5ad} does not contain obs['{required}']")
        source = obs_lookup[required]
        if source != required:
            data.obs[required] = data.obs[source]

    data = data[data.obs[BATCH_KEY].notna() & data.obs["truth"].notna()].copy()
    data.obs[BATCH_KEY] = data.obs[BATCH_KEY].astype(str)
    data.obs["truth"] = data.obs["truth"].astype(str).astype("category")
    batch_values = data.obs[BATCH_KEY].unique().tolist()
    batch_order = sorted(
        batch_values,
        key=lambda value: (
            (0, int(value[5:]))
            if value.startswith("batch") and value[5:].isdigit()
            else (1, value)
        ),
    )
    data.obs[BATCH_KEY] = pd.Categorical(
        data.obs[BATCH_KEY], categories=batch_order, ordered=True
    )

    if "counts" in data.layers:
        counts = data.layers["counts"].copy()
    elif data.raw is not None:
        raw = data.raw.to_adata()
        if raw.shape != data.shape or not raw.var_names.equals(data.var_names):
            raise ValueError("adata.raw is not aligned with the simulated matrix")
        counts = raw.X.copy()
    else:
        counts = data.X.copy()
    data.layers["counts"] = counts
    data.X = counts.copy()

    if "spatial" not in data.obsm:
        data.obsm["spatial"] = data.obs[["spatial1", "spatial2"]].to_numpy(
            dtype=np.float64
        )
    if sp.issparse(data.X):
        data.X.eliminate_zeros()

    data.obs["batch_name"] = data.obs[BATCH_KEY].astype(str)

    # Normalize expression while retaining raw counts for HVG selection.
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        layer="counts",
        n_top_genes=min(3000, data.n_vars),
        batch_key=BATCH_KEY,
        subset=True,
    )

    _, adjacency = sd.Neiber(
        data,
        k_intra=6,
        k_inter=2,
        slice_order=batch_order,
    )
    adjacency = adjacency.maximum(adjacency.T)
    operators = sd.to_torch_operators(
        sd.build_simplicial_operators(adjacency, max_order=2), device=device
    )

    sc.tl.pca(
        data,
        n_comps=min(50, data.n_obs - 1, data.n_vars - 1),
        random_state=SEED,
    )
    features = torch.as_tensor(
        np.asarray(data.obsm["X_pca"]), dtype=torch.float32, device=device
    )
    batch_category = data.obs[BATCH_KEY].cat.set_categories(batch_order)
    batch_ids = torch.as_tensor(
        batch_category.cat.codes.to_numpy(), dtype=torch.long, device=device
    )
    modality_ids = torch.zeros(data.n_obs, dtype=torch.long, device=device)

    config = sd.SpaDiffConfig(
        data_dim=features.shape[1],
        condition_input_dim=features.shape[1],
        num_batches=len(batch_order),
        num_modalities=1,
        num_scales=1000,
        topology_hidden_dim=128,
        topology_dim=64,
        propagation_steps=3,
        propagation_alpha=0.4,
        hidden_dim=128,
        dropout=0.1,
        topology_projection_dropout=0.0,
        topology_residual=True,
        topology_output_normalization="feature",
        simplex_orders=(1, 2),
        dsm_weighting="variance",
        dsm_weight=1.0,
        batch_alignment_weight=BATCH_LOSS_WEIGHT,
        batch_posterior_weight=1.0,
        prior_kl_weight=PRIOR_KL_WEIGHT,
        batch_balanced_loss=True,
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

    reference_ids = torch.zeros_like(batch_ids)
    if training.ema is not None:
        training.ema.store(model.parameters())
        training.ema.copy_to(model.parameters())
    try:
        model.eval()
        with torch.no_grad():
            topology_embedding = model.encode_condition(features, operators)
            harmonized_embedding = model.harmonize(
                observed_features=features,
                operators=operators,
                reference_batch_ids=reference_ids,
                modality_ids=modality_ids,
                strength=HARMONIZE_STRENGTH,
                guidance_scale=1.0,
                ode_steps=300,
            )
    finally:
        if training.ema is not None:
            training.ema.restore(model.parameters())

    data.obsm["spadiff"] = topology_embedding.detach().cpu().numpy()
    data.obsm["X_spadiff"] = harmonized_embedding.detach().cpu().numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.uns["integration_method"] = "SpaDiff"
    data.uns["integration_embedding"] = "X_spadiff"
    data.uns["input_h5ad"] = str(input_h5ad)
    data.write_h5ad(output_path, compression="gzip")
    print(f"Saved SpaDiff: {output_path} | shape={data.shape}", flush=True)
