from pathlib import Path
import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import SEDR
import scipy.sparse as sp
import torch


INPUT_ROOT = Path(r'D:\SpaDiff\0_data\donor3_151673_151676')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\3_batch')
LEVELS = ("low", "mid", "high")
BATCH_KEY = "batch"
SEED = 2023
FORCE_CPU = False

for level in LEVELS:
    input_h5ad = INPUT_ROOT / level / "simulated_data.h5ad"
    output_path = OUTPUT_ROOT / 'SEDR' / level / "integrated.h5ad"
    print(f"Processing SEDR: {level}", flush=True)
    SEDR.fix_seed(SEED)
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

    slices = [data[data.obs[BATCH_KEY] == batch].copy() for batch in batch_order]

    processed = []
    graph_dict = None
    for current in slices:
        current.X = current.layers["counts"].copy()
        current.layers["count"] = current.layers["counts"].copy()
        sc.pp.normalize_total(current, target_sum=1e4)
        sc.pp.log1p(current)
        sc.pp.highly_variable_genes(
            current,
            flavor="seurat_v3",
            n_top_genes=min(3000, current.n_vars),
            layer="counts",
        )
        current_graph = SEDR.graph_construction(current, 12)
        graph_dict = (
            current_graph
            if graph_dict is None
            else SEDR.combine_graph_dict(graph_dict, current_graph)
        )
        processed.append(current)

    result = ad.concat(processed, join="inner", merge="same", index_unique=None)
    result.obs[BATCH_KEY] = data.obs.loc[result.obs_names, BATCH_KEY].astype(str).values
    result.obs["truth"] = data.obs.loc[result.obs_names, "truth"].astype(str).values
    result.obsm["spatial"] = data[result.obs_names].obsm["spatial"].copy()

    sc.pp.pca(
        result,
        n_comps=min(30, result.n_obs - 1, result.n_vars - 1),
        random_state=SEED,
    )
    device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda:0"
    model = SEDR.Sedr(
        result.obsm["X_pca"], graph_dict, mode="clustering", device=device
    )
    model.train_without_dec()
    sedr_embedding, _, _, _ = model.process()
    result.obsm["SEDR"] = np.asarray(sedr_embedding)

    metadata = result.obs[[BATCH_KEY]].copy()
    try:
        harmony = hm.run_harmony(
            result.obsm["SEDR"], metadata, [BATCH_KEY], random_state=SEED
        )
    except TypeError:
        harmony = hm.run_harmony(result.obsm["SEDR"], metadata, [BATCH_KEY])

    corrected = np.asarray(harmony.Z_corr)
    expected = result.obsm["SEDR"].shape
    embedding = corrected if corrected.shape == expected else corrected.T
    if embedding.shape != expected:
        raise ValueError(f"Unexpected SEDR.Harmony shape {corrected.shape}")

    result.obsm["X_sedr_harmony"] = embedding
    result.obsm["SEDR.Harmony"] = embedding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.uns["integration_method"] = "SEDR"
    result.uns["integration_embedding"] = "X_sedr_harmony"
    result.uns["input_h5ad"] = str(input_h5ad)
    result.write_h5ad(output_path, compression="gzip")
    print(f"Saved SEDR: {output_path} | shape={result.shape}", flush=True)
