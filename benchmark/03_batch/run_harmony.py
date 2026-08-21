from pathlib import Path
import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


INPUT_ROOT = Path(r'D:\SpaDiff\0_data\donor3_151673_151676')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\3_batch')
LEVELS = ("low", "mid", "high")
BATCH_KEY = "batch"
SEED = 2023

for level in LEVELS:
    input_h5ad = INPUT_ROOT / level / "simulated_data.h5ad"
    output_path = OUTPUT_ROOT / 'Harmony' / level / "integrated.h5ad"
    print(f"Processing Harmony: {level}", flush=True)
    np.random.seed(SEED)
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

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        flavor="seurat_v3",
        n_top_genes=min(3000, data.n_vars),
        layer="counts",
        batch_key=BATCH_KEY,
    )
    sc.pp.pca(
        data,
        n_comps=min(30, data.n_obs - 1, data.n_vars - 1),
        random_state=SEED,
    )

    metadata = data.obs[[BATCH_KEY]].copy()
    try:
        harmony = hm.run_harmony(
            data.obsm["X_pca"],
            metadata,
            [BATCH_KEY],
            random_state=SEED,
        )
    except TypeError:
        harmony = hm.run_harmony(data.obsm["X_pca"], metadata, [BATCH_KEY])

    corrected = np.asarray(harmony.Z_corr)
    expected = data.obsm["X_pca"].shape
    if corrected.shape == expected:
        embedding = corrected
    elif corrected.T.shape == expected:
        embedding = corrected.T
    else:
        raise ValueError(f"Unexpected Harmony shape {corrected.shape}; expected {expected}")

    data.obsm["X_harmony"] = embedding
    data.obsm["Harmony"] = embedding
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.uns["integration_method"] = "Harmony"
    data.uns["integration_embedding"] = "X_harmony"
    data.uns["input_h5ad"] = str(input_h5ad)
    data.write_h5ad(output_path, compression="gzip")
    print(f"Saved Harmony: {output_path} | shape={data.shape}", flush=True)
