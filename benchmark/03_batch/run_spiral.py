import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from spiral.layers import MeanAggregator
from spiral.main import SPIRAL_integration
from spiral.utils import layer_map

INPUT_ROOT = Path(r'D:\SpaDiff\0_data\donor3_151673_151676')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\3_batch')
LEVELS = ("low", "mid", "high")
BATCH_KEY = "batch"
SEED = 0
FORCE_CPU = False
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
torch.set_num_interop_threads(1)

for level in LEVELS:
    input_h5ad = INPUT_ROOT / level / "simulated_data.h5ad"
    output_path = OUTPUT_ROOT / 'SPIRAL' / level / "integrated.h5ad"
    print(f"Processing SPIRAL: {level}", flush=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"Batches: {batch_order}", flush=True)
    print(
        f"Threads per SPIRAL process: {torch.get_num_threads()}",
        flush=True,
    )

    processed = []
    for current in slices:
        current.X = current.layers["counts"].copy()
        sc.pp.highly_variable_genes(
            current,
            flavor="seurat_v3",
            n_top_genes=min(1000, current.n_vars),
            layer="counts",
            subset=True,
        )
        sc.pp.normalize_total(current, target_sum=1e4)
        sc.pp.log1p(current)
        processed.append(current)

    integrated = ad.concat(processed, join="inner", merge="same", index_unique=None)
    integrated.obs[BATCH_KEY] = data.obs.loc[integrated.obs_names, BATCH_KEY].astype(str).values
    integrated.obs["truth"] = data.obs.loc[integrated.obs_names, "truth"].astype(str).values
    integrated.obsm["spatial"] = data[integrated.obs_names].obsm["spatial"].copy()
    print(f"Common HVGs across batches: {integrated.n_vars}", flush=True)

    knn = 6
    with tempfile.TemporaryDirectory(
        prefix="spiral_", dir=str(output_path.parent)
    ) as temporary:
        temporary = Path(temporary)
        feature_files = []
        edge_files = []
        metadata_files = []

        for index, batch in enumerate(batch_order):
            current = integrated[integrated.obs[BATCH_KEY] == batch].copy()
            feature_file = temporary / f"{batch}_features.csv"
            edge_file = temporary / f"{batch}_edges.txt"
            metadata_file = temporary / f"{batch}_metadata.csv"

            matrix = current.X.toarray() if hasattr(current.X, "toarray") else np.asarray(current.X)
            pd.DataFrame(
                matrix, index=current.obs_names, columns=current.var_names
            ).to_csv(feature_file)

            coordinates = np.asarray(current.obsm["spatial"])
            neighbor_indices = (
                NearestNeighbors(n_neighbors=knn + 1)
                .fit(coordinates)
                .kneighbors(coordinates, return_distance=False)
            )
            edges = np.asarray(
                [
                    (current.obs_names[i], current.obs_names[j])
                    for i in range(current.n_obs)
                    for j in neighbor_indices[i, 1:]
                ],
                dtype=str,
            )
            np.savetxt(edge_file, edges, fmt="%s")
            pd.DataFrame(
                {"batch": str(index)}, index=current.obs_names
            ).to_csv(metadata_file)

            feature_files.append(str(feature_file))
            edge_files.append(str(edge_file))
            metadata_files.append(str(metadata_file))

        n_genes = integrated.n_vars
        n_batches = len(batch_order) if len(batch_order) > 2 else 1
        parameters = SimpleNamespace(
            seed=SEED,
            AEdims=[n_genes, [512], 32],
            AEdimsR=[32, [512], n_genes],
            GSdims=[512, 32],
            zdim=32,
            znoise_dim=4,
            CLdims=[4, [], n_batches],
            DIdims=[28, [32, 16], n_batches],
            beta=1.0,
            agg_class=MeanAggregator,
            num_samples=knn,
            N_WALKS=knn,
            WALK_LEN=1,
            N_WALK_LEN=knn,
            NUM_NEG=knn,
            epochs=100,
            batch_size=1024,
            lr=1e-3,
            weight_decay=5e-4,
            alpha1=n_genes,
            alpha2=1,
            alpha3=1,
            alpha4=1,
            lamda=1,
            Q=10,
        )
        model = SPIRAL_integration(
            parameters, feature_files, edge_files, metadata_files
        )
        model.train()
        model.model.eval()

        indices = np.arange(model.feat.shape[0])
        layers, mappings = layer_map(
            indices.tolist(), model.adj, len(model.params.GSdims)
        )
        rows = model.adj.tolil().rows[layers[0]]
        model_device = next(model.model.parameters()).device
        if FORCE_CPU:
            model.model.to("cpu")
            model_device = torch.device("cpu")
        features = torch.as_tensor(
            model.feat.iloc[layers[0]].values,
            dtype=torch.float32,
            device=model_device,
        )
        with torch.no_grad():
            embeddings, _, _, _ = model.model(
                features,
                layers,
                mappings,
                rows,
                model.params.lamda,
                model.de_act,
                model.cl_act,
            )

        embedding = embeddings[-1].detach().cpu().numpy()
        embedding = embedding[:, model.params.znoise_dim :]
        embedding_frame = pd.DataFrame(embedding, index=model.feat.index)

    integrated = integrated[embedding_frame.index].copy()
    integrated.obsm["X_spiral"] = embedding_frame.loc[
        integrated.obs_names
    ].to_numpy()
    integrated.obsm["spiral"] = integrated.obsm["X_spiral"].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    integrated.uns["integration_method"] = "SPIRAL"
    integrated.uns["integration_embedding"] = "X_spiral"
    integrated.uns["input_h5ad"] = str(input_h5ad)
    integrated.write_h5ad(output_path, compression="gzip")
    print(f"Saved SPIRAL: {output_path} | shape={integrated.shape}", flush=True)
