import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from spiral.layers import MeanAggregator
from spiral.main import SPIRAL_integration
from spiral.utils import layer_map, mclust_R

from utils import ResourceMonitor, calculate_metrics, load_h5ad, update_summary


DATA_DIR = Path(r'D:\SpaDiff\0_data\2_spot_number')
OUTPUT_ROOT = Path(r'D:\SpaDiff\1_benchmark\2_spot_number')
SPOT_NUMBERS = (2000, 4000, 6000, 8000, 10000)
OUTPUT_DIR = OUTPUT_ROOT / "SPIRAL"
SEED = 0
FORCE_CPU = False
N_CLUSTERS_OVERRIDE = None
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for spot_number in SPOT_NUMBERS:
    input_h5ad = DATA_DIR / f"DLPFC_4slices_{spot_number}_spots.h5ad"
    print(f"Processing SPIRAL: {input_h5ad}", flush=True)
    data = load_h5ad(input_h5ad)
    n_clusters = int(
        N_CLUSTERS_OVERRIDE or data.obs["ground_truth"].nunique()
    )
    input_dir = OUTPUT_DIR / "spiral_input"
    os.makedirs(input_dir, exist_ok=True)

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(
        data,
        n_top_genes=min(1000, data.n_vars),
        flavor="seurat_v3",
    )
    data = data[:, data.var["highly_variable"]].copy()

    if hasattr(data.X, "toarray"):
        feature = data.X.toarray()
    else:
        feature = np.asarray(data.X)

    feature_path = input_dir / "single_slice_features.csv"
    edge_path = input_dir / "single_slice_edges.txt"
    metadata_path = input_dir / "single_slice_meta.csv"
    pd.DataFrame(
        feature,
        index=data.obs_names,
        columns=data.var_names,
    ).to_csv(feature_path)

    neighbor_indices = (
        NearestNeighbors(n_neighbors=7)
        .fit(np.asarray(data.obsm["spatial"]))
        .kneighbors(return_distance=False)
    )
    edges = np.array(
        [
            (data.obs_names[i], data.obs_names[j])
            for i in range(data.n_obs)
            for j in neighbor_indices[i, 1:]
        ],
        dtype=str,
    )
    np.savetxt(edge_path, edges, fmt="%s")
    pd.DataFrame(
        {"batch": "single_slice"},
        index=data.obs_names,
    ).to_csv(metadata_path)

    n_genes = data.n_vars
    parameters = SimpleNamespace(
        seed=SEED,
        AEdims=[n_genes, [512], 32],
        AEdimsR=[32, [512], n_genes],
        GSdims=[512, 32],
        zdim=32,
        znoise_dim=4,
        CLdims=[4, [], 1],
        DIdims=[28, [32, 16], 1],
        beta=1.0,
        agg_class=MeanAggregator,
        num_samples=6,
        N_WALKS=6,
        WALK_LEN=1,
        N_WALK_LEN=6,
        NUM_NEG=6,
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
        parameters,
        [str(feature_path)],
        [str(edge_path)],
        [str(metadata_path)],
    )
    model.train()
    model.model.eval()

    all_indices = np.arange(model.feat.shape[0])
    layers, mappings = layer_map(
        all_indices.tolist(),
        model.adj,
        len(model.params.GSdims),
    )
    adjacency_rows = model.adj.tolil().rows[layers[0]]
    device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"
    model_features = torch.tensor(
        model.feat.iloc[layers[0]].values,
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        embeddings, _, _, _ = model.model(
            model_features,
            layers,
            mappings,
            adjacency_rows,
            model.params.lamda,
            model.de_act,
            model.cl_act,
        )

    data.obsm["spiral"] = (
        embeddings[-1]
        .detach()
        .cpu()
        .numpy()[:, model.params.znoise_dim :]
    )
    data = mclust_R(
        data,
        used_obsm="spiral",
        num_cluster=n_clusters,
    )
    prediction = data.obs["mclust"].to_numpy()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = monitor.stop()

    row = {
        "method": "SPIRAL",
        "input_h5ad": str(input_h5ad),
        "status": "completed",
        "n_spots": data.n_obs,
        "runtime_seconds": runtime_seconds,
        **calculate_metrics(data.obs["ground_truth"], prediction),
        **resource_metrics,
    }
    output = OUTPUT_DIR / f"SPIRAL_{input_h5ad.stem}.h5ad"
    data.write_h5ad(output, compression="gzip")
    row["output_file"] = str(output)
    update_summary(row, OUTPUT_DIR / "SPIRAL_runtime.csv")
    print(row, flush=True)
