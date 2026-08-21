import gc
import json
import time
from pathlib import Path

import pandas as pd
import torch
import deepstkit as dt

from utils import ResourceMonitor, calculate_ari_nmi, gpu_metrics


DATA_DIR = Path(r"D:\SpaDiff\0_data\1_DLPFC")
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "DeepST"
SLICES = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)
N_CLUSTERS = {
    sample_id: 5 if sample_id in SLICES[4:8] else 7
    for sample_id in SLICES
}

SEED = 2024
PRE_EPOCHS = 500
EPOCHS = 500
PCA_COMPONENTS = 200
USE_MORPHOLOGICAL = False
USE_GPU = torch.cuda.is_available()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_rows = []

for sample_id in SLICES:
    print(f"Processing DeepST: {sample_id}", flush=True)
    dt.utils_func.seed_torch(seed=SEED)
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    model_dir = OUTPUT_DIR / "model_files" / sample_id
    model_dir.mkdir(parents=True, exist_ok=True)
    deepst = dt.main.run(
        save_path=str(model_dir),
        task="Identify_Domain",
        pre_epochs=PRE_EPOCHS,
        epochs=EPOCHS,
        use_gpu=USE_GPU,
    )
    adata = deepst._get_adata(
        platform="Visium",
        data_path=str(DATA_DIR),
        data_name=sample_id,
    )
    adata.var_names_make_unique()

    truth = pd.read_csv(
        DATA_DIR / sample_id / "truth.txt",
        sep="\t",
        header=None,
        dtype=str,
    ).dropna(how="all")
    if truth.shape[1] >= 2:
        truth.index = truth.iloc[:, 0].str.strip().str.replace(r"-1$", "", regex=True)
        truth = truth.iloc[:, -1]
        canonical_obs = adata.obs_names.str.replace(r"-1$", "", regex=True)
        adata.obs["ground_truth"] = truth.reindex(canonical_obs).to_numpy()
    elif len(truth) == adata.n_obs:
        adata.obs["ground_truth"] = truth.iloc[:, 0].to_numpy()
    else:
        raise ValueError(f"Cannot align truth.txt for {sample_id}")
    adata.obs["ground_truth"] = pd.Categorical(adata.obs["ground_truth"])

    monitor = ResourceMonitor()
    monitor.start()
    started = time.perf_counter()

    adata = deepst._get_augment(
        adata,
        spatial_type="BallTree",
        use_morphological=USE_MORPHOLOGICAL,
    )
    graph = deepst._get_graph(adata.obsm["spatial"], distType="KDTree")
    processed = deepst._data_process(adata, pca_n_comps=PCA_COMPONENTS)
    embedding = deepst._fit(data=processed, graph_dict=graph)
    adata.obsm["DeepST_embed"] = embedding
    adata = deepst._get_cluster_data(
        adata,
        n_domains=N_CLUSTERS[sample_id],
        priori=True,
    )

    prediction_column = next(
        (column for column in (
            "DeepST_refine_domain", "DeepST_domain",
            "DeepST_cluster", "domains",
        ) if column in adata.obs),
        None,
    )
    if prediction_column is None:
        raise KeyError("DeepST did not create a recognized prediction column")
    if USE_GPU:
        torch.cuda.synchronize()
    runtime_seconds = time.perf_counter() - started
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch if USE_GPU else None),
    }

    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        adata.obs[prediction_column],
    )
    adata.uns["DeepST_ARI"] = ari
    adata.uns["DeepST_NMI"] = nmi
    adata.uns["DeepST_parameters"] = json.dumps({
        "seed": SEED,
        "pre_epochs": PRE_EPOCHS,
        "epochs": EPOCHS,
        "pca_components": PCA_COMPONENTS,
        "use_morphological": USE_MORPHOLOGICAL,
    })

    output = OUTPUT_DIR / (
        f"DeepST_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output, compression="gzip")
    row = {
        "method": "DeepST",
        "slice_id": sample_id,
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_clusters": N_CLUSTERS[sample_id],
        "n_spots": adata.n_obs,
        "n_truth_spots": n_truth_spots,
        "output_file": str(output),
        **resource_metrics,
    }
    summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "DeepST_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)

    del adata, processed, embedding, graph, deepst
    gc.collect()
    if USE_GPU:
        torch.cuda.empty_cache()
