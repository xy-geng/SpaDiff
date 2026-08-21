import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import SpaGCN as spg
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
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "SpaGCN"
SEED = 42
USE_HISTOLOGY = True

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sample_ids = SLICES
summary_rows = []

for sample_id in sample_ids:
    sample_dir = DATA_DIR / sample_id
    adata = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True,
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

    spatial_coordinates = np.asarray(adata.obsm["spatial"])
    image_path = sample_dir / f"{sample_id}_full_image.tif"
    use_histology = image_path.exists() and USE_HISTOLOGY
    image = cv2.imread(str(image_path)) if use_histology else None

    adjacency = spg.calculate_adj_matrix(
        x=spatial_coordinates[:, 0],
        y=spatial_coordinates[:, 1],
        x_pixel=spatial_coordinates[:, 0],
        y_pixel=spatial_coordinates[:, 1],
        image=image,
        beta=49,
        alpha=1,
        histology=use_histology,
    )

    spg.prefilter_genes(adata, min_cells=3)
    spg.prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)

    length_scale = spg.search_l(
        0.5,
        adjacency,
        start=0.01,
        end=1000,
        tol=0.01,
        max_run=100,
    )
    resolution = spg.search_res(
        adata,
        adjacency,
        length_scale,
        N_CLUSTERS[sample_id],
        start=0.5,
        step=0.05,
        tol=5e-3,
        lr=0.05,
        max_epochs=20,
        r_seed=SEED,
        t_seed=SEED,
        n_seed=SEED,
    )

    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = spg.SpaGCN()
    model.set_l(length_scale)
    model.train(
        adata,
        adjacency,
        init_spa=True,
        init="louvain",
        res=resolution,
        tol=5e-3,
        lr=0.05,
        max_epochs=200,
    )

    prediction, _ = model.predict()
    array_coordinates = adata.obs[["array_row", "array_col"]].to_numpy()
    spatial_adjacency = spg.calculate_adj_matrix(
        x=array_coordinates[:, 0],
        y=array_coordinates[:, 1],
        histology=False,
    )
    prediction = spg.refine(
        sample_id=adata.obs_names.tolist(),
        pred=prediction.tolist(),
        dis=spatial_adjacency,
        shape="hexagon",
    )
    adata.obs["refined_pred"] = pd.Categorical(prediction)
    runtime_seconds = time.perf_counter() - start_time
    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }

    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        prediction,
    )
    adata.uns["SpaGCN_ARI"] = ari
    adata.uns["SpaGCN_NMI"] = nmi
    adata.uns["SpaGCN_parameters"] = json.dumps(
        {
            "p": 0.5,
            "beta": 49,
            "alpha": 1,
            "histology": use_histology,
            "clustering": "SpaGCN.predict followed by hexagon refinement",
        }
    )

    output_path = OUTPUT_DIR / (
        f"SpaGCN_{sample_id}_ARI_{ari:.4f}_NMI_{nmi:.4f}.h5ad"
    )
    adata.write_h5ad(output_path, compression="gzip")

    row = {
        "method": "SpaGCN",
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
        OUTPUT_DIR / "SpaGCN_DLPFC_summary.csv",
        index=False,
    )
    print(row, flush=True)
