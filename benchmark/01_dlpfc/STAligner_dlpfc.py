#!/usr/bin/env python3
"""Run STAligner independently and sequentially on DLPFC slices."""

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import STAligner
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
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "STAligner"
SEED = 42
FORCE_CPU = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sample_ids = SLICES

device = torch.device(
    "cpu"
    if FORCE_CPU or not torch.cuda.is_available()
    else "cuda"
)

summary_rows = []


for sample_id in sample_ids:
    print(
        f"\nProcessing DLPFC slice {sample_id}",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    sample_dir = DATA_DIR / sample_id

    # 读取单张切片
    adata = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
        load_images=True,
    )
    adata.var_names_make_unique(
        join="++"
    )

    # 读取真实标签
    truth = pd.read_csv(
        sample_dir / "truth.txt",
        sep="\t",
        header=None,
        index_col=0,
    ).iloc[:, 0]

    adata.obs["ground_truth"] = truth.reindex(
        adata.obs_names
    ).astype("category")

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    adata.obs["slice_id"] = sample_id

    # train_STAligner要求存在batch_name
    adata.obs["batch_name"] = pd.Categorical(
        [sample_id] * adata.n_obs
    )

    # 构建单张切片空间图
    STAligner.Cal_Spatial_Net(
        adata,
        rad_cutoff=150,
    )

    # 筛选高变基因
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=3000,
    )

    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )
    sc.pp.log1p(adata)

    adata = adata[
        :,
        adata.var["highly_variable"],
    ].copy()

    # 将单张切片邻接矩阵转换为STAligner要求的edgeList
    adjacency = np.asarray(
        adata.uns["adj"].todense()
    )

    adata.uns["edgeList"] = np.nonzero(
        adjacency
    )

    # 单张切片没有跨切片 MNN。
    adata = STAligner.train_STAligner(
        adata,
        iter_comb=None,
        verbose=True,
        knn_neigh=50,
        device=device,
        margin=1.0,
        random_seed=SEED,
    )

    # 每张切片单独使用mclust
    adata = STAligner.mclust_R(
        adata,
        num_cluster=N_CLUSTERS[sample_id],
        used_obsm="STAligner",
        random_seed=SEED,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    resource_metrics = {
        **monitor.stop(),
        **gpu_metrics(torch),
    }

    # 计算ARI和NMI
    ari, nmi, n_truth_spots = calculate_ari_nmi(
        adata.obs["ground_truth"],
        adata.obs["mclust"],
    )

    adata.uns["STAligner_ARI"] = ari
    adata.uns["STAligner_NMI"] = nmi
    adata.uns["STAligner_n_clusters"] = (
        N_CLUSTERS[sample_id]
    )
    adata.uns["STAligner_seed"] = SEED
    adata.uns["STAligner_knn_neigh"] = 50
    adata.uns["STAligner_mode"] = (
        "single_slice_graph_autoencoder_pretraining"
    )

    # 保存h5ad
    output_path = OUTPUT_DIR / (
        f"STAligner_{sample_id}"
        f"_ARI_{ari:.4f}"
        f"_NMI_{nmi:.4f}.h5ad"
    )

    del adata.uns["edgeList"]
    adata.write_h5ad(
        output_path,
        compression="gzip",
    )

    summary_row = {
        "method": "STAligner",
        "slice_id": sample_id,
        "status": "completed",
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_clusters": N_CLUSTERS[sample_id],
        "knn_neigh": 50,
        "mode": "single_slice_pretraining",
        "n_spots": adata.n_obs,
        "n_truth_spots": n_truth_spots,
        "output_file": str(output_path),
        **resource_metrics,
    }

    summary_rows.append(
        summary_row
    )

    # 每完成一张立即更新summary
    summary_path = (
        OUTPUT_DIR
        / "STAligner_DLPFC_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
    )

    print(
        summary_row,
        flush=True,
    )

    # 释放当前切片资源
    del adata
    del adjacency

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


print(
    "\nAll requested STAligner slices completed.",
    flush=True,
)

print(
    f"Summary: "
    f"{OUTPUT_DIR / 'STAligner_DLPFC_summary.csv'}",
    flush=True,
)
