import gc
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.neighbors import NearestNeighbors
from spiral.layers import MeanAggregator
from spiral.main import SPIRAL_integration
from spiral.utils import layer_map, mclust_R

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
OUTPUT_DIR = Path(r"D:\SpaDiff\1_benchmark\DLPFC") / "SPIRAL"
SEED = 0

if not torch.cuda.is_available():
    raise RuntimeError(
        "SPIRAL requires CUDA, but PyTorch did not detect an available GPU."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sample_ids = SLICES

summary_rows = []


for sample_id in sample_ids:
    print(
        f"\nProcessing DLPFC slice {sample_id}",
        flush=True,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    sample_dir = DATA_DIR / sample_id
    input_dir = (
        OUTPUT_DIR
        / "spiral_input"
        / sample_id
    )
    input_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 读取单张切片
    adata = sc.read_visium(
        sample_dir,
        count_file="filtered_feature_bc_matrix.h5",
    )
    adata.var_names_make_unique()

    truth = pd.read_csv(
        sample_dir / "truth.txt",
        sep="\t",
        header=None,
        index_col=0,
    ).iloc[:, 0]

    adata.obs["ground_truth"] = truth.reindex(
        adata.obs_names
    )

    adata.obs_names = [
        f"{barcode}_{sample_id}"
        for barcode in adata.obs_names
    ]

    monitor = ResourceMonitor()
    monitor.start()
    start_time = time.perf_counter()

    # 表达矩阵预处理
    sc.pp.normalize_total(
        adata,
        target_sum=1e4,
    )
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1000,
        flavor="seurat_v3",
    )

    adata = adata[
        :,
        adata.var["highly_variable"],
    ].copy()

    # 生成SPIRAL feature文件
    if hasattr(adata.X, "toarray"):
        expression = adata.X.toarray()
    else:
        expression = np.asarray(adata.X)

    feature_frame = pd.DataFrame(
        expression,
        index=adata.obs_names,
        columns=adata.var_names,
    )

    feature_path = (
        input_dir
        / f"{sample_id}_features.csv"
    )
    feature_frame.to_csv(feature_path)

    # 构建空间邻接边
    neighbors = NearestNeighbors(
        n_neighbors=7,
    )
    neighbors.fit(
        adata.obsm["spatial"]
    )

    neighbor_indices = neighbors.kneighbors(
        return_distance=False,
    )

    edges = np.array(
        [
            (
                adata.obs_names[i],
                adata.obs_names[j],
            )
            for i in range(adata.n_obs)
            for j in neighbor_indices[i, 1:]
        ],
        dtype=str,
    )

    edge_path = (
        input_dir
        / f"{sample_id}_edges.txt"
    )
    np.savetxt(
        edge_path,
        edges,
        fmt="%s",
    )

    # 生成metadata文件
    metadata_path = (
        input_dir
        / f"{sample_id}_meta.csv"
    )

    pd.DataFrame(
        {"batch": sample_id},
        index=adata.obs_names,
    ).to_csv(metadata_path)

    # SPIRAL参数
    n_genes = adata.n_vars

    parameters = SimpleNamespace(
        seed=SEED,
        AEdims=[
            n_genes,
            [512],
            32,
        ],
        AEdimsR=[
            32,
            [512],
            n_genes,
        ],
        GSdims=[
            512,
            32,
        ],
        zdim=32,
        znoise_dim=4,
        CLdims=[
            4,
            [],
            1,
        ],
        DIdims=[
            28,
            [32, 16],
            1,
        ],
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

    # 每张切片创建一个独立模型
    model = SPIRAL_integration(
        parameters,
        [str(feature_path)],
        [str(edge_path)],
        [str(metadata_path)],
    )

    model.train()
    model.model.eval()

    # 提取SPIRAL嵌入
    all_indices = np.arange(
        model.feat.shape[0]
    )

    layers, mappings = layer_map(
        all_indices.tolist(),
        model.adj,
        len(model.params.GSdims),
    )

    adjacency_rows = (
        model.adj
        .tolil()
        .rows[layers[0]]
    )

    model_features = torch.tensor(
        model.feat.iloc[
            layers[0]
        ].values,
        dtype=torch.float32,
        device="cuda",
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

    embedding = (
        embeddings[-1]
        .detach()
        .cpu()
        .numpy()[
            :,
            model.params.znoise_dim:,
        ]
    )

    # 构建结果AnnData
    result = ad.AnnData(
        model.feat.copy()
    )

    result.obsm["spiral"] = embedding

    result.obs["ground_truth"] = (
        adata.obs.loc[
            result.obs_names,
            "ground_truth",
        ].values
    )

    # 使用SPIRAL内置mclust接口
    result = mclust_R(
        result,
        used_obsm="spiral",
        num_cluster=N_CLUSTERS[sample_id],
    )

    # 统计运行时间和资源
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
    ari, nmi, n_truth_spots = (
        calculate_ari_nmi(
            result.obs["ground_truth"],
            result.obs["mclust"],
        )
    )

    result.uns["SPIRAL_ARI"] = ari
    result.uns["SPIRAL_NMI"] = nmi
    result.uns["SPIRAL_n_clusters"] = (
        N_CLUSTERS[sample_id]
    )
    result.uns["SPIRAL_seed"] = SEED

    # 保存h5ad
    output_path = OUTPUT_DIR / (
        f"SPIRAL_{sample_id}"
        f"_ARI_{ari:.4f}"
        f"_NMI_{nmi:.4f}.h5ad"
    )

    result.write_h5ad(
        output_path,
        compression="gzip",
    )

    summary_row = {
        "method": "SPIRAL",
        "slice_id": sample_id,
        "status": "completed",
        "ari": ari,
        "nmi": nmi,
        "runtime_seconds": runtime_seconds,
        "n_clusters": N_CLUSTERS[sample_id],
        "n_spots": result.n_obs,
        "n_truth_spots": n_truth_spots,
        "output_file": str(output_path),
        **resource_metrics,
    }

    summary_rows.append(
        summary_row
    )

    # 每完成一张切片立即更新summary
    summary_path = (
        OUTPUT_DIR
        / "SPIRAL_DLPFC_summary.csv"
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

    # 释放当前切片的内存和显存
    del model
    del model_features
    del embeddings
    del embedding
    del result
    del adata
    del expression
    del feature_frame

    gc.collect()
    torch.cuda.empty_cache()


print(
    "\nAll requested SPIRAL slices completed.",
    flush=True,
)

print(
    f"Summary: "
    f"{OUTPUT_DIR / 'SPIRAL_DLPFC_summary.csv'}",
    flush=True,
)
