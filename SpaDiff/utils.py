from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def mclust_R(
    adata,
    num_cluster,
    modelNames="EEE",
    used_obsm="emb",
    pca_num=30,
    random_seed=200,
):
    from sklearn.decomposition import PCA

    values = np.asarray(adata.obsm[used_obsm])
    components = min(pca_num, values.shape[0] - 1, values.shape[1])
    embedding = PCA(n_components=components, random_state=random_seed).fit_transform(
        values
    )
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()
    robjects.r["set.seed"](random_seed)
    result = robjects.r["Mclust"](
        rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames
    )
    return np.asarray(result[-2], dtype=int)


def cal_purity(y_true, y_pred):
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    if y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same size")
    matrix = confusion_matrix(y_true, y_pred)
    return float(np.max(matrix, axis=0).sum() / matrix.sum())



def _adjust_clustering_resolution(
    adata,
    target_n_clusters,
    clustering_func,
    key_added,
    use_rep="X_pca",
    n_neighbors=15,
    random_state=0,
    resolution_bounds=(0.01, 5.0),
    tolerance=0,
    max_iterations=25,
    verbose=True,
    neighbors_key=None,
    clustering_kwargs=None,
):

    import scanpy as sc

    if isinstance(target_n_clusters, bool) or not isinstance(
        target_n_clusters, (int, np.integer)
    ):
        raise TypeError("target_n_clusters must be an integer")
    if target_n_clusters <= 0:
        raise ValueError("target_n_clusters must be positive")
    if adata.n_obs < 2:
        raise ValueError("at least two observations are required for graph clustering")
    if use_rep is not None and use_rep != "X" and use_rep not in adata.obsm:
        raise KeyError(f"representation {use_rep!r} was not found in adata.obsm")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)):
        raise TypeError("n_neighbors must be an integer")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not callable(clustering_func):
        raise TypeError("clustering_func must be callable")

    try:
        res_low, res_high = (float(value) for value in resolution_bounds)
    except (TypeError, ValueError) as error:
        raise ValueError("resolution_bounds must contain two finite numbers") from error
    if not np.isfinite(res_low) or not np.isfinite(res_high):
        raise ValueError("resolution bounds must be finite")
    if not 0.0 < res_low < res_high:
        raise ValueError("resolution_bounds must satisfy 0 < low < high")

    effective_neighbors = min(int(n_neighbors), adata.n_obs - 1)
    neighbor_options = {
        "use_rep": use_rep,
        "n_neighbors": effective_neighbors,
        "random_state": random_state,
    }
    if neighbors_key is not None:
        neighbor_options["key_added"] = neighbors_key
    sc.pp.neighbors(adata, **neighbor_options)

    cluster_options = dict(clustering_kwargs or {})
    protected = {"resolution", "key_added", "random_state", "neighbors_key"}
    overlap = protected.intersection(cluster_options)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"clustering_kwargs must not override: {names}")
    if neighbors_key is not None:
        cluster_options["neighbors_key"] = neighbors_key

    history = []
    best = None

    def evaluate(resolution):
        nonlocal best
        clustering_func(
            adata,
            resolution=float(resolution),
            key_added=key_added,
            random_state=random_state,
            **cluster_options,
        )
        n_clusters = int(adata.obs[key_added].nunique(dropna=True))
        if n_clusters == 0:
            raise RuntimeError("clustering produced no non-null labels")
        labels = adata.obs[key_added].copy()
        record = {"resolution": float(resolution), "n_clusters": n_clusters}
        history.append(record)
        # Prefer the smallest error; on equal error retain a non-undershooting
        # partition, then the lower resolution for a conservative solution.
        rank = (
            abs(n_clusters - target_n_clusters),
            n_clusters < target_n_clusters,
            float(resolution),
        )
        if best is None or rank < best[0]:
            best = (rank, float(resolution), n_clusters, labels)
        if verbose:
            print(
                f"  resolution={resolution:.6g} -> {n_clusters} clusters "
                f"(target={target_n_clusters})"
            )
        return n_clusters

    if verbose:
        print(
            f"Searching resolution in [{res_low:g}, {res_high:g}] for "
            f"{target_n_clusters} clusters..."
        )

    low_count = evaluate(res_low)
    if abs(low_count - target_n_clusters) > tolerance:
        evaluate(res_high)

    for _ in range(max_iterations):
        if best is not None and best[0][0] <= tolerance:
            break
        current_res = 0.5 * (res_low + res_high)
        if np.isclose(current_res, res_low) or np.isclose(current_res, res_high):
            break
        current_count = evaluate(current_res)

        if current_count < target_n_clusters:
            res_low = current_res
        else:
            res_high = current_res

    _, best_res, closest_n_clusters, best_labels = best
    adata.obs[key_added] = best_labels
    adata.uns[f"{key_added}_resolution_search"] = {
        "target_n_clusters": int(target_n_clusters),
        "selected_resolution": float(best_res),
        "selected_n_clusters": int(closest_n_clusters),
        "tolerance": int(tolerance),
        "resolutions": np.asarray(
            [item["resolution"] for item in history], dtype=np.float64
        ),
        "n_clusters": np.asarray(
            [item["n_clusters"] for item in history], dtype=np.int64
        ),
    }
    if verbose:
        status = (
            "target reached"
            if abs(closest_n_clusters - target_n_clusters) <= tolerance
            else "closest solution"
        )
        print(
            f"Selected resolution={best_res:.6g}: {closest_n_clusters} clusters "
            f"({status})."
        )
    return adata


def adjust_louvain_resolution(
    adata,
    target_n_clusters,
    use_rep="X_pca",
    key_added="louvain",
    n_neighbors=15,
    random_state=0,
    resolution_bounds=(0.01, 5.0),
    tolerance=0,
    max_iterations=25,
    verbose=True,
    neighbors_key=None,
    clustering_kwargs=None,
):
    import scanpy as sc

    if verbose:
        print(f"Adjusting Louvain toward {target_n_clusters} clusters.")
    return _adjust_clustering_resolution(
        adata=adata,
        target_n_clusters=target_n_clusters,
        clustering_func=sc.tl.louvain,
        key_added=key_added,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        random_state=random_state,
        resolution_bounds=resolution_bounds,
        tolerance=tolerance,
        max_iterations=max_iterations,
        verbose=verbose,
        neighbors_key=neighbors_key,
        clustering_kwargs=clustering_kwargs,
    )
