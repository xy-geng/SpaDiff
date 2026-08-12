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
    """Preserved optional R/mclust bridge with safe PCA dimensionality."""
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


def tfidf(X):
    """TF-IDF normalization retained from the original ATAC preprocessing."""
    import scipy.sparse

    column_sum = np.asarray(X.sum(axis=0)).ravel()
    idf = np.divide(
        X.shape[0],
        column_sum,
        out=np.zeros_like(column_sum, dtype=float),
        where=column_sum > 0,
    )
    if scipy.sparse.issparse(X):
        row_sum = np.asarray(X.sum(axis=1)).ravel()
        inv = np.divide(
            1.0, row_sum, out=np.zeros_like(row_sum, dtype=float), where=row_sum > 0
        )
        return X.multiply(inv[:, None]).multiply(idf)
    row_sum = X.sum(axis=1, keepdims=True)
    tf = np.divide(X, row_sum, out=np.zeros_like(X, dtype=float), where=row_sum > 0)
    return tf * idf


def lsi(
    adata, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs
):
    """LSI retained from the original code with zero-variance protection."""
    import sklearn.preprocessing
    import sklearn.utils.extmath

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    normalized = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(
        tfidf(adata_use.X)
    )
    normalized = np.log1p(normalized * 1e4)
    values = sklearn.utils.extmath.randomized_svd(normalized, n_components, **kwargs)[0]
    values -= values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, ddof=1, keepdims=True)
    values = np.divide(values, scale, out=np.zeros_like(values), where=scale > 0)
    adata.obsm["X_lsi"] = values[:, 1:]


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
    """Tune a graph-clustering resolution toward a requested cluster count.
    """
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
        # Louvain/Leiden cluster counts are normally non-decreasing with
        # resolution. The best visited solution is still retained if a backend
        # exhibits a small non-monotonic jump.
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
    """Run Louvain while tuning resolution toward ``target_n_clusters``."""
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
