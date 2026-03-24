import argparse
def get_args():
    parser = argparse.ArgumentParser(description="SpaDiff")
    
    # Global
    parser.add_argument('--num_pcs', type=int, default=30, help='Number of principal components for PCA')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device number') 
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--domains', type=int, default=7, help='Number of clusters for KMeans and self-supervised clustering')
    
    # HiGCN
    parser.add_argument('--RPMAX', type=int, default=100, help='Repeat times (not directly used in core model, maybe for experiments)')
    parser.add_argument('--Order', type=int, default=2, help='Max simplex dimension for Laplacians (e.g., 2 for L1 and L2)')
    parser.add_argument('--K', type=int, default=5, help='Number of propagation steps in HiGCN_prop')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha teleport probability in HiGCN_prop')
    parser.add_argument('--dprate', type=float, default=0.2, help='Dropout rate after linear layers in HiGCN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for input and pre-output in HiGCN')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension size for HiGCN embedding')

    parser.add_argument('--h_attention', type=int, default=128, help='')
    parser.add_argument('--h_decoder', type=int, default=128, help='')
    parser.add_argument('--recon_weight', type=float, default=0.5, help='')
    
    # DEC
    parser.add_argument('--init_method', type=str, default='kmeans', help='Initial clustering method (KMeans is used)')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters for KMeans and self-supervised clustering')
    parser.add_argument('--n_neighbors', type=int, default=10, help='Number of neighbors for kNN graph construction')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs (user default was 10, common is 100-500)') 
    parser.add_argument('--update_interval', type=int, default=10, help='Frequency (epochs) to update target distribution p in training')
    parser.add_argument('--trajectory_log_interval', type=int, default=50, help='Frequency (epochs) to log clustering trajectory in training')
    parser.add_argument('--alpha_dec', type=float, default=1, help='Alpha for student-t distribution in cal_q')
    parser.add_argument('--tol_dec', type=float, default=1e-4, help='')

    # spatial_reconstruction 
    parser.add_argument('--rec_alpha', type=float, default=1.0, help='Alpha weight for aggregated expression in spatial_reconstruction')
    parser.add_argument('--rec_normalize_total', type=bool, default=True, help='Whether to normalize total in spatial_reconstruction')
    
    # spatial
    parser.add_argument('--k_intra', type=int, default=6, help='')
    parser.add_argument('--k_inter', type=int, default=2, help='')

    return parser

# -------------------------------------------------------------------------
import numpy as np
import scanpy as sc
def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb',pca_num = 30,random_seed = 200):
    
    np.random.seed(random_seed)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_num, random_state=random_seed) 
    embedding = pca.fit_transform(adata.obsm[used_obsm].copy())

    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(embedding), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    # adata.obs['mclust'] = mclust_res
    # adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    # adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return mclust_res


def _adjust_clustering_resolution(
    adata,
    target_n_clusters, 
    clustering_func, 
    key_added, 
    use_rep='X_pca', 
    n_neighbors=15, 
    random_state=0, 
    resolution_bounds=(0.01, 5.0), 
    tolerance=0, 
    max_iterations=25, 
    verbose=True 
):

    if target_n_clusters <= 0: 
        raise ValueError("target_n_clusters must be a positive integer.")
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state)

    res_low, res_high = resolution_bounds
    current_res = (res_low + res_high) / 2.0 
    best_res = current_res 
    closest_n_clusters = -1 

    for i in range(max_iterations):
        if verbose:
            print(f"Iteration {i+1}/{max_iterations}: Trying resolution = {current_res:.4f}")
        clustering_func(adata, resolution=current_res, key_added=key_added, random_state=random_state)
        current_n_clusters = adata.obs[key_added].nunique() 
        if verbose:
            print(f"  -> Found {current_n_clusters} clusters.")
        if closest_n_clusters == -1 or \
           abs(current_n_clusters - target_n_clusters) < abs(closest_n_clusters - target_n_clusters) or \
           (abs(current_n_clusters - target_n_clusters) == abs(closest_n_clusters - target_n_clusters) and current_n_clusters >= target_n_clusters): 
            closest_n_clusters = current_n_clusters
            best_res = current_res

        if abs(current_n_clusters - target_n_clusters) <= tolerance: 
            if verbose:
                print(f"Target met: Found {current_n_clusters} clusters (target: {target_n_clusters}) with resolution {current_res:.4f}.")
            return adata 
        if current_n_clusters < target_n_clusters: 
            res_low = current_res
        else: 
            res_high = current_res
        
        new_res = (res_low + res_high) / 2.0 
        if np.isclose(new_res, current_res): 
             if verbose:
                print("Resolution search converged or stuck.")
             break 
        current_res = new_res
    if verbose:
        print(f"Max iterations reached or search converged. Best attempt: resolution={best_res:.4f} gave {closest_n_clusters} clusters (target: {target_n_clusters}).")
    clustering_func(adata, resolution=best_res, key_added=key_added, random_state=random_state)
    return adata


def adjust_louvain_resolution(
    adata,
    target_n_clusters, 
    use_rep='X_pca', 
    key_added='louvain', 
    n_neighbors=15, 
    random_state=0, 
    resolution_bounds=(0.01, 5.0),
    tolerance=0, 
    max_iterations=25, 
    verbose=True 
):
    if verbose:
        print(f"Adjusting Louvain clustering for target of {target_n_clusters} clusters...")
    return _adjust_clustering_resolution( 
        adata=adata, target_n_clusters=target_n_clusters, clustering_func=sc.tl.louvain, 
        key_added=key_added, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state,
        resolution_bounds=resolution_bounds, tolerance=tolerance, max_iterations=max_iterations,
        verbose=verbose
    )


def adjust_leiden_resolution(
    adata,
    target_n_clusters, 
    use_rep='X_pca', 
    key_added='leiden', 
    n_neighbors=15, 
    random_state=0, 
    resolution_bounds=(0.01, 5.0), 
    tolerance=0, 
    max_iterations=25, 
    verbose=True 
):
    if verbose:
        print(f"Adjusting Leiden clustering for target of {target_n_clusters} clusters...")
    return _adjust_clustering_resolution( 
        adata=adata, target_n_clusters=target_n_clusters, clustering_func=sc.tl.leiden, 
        key_added=key_added, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state,
        resolution_bounds=resolution_bounds, tolerance=tolerance, max_iterations=max_iterations,
        verbose=verbose
    )


# -------------------------------------------------------------------------
def cal_purity(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int_).ravel() 
    y_pred = np.asarray(y_pred, dtype=np.int_).ravel() 
    if y_true.size != y_pred.size: 
        raise ValueError("y_true and y_pred must have the same size.")
    from sklearn.metrics import confusion_matrix 
    con_mat = confusion_matrix(y_true, y_pred)
    purity = np.sum(np.amax(con_mat, axis=0)) / np.sum(con_mat)
    return purity


# -------------------------------------------------------------------------
import anndata
import sklearn
import scipy
from typing import Optional
def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    """
    LSI analysis 
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X) #.tocsr()

    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    """
    TF-IDF normalization
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   