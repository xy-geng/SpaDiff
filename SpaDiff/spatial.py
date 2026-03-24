
from typing import Optional, Tuple
from anndata import AnnData

import numpy as np
import scanpy as sc

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix


import numpy as np
import networkx as nx
import torch
from torch_sparse import SparseTensor, mul 



def HL_Loader(coord: np.ndarray, nbrs_adj_matrix, Order=2):
    num_nodes = coord.shape[0]
    if hasattr(nbrs_adj_matrix, "tocoo"): 
        edges = np.array(nbrs_adj_matrix.tocoo().nonzero()).T
    else: 
        edges = np.array(np.nonzero(nbrs_adj_matrix)).T 

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    HL = creat_L2(G, max_clique_size= Order)

    print("Finished calculating Higher-order Laplacians!")
    return HL


def creat_L2(G: nx.Graph, max_clique_size: int = 2):

    print(f"Calculating Laplacians up to order {max_clique_size}...")
   
    num_nodes = G.number_of_nodes()

    larger_nei = {} 
    for v_i in G.nodes:
        larger_nei[v_i] = {nei_j for nei_j in G.neighbors(v_i) if nei_j > v_i}

    _D, L = {}, {} 
    for i in range(1, max_clique_size + 1):
        _D[i] = torch.zeros(num_nodes)
        L[i] = torch.zeros((num_nodes, num_nodes))

    for edge in G.edges:
        u, v = edge
        if u == v: 
            continue
        # 1阶
        _D[1][u] += 1
        _D[1][v] += 1
        L[1][u, u] += 1
        L[1][v, v] += 1
        L[1][u, v] += 1 
        L[1][v, u] += 1 
       
        # 2阶
        com_nei = larger_nei[u] & larger_nei[v]
        _D[2][u] += len(com_nei)
        _D[2][v] += len(com_nei)
        L[2][u, u] += len(com_nei)
        L[2][v, v] += len(com_nei)
        L[2][u, v] += len(com_nei)
        L[2][v, u] += len(com_nei)

        for i in com_nei:
            L[2][u, i] += 1
            L[2][i, u] += 1
            L[2][v, i] += 1
            L[2][i, v] += 1
            L[2][i, i] += 1
            _D[2][i] += 1
    # print("n0:{}, n1:{}, n2:{}".format(G.number_of_nodes(), G.number_of_edges(), sum(_D[2]) / 3))
    new_triangle = sum(nx.triangles(G).values()) / 3
    print("总共三角形数：", new_triangle)

    for k_ in range(1, max_clique_size + 1):
        L[k_] = SparseTensor.from_dense(L[k_])
        D_inv_sqrt = _D[k_].pow_(-0.5)
        D_inv_sqrt.masked_fill_(D_inv_sqrt == float('inf'), 0.)
        L[k_] = mul(L[k_], D_inv_sqrt.view(1, -1))
        L[k_] = mul(L[k_], D_inv_sqrt.view(-1, 1) / (k_ + 1))
    return L

def spatial_reconstruction(
    adata: AnnData,
    alpha: float = 1,
    n_neighbors: int = 10,
    n_pcs: int = 15,
    use_highly_variable: Optional[bool] = None,
    copy: bool = False,
):
  
    adata = adata.copy() if copy else adata

    sc.pp.pca(adata, 
              n_comps=n_pcs, 
              use_highly_variable=use_highly_variable)
    
    coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors,metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)
    HL = HL_Loader(coord,nbrs) 
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists
    
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X
    adata.X = csr_matrix(X_rec)
    
    del adata.obsm['X_pca']
    
    return adata, HL

 

def spatial_rec_multi(
    adata: AnnData,
    alpha: float = 1,
    n_pcs: int = 15,
    k_a: int = 6,
    k_e: int = 2,
    use_highly_variable: Optional[bool] = None,
    copy: bool = False,
) -> Tuple[AnnData, object]:
  
    adata = adata.copy() if copy else adata
    
    sc.pp.pca(adata, 
              n_comps=n_pcs, 
              use_highly_variable=use_highly_variable)
    
    coord, nbrs = Neiber(adata, k_a, k_e)
    
    HL = HL_Loader(coord,nbrs)  
    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X
    
    adata.X = csr_matrix(X_rec)
    
    del adata.obsm['X_pca']
    
    return adata, HL



import numpy as np
import scipy.sparse as sp
def Neiber(adata, k_intra = 6, k_inter = 2):
    coord = adata.obsm["spatial"] 
    slice_ids = adata.obs["batch_name"].values 
    N = coord.shape[0]
    adj = sp.lil_matrix((N, N)) 
    unique_slices = np.unique(slice_ids) 
    for sid in unique_slices:
        idx = np.where(slice_ids == sid)[0]
        coord_slice = coord[idx]

        neigh = NearestNeighbors(
            n_neighbors=k_intra, 
            metric="euclidean"
        )
        neigh.fit(coord_slice)
        knn = neigh.kneighbors_graph(coord_slice, mode="connectivity")
        knn.setdiag(0) 
        knn.eliminate_zeros()
        adj[np.ix_(idx, idx)] = knn 
    slice_order = list(unique_slices) 
    for i in range(len(slice_order) - 1):
        s1 = slice_order[i]
        s2 = slice_order[i + 1]

        idx1 = np.where(slice_ids == s1)[0]
        idx2 = np.where(slice_ids == s2)[0]

        coord1 = coord[idx1]
        coord2 = coord[idx2]
        neigh = NearestNeighbors(
            n_neighbors=k_inter,
            metric="euclidean"
        )
        neigh.fit(coord2)
        # s1 -> s2
        knn_12 = neigh.kneighbors_graph(coord1, mode="connectivity")
        adj[np.ix_(idx1, idx2)] = knn_12
        # s2 -> s1
        neigh.fit(coord1)
        knn_21 = neigh.kneighbors_graph(coord2, mode="connectivity")
        adj[np.ix_(idx2, idx1)] = knn_21

    nbrs_adj = adj.tocsr()
    return coord, nbrs_adj
