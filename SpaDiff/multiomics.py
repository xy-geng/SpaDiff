from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import SpaDiffConfig
from .spadiff import SpaDiff


def standardize_columns(values, eps: float = 1e-6) -> np.ndarray:
    """Standardize latent dimensions across spots without row-wise distortion."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [spots, features]")
    mean = array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, ddof=0, keepdims=True)
    standardized = np.divide(
        array - mean,
        scale,
        out=np.zeros_like(array),
        where=scale > eps,
    )
    return np.asarray(standardized, dtype=np.float32)


def align_paired_anndata(
    adata_rna,
    adata_atac,
    *,
    copy: bool = True,
    check_spatial: bool = True,
    spatial_key: str = "spatial",
    spatial_atol: float = 1e-5,
):
    """Align paired modalities by spot name and verify their coordinates."""

    rna_names = adata_rna.obs_names
    atac_names = adata_atac.obs_names
    if not rna_names.is_unique or not atac_names.is_unique:
        raise ValueError("paired RNA/ATAC spot names must be unique")

    missing_in_atac = rna_names.difference(atac_names)
    missing_in_rna = atac_names.difference(rna_names)
    if len(missing_in_atac) or len(missing_in_rna):
        raise ValueError(
            "RNA and ATAC do not contain the same spots: "
            f"missing_in_atac={len(missing_in_atac)}, "
            f"missing_in_rna={len(missing_in_rna)}"
        )

    rna = adata_rna.copy() if copy else adata_rna
    atac = adata_atac[rna_names].copy()
    if not np.array_equal(np.asarray(rna.obs_names), np.asarray(atac.obs_names)):
        raise RuntimeError("failed to align ATAC spots to RNA order")

    if check_spatial:
        if spatial_key not in rna.obsm or spatial_key not in atac.obsm:
            raise KeyError(f"both modalities must contain obsm[{spatial_key!r}]")
        rna_spatial = np.asarray(rna.obsm[spatial_key], dtype=np.float64)
        atac_spatial = np.asarray(atac.obsm[spatial_key], dtype=np.float64)
        if rna_spatial.shape != atac_spatial.shape or not np.allclose(
            rna_spatial, atac_spatial, atol=spatial_atol, rtol=0.0
        ):
            maximum = (
                float(np.max(np.abs(rna_spatial - atac_spatial)))
                if rna_spatial.shape == atac_spatial.shape
                else float("inf")
            )
            raise ValueError(
                "paired RNA/ATAC spatial coordinates are inconsistent; "
                f"maximum absolute difference={maximum:.6g}"
            )
    return rna, atac


def robust_atac_lsi(
    adata,
    *,
    n_components: int = 30,
    min_cells: int = 10,
    scale_factor: float = 1e4,
    drop_depth_component: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    """Compute TF-IDF/LSI using U*Sigma and column-wise standardization.

    Peaks observed in fewer than ``min_cells`` spots are removed.  The component
    most correlated with log library size is treated as the depth component and
    removed, mirroring the common practice of excluding depth-dominated LSI
    dimensions while avoiding the assumption that it must always be component 1.
    """

    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if min_cells <= 0:
        raise ValueError("min_cells must be positive")

    matrix = (
        adata.X.tocsr().astype(np.float64, copy=True)
        if sp.issparse(adata.X)
        else sp.csr_matrix(np.asarray(adata.X, dtype=np.float64))
    )
    if matrix.shape[0] < 3:
        raise ValueError("at least three spots are required for LSI")
    if matrix.data.size and np.min(matrix.data) < 0:
        raise ValueError("ATAC counts must be non-negative")

    detected = np.asarray(matrix.getnnz(axis=0)).ravel()
    selected = detected >= min_cells
    if selected.sum() < 3:
        raise ValueError("too few ATAC peaks remain after min_cells filtering")
    matrix = matrix[:, selected]

    library_size = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(library_size <= 0):
        count = int(np.sum(library_size <= 0))
        raise ValueError(f"{count} ATAC spots have zero counts after peak filtering")

    term_frequency = sp.diags(1.0 / library_size) @ matrix
    document_frequency = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.float64)
    inverse_document_frequency = np.log1p(
        matrix.shape[0] / np.maximum(document_frequency, 1.0)
    )
    tfidf = (term_frequency @ sp.diags(inverse_document_frequency)).tocsr()
    tfidf.data = np.log1p(tfidf.data * scale_factor)

    maximum_components = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    requested = n_components + int(drop_depth_component)
    fitted_components = min(requested, maximum_components)
    if fitted_components < 2:
        raise ValueError("ATAC matrix is too small for the requested LSI")

    decomposition = TruncatedSVD(
        n_components=fitted_components,
        algorithm="randomized",
        random_state=random_state,
    )
    values = decomposition.fit_transform(tfidf)

    log_depth = np.log1p(library_size)
    correlations = np.zeros(values.shape[1], dtype=np.float64)
    if np.std(log_depth) > 0:
        for index in range(values.shape[1]):
            if np.std(values[:, index]) > 0:
                correlations[index] = np.corrcoef(values[:, index], log_depth)[0, 1]
    correlations = np.nan_to_num(correlations)

    depth_component: Optional[int] = None
    kept = list(range(values.shape[1]))
    if drop_depth_component and len(kept) > 1:
        depth_component = int(np.argmax(np.abs(correlations)))
        kept.remove(depth_component)
    kept = kept[:n_components]
    values = standardize_columns(values[:, kept])

    diagnostics = {
        "n_input_peaks": int(adata.n_vars),
        "n_selected_peaks": int(selected.sum()),
        "n_components": int(values.shape[1]),
        "depth_component": depth_component,
        "depth_correlations": correlations,
        "kept_components": np.asarray(kept, dtype=np.int64),
        "explained_variance_ratio": decomposition.explained_variance_ratio_,
    }
    return values, diagnostics


def build_spatial_adjacency(
    coordinates,
    *,
    n_neighbors: int = 6,
    metric: str = "euclidean",
):
    """Build a symmetric kNN graph while explicitly excluding each spot itself."""

    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    values = np.asarray(coordinates, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("coordinates must have shape [spots, dimensions]")
    k = min(int(n_neighbors), values.shape[0] - 1)
    if k <= 0:
        raise ValueError("n_neighbors must be positive")

    estimator = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(values)
    _, indices = estimator.kneighbors(values)
    rows = []
    columns = []
    for row, neighbors in enumerate(indices):
        selected = [int(value) for value in neighbors if int(value) != row][:k]
        rows.extend([row] * len(selected))
        columns.extend(selected)
    adjacency = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(values.shape[0], values.shape[0]),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


class ModalityAdapter(nn.Module):
    """Map one modality-specific latent coordinate system to a shared input space."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.skip = (
            nn.Identity()
            if input_dim == output_dim
            else nn.Linear(input_dim, output_dim, bias=False)
        )
        self.output_norm = nn.LayerNorm(output_dim)
        self.residual_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(self, values: Tensor) -> Tensor:
        gate = torch.sigmoid(self.residual_gate)
        return self.output_norm(self.network(values) + gate * self.skip(values))


def paired_vicreg_loss(
    first: Tensor,
    second: Tensor,
    *,
    variance_weight: float = 0.1,
    covariance_weight: float = 0.01,
    eps: float = 1e-4,
) -> dict[str, Tensor]:
    """Align paired spots while variance/covariance terms prevent collapse."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("paired embeddings must have identical [spots, features] shape")
    if first.shape[0] < 2:
        raise ValueError("paired alignment requires at least two spots")

    first_unit = F.normalize(first, dim=-1)
    second_unit = F.normalize(second, dim=-1)
    invariance = 1.0 - (first_unit * second_unit).sum(dim=-1).mean()

    first_std = torch.sqrt(first.var(dim=0, unbiased=False) + eps)
    second_std = torch.sqrt(second.var(dim=0, unbiased=False) + eps)
    variance = 0.5 * (
        F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
    )

    def covariance_penalty(values: Tensor) -> Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
        diagonal = torch.diag(torch.diagonal(covariance))
        return (covariance - diagonal).square().sum() / values.shape[1]

    covariance = 0.5 * (
        covariance_penalty(first) + covariance_penalty(second)
    )
    total = invariance + variance_weight * variance + covariance_weight * covariance
    return {
        "loss": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


class PairedMultiOmicsSpaDiff(nn.Module):
    """Non-invasive paired RNA/ATAC extension of the existing SpaDiff model.

    The expected training layout is modality-major and paired by row:
    ``target=[RNA, ATAC]`` and ``condition=[ATAC, RNA]``.  The core SpaDiff loss
    remains intact; an explicit paired loss is added between topology embeddings
    produced from the two measurements of each spot.
    """

    def __init__(
        self,
        config: SpaDiffConfig,
        *,
        raw_input_dim: int,
        adapter_hidden_dim: int = 64,
        adapter_dropout: float = 0.05,
        pair_weight: float = 0.25,
        pair_variance_weight: float = 0.1,
        pair_covariance_weight: float = 0.01,
    ):
        super().__init__()
        if config.num_modalities != 2:
            raise ValueError("PairedMultiOmicsSpaDiff currently requires two modalities")
        if raw_input_dim <= 0 or pair_weight < 0:
            raise ValueError("raw_input_dim must be positive and pair_weight non-negative")

        self.core = SpaDiff(config)
        self.raw_input_dim = int(raw_input_dim)
        self.pair_weight = float(pair_weight)
        self.pair_variance_weight = float(pair_variance_weight)
        self.pair_covariance_weight = float(pair_covariance_weight)
        self.adapters = nn.ModuleList(
            ModalityAdapter(
                self.raw_input_dim,
                adapter_hidden_dim,
                config.condition_input_dim,
                adapter_dropout,
            )
            for _ in range(2)
        )

    @property
    def config(self) -> SpaDiffConfig:
        return self.core.config

    def adapt_condition(
        self, features: Tensor, source_modality_ids: Tensor
    ) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.raw_input_dim:
            raise ValueError(
                f"raw condition features must have shape [N, {self.raw_input_dim}]"
            )
        if source_modality_ids.shape != (features.shape[0],):
            raise ValueError("source_modality_ids must have shape [N]")
        output = features.new_empty(
            (features.shape[0], self.config.condition_input_dim)
        )
        for modality, adapter in enumerate(self.adapters):
            mask = source_modality_ids == modality
            if mask.any():
                output[mask] = adapter(features[mask])
        if source_modality_ids.numel() and (
            source_modality_ids.min().item() < 0
            or source_modality_ids.max().item() > 1
        ):
            raise ValueError("source modality id must be 0 (RNA) or 1 (ATAC)")
        return output

    def loss(
        self,
        target_features: Tensor,
        operators,
        batch_ids: Tensor,
        modality_ids: Tensor,
        *,
        condition_features: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if condition_features is None:
            raise ValueError("paired training requires explicit condition_features")
        source_modality_ids = 1 - modality_ids.long()
        adapted = self.adapt_condition(condition_features, source_modality_ids)
        output = self.core.loss(
            target_features,
            operators,
            batch_ids,
            modality_ids,
            condition_features=adapted,
        )

        topology = output["topology"]
        from_atac = topology[modality_ids == 0]
        from_rna = topology[modality_ids == 1]
        if from_atac.shape != from_rna.shape:
            raise ValueError("paired training requires equal RNA and ATAC spot counts")
        paired = paired_vicreg_loss(
            from_rna,
            from_atac,
            variance_weight=self.pair_variance_weight,
            covariance_weight=self.pair_covariance_weight,
        )

        result = dict(output)
        result["core_loss"] = output["loss"]
        result["paired_alignment_loss"] = paired["loss"]
        result["paired_invariance_loss"] = paired["invariance"]
        result["paired_variance_loss"] = paired["variance"]
        result["paired_covariance_loss"] = paired["covariance"]
        result["weighted_paired_alignment_loss"] = self.pair_weight * paired["loss"]
        result["loss"] = output["loss"] + result["weighted_paired_alignment_loss"]
        return result

    @torch.no_grad()
    def encode_modalities(
        self,
        rna_features: Tensor,
        atac_features: Tensor,
        operators,
    ) -> tuple[Tensor, Tensor]:
        if rna_features.shape != atac_features.shape:
            raise ValueError("paired RNA and ATAC latent matrices must have equal shape")
        rna_ids = torch.zeros(
            rna_features.shape[0], dtype=torch.long, device=rna_features.device
        )
        atac_ids = torch.ones(
            atac_features.shape[0], dtype=torch.long, device=atac_features.device
        )
        adapted_rna = self.adapt_condition(rna_features, rna_ids)
        adapted_atac = self.adapt_condition(atac_features, atac_ids)
        return (
            self.core.encode_condition(adapted_rna, operators),
            self.core.encode_condition(adapted_atac, operators),
        )

    @torch.no_grad()
    def generate(
        self,
        condition_features: Tensor,
        operators,
        target_batch_ids: Tensor,
        target_modality_ids: Tensor,
        *,
        source_modality_ids: Tensor,
        guidance_scale: float = 1.0,
        guidance_target: str = "all",
        ode_steps: Optional[int] = None,
    ) -> Tensor:
        adapted = self.adapt_condition(condition_features, source_modality_ids)
        return self.core.generate(
            adapted,
            operators,
            target_batch_ids,
            target_modality_ids,
            guidance_scale=guidance_scale,
            guidance_target=guidance_target,
            ode_steps=ode_steps,
        )


def consensus_disagreement_embedding(
    rna_embedding,
    atac_embedding,
    *,
    n_components: int = 20,
    disagreement_weight: float = 0.25,
    random_state: int = 42,
) -> np.ndarray:
    """Preserve shared signal and residual modality disagreement before PCA."""

    from sklearn.decomposition import PCA

    rna = standardize_columns(rna_embedding)
    atac = standardize_columns(atac_embedding)
    if rna.shape != atac.shape:
        raise ValueError("RNA and ATAC embeddings must have identical shape")
    consensus = 0.5 * (rna + atac)
    disagreement = np.abs(rna - atac)
    combined = np.concatenate(
        (consensus, float(disagreement_weight) * disagreement), axis=1
    )
    components = min(n_components, combined.shape[0] - 1, combined.shape[1])
    if components <= 0:
        raise ValueError("not enough observations for fused PCA")
    fused = PCA(n_components=components, random_state=random_state).fit_transform(
        combined
    )
    return standardize_columns(fused)


def _row_normalize(matrix):
    import scipy.sparse as sp

    matrix = matrix.tocsr().astype(np.float64)
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    inverse = np.divide(
        1.0,
        row_sum,
        out=np.zeros_like(row_sum, dtype=np.float64),
        where=row_sum > 0,
    )
    return (sp.diags(inverse) @ matrix).tocsr()


def _latent_for_neighbors(values, n_components: int, random_state: int):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    standardized = standardize_columns(values)
    components = min(n_components, standardized.shape[0] - 1, standardized.shape[1])
    if components < standardized.shape[1]:
        standardized = PCA(
            n_components=components, random_state=random_state
        ).fit_transform(standardized)
    return normalize(standardized, norm="l2")


def _adaptive_knn_graph(values: np.ndarray, n_neighbors: int):
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    n = values.shape[0]
    k = min(int(n_neighbors), n - 1)
    if k <= 0:
        raise ValueError("n_neighbors must be positive")
    estimator = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(values)
    distances, indices = estimator.kneighbors(values)
    rows = []
    columns = []
    weights = []
    for row in range(n):
        keep = indices[row] != row
        local_indices = indices[row][keep][:k]
        local_distances = distances[row][keep][:k]
        sigma = max(float(local_distances[-1]), 1e-6)
        local_weights = np.exp(-np.square(local_distances / sigma))
        rows.extend([row] * len(local_indices))
        columns.extend(local_indices.tolist())
        weights.extend(local_weights.tolist())
    graph = sp.csr_matrix(
        (np.asarray(weights), (rows, columns)), shape=(n, n), dtype=np.float64
    )
    return _row_normalize(graph)


def build_adaptive_multimodal_connectivity(
    rna_embedding,
    atac_embedding,
    spatial_adjacency,
    *,
    n_neighbors: int = 15,
    neighbor_components: int = 20,
    spatial_weight: float = 0.30,
    modality_temperature: float = 1.0,
    random_state: int = 42,
):
    """Build a WNN-inspired, spot-adaptive RNA/ATAC plus spatial graph.

    Each modality's neighbors predict the other modality.  Lower cross-modal
    prediction error gives that modality more weight for the spot.  The weighted
    molecular graph is then mixed with the physical spatial graph.
    """

    import scipy.sparse as sp

    if not 0.0 <= spatial_weight < 1.0:
        raise ValueError("spatial_weight must lie in [0, 1)")
    if modality_temperature <= 0:
        raise ValueError("modality_temperature must be positive")

    rna = _latent_for_neighbors(rna_embedding, neighbor_components, random_state)
    atac = _latent_for_neighbors(atac_embedding, neighbor_components, random_state)
    if rna.shape[0] != atac.shape[0]:
        raise ValueError("RNA and ATAC embeddings must contain the same spots")

    rna_graph = _adaptive_knn_graph(rna, n_neighbors)
    atac_graph = _adaptive_knn_graph(atac, n_neighbors)
    rna_cross_error = np.mean(np.square(rna_graph @ atac - atac), axis=1)
    atac_cross_error = np.mean(np.square(atac_graph @ rna - rna), axis=1)

    errors = np.column_stack((rna_cross_error, atac_cross_error))
    scale = np.median(errors, axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    logits = -(errors / scale) / modality_temperature
    logits -= logits.max(axis=1, keepdims=True)
    modality_weights = np.exp(logits)
    modality_weights /= modality_weights.sum(axis=1, keepdims=True)

    molecular = (
        sp.diags(modality_weights[:, 0]) @ rna_graph
        + sp.diags(modality_weights[:, 1]) @ atac_graph
    )
    molecular = 0.5 * (molecular + molecular.T)
    molecular = _row_normalize(molecular)

    spatial = _row_normalize(spatial_adjacency)
    spatial = 0.5 * (spatial + spatial.T)
    spatial = _row_normalize(spatial)
    fused = (1.0 - spatial_weight) * molecular + spatial_weight * spatial
    fused = (0.5 * (fused + fused.T)).tocsr()
    fused.setdiag(0)
    fused.eliminate_zeros()
    return fused, {
        "rna_weight": modality_weights[:, 0].astype(np.float32),
        "atac_weight": modality_weights[:, 1].astype(np.float32),
        "rna_cross_prediction_error": rna_cross_error.astype(np.float32),
        "atac_cross_prediction_error": atac_cross_error.astype(np.float32),
    }


def cluster_leiden_to_target(
    adata,
    adjacency,
    *,
    target_n_clusters: int,
    key_added: str = "spadiff_leiden",
    resolution_bounds: tuple[float, float] = (0.05, 3.0),
    max_iterations: int = 25,
    random_state: int = 42,
    verbose: bool = True,
):
    """Run Leiden on a supplied fused graph and retain the closest cluster count."""

    import scanpy as sc

    if target_n_clusters <= 0:
        raise ValueError("target_n_clusters must be positive")
    low, high = (float(value) for value in resolution_bounds)
    if not 0.0 < low < high:
        raise ValueError("resolution_bounds must satisfy 0 < low < high")

    history = []
    best = None

    def evaluate(resolution: float):
        nonlocal best
        sc.tl.leiden(
            adata,
            adjacency=adjacency,
            resolution=float(resolution),
            key_added=key_added,
            random_state=random_state,
            directed=False,
        )
        count = int(adata.obs[key_added].nunique())
        labels = adata.obs[key_added].copy()
        rank = (
            abs(count - target_n_clusters),
            count < target_n_clusters,
            float(resolution),
        )
        if best is None or rank < best[0]:
            best = (rank, float(resolution), count, labels)
        history.append((float(resolution), count))
        if verbose:
            print(f"resolution={resolution:.6g} -> {count} clusters")
        return count

    evaluate(low)
    evaluate(high)
    for _ in range(max_iterations):
        if best[0][0] == 0:
            break
        midpoint = 0.5 * (low + high)
        if np.isclose(midpoint, low) or np.isclose(midpoint, high):
            break
        count = evaluate(midpoint)
        if count < target_n_clusters:
            low = midpoint
        else:
            high = midpoint

    _, resolution, count, labels = best
    adata.obs[key_added] = labels
    adata.uns[f"{key_added}_resolution_search"] = {
        "target_n_clusters": int(target_n_clusters),
        "selected_resolution": float(resolution),
        "selected_n_clusters": int(count),
        "resolutions": np.asarray([item[0] for item in history]),
        "n_clusters": np.asarray([item[1] for item in history]),
    }
    if verbose:
        print(f"selected resolution={resolution:.6g}: {count} clusters")
    return adata
