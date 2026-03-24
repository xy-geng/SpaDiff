"""Higher-order encoders and preserved autoencoder components."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def operator_matmul(operator: Tensor, x: Tensor) -> Tensor:
    return torch.sparse.mm(operator, x) if operator.layout != torch.strided else operator @ x


def _get_operator(operators, order: int) -> Tensor:
    if isinstance(operators, Mapping):
        return operators[order]
    # Corresponds to original SpaDiff_code/model.py: HiGCN.forward, where
    # HL[1] is the edge operator and HL[2] is the triangle operator.
    return operators[order]


class PolynomialPropagation(nn.Module):
    """Manuscript Eq. (9)-(10), with an optional learnable legacy mode."""

    def __init__(self, steps: int, alpha: float, learnable: bool = False):
        super().__init__()
        coefficients = [alpha * (1.0 - alpha) ** k for k in range(steps)]
        coefficients.append((1.0 - alpha) ** steps)
        values = torch.tensor(coefficients, dtype=torch.float32)
        self.steps = steps
        self.learnable = learnable
        if learnable:
            # Corresponds to original HiGCN_prop.fW, but initialized so the
            # endpoint follows manuscript Eq. (10), not alpha*(1-alpha)^L.
            self.logits = nn.Parameter(values.clamp_min(1e-12).log())
        else:
            self.register_buffer("coefficients", values)

    def forward(self, x: Tensor, operator: Tensor) -> Tensor:
        weights = F.softmax(self.logits, dim=0) if self.learnable else self.coefficients
        propagated = x
        result = weights[0] * x
        for step in range(1, self.steps + 1):
            propagated = operator_matmul(operator, propagated)
            result = result + weights[step] * propagated
        return result


class TopologyEncoder(nn.Module):
    """Multi-channel edge/triangle encoder corresponding to Eq. (8)-(11)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        orders: Sequence[int] = (1, 2),
        steps: int = 5,
        alpha: float = 0.2,
        dropout: float = 0.1,
        projection_dropout: float | None = None,
        learnable_propagation: bool = False,
    ):
        super().__init__()
        self.orders = tuple(orders)
        self.dropout = dropout
        self.projection_dropout = (
            dropout if projection_dropout is None else projection_dropout
        )
        self.projections = nn.ModuleDict(
            {str(order): nn.Linear(input_dim, hidden_dim) for order in self.orders}
        )
        self.propagations = nn.ModuleDict(
            {
                str(order): PolynomialPropagation(steps, alpha, learnable_propagation)
                for order in self.orders
            }
        )
        # Corresponds to original HiGCN.lin_out and manuscript Eq. (11).
        self.fusion = nn.Linear(len(self.orders) * hidden_dim, output_dim)

    def forward(self, features: Tensor, operators) -> Tensor:
        dropped = F.dropout(features, self.dropout, training=self.training)
        channels = []
        for order in self.orders:
            hidden = F.silu(self.projections[str(order)](dropped))
            hidden = F.dropout(
                hidden, self.projection_dropout, training=self.training
            )
            channels.append(
                self.propagations[str(order)](hidden, _get_operator(operators, order))
            )
        return self.fusion(torch.cat(channels, dim=-1))


class OriginalHiGCNPropagation(nn.Module):
    """Exact propagation parameterization used by ``SpaDiff_code/model.py``.

    In particular, the original implementation applies softmax to the raw
    geometric coefficients, rather than to their logarithms.  Keeping that
    behavior is necessary for the k2=0 reproduction endpoint.
    """

    def __init__(self, steps: int, alpha: float):
        super().__init__()
        self.steps = steps
        self.weights = nn.Parameter(torch.empty(steps + 1))
        with torch.no_grad():
            for step in range(steps + 1):
                self.weights[step] = alpha * (1.0 - alpha) ** step

    def forward(self, x: Tensor, operator: Tensor) -> Tensor:
        weights = F.softmax(self.weights, dim=0)
        result = weights[0] * x
        propagated = x
        for step in range(self.steps):
            propagated = operator_matmul(operator, propagated)
            result = result + weights[step + 1] * propagated
        return result


class OriginalHiGCNEncoder(nn.Module):
    """HiGCN path matching the original single-modality DLPFC model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        orders: Sequence[int] = (1, 2),
        steps: int = 5,
        alpha: float = 0.2,
        dropout: float = 0.5,
        projection_dropout: float = 0.2,
    ):
        super().__init__()
        self.orders = tuple(orders)
        self.dropout = dropout
        self.projection_dropout = projection_dropout
        self.projections = nn.ModuleDict()
        self.propagations = nn.ModuleDict()
        # Preserve the original construction order: Linear, propagation,
        # Linear, propagation, ..., then the output Linear.
        for order in self.orders:
            self.projections[str(order)] = nn.Linear(input_dim, hidden_dim)
            self.propagations[str(order)] = OriginalHiGCNPropagation(steps, alpha)
        self.fusion = nn.Linear(len(self.orders) * hidden_dim, output_dim)

    def forward(self, features: Tensor, operators) -> Tensor:
        features = F.dropout(features, self.dropout, training=self.training)
        channels = []
        for order in self.orders:
            hidden = self.projections[str(order)](features)
            if self.projection_dropout > 0.0:
                hidden = F.dropout(
                    hidden, self.projection_dropout, training=self.training
                )
            channels.append(
                self.propagations[str(order)](
                    hidden, _get_operator(operators, order)
                )
            )
        return self.fusion(torch.cat(channels, dim=-1))


class OriginalDECObjective(nn.Module):
    """Corrected DEC objective with selectable KMeans/mclust initialization."""

    def __init__(
        self,
        num_clusters: int,
        embedding_dim: int,
        alpha: float = 1.0,
        init_method: str = "kmeans",
        mclust_model_names: str = "EEE",
        mclust_pca_dim: int = 30,
    ):
        super().__init__()

        init_method = init_method.lower()

        if init_method not in {"kmeans", "mclust"}:
            raise ValueError(
                "init_method must be either 'kmeans' or 'mclust', "
                f"got {init_method!r}"
            )
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if mclust_pca_dim <= 0:
            raise ValueError("mclust_pca_dim must be positive")
        if not mclust_model_names:
            raise ValueError("mclust_model_names cannot be empty")

        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.alpha = alpha

        self.init_method = init_method
        self.mclust_model_names = mclust_model_names
        self.mclust_pca_dim = mclust_pca_dim

        # 修正原始 DEC 的缺陷：
        # centers 在创建优化器前已经是 Parameter，因此会被优化器更新。
        self.centers = nn.Parameter(
            torch.empty(num_clusters, embedding_dim)
        )
        nn.init.xavier_uniform_(self.centers)

        self.initialized = False

    @torch.no_grad()
    def initialize(
        self,
        embedding: Tensor,
        random_state: int = 42,
    ) -> Tensor:
        """使用配置指定的 KMeans 或 mclust 初始化 DEC 中心。"""

        array = embedding.detach().cpu().numpy()

        if self.init_method == "kmeans":
            labels, centers = self._initialize_kmeans(
                array,
                random_state=random_state,
            )

        elif self.init_method == "mclust":
            labels, centers = self._initialize_mclust(
                array,
                random_state=random_state,
            )

        else:
            raise RuntimeError(
                f"unsupported DEC initialization method: "
                f"{self.init_method!r}"
            )

        if centers.shape != (
            self.num_clusters,
            self.embedding_dim,
        ):
            raise RuntimeError(
                "invalid DEC center shape after initialization: "
                f"expected "
                f"{(self.num_clusters, self.embedding_dim)}, "
                f"got {centers.shape}"
            )

        self.centers.copy_(
            torch.as_tensor(
                centers,
                device=embedding.device,
                dtype=embedding.dtype,
            )
        )

        self.initialized = True

        return torch.as_tensor(
            labels,
            device=embedding.device,
            dtype=torch.long,
        )

    def _initialize_kmeans(
        self,
        array: np.ndarray,
        *,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用 KMeans 初始化类别和聚类中心。"""

        from sklearn.cluster import KMeans

        estimator = KMeans(
            n_clusters=self.num_clusters,
            n_init=10,
            random_state=random_state,
        )

        labels = estimator.fit_predict(array).astype(
            np.int64,
            copy=False,
        )

        centers = np.asarray(
            estimator.cluster_centers_,
            dtype=array.dtype,
        )

        return labels, centers

    def _initialize_mclust(
        self,
        array: np.ndarray,
        *,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用 R/mclust 初始化类别，再在原始嵌入空间计算中心。"""

        from sklearn.decomposition import PCA

        components = min(
            self.mclust_pca_dim,
            array.shape[0] - 1,
            array.shape[1],
        )

        if components <= 0:
            raise ValueError(
                "mclust initialization requires at least two "
                "samples and one embedding dimension"
            )

        embedding_pca = PCA(
            n_components=components,
            random_state=random_state,
        ).fit_transform(array)


        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        robjects.r.library("mclust")
        rpy2.robjects.numpy2ri.activate()
        robjects.r["set.seed"](random_state)

        result = robjects.r["Mclust"](
            rpy2.robjects.numpy2ri.numpy2rpy(
                embedding_pca
            ),
            self.num_clusters,
            self.mclust_model_names,
        )

        # mclust 分类标签从 1 开始，转换成 0-based。
        labels = (
            np.asarray(
                result[-2],
                dtype=np.int64,
            ).reshape(-1)
            - 1
        )

        expected_labels = np.arange(
            self.num_clusters,
            dtype=np.int64,
        )
        actual_labels = np.unique(labels)

        if not np.array_equal(
            actual_labels,
            expected_labels,
        ):
            raise RuntimeError(
                "mclust did not produce every requested cluster: "
                f"expected={expected_labels.tolist()}, "
                f"actual={actual_labels.tolist()}"
            )

        # mclust 在 PCA 空间中进行分类，但 DEC 在原始拓扑嵌入
        # 空间中计算距离，因此中心必须在原始 array 中重新计算。
        centers = np.stack(
            [
                array[labels == cluster_id].mean(axis=0)
                for cluster_id in range(self.num_clusters)
            ],
            axis=0,
        ).astype(array.dtype, copy=False)

        return labels, centers

    def soft_assign(
        self,
        embedding: Tensor,
    ) -> Tensor:
        """计算标准 DEC Student-t 软聚类概率 q。"""

        if not self.initialized:
            raise RuntimeError(
                "DEC cluster centers have not been initialized"
            )

        if embedding.ndim != 2:
            raise ValueError(
                "embedding must be a two-dimensional tensor"
            )

        if embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"embedding width must be {self.embedding_dim}, "
                f"got {embedding.shape[1]}"
            )

        squared_distance = (
            embedding.unsqueeze(1)
            - self.centers.unsqueeze(0)
        ).square().sum(dim=2)

        # 标准 DEC Student-t 核：
        # q_ij ∝ (1 + ||z_i - μ_j||² / alpha)
        #        ^ (-(alpha + 1) / 2)
        numerator = (
            1.0 + squared_distance / self.alpha
        ).pow(-(self.alpha + 1.0) / 2.0)

        return numerator / numerator.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

    @staticmethod
    def target_distribution(
        q: Tensor,
    ) -> Tensor:
        """根据当前软聚类概率 q 计算 DEC 目标分布 p。"""

        cluster_frequency = q.sum(
            dim=0,
            keepdim=True,
        ).clamp_min(1e-12)

        weight = q.square() / cluster_frequency

        return weight / weight.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

    @staticmethod
    def loss(
        p: Tensor,
        q: Tensor,
    ) -> Tensor:
        """计算数值稳定的 KL(P || Q)。"""

        p_safe = p.clamp_min(1e-12)
        q_safe = q.clamp_min(1e-12)

        return torch.sum(
            p_safe * (
                p_safe.log() - q_safe.log()
            ),
            dim=1,
        ).mean()


