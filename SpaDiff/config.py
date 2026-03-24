from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpaDiffConfig:
    """Hyperparameters matching manuscript 
    """

    data_dim: int = 50
    condition_input_dim: int = 50
    hidden_dim: int = 128
    topology_hidden_dim: int = 64
    topology_dim: int = 64
    time_embedding_dim: int = 64
    condition_embedding_dim: int = 32
    score_depth: int = 4
    dropout: float = 0.1
    topology_projection_dropout: float | None = None

    simplex_orders: tuple[int, ...] = (1, 2)
    propagation_steps: int = 5
    propagation_alpha: float = 0.2
    learnable_propagation: bool = False


    k1: float = 1.0
    k2: float = 0.0
    num_clusters: int = 7
    dec_alpha: float = 1.0
    dec_update_interval: int = 10
    dec_tolerance: float = 1e-4

    # DEC 初始化方法："kmeans" 或 "mclust"
    dec_init_method: str = "kmeans"

    # 仅在 dec_init_method="mclust" 时使用。
    dec_mclust_model_names: str = "EEE"
    dec_mclust_pca_dim: int = 30

    # 同时作为 KMeans 或 mclust 初始化的随机种子。
    random_seed: int = 42

    num_batches: int = 1
    num_modalities: int = 1

    condition_dropout_joint: float = 0.10
    condition_dropout_topology: float = 0.02
    condition_dropout_batch: float = 0.05
    condition_dropout_modality: float = 0.05


    beta_min: float = 0.1
    beta_max: float = 20.0
    num_scales: int = 1000
    sampling_eps: float = 1e-3

    def validate(self) -> None:
        positive = (
            "data_dim",
            "condition_input_dim",
            "hidden_dim",
            "topology_hidden_dim",
            "topology_dim",
            "time_embedding_dim",
            "condition_embedding_dim",
            "score_depth",
            "propagation_steps",
            "num_clusters",
            "dec_update_interval",
            "dec_mclust_pca_dim",
            "num_batches",
            "num_modalities",
            "num_scales",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.simplex_orders or any(order < 1 for order in self.simplex_orders):
            raise ValueError("simplex_orders must contain positive simplex orders")
        if not 0.0 < self.propagation_alpha < 1.0:
            raise ValueError("propagation_alpha must lie in (0, 1)")
        for name in (
            "dropout",
            "condition_dropout_joint",
            "condition_dropout_topology",
            "condition_dropout_batch",
            "condition_dropout_modality",
        ):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if self.topology_projection_dropout is not None and not (
            0.0 <= self.topology_projection_dropout < 1.0
        ):
            raise ValueError("topology_projection_dropout must lie in [0, 1)")
        if self.k1 < 0.0 or self.k2 < 0.0:
            raise ValueError("k1 and k2 must be non-negative")
        if self.k1 == 0.0 and self.k2 == 0.0:
            raise ValueError("at least one of k1 and k2 must be positive")
        if not isinstance(self.dec_init_method, str):
            raise ValueError(
                "dec_init_method must be a string"
            )

        if self.dec_init_method.lower() not in {
            "kmeans",
            "mclust",
        }:
            raise ValueError(
                "dec_init_method must be either "
                "'kmeans' or 'mclust'"
            )

        if not isinstance(
            self.dec_mclust_model_names,
            str,
        ) or not self.dec_mclust_model_names:
            raise ValueError(
                "dec_mclust_model_names must be "
                "a non-empty string"
            )
        if self.dec_alpha <= 0.0:
            raise ValueError("dec_alpha must be positive")
        if self.dec_tolerance < 0.0:
            raise ValueError("dec_tolerance must be non-negative")
        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError("require 0 < beta_min < beta_max")
        if not 0.0 < self.sampling_eps < 1.0:
            raise ValueError("sampling_eps must lie in (0, 1)")
