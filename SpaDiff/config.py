from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpaDiffConfig:
    """Configuration for the paper-aligned SpaDiff implementation.

    ``data_dim`` is the PCA/LSI target width. ``condition_input_dim`` is the
    width of the feature matrix used to construct the fused simplicial
    condition ``H``.  The batch variable from the paper is represented by the
    joint ``(batch_id, modality_id)`` technical condition so that the same
    objective also applies to multi-omics experiments.

    ``dsm_weighting`` makes the paper's otherwise implicit time weighting
    explicit:

    - ``"score"``: literal unweighted score error in Eq. (18)/(19).
    - ``"variance"``: variance-weighted score error, exactly epsilon-MSE.
    - ``"likelihood"``: diffusion-squared likelihood weighting.

    """

    data_dim: int = 50
    condition_input_dim: int = 50
    hidden_dim: int = 128
    topology_hidden_dim: int = 128
    topology_dim: int = 64
    technical_hidden_dim: int = 64
    time_embedding_dim: int = 64
    condition_embedding_dim: int = 32
    score_depth: int = 4
    dropout: float = 0.1
    topology_projection_dropout: float | None = 0.0

    simplex_orders: tuple[int, ...] = (1, 2)
    propagation_steps: int = 5
    propagation_alpha: float = 0.4
    learnable_propagation: bool = False
    topology_residual: bool = True
    topology_output_normalization: str = "feature"

    num_batches: int = 1
    num_modalities: int = 1


    use_topology_condition: bool = True
    use_batch_condition: bool = True


    dsm_weighting: str = "variance"
    dsm_weight: float = 1.0
    batch_alignment_weight: float = 0.5


    batch_posterior_weight: float = 1.0
    prior_kl_weight: float = 1.0
    adversarial_strength: float = 1.0
    batch_balanced_loss: bool = True


    condition_dropout_joint: float = 0.0
    condition_dropout_topology: float = 0.0
    condition_dropout_batch: float = 0.0
    condition_dropout_modality: float = 0.0


    beta_min: float = 0.1
    beta_max: float = 20.0
    num_scales: int = 1000
    training_eps: float = 1e-3
    sampling_eps: float = 1e-3

    @property
    def num_technical_conditions(self) -> int:
        return self.num_batches * self.num_modalities

    def validate(self) -> None:
        positive = (
            "data_dim",
            "condition_input_dim",
            "hidden_dim",
            "topology_hidden_dim",
            "topology_dim",
            "technical_hidden_dim",
            "time_embedding_dim",
            "condition_embedding_dim",
            "score_depth",
            "propagation_steps",
            "num_batches",
            "num_modalities",
            "num_scales",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

        if not self.simplex_orders or any(order < 0 for order in self.simplex_orders):
            raise ValueError("simplex_orders must contain non-negative simplex orders")
        if not 0.0 < self.propagation_alpha < 1.0:
            raise ValueError("propagation_alpha must lie in (0, 1)")
        if self.topology_output_normalization.lower() not in {
            "none",
            "feature",
            "layer",
        }:
            raise ValueError(
                "topology_output_normalization must be 'none', 'feature' or 'layer'"
            )

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

        if self.dsm_weighting.lower() not in {"score", "variance", "likelihood"}:
            raise ValueError(
                "dsm_weighting must be 'score', 'variance' or 'likelihood'"
            )
        for name in (
            "dsm_weight",
            "batch_alignment_weight",
            "batch_posterior_weight",
            "prior_kl_weight",
            "adversarial_strength",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if (
            self.dsm_weight == 0.0
            and self.batch_alignment_weight == 0.0
            and self.prior_kl_weight == 0.0
        ):
            raise ValueError("at least one outer loss weight must be positive")

        if not 0.0 < self.beta_min < self.beta_max:
            raise ValueError("require 0 < beta_min < beta_max")
        for name in ("training_eps", "sampling_eps"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
