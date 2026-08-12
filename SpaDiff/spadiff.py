from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import SpaDiffConfig
from .diffusion import ConditionalScoreNetwork, conditional_dsm_loss
from .model import (
    TechnicalConditionObjective,
    TopologyEncoder,
    empirical_prior_kl,
    technical_condition_ids,
)
from .sampling import probability_flow_sample
from .sde import VPSDE, expand_like


class SpaDiff(nn.Module):
    """Unified score model conditioned on fused topology and technical labels.

    The VP forward perturbation kernel is condition-independent, as in standard
    conditional diffusion.  The topology ``H`` and technical condition ``b``
    parameterize the reverse score.  Batch/modality invariance of ``H`` is
    learned through the SI distribution-ratio term implemented with an
    adversarial technical-condition predictor.
    """

    def __init__(self, config: SpaDiffConfig):
        super().__init__()
        config.validate()
        self.config = config
        projection_dropout = (
            config.dropout
            if config.topology_projection_dropout is None
            else config.topology_projection_dropout
        )
        # 多阶单纯复形拓扑编码器，输出融合条件 H。
        self.topology_encoder = TopologyEncoder(
            input_dim=config.condition_input_dim,
            hidden_dim=config.topology_hidden_dim,
            output_dim=config.topology_dim,
            orders=config.simplex_orders,
            steps=config.propagation_steps,
            alpha=config.propagation_alpha,
            dropout=config.dropout,
            projection_dropout=projection_dropout,
            learnable_propagation=config.learnable_propagation,
            residual=config.topology_residual,
            output_normalization=config.topology_output_normalization,
        )
        # 以 (x_t, t, H, b, modality) 为条件的扩散分数网络。
        self.score_model = ConditionalScoreNetwork(
            data_dim=config.data_dim,
            topology_dim=config.topology_dim,
            hidden_dim=config.hidden_dim,
            time_embedding_dim=config.time_embedding_dim,
            condition_embedding_dim=config.condition_embedding_dim,
            num_batches=config.num_batches,
            num_modalities=config.num_modalities,
            depth=config.score_depth,
            dropout=config.dropout,
        )
        # 批次/模态后验与对抗预测器，实现损失第 2 项。
        self.technical_objective = TechnicalConditionObjective(
            data_dim=config.data_dim,
            topology_dim=config.topology_dim,
            num_conditions=config.num_technical_conditions,
            hidden_dim=config.technical_hidden_dim,
            adversarial_strength=config.adversarial_strength,
        )
        # VP-SDE 前向加噪核及对应的反向生成动力学。
        self.sde = VPSDE(
            beta_min=config.beta_min,
            beta_max=config.beta_max,
            num_scales=config.num_scales,
        )

    def _validate_labels(self, batch_ids: Tensor, modality_ids: Tensor, n: int) -> None:
        if batch_ids.shape != (n,) or modality_ids.shape != (n,):
            raise ValueError("batch_ids and modality_ids must each have shape [N]")
        if batch_ids.numel() and (
            batch_ids.min().item() < 0
            or batch_ids.max().item() >= self.config.num_batches
        ):
            raise ValueError("batch id is outside the configured range")
        if modality_ids.numel() and (
            modality_ids.min().item() < 0
            or modality_ids.max().item() >= self.config.num_modalities
        ):
            raise ValueError("modality id is outside the configured range")

    def encode_condition(self, features: Tensor, operators) -> Tensor:
        if features.ndim != 2:
            raise ValueError("condition features must have shape [N, F]")
        if features.shape[-1] != self.config.condition_input_dim:
            raise ValueError(
                f"condition feature width must be {self.config.condition_input_dim}, "
                f"got {features.shape[-1]}"
            )
        return self.topology_encoder(features, operators)

    def loss(
        self,
        target_features: Tensor,
        operators,
        batch_ids: Tensor,
        modality_ids: Tensor,
        *,
        condition_features: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Evaluate the paper/SI-aligned joint training objective.

        The returned ``loss`` has exactly three paper-level terms:

        ``dsm_weight * DSM``
        ``+ batch_alignment_weight * (L_ratio + batch_posterior_weight * L_q)``
        ``+ prior_kl_weight * KL(q(H|b) || p(H))``.

        ``L_q`` is an auxiliary sub-loss required to identify q_phi(b|x0),
        rather than a fourth manuscript loss term.
        """

        if (
            target_features.ndim != 2
            or target_features.shape[1] != self.config.data_dim
        ):
            raise ValueError(
                f"target_features must have shape [N, {self.config.data_dim}]"
            )
        source = target_features if condition_features is None else condition_features
        if source.shape[0] != target_features.shape[0]:
            raise ValueError(
                "target and condition features must contain the same spots"
            )
        if source.device != target_features.device:
            raise ValueError(
                "target_features and condition_features must share a device"
            )

        batch_ids = batch_ids.to(device=target_features.device, dtype=torch.long)
        modality_ids = modality_ids.to(device=target_features.device, dtype=torch.long)
        self._validate_labels(batch_ids, modality_ids, target_features.shape[0])
        condition_ids = technical_condition_ids(
            batch_ids, modality_ids, self.config.num_modalities
        )
        # 前向模型：从输入特征和多阶空间算子得到融合拓扑表示 H。
        topology = self.encode_condition(source, operators)

        training_scale = 1.0 if self.training else 0.0
        # 损失第 1 项：拓扑和批次条件下的去噪分数匹配损失 L_DSM。
        dsm = conditional_dsm_loss(
            self.score_model,
            self.sde,
            target_features,
            topology,
            batch_ids,
            modality_ids,
            eps=self.config.training_eps,
            weighting=self.config.dsm_weighting,
            loss_group_ids=condition_ids,
            batch_balanced=self.config.batch_balanced_loss,
            joint_dropout=training_scale * self.config.condition_dropout_joint,
            topology_dropout=training_scale * self.config.condition_dropout_topology,
            batch_dropout=training_scale * self.config.condition_dropout_batch,
            modality_dropout=training_scale * self.config.condition_dropout_modality,
            use_topology_condition=self.config.use_topology_condition,
            use_batch_condition=self.config.use_batch_condition,
        )
        # 损失第 2 项：批次分布比值的可计算对抗形式。
        technical = self.technical_objective(
            target_features,
            topology,
            condition_ids,
            balanced=self.config.batch_balanced_loss,
        )
        # 损失第 3 项：KL(q_phi(H|b) || p(H))。
        # 实际训练可省略，因此权重为 0 时连经验矩也不计算。
        prior = (
            empirical_prior_kl(topology, condition_ids)
            if self.config.prior_kl_weight > 0.0
            else topology.sum() * 0.0
        )

        # q_phi(b|x0) 的监督损失属于第 2 项的内部辅助子项。
        batch_loss = (
            technical["alignment_loss"]
            + self.config.batch_posterior_weight * technical["posterior_loss"]
        )

        # 三个显式外层权重分别控制：DSM、批次对齐、潜表示 KL。
        weighted_dsm = self.config.dsm_weight * dsm["loss"]
        weighted_batch = self.config.batch_alignment_weight * batch_loss
        weighted_prior = self.config.prior_kl_weight * prior
        total = weighted_dsm + weighted_batch + weighted_prior
        return {
            "loss": total,
            "dsm_loss": dsm["loss"],
            "batch_loss": batch_loss,
            "batch_alignment_loss": technical["alignment_loss"],
            "batch_posterior_loss": technical["posterior_loss"],
            "prior_kl_loss": prior,
            "weighted_dsm_loss": weighted_dsm,
            "weighted_batch_loss": weighted_batch,
            "weighted_prior_kl_loss": weighted_prior,
            "noise_mse": dsm["noise_mse"],
            "score_mse": dsm["score_mse"],
            "mean_time": dsm["mean_time"],
            "posterior_accuracy": technical["posterior_accuracy"],
            "topology_batch_accuracy": technical["topology_accuracy"],
            "topology": topology,
        }

    @torch.no_grad()
    def generate(
        self,
        condition_features: Tensor,
        operators,
        target_batch_ids: Tensor,
        target_modality_ids: Tensor,
        *,
        guidance_scale: float = 1.0,
        guidance_target: str = "all",
        ode_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate target PCA/LSI features from topology/source conditions."""

        was_training = self.training
        try:
            self.eval()
            topology = self.encode_condition(condition_features, operators)
            target_batch_ids = target_batch_ids.to(
                device=condition_features.device, dtype=torch.long
            )
            target_modality_ids = target_modality_ids.to(
                device=condition_features.device, dtype=torch.long
            )
            self._validate_labels(
                target_batch_ids, target_modality_ids, condition_features.shape[0]
            )
            return probability_flow_sample(
                self.score_model,
                self.sde,
                topology,
                target_batch_ids,
                target_modality_ids,
                self.config.data_dim,
                steps=ode_steps,
                guidance_scale=guidance_scale,
                guidance_target=guidance_target,
                eps=self.config.sampling_eps,
                use_topology_condition=self.config.use_topology_condition,
                use_batch_condition=self.config.use_batch_condition,
            )
        finally:
            self.train(was_training)

    @torch.no_grad()
    def harmonize(
        self,
        observed_features: Tensor,
        operators,
        reference_batch_ids: Tensor,
        modality_ids: Tensor,
        *,
        condition_features: Optional[Tensor] = None,
        strength: float = 0.5,
        guidance_scale: float = 1.0,
        ode_steps: Optional[int] = None,
    ) -> Tensor:
        """Denoise observations under a chosen reference technical condition."""

        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must lie in (0, 1]")
        condition_source = (
            observed_features if condition_features is None else condition_features
        )
        if condition_source.shape[0] != observed_features.shape[0]:
            raise ValueError(
                "observed and condition features must contain the same spots"
            )
        if condition_source.device != observed_features.device:
            raise ValueError(
                "observed_features and condition_features must share a device"
            )

        was_training = self.training
        try:
            self.eval()
            topology = self.encode_condition(condition_source, operators)
            reference_batch_ids = reference_batch_ids.to(
                device=observed_features.device, dtype=torch.long
            )
            modality_ids = modality_ids.to(
                device=observed_features.device, dtype=torch.long
            )
            self._validate_labels(
                reference_batch_ids, modality_ids, observed_features.shape[0]
            )
            t_start = max(self.config.sampling_eps, strength * self.sde.T)
            t = torch.full(
                (observed_features.shape[0],),
                t_start,
                device=observed_features.device,
                dtype=observed_features.dtype,
            )
            mean, std = self.sde.marginal_prob(observed_features, t)
            initial = mean + expand_like(std, observed_features) * torch.randn_like(
                observed_features
            )
            return probability_flow_sample(
                self.score_model,
                self.sde,
                topology,
                reference_batch_ids,
                modality_ids,
                self.config.data_dim,
                steps=ode_steps,
                guidance_scale=guidance_scale,
                guidance_target="labels",
                eps=self.config.sampling_eps,
                initial=initial,
                start_time=t_start,
                use_topology_condition=self.config.use_topology_condition,
                use_batch_condition=self.config.use_batch_condition,
            )
        finally:
            self.train(was_training)
