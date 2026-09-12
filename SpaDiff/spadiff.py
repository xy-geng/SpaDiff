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
from .workflow import SpaDiffWorkflowMixin


class SpaDiff(SpaDiffWorkflowMixin, nn.Module):
    """Unified score model conditioned on fused topology and technical labels.
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

        self.technical_objective = TechnicalConditionObjective(
            data_dim=config.data_dim,
            topology_dim=config.topology_dim,
            num_conditions=config.num_technical_conditions,
            hidden_dim=config.technical_hidden_dim,
            adversarial_strength=config.adversarial_strength,
        )

        self.sde = VPSDE(
            beta_min=config.beta_min,
            beta_max=config.beta_max,
            num_scales=config.num_scales,
        )
        self._reset_workflow_state()

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

    def encode_condition(
        self, features: Tensor, operators, *, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if features.ndim != 2:
            raise ValueError("condition features must have shape [N, F]")
        if features.shape[-1] != self.config.condition_input_dim:
            raise ValueError(
                f"condition feature width must be {self.config.condition_input_dim}, "
                f"got {features.shape[-1]}"
            )
        return self.topology_encoder(
            features, operators, return_attention=return_attention
        )

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

        topology = self.encode_condition(source, operators)
        return self.loss_from_topology(
            target_features,
            topology,
            batch_ids,
            modality_ids,
        )

    def encode_paired_multiomics(
        self,
        rna_features: Tensor,
        atac_features: Tensor,
        operators,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode paired RNA and ATAC views and return their exact mean.
        """

        if self.config.num_modalities != 2:
            raise ValueError(
                "paired RNA-ATAC integration requires num_modalities=2"
            )
        if self.config.data_dim != self.config.condition_input_dim:
            raise ValueError(
                "paired RNA-ATAC integration requires data_dim to equal "
                "condition_input_dim"
            )
        expected_width = self.config.data_dim
        for name, features in (
            ("rna_features", rna_features),
            ("atac_features", atac_features),
        ):
            if features.ndim != 2 or features.shape[1] != expected_width:
                raise ValueError(
                    f"{name} must have shape [N, {expected_width}]"
                )
            if not torch.isfinite(features).all():
                raise ValueError(f"{name} must contain only finite values")
        if rna_features.shape != atac_features.shape:
            raise ValueError(
                "paired RNA and ATAC features must have identical shapes"
            )
        if rna_features.device != atac_features.device:
            raise ValueError("paired RNA and ATAC features must share a device")
        if rna_features.dtype != atac_features.dtype:
            raise ValueError("paired RNA and ATAC features must share a dtype")

        topology_rna = self.encode_condition(rna_features, operators)
        topology_atac = self.encode_condition(atac_features, operators)
        topology_joint = 0.5 * (topology_rna + topology_atac)
        return topology_rna, topology_atac, topology_joint

    def paired_multiomics_loss(
        self,
        rna_features: Tensor,
        atac_features: Tensor,
        operators,
        batch_ids: Tensor,
    ) -> dict[str, Tensor]:
        """Evaluate the joint objective for paired RNA and ATAC instances.
        """

        topology_rna, topology_atac, topology_joint = (
            self.encode_paired_multiomics(
                rna_features,
                atac_features,
                operators,
            )
        )
        n_spots = rna_features.shape[0]
        batch_ids = batch_ids.to(device=rna_features.device, dtype=torch.long)
        if batch_ids.shape != (n_spots,):
            raise ValueError(f"batch_ids must have shape [{n_spots}]")

        all_features = torch.cat((rna_features, atac_features), dim=0)
        all_topology = torch.cat((topology_rna, topology_atac), dim=0)
        all_batch_ids = torch.cat((batch_ids, batch_ids), dim=0)
        all_modality_ids = torch.cat(
            (
                torch.zeros(n_spots, dtype=torch.long, device=rna_features.device),
                torch.ones(n_spots, dtype=torch.long, device=rna_features.device),
            ),
            dim=0,
        )
        output = self.loss_from_topology(
            all_features,
            all_topology,
            all_batch_ids,
            all_modality_ids,
        )
        output.update(
            {
                "topology_rna": topology_rna,
                "topology_atac": topology_atac,
                "topology_joint": topology_joint,
            }
        )
        return output

    def loss_from_topology(
        self,
        target_features: Tensor,
        topology: Tensor,
        batch_ids: Tensor,
        modality_ids: Tensor,
    ) -> dict[str, Tensor]:
        """Evaluate the joint objective from an already encoded topology.
        """

        if (
            target_features.ndim != 2
            or target_features.shape[1] != self.config.data_dim
        ):
            raise ValueError(
                f"target_features must have shape [N, {self.config.data_dim}]"
            )
        if topology.shape != (
            target_features.shape[0],
            self.config.topology_dim,
        ):
            raise ValueError(
                "topology must have shape "
                f"[N, {self.config.topology_dim}]"
            )
        if topology.device != target_features.device:
            raise ValueError("target_features and topology must share a device")

        batch_ids = batch_ids.to(device=target_features.device, dtype=torch.long)
        modality_ids = modality_ids.to(device=target_features.device, dtype=torch.long)
        self._validate_labels(batch_ids, modality_ids, target_features.shape[0])
        condition_ids = technical_condition_ids(
            batch_ids, modality_ids, self.config.num_modalities
        )

        training_scale = 1.0 if self.training else 0.0

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

        technical = self.technical_objective(
            target_features,
            topology,
            condition_ids,
            balanced=self.config.batch_balanced_loss,
        )

        prior = (
            empirical_prior_kl(topology, condition_ids)
            if self.config.prior_kl_weight > 0.0
            else topology.sum() * 0.0
        )

        batch_loss = (
            technical["alignment_loss"]
            + self.config.batch_posterior_weight * technical["posterior_loss"]
        )

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
        strength: float = 0.10,
        guidance_scale: float = 1.0,
        ode_steps: Optional[int] = 300,
    ) -> Tensor:

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
