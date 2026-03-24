"""End-to-end topology-conditioned SpaDiff model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import SpaDiffConfig
from .diffusion import ConditionalScoreNetwork, conditional_dsm_loss
from .model import OriginalDECObjective, OriginalHiGCNEncoder, TopologyEncoder
from .sampling import probability_flow_sample
from .sde import VPSDE, expand_like


class SpaDiff(nn.Module):
    """Conditional VP-SDE for topology-aware spatial-omics generation.

    The forward perturbation kernel is intentionally condition-independent.
    Conditions H, batch and modality parameterize the reverse score, which is
    the standard conditional diffusion formulation and should be reflected in
    the manuscript's Eq. (12)-(18).
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
        if config.k1 > 0.0:
            # The DEC and diffusion terms share this original-compatible
            # encoder, so both losses update the same spatial representation.
            self.topology_encoder = OriginalHiGCNEncoder(
                input_dim=config.condition_input_dim,
                hidden_dim=config.topology_hidden_dim,
                output_dim=config.topology_dim,
                orders=config.simplex_orders,
                steps=config.propagation_steps,
                alpha=config.propagation_alpha,
                dropout=config.dropout,
                projection_dropout=projection_dropout,
            )
            self.original_objective = OriginalDECObjective(
                num_clusters=config.num_clusters,
                embedding_dim=config.topology_dim,
                alpha=config.dec_alpha,
                init_method=config.dec_init_method,
                mclust_model_names=(config.dec_mclust_model_names),
                mclust_pca_dim=config.dec_mclust_pca_dim,
            )
        else:
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
            )
            self.original_objective = None

        # Do not even instantiate the diffusion branch at k2=0.  Besides
        # avoiding useless work, this preserves the original random-number
        # stream for HiGCN initialization and dropout during reproduction.
        if config.k2 > 0.0:
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
            self.sde = VPSDE(
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                num_scales=config.num_scales,
            )
        else:
            self.score_model = None
            self.sde = None

    def encode_condition(self, features: Tensor, operators) -> Tensor:
        if features.shape[-1] != self.config.condition_input_dim:
            raise ValueError(
                f"condition feature width must be {self.config.condition_input_dim}, "
                f"got {features.shape[-1]}"
            )
        return self.topology_encoder(features, operators)

    @torch.no_grad()
    def initialize_original_objective(
        self, features: Tensor, operators
    ) -> Tensor:
        if self.original_objective is None:
            raise RuntimeError("k1 must be positive to initialize the original objective")
        embedding = self.encode_condition(features, operators)
        return self.original_objective.initialize(
            embedding, random_state=self.config.random_seed
        )

    def original_forward(
        self, features: Tensor, operators
    ) -> tuple[Tensor, Tensor]:
        if self.original_objective is None:
            raise RuntimeError("k1 must be positive to use the original objective")
        embedding = self.encode_condition(features, operators)
        return embedding, self.original_objective.soft_assign(embedding)

    def original_loss(
        self, features: Tensor, operators, target_distribution: Tensor
    ) -> dict[str, Tensor]:
        embedding, q = self.original_forward(features, operators)
        loss = self.original_objective.loss(target_distribution, q)
        return {"loss": loss, "embedding": embedding, "q": q}

    @torch.no_grad()
    def original_predict(
        self,
        features: Tensor,
        operators,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """在关闭 dropout 的情况下返回标签、概率和嵌入。"""

        was_training = self.training

        try:
            self.eval()

            embedding, q = self.original_forward(
                features,
                operators,
            )

            labels = q.argmax(dim=1)

        finally:
            self.train(was_training)

        return labels, q, embedding
    
    def diffusion_loss(
        self,
        target_features: Tensor,
        operators,
        batch_ids: Tensor,
        modality_ids: Tensor,
        *,
        condition_features: Optional[Tensor] = None,
        likelihood_weighting: bool = False,
    ) -> dict[str, Tensor]:
        """Train p(target | topology(source), batch, modality).

        If ``condition_features`` is omitted, target features also construct H.
        For cross-modal generation, pass paired source-modality features here.
        """
        if self.score_model is None or self.sde is None:
            raise RuntimeError("k2 must be positive to compute diffusion loss")
        source = target_features if condition_features is None else condition_features
        if source.device != target_features.device:
            raise ValueError("target_features and condition_features must share a device")
        topology = self.encode_condition(source, operators)
        batch_ids = batch_ids.to(device=target_features.device, dtype=torch.long)
        modality_ids = modality_ids.to(device=target_features.device, dtype=torch.long)
        training_scale = 1.0 if self.training else 0.0
        return conditional_dsm_loss(
            self.score_model,
            self.sde,
            target_features,
            topology,
            batch_ids,
            modality_ids,
            eps=self.config.sampling_eps,
            likelihood_weighting=likelihood_weighting,
            joint_dropout=training_scale * self.config.condition_dropout_joint,
            topology_dropout=training_scale * self.config.condition_dropout_topology,
            batch_dropout=training_scale * self.config.condition_dropout_batch,
            modality_dropout=training_scale * self.config.condition_dropout_modality,
        )

    @torch.no_grad()
    def generate(
        self,
        condition_features: Tensor,
        operators,
        target_batch_ids: Tensor,
        target_modality_ids: Tensor,
        *,
        sampler: str = "ode",
        guidance_scale: float = 1.0,
        guidance_target: str = "all",
        # TUTORIAL-UNUSED: PC-only arguments are retained in the commented sampler block.
        # corrector_steps: int = 1,
        # snr: float = 0.1,
        ode_steps: Optional[int] = None,
    ) -> Tensor:
        """Generate target PCA/LSI features from source/topology conditions."""
        if self.score_model is None or self.sde is None:
            raise RuntimeError("generation is unavailable when k2=0")
        was_training = self.training
        self.eval()
        topology = self.encode_condition(condition_features, operators)
        target_batch_ids = target_batch_ids.to(
            device=condition_features.device, dtype=torch.long
        )
        target_modality_ids = target_modality_ids.to(
            device=condition_features.device, dtype=torch.long
        )

        if sampler == "ode":
            result = probability_flow_sample(
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
            )
        else:
            raise ValueError("tutorial-supported sampler must be 'ode'")
        self.train(was_training)
        return result

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
        sampler: str = "ode",
        guidance_scale: float = 1.0,
        ode_steps: Optional[int] = None,
    ) -> Tensor:
        """Noise observed data, then denoise under a reference batch condition."""
        if self.score_model is None or self.sde is None:
            raise RuntimeError("harmonization is unavailable when k2=0")
        if not 0.0 < strength <= 1.0:
            raise ValueError("strength must lie in (0, 1]")
        was_training = self.training
        self.eval()
        condition_source = observed_features if condition_features is None else condition_features
        if condition_source.device != observed_features.device:
            raise ValueError("observed_features and condition_features must share a device")
        topology = self.encode_condition(condition_source, operators)
        reference_batch_ids = reference_batch_ids.to(
            device=observed_features.device, dtype=torch.long
        )
        modality_ids = modality_ids.to(device=observed_features.device, dtype=torch.long)
        t_start = max(self.config.sampling_eps, strength * self.sde.T)
        t = torch.full(
            (observed_features.shape[0],),
            t_start,
            device=observed_features.device,
            dtype=observed_features.dtype,
        )
        mean, std = self.sde.marginal_prob(observed_features, t)
        initial = mean + expand_like(std, observed_features) * torch.randn_like(observed_features)

        if sampler == "ode":
            result = probability_flow_sample(
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
            )
        else:
            raise ValueError("tutorial-supported sampler must be 'ode'")
        self.train(was_training)
        return result
