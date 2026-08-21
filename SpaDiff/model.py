"""Higher-order topology encoder and technical objectives."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def operator_matmul(operator: Tensor, x: Tensor) -> Tensor:
    return (
        torch.sparse.mm(operator, x)
        if operator.layout != torch.strided
        else operator @ x
    )


def _get_operator(operators, order: int) -> Tensor:
    if isinstance(operators, Mapping):
        return operators[order]
    return operators[order]


class PolynomialPropagation(nn.Module):
    """Polynomial higher-order propagation from manuscript Eq. (9)-(10)."""

    def __init__(self, steps: int, alpha: float, learnable: bool = False):
        super().__init__()
        coefficients = [alpha * (1.0 - alpha) ** k for k in range(steps)]
        coefficients.append((1.0 - alpha) ** steps)
        values = torch.tensor(coefficients, dtype=torch.float32)
        self.steps = steps
        self.learnable = learnable
        if learnable:
            self.logits = nn.Parameter(values.clamp_min(1e-12).log())
        else:
            self.register_buffer("coefficients", values)

    def forward(self, x: Tensor, operator: Tensor) -> Tensor:
        # 模型组成：实现论文式 (9)-(10) 的多跳高阶邻域多项式传播。
        weights = F.softmax(self.logits, dim=0) if self.learnable else self.coefficients
        propagated = x
        result = weights[0] * x
        for step in range(1, self.steps + 1):
            propagated = operator_matmul(operator, propagated)
            result = result + weights[step] * propagated
        return result


class TopologyEncoder(nn.Module):
    """Multi-channel edge/triangle encoder with order attention."""

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
        residual: bool = True,
        output_normalization: str = "feature",
    ):
        super().__init__()
        self.orders = tuple(orders)
        self.dropout = dropout
        self.projection_dropout = (
            dropout if projection_dropout is None else projection_dropout
        )
        self.output_normalization = output_normalization.lower()
        if self.output_normalization not in {"none", "feature", "layer"}:
            raise ValueError(
                "output_normalization must be 'none', 'feature' or 'layer'"
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
        self.channel_norms = nn.ModuleDict(
            {str(order): nn.LayerNorm(hidden_dim) for order in self.orders}
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.fusion = nn.Linear(hidden_dim, output_dim)
        if residual:
            if input_dim == output_dim:
                self.residual_projection = nn.Identity()
            else:
                self.residual_projection = nn.Linear(input_dim, output_dim, bias=False)
                nn.init.orthogonal_(self.residual_projection.weight)
            nn.init.xavier_uniform_(self.fusion.weight, gain=0.1)
            nn.init.zeros_(self.fusion.bias)
        else:
            self.residual_projection = None

        self.layer_norm = (
            nn.LayerNorm(output_dim) if self.output_normalization == "layer" else None
        )

    def forward(
        self, features: Tensor, operators, *, return_attention: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        # 每个单纯形阶数（边、三角形等）使用独立投影与传播通道。
        dropped = F.dropout(features, self.dropout, training=self.training)
        channels = []
        for order in self.orders:
            hidden = self.projections[str(order)](dropped)
            hidden = F.dropout(hidden, self.projection_dropout, training=self.training)
            hidden = self.propagations[str(order)](
                hidden, _get_operator(operators, order)
            )
            hidden = F.silu(hidden)
            channels.append(self.channel_norms[str(order)](hidden))

        # 对同一 spot 的不同单纯形阶进行 softmax 注意力，再按论文式 (13)-(15)
        # 加权求和得到融合的拓扑条件 H。
        stacked = torch.stack(channels, dim=1)
        attention = F.softmax(self.attention(stacked).squeeze(-1), dim=1)
        fused = torch.sum(attention.unsqueeze(-1) * stacked, dim=1)
        output = self.fusion(fused)
        if self.residual_projection is not None:
            output = output + self.residual_projection(features)
        if self.output_normalization == "feature":
            mean = output.mean(dim=0, keepdim=True)
            variance = output.var(dim=0, keepdim=True, unbiased=False)
            output = (output - mean) * torch.rsqrt(variance + 1e-5)
        elif self.layer_norm is not None:
            output = self.layer_norm(output)
        if return_attention:
            return output, attention
        return output


def technical_condition_ids(
    batch_ids: Tensor, modality_ids: Tensor, num_modalities: int
) -> Tensor:

    if batch_ids.shape != modality_ids.shape:
        raise ValueError("batch_ids and modality_ids must have identical shapes")
    return batch_ids.long() * num_modalities + modality_ids.long()


def balanced_mean(values: Tensor, group_ids: Tensor, enabled: bool = True) -> Tensor:
    """Average groups equally so large slices do not dominate the objective."""

    if values.ndim != 1 or group_ids.ndim != 1 or values.shape != group_ids.shape:
        raise ValueError("values and group_ids must both have shape [N]")
    if not enabled:
        return values.mean()
    groups = torch.unique(group_ids)
    if groups.numel() == 0:
        raise ValueError("cannot reduce an empty batch")
    return torch.stack([values[group_ids == group].mean() for group in groups]).mean()


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        # 损失第 2 项：分类器按正常方向学习批次；拓扑编码器收到反向梯度，
        # 从而尽量移除 H 中可识别批次/模态的信息。
        return -ctx.strength * gradient, None


def gradient_reverse(value: Tensor, strength: float = 1.0) -> Tensor:
    return _GradientReversal.apply(value, strength)


class TechnicalConditionObjective(nn.Module):
    """Operational form of the SI q(b|x0) / p(b|H) alignment term.

    """

    def __init__(
        self,
        data_dim: int,
        topology_dim: int,
        num_conditions: int,
        hidden_dim: int,
        adversarial_strength: float = 1.0,
    ):
        super().__init__()
        self.num_conditions = num_conditions
        self.adversarial_strength = adversarial_strength
        if num_conditions > 1:
            self.data_posterior = nn.Sequential(
                nn.LayerNorm(data_dim),
                nn.Linear(data_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_conditions),
            )
            self.topology_predictor = nn.Sequential(
                nn.LayerNorm(topology_dim),
                nn.Linear(topology_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_conditions),
            )
        else:
            self.data_posterior = None
            self.topology_predictor = None

    def forward(
        self,
        clean: Tensor,
        topology: Tensor,
        condition_ids: Tensor,
        *,
        balanced: bool = True,
    ) -> dict[str, Tensor]:
        zero = topology.sum() * 0.0
        if self.num_conditions == 1:
            return {
                "alignment_loss": zero,
                "posterior_loss": zero,
                "posterior_accuracy": zero.detach(),
                "topology_accuracy": zero.detach(),
            }

        if (
            condition_ids.min().item() < 0
            or condition_ids.max().item() >= self.num_conditions
        ):
            raise ValueError("technical condition id is outside the configured range")

        # 损失第 2 项的辅助子项：用真实技术标签监督 q_phi(b|x0)。
        # 没有这一步，比值项中的 q_phi(b|x0) 没有可辨识的学习目标。
        posterior_logits = self.data_posterior(clean)
        posterior_per_row = F.cross_entropy(
            posterior_logits, condition_ids, reduction="none"
        )
        posterior_loss = balanced_mean(
            posterior_per_row, condition_ids, enabled=balanced
        )

        # 损失第 2 项的比值/对齐主体：用 q_phi(b|x0) 作为软目标，训练 p(b|H)。
        # detach 防止对齐分支反过来破坏已经由真实标签监督的 q_phi。
        posterior_probability = F.softmax(posterior_logits.detach(), dim=-1)
        topology_logits = self.topology_predictor(
            gradient_reverse(topology, self.adversarial_strength)
        )
        # 梯度反转使预测器最小化该 KL，而拓扑编码器对抗性地弱化批次信息。
        ratio_per_row = F.kl_div(
            F.log_softmax(topology_logits, dim=-1),
            posterior_probability,
            reduction="none",
        ).sum(dim=-1)
        alignment_loss = balanced_mean(ratio_per_row, condition_ids, enabled=balanced)

        return {
            "alignment_loss": alignment_loss,
            "posterior_loss": posterior_loss,
            "posterior_accuracy": (posterior_logits.argmax(dim=-1) == condition_ids)
            .float()
            .mean()
            .detach(),
            "topology_accuracy": (topology_logits.argmax(dim=-1) == condition_ids)
            .float()
            .mean()
            .detach(),
        }


def empirical_prior_kl(
    topology: Tensor,
    condition_ids: Tensor,
    *,
    eps: float = 1e-5,
) -> Tensor:
    """Approximate KL(q_phi(H|b) || p(H)) with diagonal empirical Gaussians.
    """

    if topology.ndim != 2 or condition_ids.shape != (topology.shape[0],):
        raise ValueError("topology must be [N, D] and condition_ids must be [N]")
    groups = torch.unique(condition_ids)
    if groups.numel() <= 1:
        return topology.sum() * 0.0

    # 损失第 3 项：用对角高斯的经验矩近似 q_phi(H|b) 与共享先验 p(H)。
    pooled_mean = topology.mean(dim=0).detach()
    pooled_variance = topology.var(dim=0, unbiased=False).clamp_min(eps).detach()
    terms = []
    for group in groups:
        values = topology[condition_ids == group]
        group_mean = values.mean(dim=0)
        group_variance = values.var(dim=0, unbiased=False).clamp_min(eps)

        kl = 0.5 * (
            group_variance / pooled_variance
            + (group_mean - pooled_mean).square() / pooled_variance
            - 1.0
            + pooled_variance.log()
            - group_variance.log()
        )
        terms.append(kl.mean())
    return torch.stack(terms).mean()
