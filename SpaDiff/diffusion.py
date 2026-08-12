"""Conditional score network and denoising score-matching objective."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import balanced_mean
from .sde import SDE, expand_like


def sinusoidal_time_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    """Continuous-time positional embedding for t in [0, 1]."""
    # 把连续扩散时刻 t 编码为正余弦向量，供分数网络使用。
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    angles = (t * 1000.0)[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_scale_shift = nn.Linear(condition_dim, 2 * hidden_dim)
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # A near-identity start improves score-network stability.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        # 用时间、批次和模态条件生成缩放/平移量，调制隐藏特征。
        scale, shift = self.to_scale_shift(condition).chunk(2, dim=-1)
        hidden = self.norm(x) * (1.0 + scale) + shift
        return x + self.net(hidden)


class ConditionalScoreNetwork(nn.Module):
    """Predict epsilon from (x_t, t, topology H, batch, modality).

    """

    def __init__(
        self,
        data_dim: int,
        topology_dim: int,
        hidden_dim: int,
        time_embedding_dim: int,
        condition_embedding_dim: int,
        num_batches: int,
        num_modalities: int,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.data_dim = data_dim
        self.topology_dim = topology_dim
        self.time_embedding_dim = time_embedding_dim
        self.num_batches = num_batches
        self.num_modalities = num_modalities
        self.null_topology = nn.Parameter(torch.zeros(1, topology_dim))
        self.batch_embedding = nn.Embedding(num_batches + 1, condition_embedding_dim)
        self.modality_embedding = nn.Embedding(
            num_modalities + 1, condition_embedding_dim
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, condition_embedding_dim),
            nn.SiLU(),
            nn.Linear(condition_embedding_dim, condition_embedding_dim),
        )
        self.input_projection = nn.Linear(data_dim + topology_dim, hidden_dim)
        combined_condition_dim = 3 * condition_embedding_dim
        self.blocks = nn.ModuleList(
            FiLMResidualBlock(hidden_dim, combined_condition_dim, dropout)
            for _ in range(depth)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def null_labels(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.full_like(like, self.num_batches),
            torch.full_like(like, self.num_modalities),
        )

    def forward(
        self,
        noisy: Tensor,
        t: Tensor,
        topology: Tensor,
        batch_ids: Tensor,
        modality_ids: Tensor,
        *,
        drop_topology: Optional[Tensor] = None,
    ) -> Tensor:
        n = noisy.shape[0]
        if topology.shape != (n, self.topology_dim):
            raise ValueError(
                f"topology must have shape {(n, self.topology_dim)}, got {tuple(topology.shape)}"
            )
        if t.shape != (n,) or batch_ids.shape != (n,) or modality_ids.shape != (n,):
            raise ValueError("t, batch_ids and modality_ids must each have shape [N]")
        if drop_topology is not None:
            topology = torch.where(
                drop_topology[:, None], self.null_topology.expand_as(topology), topology
            )
        # 分别编码扩散时间 t、批次 b 和模态，再拼成 FiLM 条件。
        time = self.time_mlp(sinusoidal_time_embedding(t, self.time_embedding_dim))
        condition = torch.cat(
            (
                time,
                self.batch_embedding(batch_ids.long()),
                self.modality_embedding(modality_ids.long()),
            ),
            dim=-1,
        )
        # 把加噪数据 x_t 与融合拓扑表示 H 拼接后送入条件残差网络。
        hidden = self.input_projection(torch.cat((noisy, topology), dim=-1))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden)


def apply_condition_dropout(
    model: ConditionalScoreNetwork,
    batch_ids: Tensor,
    modality_ids: Tensor,
    *,
    joint_probability: float,
    topology_probability: float,
    batch_probability: float,
    modality_probability: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Independently drop H, batch and modality for classifier-free guidance."""
    n = batch_ids.shape[0]
    device = batch_ids.device
    joint_drop = torch.rand(n, device=device) < joint_probability
    topology_drop = joint_drop | (torch.rand(n, device=device) < topology_probability)
    batch_drop = joint_drop | (torch.rand(n, device=device) < batch_probability)
    modality_drop = joint_drop | (torch.rand(n, device=device) < modality_probability)
    null_batch, null_modality = model.null_labels(batch_ids)
    return (
        topology_drop,
        torch.where(batch_drop, null_batch, batch_ids),
        torch.where(modality_drop, null_modality, modality_ids),
    )


def conditional_dsm_loss(
    model: ConditionalScoreNetwork,
    sde: SDE,
    clean: Tensor,
    topology: Tensor,
    batch_ids: Tensor,
    modality_ids: Tensor,
    *,
    eps: float = 1e-3,
    weighting: Literal["score", "variance", "likelihood"] = "variance",
    loss_group_ids: Optional[Tensor] = None,
    batch_balanced: bool = True,
    joint_dropout: float = 0.0,
    topology_dropout: float = 0.0,
    batch_dropout: float = 0.0,
    modality_dropout: float = 0.0,
    use_topology_condition: bool = True,
    use_batch_condition: bool = True,
) -> dict[str, Tensor]:
    """Conditional continuous-time DSM with an explicit time weighting.
    """
    weighting = weighting.lower()
    if weighting not in {"score", "variance", "likelihood"}:
        raise ValueError("weighting must be 'score', 'variance' or 'likelihood'")
    # 损失第 1 项（DSM）：随机采样 t，并按 VP-SDE 的闭式扰动核构造 x_t。
    n = clean.shape[0]
    t = torch.rand(n, device=clean.device, dtype=clean.dtype) * (sde.T - eps) + eps
    noise = torch.randn_like(clean)
    mean, std = sde.marginal_prob(clean, t)
    perturbed = mean + expand_like(std, clean) * noise
    topology_drop, used_batch, used_modality = apply_condition_dropout(
        model,
        batch_ids,
        modality_ids,
        joint_probability=joint_dropout,
        topology_probability=topology_dropout,
        batch_probability=batch_dropout,
        modality_probability=modality_dropout,
    )
    # 组件消融时使用模型预留的空条件，而不是把某个真实批次误当作空批次。
    if not use_topology_condition:
        topology_drop = torch.ones_like(topology_drop, dtype=torch.bool)
    if not use_batch_condition:
        null_batch, _ = model.null_labels(batch_ids)
        used_batch = null_batch
    # 分数网络实际预测 epsilon；下面再用 s_theta=-epsilon_theta/std 转为分数。
    predicted_noise = model(
        perturbed,
        t,
        topology,
        used_batch,
        used_modality,
        drop_topology=topology_drop,
    )
    score = -predicted_noise / expand_like(std.clamp_min(1e-12), clean)
    feature_dims = tuple(range(1, clean.ndim))
    # 真实条件分数为 grad log p(x_t|x_0)=-epsilon/std。
    # 因此残差 s_theta-grad log p 等于 score+epsilon/std。
    score_residual = score + noise / expand_like(std.clamp_min(1e-12), clean)
    score_per_row = score_residual.square().mean(dim=feature_dims)
    noise_per_row = (predicted_noise - noise).square().mean(dim=feature_dims)

    # score 对应式 (18)/(19) 的形式；variance 等价于稳定的 epsilon-MSE。
    if weighting == "score":
        per_row = score_per_row
    elif weighting == "likelihood":
        diffusion = sde.sde(torch.zeros_like(clean), t)[1]
        per_row = score_per_row * diffusion.square()
    else:
        per_row = noise_per_row

    # 多切片/多模态时先在每个技术组内求均值，再对组等权平均，避免大切片支配损失。
    if loss_group_ids is None:
        loss = per_row.mean()
    else:
        loss = balanced_mean(per_row, loss_group_ids, enabled=batch_balanced)
    return {
        "loss": loss,
        "score_mse": score_per_row.mean().detach(),
        "noise_mse": noise_per_row.mean().detach(),
        "mean_time": t.mean().detach(),
    }


def make_score_fn(
    model: ConditionalScoreNetwork,
    sde: SDE,
    topology: Tensor,
    batch_ids: Tensor,
    modality_ids: Tensor,
    *,
    guidance_scale: float = 1.0,
    guidance_target: Literal["all", "labels"] = "all",
    use_topology_condition: bool = True,
    use_batch_condition: bool = True,
):
    """Wrap epsilon prediction as a true conditional score function.

    This mirrors score_sde_pytorch/models/utils.py::get_score_fn, while also
    binding SpaDiff's topology, batch and modality conditions.
    """
    if guidance_target not in ("all", "labels"):
        raise ValueError("guidance_target must be 'all' or 'labels'")

    def score_fn(x: Tensor, t: Tensor) -> Tensor:
        # 训练与采样必须使用同一套组件开关，避免消融条件在推断时被重新启用。
        used_batch = batch_ids
        drop_topology = None
        if not use_batch_condition:
            used_batch, _ = model.null_labels(batch_ids)
        if not use_topology_condition:
            drop_topology = torch.ones_like(batch_ids, dtype=torch.bool)
        conditional = model(
            x,
            t,
            topology,
            used_batch,
            modality_ids,
            drop_topology=drop_topology,
        )
        if guidance_scale == 1.0:
            predicted_noise = conditional
        else:
            null_batch, null_modality = model.null_labels(batch_ids)
            drop_topology = torch.ones_like(batch_ids, dtype=torch.bool)
            if guidance_target == "labels" and use_topology_condition:
                drop_topology.zero_()
            unconditional = model(
                x,
                t,
                topology,
                null_batch,
                null_modality,
                drop_topology=drop_topology,
            )
            predicted_noise = unconditional + guidance_scale * (
                conditional - unconditional
            )
        _, std = sde.marginal_prob(torch.zeros_like(x), t)
        return -predicted_noise / expand_like(std.clamp_min(1e-12), x)

    return score_fn
