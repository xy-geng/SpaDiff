"""Training utilities for the tutorial's conditional SpaDiff path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor



class ExponentialMovingAverage:
    """Small EMA utility following the training pattern in score_sde_pytorch."""

    def __init__(self, parameters, decay: float = 0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = decay
        self.shadow = [p.detach().clone() for p in parameters if p.requires_grad]
        self.backup = None

    @torch.no_grad()
    def update(self, parameters) -> None:
        values = [p for p in parameters if p.requires_grad]
        if len(values) != len(self.shadow):
            raise ValueError("parameter set changed after EMA initialization")
        for shadow, parameter in zip(self.shadow, values):
            shadow.lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def store(self, parameters) -> None:
        self.backup = [p.detach().clone() for p in parameters if p.requires_grad]

    @torch.no_grad()
    def copy_to(self, parameters) -> None:
        values = [p for p in parameters if p.requires_grad]
        for parameter, shadow in zip(values, self.shadow):
            parameter.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters) -> None:
        if self.backup is None:
            raise RuntimeError("EMA.store must be called before restore")
        values = [p for p in parameters if p.requires_grad]
        for parameter, backup in zip(values, self.backup):
            parameter.copy_(backup)
        self.backup = None



@dataclass
class TrainingResult:
    losses: list[float]
    best_loss: float
    ema: Optional[ExponentialMovingAverage]
    original_losses: list[float]
    diffusion_losses: list[float]
    stopped_early: bool = False


def train_spadiff(
    model,
    target_features: Tensor,
    operators,
    batch_ids: Tensor,
    modality_ids: Tensor,
    *,
    condition_features: Optional[Tensor] = None,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    ema_decay: Optional[float] = 0.999,
    likelihood_weighting: bool = False,
    verbose_every: int = 25,
) -> TrainingResult:
    """Optimize ``k1 * original DEC-KL + k2 * diffusion DSM``.

    When ``k2 == 0`` this deliberately follows the original DLPFC optimizer,
    DEC initialization and early-stop behavior.  The diffusion branch is not
    evaluated at all, which is essential for RNG-compatible reproduction.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    k1 = float(model.config.k1)
    k2 = float(model.config.k2)
    original_only = k1 > 0.0 and k2 == 0.0
    optimizer_class = torch.optim.Adam if original_only else torch.optim.AdamW
    optimizer = optimizer_class(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ema = (
        ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        if ema_decay is not None and k2 > 0.0
        else None
    )
    history: list[float] = []
    original_history: list[float] = []
    diffusion_history: list[float] = []
    best = float("inf")
    stopped_early = False
    model.train()

    previous_labels = None
    target_distribution = None
    if k1 > 0.0:
        previous_labels = model.initialize_original_objective(
            target_features, operators
        )

    for epoch in range(epochs):
        original_output = None

        if k1 > 0.0:
            if epoch % model.config.dec_update_interval == 0:
                was_training = model.training
                try:
                    model.eval()
                    with torch.no_grad():
                        _, update_q = model.original_forward(
                            target_features,
                            operators,
                        )

                        target_distribution = (
                            model.original_objective
                            .target_distribution(update_q)
                            .detach()
                        )

                        current_labels = update_q.argmax(dim=1)

                finally:
                    model.train(was_training)

                # 只比较相邻两次目标分布更新时的标签，再比较每一个 epoch 中带随机 dropout 的标签。
                if previous_labels is not None and epoch > 0:
                    changed = (
                        current_labels != previous_labels
                    ).float().mean().item()

                    if changed < model.config.dec_tolerance:
                        stopped_early = True

                        if verbose_every:
                            print(
                                f"DEC early stopping at epoch {epoch + 1}: "
                                f"label change={changed:.6f} < "
                                f"tolerance={model.config.dec_tolerance:.6f}"
                            )

                        break

                previous_labels = current_labels.detach()

            # 训练损失仍在 train 模式下计算，可以继续使用 dropout。
            original_output = model.original_loss(
                target_features,
                operators,
                target_distribution,
            )
            original_loss = original_output["loss"]

        else:
            original_loss = target_features.new_zeros(())

        if k2 > 0.0:
            diffusion_output = model.diffusion_loss(
                target_features,
                operators,
                batch_ids,
                modality_ids,
                condition_features=condition_features,
                likelihood_weighting=likelihood_weighting,
            )
            diffusion_loss = diffusion_output["loss"]
        else:
            diffusion_loss = target_features.new_zeros(())

        # The requested hybrid objective.
        loss = k1 * original_loss + k2 * diffusion_loss
        optimizer.zero_grad(set_to_none=not original_only)
        loss.backward()
        # The original DEC code did not clip gradients.
        if not original_only and grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model.parameters())
        value = float(loss.detach().cpu())
        original_value = float(original_loss.detach().cpu())
        diffusion_value = float(diffusion_loss.detach().cpu())
        history.append(value)
        original_history.append(original_value)
        diffusion_history.append(diffusion_value)
        best = min(best, value)
        if verbose_every and (epoch == 0 or (epoch + 1) % verbose_every == 0):
            print(
                f"Epoch {epoch + 1:04d}/{epochs} | total={value:.6f} | "
                f"original={original_value:.6f} | diffusion={diffusion_value:.6f}"
            )

        if original_only and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return TrainingResult(
        losses=history,
        best_loss=best,
        ema=ema,
        original_losses=original_history,
        diffusion_losses=diffusion_history,
        stopped_early=stopped_early,
    )

