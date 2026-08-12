from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class ExponentialMovingAverage:
    """EMA utility for stable score-model sampling."""

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
        if len(values) != len(self.shadow):
            raise ValueError("parameter set changed after EMA initialization")
        for parameter, shadow in zip(values, self.shadow):
            parameter.copy_(shadow)

    @torch.no_grad()
    def restore(self, parameters) -> None:
        if self.backup is None:
            raise RuntimeError("EMA.store must be called before restore")
        values = [p for p in parameters if p.requires_grad]
        if len(values) != len(self.backup):
            raise ValueError("parameter set changed after EMA.store")
        for parameter, backup in zip(values, self.backup):
            parameter.copy_(backup)
        self.backup = None


@dataclass
class TrainingResult:
    losses: list[float]
    best_loss: float
    best_epoch: int
    ema: Optional[ExponentialMovingAverage]
    dsm_losses: list[float]
    batch_alignment_losses: list[float]
    batch_posterior_losses: list[float]
    prior_kl_losses: list[float]
    diagnostics: list[dict]
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
    grad_clip: Optional[float] = 1.0,
    ema_decay: Optional[float] = 0.999,
    verbose_every: int = 25,
    checkpoint_epochs: Optional[Sequence[int]] = None,
    checkpoint_callback: Optional[Callable[[object, int, dict], bool | None]] = None,
) -> TrainingResult:
    """Optimize the DSM, batch-ratio and optional latent-prior terms.

    There is intentionally no clustering-driven early stop: the manuscript
    specifies diffusion training for 500 epochs, and a downstream clustering
    criterion must not terminate score learning.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if grad_clip is not None and grad_clip <= 0.0:
        raise ValueError("grad_clip must be positive or None")
    callback_epochs = None
    if checkpoint_epochs is not None:
        callback_epochs = {int(value) for value in checkpoint_epochs}
        if not callback_epochs or any(value <= 0 for value in callback_epochs):
            raise ValueError("checkpoint_epochs must contain positive integers")
        if checkpoint_callback is None:
            raise ValueError(
                "checkpoint_callback is required when checkpoint_epochs are set"
            )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    ema = (
        ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        if ema_decay is not None
        else None
    )

    history: list[float] = []
    dsm_history: list[float] = []
    alignment_history: list[float] = []
    posterior_history: list[float] = []
    prior_history: list[float] = []
    diagnostics: list[dict] = []
    best = float("inf")
    best_epoch = 0
    stopped_early = False
    model.train()

    for epoch in range(epochs):
        # model.loss 内部已经按三个外层权重组合 DSM、批次对齐和 KL 项。
        output = model.loss(
            target_features,
            operators,
            batch_ids,
            modality_ids,
            condition_features=condition_features,
        )
        # 这里只对加权后的总损失反向传播；各原始子项保留用于诊断量级。
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch + 1}: {loss.item()}"
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model.parameters())

        values = {
            "total_loss": float(loss.detach().cpu()),
            "dsm_loss": float(output["dsm_loss"].detach().cpu()),
            "batch_loss": float(output["batch_loss"].detach().cpu()),
            "batch_alignment_loss": float(
                output["batch_alignment_loss"].detach().cpu()
            ),
            "batch_posterior_loss": float(
                output["batch_posterior_loss"].detach().cpu()
            ),
            "prior_kl_loss": float(output["prior_kl_loss"].detach().cpu()),
            "weighted_dsm_loss": float(output["weighted_dsm_loss"].detach().cpu()),
            "weighted_batch_loss": float(
                output["weighted_batch_loss"].detach().cpu()
            ),
            "weighted_prior_kl_loss": float(
                output["weighted_prior_kl_loss"].detach().cpu()
            ),
            "noise_mse": float(output["noise_mse"].detach().cpu()),
            "score_mse": float(output["score_mse"].detach().cpu()),
            "posterior_accuracy": float(output["posterior_accuracy"].detach().cpu()),
            "topology_batch_accuracy": float(
                output["topology_batch_accuracy"].detach().cpu()
            ),
        }
        history.append(values["total_loss"])
        dsm_history.append(values["dsm_loss"])
        alignment_history.append(values["batch_alignment_loss"])
        posterior_history.append(values["batch_posterior_loss"])
        prior_history.append(values["prior_kl_loss"])
        diagnostics.append({"epoch": epoch + 1, **values})
        if values["total_loss"] < best:
            best = values["total_loss"]
            best_epoch = epoch + 1

        if verbose_every and (epoch == 0 or (epoch + 1) % verbose_every == 0):
            print(
                f"Epoch {epoch + 1:04d}/{epochs} | "
                f"total={values['total_loss']:.6f} | "
                f"dsm={values['dsm_loss']:.6f} | "
                f"align={values['batch_alignment_loss']:.6f} | "
                f"posterior={values['batch_posterior_loss']:.6f} | "
                f"prior={values['prior_kl_loss']:.6f}"
            )

        if (
            checkpoint_callback is not None
            and callback_epochs is not None
            and (epoch + 1) in callback_epochs
        ):
            should_stop = checkpoint_callback(model, epoch + 1, values)
            if should_stop:
                stopped_early = True
                break

    return TrainingResult(
        losses=history,
        best_loss=best,
        best_epoch=best_epoch,
        ema=ema,
        dsm_losses=dsm_history,
        batch_alignment_losses=alignment_history,
        batch_posterior_losses=posterior_history,
        prior_kl_losses=prior_history,
        diagnostics=diagnostics,
        stopped_early=stopped_early,
    )
