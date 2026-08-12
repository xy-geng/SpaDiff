"""Continuous SDE definitions adapted from score_sde_pytorch for [N, F] data.

Reference design:
https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py

This is an adaptation, not a verbatim copy. The important changes are
two-dimensional broadcasting, condition-aware score closures, generator-aware
sampling, and explicit reverse-SDE versus probability-flow-ODE semantics.
"""

from __future__ import annotations

import abc
from typing import Callable

import torch
from torch import Tensor


ScoreFn = Callable[[Tensor, Tensor], Tensor]


def expand_like(vector: Tensor, target: Tensor) -> Tensor:
    """Append singleton dimensions until ``vector`` broadcasts over target."""
    return vector.reshape(vector.shape[0], *([1] * (target.ndim - 1)))


class SDE(abc.ABC):
    def __init__(self, num_scales: int):
        self.N = num_scales

    @property
    @abc.abstractmethod
    def T(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def prior_sampling(
        self, shape, *, device=None, dtype=None, generator=None
    ) -> Tensor:
        raise NotImplementedError

    def reverse(self, score_fn: ScoreFn, probability_flow: bool = False):
        return ReverseSDE(self, score_fn, probability_flow)


class ReverseSDE:
    """Reverse-time SDE, or probability-flow ODE when requested."""

    def __init__(self, forward_sde: SDE, score_fn: ScoreFn, probability_flow: bool):
        self.forward_sde = forward_sde
        self.score_fn = score_fn
        self.probability_flow = probability_flow
        self.N = forward_sde.N
        self.T = forward_sde.T

    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        drift, diffusion = self.forward_sde.sde(x, t)
        score = self.score_fn(x, t)
        factor = 0.5 if self.probability_flow else 1.0
        drift = drift - expand_like(diffusion.square(), x) * score * factor
        if self.probability_flow:
            diffusion = torch.zeros_like(diffusion)
        return drift, diffusion


class VPSDE(SDE):
    """Variance-preserving SDE, the continuous limit of DDPM."""

    def __init__(
        self, beta_min: float = 0.1, beta_max: float = 20.0, num_scales: int = 1000
    ):
        super().__init__(num_scales)
        if not 0.0 < beta_min < beta_max:
            raise ValueError("require 0 < beta_min < beta_max")
        if beta_max / num_scales >= 1.0:
            raise ValueError(
                "num_scales must exceed beta_max so every discrete VP beta is below 1"
            )
        self.beta_min = beta_min
        self.beta_max = beta_max

    @property
    def T(self) -> float:
        return 1.0

    def sde(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        drift = -0.5 * expand_like(beta_t, x) * x
        return drift, beta_t.sqrt()

    def marginal_prob(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        log_mean = (
            -0.25 * t.square() * (self.beta_max - self.beta_min)
            - 0.5 * t * self.beta_min
        )
        mean = expand_like(log_mean.exp(), x) * x
        std = (1.0 - (2.0 * log_mean).exp()).clamp_min(0.0).sqrt()
        return mean, std

    def prior_sampling(
        self, shape, *, device=None, dtype=None, generator=None
    ) -> Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)
