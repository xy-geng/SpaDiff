

from __future__ import annotations


from typing import Optional

import torch
from torch import Tensor

from .diffusion import make_score_fn
from .sde import SDE



@torch.no_grad()
def probability_flow_sample(
    model,
    sde: SDE,
    topology: Tensor,
    batch_ids: Tensor,
    modality_ids: Tensor,
    feature_dim: int,
    *,
    steps: Optional[int] = None,
    guidance_scale: float = 1.0,
    guidance_target: str = "all",
    eps: float = 1e-3,
    initial: Optional[Tensor] = None,
    start_time: float = 1.0,
) -> Tensor:
    """Deterministic Heun solver for the probability-flow ODE.

    This avoids adding SciPy/torchdiffeq as a mandatory dependency. It is a
    fixed-step reference implementation; use an adaptive ODE solver for
    likelihood work.
    """
    score_fn = make_score_fn(
        model,
        sde,
        topology,
        batch_ids,
        modality_ids,
        guidance_scale=guidance_scale,
        guidance_target=guidance_target,
    )
    rsde = sde.reverse(score_fn, probability_flow=True)
    x = (
        sde.prior_sampling(
            (topology.shape[0], feature_dim),
            device=topology.device,
            dtype=topology.dtype,
        )
        if initial is None
        else initial.clone()
    )
    n_steps = steps or max(1, round(sde.N * start_time))
    grid = torch.linspace(start_time, eps, n_steps + 1, device=x.device, dtype=x.dtype)
    for index in range(n_steps):
        t0, t1 = grid[index], grid[index + 1]
        dt = t1 - t0
        vec0 = t0.expand(x.shape[0])
        drift0 = rsde.sde(x, vec0)[0]
        proposal = x + drift0 * dt
        vec1 = t1.expand(x.shape[0])
        drift1 = rsde.sde(proposal, vec1)[0]
        x = x + 0.5 * (drift0 + drift1) * dt
    return x
