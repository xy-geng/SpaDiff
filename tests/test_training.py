from __future__ import annotations

import inspect

import torch

from SpaDiff import SpaDiff, SpaDiffConfig
from SpaDiff.train import train_spadiff


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def loss(self, *args, **kwargs):
        loss = self.weight.square()
        zero = loss * 0.0
        return {
            "loss": loss,
            "dsm_loss": loss,
            "batch_loss": zero,
            "batch_alignment_loss": zero,
            "batch_posterior_loss": zero,
            "prior_kl_loss": zero,
            "weighted_dsm_loss": loss,
            "weighted_batch_loss": zero,
            "weighted_prior_kl_loss": zero,
            "noise_mse": zero,
            "score_mse": zero,
            "posterior_accuracy": zero,
            "topology_batch_accuracy": zero,
        }


def run_dummy(*, progress):
    model = DummyModel()
    features = torch.zeros((2, 1))
    labels = torch.zeros(2, dtype=torch.long)
    return train_spadiff(
        model,
        features,
        {},
        labels,
        labels,
        epochs=2,
        ema_decay=None,
        progress=progress,
    )


def test_training_progress_does_not_print_losses(capsys):
    result = run_dummy(progress=True)
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "Training SpaDiff" in combined
    assert "dsm=" not in combined
    assert "align=" not in combined
    assert len(result.losses) == 2
    assert len(result.diagnostics) == 2


def test_training_can_be_silent(capsys):
    run_dummy(progress=False)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_reference_defaults_match_the_tutorial():
    config = SpaDiffConfig()
    assert config.topology_hidden_dim == 128
    assert config.propagation_alpha == 0.4
    assert config.topology_projection_dropout == 0.0
    assert config.batch_alignment_weight == 0.5
    assert config.prior_kl_weight == 1.0

    train_signature = inspect.signature(train_spadiff)
    assert train_signature.parameters["epochs"].default == 500
    assert train_signature.parameters["ema_decay"].default == 0.990
    harmonize_signature = inspect.signature(SpaDiff.harmonize)
    assert harmonize_signature.parameters["strength"].default == 0.10
    assert harmonize_signature.parameters["ode_steps"].default == 300
