from __future__ import annotations

import torch

from SpaDiff.model import TopologyEncoder


def identity_operator(size: int):
    index = torch.arange(size, dtype=torch.long).repeat(2, 1)
    return torch.sparse_coo_tensor(
        index, torch.ones(size), (size, size)
    ).coalesce()


def test_topology_encoder_returns_normalized_order_attention():
    encoder = TopologyEncoder(
        input_dim=3,
        hidden_dim=5,
        output_dim=4,
        orders=(1, 2),
        steps=2,
        alpha=0.4,
        dropout=0.0,
        projection_dropout=0.0,
        residual=False,
        output_normalization="none",
    )
    features = torch.randn(6, 3)
    operator = identity_operator(6)
    output, attention = encoder(
        features, {1: operator, 2: operator}, return_attention=True
    )

    assert output.shape == (6, 4)
    assert attention.shape == (6, 2)
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(6))
    assert torch.all(attention >= 0.0)
    assert torch.all(attention <= 1.0)


def test_single_order_attention_is_one():
    encoder = TopologyEncoder(
        2,
        3,
        2,
        orders=(0,),
        steps=1,
        dropout=0.0,
        projection_dropout=0.0,
        residual=False,
        output_normalization="none",
    )
    operator = identity_operator(4)
    _, attention = encoder(
        torch.randn(4, 2), {0: operator}, return_attention=True
    )
    torch.testing.assert_close(attention, torch.ones_like(attention))
