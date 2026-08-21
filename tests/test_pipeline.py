from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

import SpaDiff as sd


def small_config():
    return sd.SpaDiffConfig(
        data_dim=3,
        condition_input_dim=3,
        hidden_dim=8,
        topology_hidden_dim=8,
        topology_dim=3,
        technical_hidden_dim=8,
        time_embedding_dim=8,
        condition_embedding_dim=4,
        score_depth=1,
        dropout=0.0,
        topology_projection_dropout=0.0,
        simplex_orders=(0,),
        propagation_steps=1,
        topology_output_normalization="none",
        num_batches=2,
        num_modalities=1,
        prior_kl_weight=0.0,
        num_scales=32,
    )


def identity_operator(n):
    indices = torch.arange(n, dtype=torch.long).repeat(2, 1)
    values = torch.ones(n)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def manual_workflow(model, features, operators):
    batch_ids = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    modality_ids = torch.zeros(6, dtype=torch.long)
    training = sd.train_spadiff(
        model,
        features,
        operators,
        batch_ids,
        modality_ids,
        epochs=1,
        ema_decay=0.990,
        progress=False,
    )
    reference_ids = torch.zeros_like(batch_ids)
    training.ema.store(model.parameters())
    training.ema.copy_to(model.parameters())
    try:
        harmonized = model.harmonize(
            features,
            operators,
            reference_ids,
            modality_ids,
            strength=0.10,
            guidance_scale=1.0,
            ode_steps=2,
        )
        model.eval()
        with torch.no_grad():
            topology = model.encode_condition(features, operators)
    finally:
        training.ema.restore(model.parameters())
    return topology.numpy(), harmonized.numpy()


def test_model_workflow_matches_manual_workflow_and_copy_semantics(multislice_adata):
    adata = multislice_adata.copy()
    adata.obs = adata.obs.iloc[:6].copy()
    adata.obs["batch_name"] = ["a", "a", "a", "b", "b", "b"]
    adata.obsm["spatial"] = adata.obsm["spatial"][:6]
    adata.n_obs = 6
    features = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.4, 0.1],
            [0.8, 0.7, 0.9],
            [0.9, 0.8, 0.7],
            [0.7, 0.9, 0.8],
        ],
        dtype=torch.float32,
    )
    operators = {0: identity_operator(6)}

    torch.manual_seed(7)
    manual_model = sd.SpaDiff(small_config())
    workflow_model = deepcopy(manual_model)
    torch.manual_seed(123)
    expected_topology, expected_harmonized = manual_workflow(
        manual_model, features, operators
    )

    torch.manual_seed(123)
    output = workflow_model.fit_transform(
        adata,
        features,
        operators,
        batch_order=["a", "b"],
        reference_batch="a",
        copy=True,
        epochs=1,
        progress=False,
        ode_steps=2,
    )

    assert output is not adata
    assert "spadiff" not in adata.obsm
    np.testing.assert_allclose(output.obsm["spadiff"], expected_topology, atol=1e-6)
    np.testing.assert_allclose(
        output.obsm["X_spadiff"], expected_harmonized, atol=1e-6
    )
    assert workflow_model.training_result_ is not None
    assert workflow_model.batch_categories_ == ("a", "b")
    assert workflow_model.modality_categories_ == (0,)
    assert workflow_model.reference_batch_ == "a"


def test_model_workflow_validates_model_category_counts(multislice_adata):
    model = sd.SpaDiff(small_config())
    features = torch.zeros((multislice_adata.n_obs, 3))
    operators = {0: identity_operator(multislice_adata.n_obs)}
    try:
        model.fit_transform(
            multislice_adata,
            features,
            operators,
            batch_order=["slice_a", "slice_b", "slice_c"],
            epochs=1,
            progress=False,
            ode_steps=1,
        )
    except ValueError as error:
        assert "num_batches" in str(error)
    else:
        raise AssertionError("expected a num_batches validation error")


def test_single_batch_workflow_and_deprecated_adapter(multislice_adata):
    config = small_config()
    config.num_batches = 1
    model = sd.SpaDiff(config)
    features = torch.zeros((multislice_adata.n_obs, 3))
    operators = {0: identity_operator(multislice_adata.n_obs)}

    with pytest.deprecated_call(match="SpaDiffPipeline is deprecated"):
        adapter = sd.SpaDiffPipeline(model)

    output = adapter.fit_transform(
        multislice_adata,
        features,
        operators,
        batch_key=None,
        epochs=1,
        progress=False,
        ode_steps=1,
    )
    assert output is multislice_adata
    assert adapter.training_result_ is model.training_result_
    assert model.batch_categories_ == (0,)
