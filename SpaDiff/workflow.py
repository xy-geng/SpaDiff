"""Internal AnnData workflow mixed into the public SpaDiff model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .train import train_spadiff


class SpaDiffWorkflowMixin:
    """High-level fit/embedding orchestration for a SpaDiff model instance."""

    def _reset_workflow_state(self) -> None:
        self.training_result_ = None
        self.batch_categories_ = None
        self.modality_categories_ = None
        self.reference_batch_ = None

    def _model_device_and_dtype(self):
        try:
            parameter = next(self.parameters())
        except StopIteration:
            return torch.device("cpu"), torch.float32
        return parameter.device, parameter.dtype

    @staticmethod
    def _ordered_categories(values, requested_order, *, label: str) -> list:
        if hasattr(values, "isna") and bool(values.isna().any()):
            raise ValueError(f"{label} labels contain missing values")
        observed = list(dict.fromkeys(values.tolist()))
        if requested_order is None:
            return observed
        requested = list(requested_order)
        if not requested:
            raise ValueError(f"{label}_order must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError(f"{label}_order must not contain duplicates")
        if set(requested) != set(observed):
            missing = [value for value in observed if value not in requested]
            unknown = [value for value in requested if value not in observed]
            raise ValueError(
                f"{label}_order must contain every observed label exactly once; "
                f"missing={missing}, unknown={unknown}"
            )
        return requested

    @classmethod
    def _encode_labels(
        cls,
        adata,
        *,
        key: str,
        requested_order: Optional[Sequence],
        label: str,
        device,
    ) -> tuple[Tensor, tuple]:
        if key not in adata.obs:
            raise KeyError(f"adata.obs does not contain {key!r}")
        values = adata.obs[key]
        categories = cls._ordered_categories(
            values, requested_order, label=label
        )
        categorical = pd.Categorical(values, categories=categories, ordered=True)
        codes = np.asarray(categorical.codes, dtype=np.int64)
        if np.any(codes < 0):
            raise ValueError(f"failed to encode every {label} label")
        return torch.as_tensor(codes, dtype=torch.long, device=device), tuple(
            categories
        )

    @staticmethod
    def _as_feature_tensor(
        values,
        *,
        name: str,
        n_obs: int,
        width: int,
        device,
        dtype,
    ):
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        if tensor.ndim != 2 or tensor.shape != (n_obs, width):
            raise ValueError(
                f"{name} must have shape [{n_obs}, {width}], "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        return tensor

    def _prepare_operators(self, operators, *, n_obs: int, device, dtype):
        if not isinstance(operators, Mapping):
            raise TypeError("operators must be an order-keyed mapping")
        prepared = {}
        for order in self.config.simplex_orders:
            if order not in operators:
                raise KeyError(f"operators does not contain simplex order {order}")
            operator = operators[order]
            if not isinstance(operator, Tensor):
                raise TypeError(f"operator {order} must be a torch.Tensor")
            if tuple(operator.shape) != (n_obs, n_obs):
                raise ValueError(
                    f"operator {order} must have shape [{n_obs}, {n_obs}]"
                )
            prepared[order] = operator.to(device=device, dtype=dtype)
        return prepared

    def fit_transform(
        self,
        adata,
        features,
        operators,
        *,
        condition_features=None,
        batch_key: Optional[str] = "batch_name",
        batch_order: Optional[Sequence] = None,
        modality_key: Optional[str] = None,
        modality_order: Optional[Sequence] = None,
        reference_batch=None,
        copy: bool = False,
        topology_key: str = "spadiff",
        harmonized_key: str = "X_spadiff",
        epochs: int = 500,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: Optional[float] = 1.0,
        ema_decay: Optional[float] = 0.990,
        progress: bool = True,
        strength: float = 0.10,
        guidance_scale: float = 1.0,
        ode_steps: Optional[int] = 300,
    ):
        """Fit this model and return AnnData containing both embeddings.

        This method intentionally does not replace ``nn.Module.train``. It
        owns the full user-facing optimization workflow while the existing
        ``loss``, ``harmonize`` and ``encode_condition`` methods remain usable
        as low-level building blocks.
        """

        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        if not topology_key or not harmonized_key:
            raise ValueError("embedding keys must be non-empty strings")
        if topology_key == harmonized_key:
            raise ValueError("topology_key and harmonized_key must be different")
        self._reset_workflow_state()
        output = adata.copy() if copy else adata
        device, dtype = self._model_device_and_dtype()
        target = self._as_feature_tensor(
            features,
            name="features",
            n_obs=output.n_obs,
            width=self.config.data_dim,
            device=device,
            dtype=dtype,
        )
        condition = (
            None
            if condition_features is None
            else self._as_feature_tensor(
                condition_features,
                name="condition_features",
                n_obs=output.n_obs,
                width=self.config.condition_input_dim,
                device=device,
                dtype=dtype,
            )
        )
        if condition is None and (
            self.config.condition_input_dim != self.config.data_dim
        ):
            raise ValueError(
                "condition_features is required when condition_input_dim differs "
                "from data_dim"
            )
        prepared_operators = self._prepare_operators(
            operators, n_obs=output.n_obs, device=device, dtype=dtype
        )

        if batch_key is None:
            if batch_order is not None:
                raise ValueError("batch_order requires batch_key")
            batch_categories = (0,)
            batch_ids = torch.zeros(output.n_obs, dtype=torch.long, device=device)
        else:
            batch_ids, batch_categories = self._encode_labels(
                output,
                key=batch_key,
                requested_order=batch_order,
                label="batch",
                device=device,
            )
        if len(batch_categories) != self.config.num_batches:
            raise ValueError(
                "model num_batches does not match the observed batch categories: "
                f"configured={self.config.num_batches}, "
                f"observed={len(batch_categories)}"
            )

        if modality_key is None:
            if modality_order is not None:
                raise ValueError("modality_order requires modality_key")
            modality_categories = (0,)
            modality_ids = torch.zeros(
                output.n_obs, dtype=torch.long, device=device
            )
        else:
            modality_ids, modality_categories = self._encode_labels(
                output,
                key=modality_key,
                requested_order=modality_order,
                label="modality",
                device=device,
            )
        if len(modality_categories) != self.config.num_modalities:
            raise ValueError(
                "model num_modalities does not match the observed modality "
                f"categories: configured={self.config.num_modalities}, "
                f"observed={len(modality_categories)}"
            )

        selected_reference = (
            batch_categories[0] if reference_batch is None else reference_batch
        )
        if selected_reference not in batch_categories:
            raise ValueError(
                f"reference_batch {selected_reference!r} is not an observed batch"
            )
        reference_code = batch_categories.index(selected_reference)
        reference_ids = torch.full_like(batch_ids, reference_code)

        training = train_spadiff(
            self,
            target,
            prepared_operators,
            batch_ids,
            modality_ids,
            condition_features=condition,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            ema_decay=ema_decay,
            progress=progress,
        )

        if training.ema is not None:
            training.ema.store(self.parameters())
            training.ema.copy_to(self.parameters())
        try:
            harmonized = self.harmonize(
                observed_features=target,
                operators=prepared_operators,
                reference_batch_ids=reference_ids,
                modality_ids=modality_ids,
                condition_features=condition,
                strength=strength,
                guidance_scale=guidance_scale,
                ode_steps=ode_steps,
            )
            source = target if condition is None else condition
            self.eval()
            with torch.no_grad():
                topology = self.encode_condition(source, prepared_operators)
        finally:
            if training.ema is not None:
                training.ema.restore(self.parameters())

        output.obsm[topology_key] = topology.detach().cpu().numpy()
        output.obsm[harmonized_key] = harmonized.detach().cpu().numpy()
        self.training_result_ = training
        self.batch_categories_ = batch_categories
        self.modality_categories_ = modality_categories
        self.reference_batch_ = selected_reference
        return output
