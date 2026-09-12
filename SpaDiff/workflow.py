"""Internal AnnData workflow mixed into the public SpaDiff model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .train import train_paired_multiomics, train_spadiff


class SpaDiffWorkflowMixin:

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

    @staticmethod
    def _validate_paired_observations(
        adata_rna,
        adata_atac,
        *,
        spatial_key: str,
        coordinate_rtol: float,
        coordinate_atol: float,
    ) -> None:
        if adata_rna.n_obs != adata_atac.n_obs:
            raise ValueError(
                "paired RNA and ATAC objects must contain the same number of spots"
            )
        if adata_rna.n_obs < 2:
            raise ValueError("paired multi-omics integration requires at least two spots")
        if not np.array_equal(
            np.asarray(adata_rna.obs_names),
            np.asarray(adata_atac.obs_names),
        ):
            raise ValueError(
                "paired RNA and ATAC obs_names must be identical and in the same order"
            )
        for name, adata in (("RNA", adata_rna), ("ATAC", adata_atac)):
            if spatial_key not in adata.obsm:
                raise KeyError(f"{name} adata.obsm does not contain {spatial_key!r}")
        rna_spatial = np.asarray(adata_rna.obsm[spatial_key], dtype=np.float64)
        atac_spatial = np.asarray(adata_atac.obsm[spatial_key], dtype=np.float64)
        if rna_spatial.shape != atac_spatial.shape:
            raise ValueError(
                "paired RNA and ATAC spatial coordinates must have identical shapes"
            )
        if not np.isfinite(rna_spatial).all() or not np.isfinite(atac_spatial).all():
            raise ValueError("paired spatial coordinates must contain only finite values")
        if not np.allclose(
            rna_spatial,
            atac_spatial,
            rtol=coordinate_rtol,
            atol=coordinate_atol,
        ):
            maximum = float(np.max(np.abs(rna_spatial - atac_spatial)))
            raise ValueError(
                "paired RNA and ATAC spatial coordinates differ; "
                f"maximum absolute difference={maximum:.6g}"
            )

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

    def fit_transform_multiomics(
        self,
        adata_rna,
        adata_atac,
        rna_features,
        atac_features,
        operators,
        *,
        spatial_key: str = "spatial",
        coordinate_rtol: float = 1e-5,
        coordinate_atol: float = 1e-5,
        copy: bool = False,
        rna_key: str = "H_rna",
        atac_key: str = "H_atac",
        joint_key: str = "spadiff_joint",
        compatibility_key: str = "spadiff",
        epochs: int = 500,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: Optional[float] = 1.0,
        ema_decay: Optional[float] = 0.990,
        progress: bool = True,
    ):
        """Fit paired RNA--ATAC views and return their spot-wise mean embedding.
        """

        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        if coordinate_rtol < 0.0 or coordinate_atol < 0.0:
            raise ValueError("coordinate tolerances must be non-negative")
        embedding_keys = (rna_key, atac_key, joint_key, compatibility_key)
        if any(not isinstance(key, str) or not key for key in embedding_keys):
            raise ValueError("embedding keys must be non-empty strings")
        if len(set(embedding_keys)) != len(embedding_keys):
            raise ValueError("paired multi-omics embedding keys must be distinct")
        if self.config.num_batches != 1:
            raise ValueError(
                "this paired single-dataset workflow requires num_batches=1"
            )
        if self.config.num_modalities != 2:
            raise ValueError(
                "paired RNA-ATAC integration requires num_modalities=2"
            )
        if self.config.data_dim != self.config.condition_input_dim:
            raise ValueError(
                "paired RNA-ATAC integration requires data_dim to equal "
                "condition_input_dim"
            )

        self._validate_paired_observations(
            adata_rna,
            adata_atac,
            spatial_key=spatial_key,
            coordinate_rtol=coordinate_rtol,
            coordinate_atol=coordinate_atol,
        )
        self._reset_workflow_state()
        output = adata_rna.copy() if copy else adata_rna
        device, dtype = self._model_device_and_dtype()
        rna = self._as_feature_tensor(
            rna_features,
            name="rna_features",
            n_obs=adata_rna.n_obs,
            width=self.config.data_dim,
            device=device,
            dtype=dtype,
        )
        atac = self._as_feature_tensor(
            atac_features,
            name="atac_features",
            n_obs=adata_atac.n_obs,
            width=self.config.data_dim,
            device=device,
            dtype=dtype,
        )
        prepared_operators = self._prepare_operators(
            operators,
            n_obs=adata_rna.n_obs,
            device=device,
            dtype=dtype,
        )
        batch_ids = torch.zeros(adata_rna.n_obs, dtype=torch.long, device=device)

        training = train_paired_multiomics(
            self,
            rna,
            atac,
            prepared_operators,
            batch_ids,
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
            self.eval()
            with torch.no_grad():
                topology_rna, topology_atac, topology_joint = (
                    self.encode_paired_multiomics(
                        rna,
                        atac,
                        prepared_operators,
                    )
                )
        finally:
            if training.ema is not None:
                training.ema.restore(self.parameters())

        rna_values = topology_rna.detach().cpu().numpy()
        atac_values = topology_atac.detach().cpu().numpy()
        joint_values = topology_joint.detach().cpu().numpy()
        output.obsm[rna_key] = rna_values
        output.obsm[atac_key] = atac_values
        output.obsm[joint_key] = joint_values
        output.obsm[compatibility_key] = joint_values.copy()
        output.uns["spadiff_multiomics"] = {
            "modalities": np.asarray(("RNA", "ATAC"), dtype=str),
            "n_paired_spots": int(output.n_obs),
            "rna_key": rna_key,
            "atac_key": atac_key,
            "joint_key": joint_key,
            "joint_formula": f"0.5 * ({rna_key} + {atac_key})",
            "dsm_weight": float(self.config.dsm_weight),
            "batch_alignment_weight": float(
                self.config.batch_alignment_weight
            ),
            "prior_kl_weight": float(self.config.prior_kl_weight),
        }
        self.training_result_ = training
        self.batch_categories_ = (0,)
        self.modality_categories_ = ("RNA", "ATAC")
        self.reference_batch_ = 0
        return output
