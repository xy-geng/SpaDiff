"""Backward-compatible wrapper for the former pipeline API."""

from __future__ import annotations

import warnings


class SpaDiffPipeline:
    """Deprecated adapter that forwards workflow calls to :class:`SpaDiff`.

    New code should call ``model.fit_transform(...)`` directly. Keeping this
    small adapter prevents existing scripts from breaking while maintaining a
    single implementation of the high-level workflow on the model itself.
    """

    def __init__(self, model):
        if not hasattr(model, "fit_transform"):
            raise TypeError("model must provide a fit_transform method")
        warnings.warn(
            "SpaDiffPipeline is deprecated; call SpaDiff.fit_transform "
            "directly instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model

    def fit_transform(self, *args, **kwargs):
        """Forward to the wrapped model's unified workflow."""

        return self.model.fit_transform(*args, **kwargs)

    @property
    def training_result_(self):
        return self.model.training_result_

    @property
    def batch_categories_(self):
        return self.model.batch_categories_

    @property
    def modality_categories_(self):
        return self.model.modality_categories_

    @property
    def reference_batch_(self):
        return self.model.reference_batch_
