"""Flax Implementations for Isotonic Regression."""

from __future__ import annotations

from jax import Array
import jax.numpy as jnp


class IsotonicRegression:
    """Calibrator that uses isotonic regression."""

    def __init__(self) -> None:
        """Set up class variables."""
        super().__init__()
        self.bin_boundaries: Array | None = None
        self.bin_probs: Array | None = None
        self.is_fitted = False

    def fit(self, y_pred: Array, y_binary: Array) -> IsotonicRegression:
        """Fit the regression function to the model outputs.

        Args:
            y_pred: The set that should be used for the calibration
            y_binary: The true labelxs

        """
        self.bin_probs = self._pav(y_pred, y_binary)
        boundaries_list = [
            y_pred[i] for i in range(1, len(self.bin_probs)) if self.bin_probs[i] != self.bin_probs[i - 1]
        ]
        self.bin_boundaries = jnp.array(boundaries_list)
        self.is_fitted = True
        return self

    def predict(self, predictions: Array) -> Array:
        """Make calibrated predictions on the input x.

        Args:
            predictions: The input for the model to make predictions on

        Returns:
            calibrated: The calibrated probabilities for the prediction

        """
        if not self.is_fitted:
            msg = "Calibrator must be fitted before Calibration"
            raise ValueError(msg)

        if self.bin_boundaries is None:
            msg = "IsotonicRegression is not fitted"
            raise RuntimeError(msg)

        if self.bin_probs is None:
            msg = "IsotonicRegression is not fitted"
            raise RuntimeError(msg)

        calibrated = []

        for pred in predictions:
            bin_id = jnp.searchsorted(self.bin_boundaries, pred)
            calibrated.append(jnp.unique(self.bin_probs)[bin_id])

        return jnp.array(calibrated)

    def _pav(self, x: Array, y: Array) -> Array:
        xy = jnp.array([x, y])
        ind = jnp.argsort(xy[0, :])
        xy_xsorted = xy[:, ind]
        if jnp.all(xy_xsorted[1][:-1] <= xy_xsorted[1][1:]):
            return xy_xsorted[1]
        for i in range(1, len(x)):
            if xy_xsorted[1, i] < xy_xsorted[1, i - 1]:
                mean = (xy_xsorted[1, i] + xy_xsorted[1, i - 1]) / 2
                xy_xsorted = xy_xsorted.at[1, i].set(mean)
                xy_xsorted = xy_xsorted.at[1, i - 1].set(mean)
        return self._pav(xy_xsorted[0], xy_xsorted[1])
