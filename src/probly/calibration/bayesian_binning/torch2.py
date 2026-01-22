"""The BBQ Calibrator with Torch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from probly.calibration.bayesian_binning.common import register_bayesian_binning_factory

from .utils import betaln


class BayesianBinningQuantiles:
    """Calibrator using Bayesian Binning into Quantiles (BBQ)."""

    def __init__(self, max_bins: int = 10) -> None:
        """Initialize BBQ calibrator.

        Args:
            max_bins: Maximum number of bins to consider (tries 2 to max_bins)
        """
        self.max_bins = max_bins
        self.bin_edges: list[Tensor] = []
        self.system_bin_probs: list[Tensor] = []
        self.system_weights: Tensor | None = None
        self.is_fitted = False

    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> BayesianBinningQuantiles:
        """Fit the BBQ calibrator.

        Args:
            calibration_set: Predicted probabilities, shape (n_samples,)
            truth_labels: True binary labels (0 or 1), shape (n_samples,)

        Returns:
            self for method chaining
        """
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have same length"
            raise ValueError(msg)
        if calibration_set.shape[0] == 0:
            msg = "calibration_set cannot be empty"
            raise ValueError(msg)

        self.system_bin_probs = []
        self.bin_edges = []
        system_log_scores = []

        for num_bins in range(2, self.max_bins + 1):
            # Quantile-based bin edges
            edges = torch.quantile(calibration_set, torch.linspace(0, 1, num_bins + 1))
            edges[0] = 0.0
            edges[-1] = 1.0
            self.bin_edges.append(edges)

            # Assign samples to bins
            bin_ids = torch.bucketize(calibration_set, edges) - 1
            bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

            # Vectorized bin counting
            bin_counts = torch.bincount(bin_ids, minlength=num_bins)
            bin_positives = torch.bincount(
                bin_ids,
                weights=truth_labels.float(),
                minlength=num_bins,
            ).long()

            # Bayesian smoothed probabilities (Beta(1,1) prior)
            # Works for all bins including empty ones
            bin_probs = (bin_positives.float() + 1.0) / (bin_counts.float() + 2.0)
            self.system_bin_probs.append(bin_probs)

            # Compute marginal likelihood for each bin
            # log P(D_j | S=s) = log B(k+1, n-k+1) - log B(1, 1)
            k = bin_positives.float()
            n = bin_counts.float()

            log_bin_scores = betaln(k + 1.0, n - k + 1.0) - betaln(
                torch.tensor(1.0, dtype=torch.float32),
                torch.tensor(1.0, dtype=torch.float32),
            )

            # System score: product of bin scores (sum in log-space)
            system_log_score = log_bin_scores.sum()
            system_log_scores.append(system_log_score)

        # Normalize using softmax (numerically stable)
        log_scores = torch.stack(system_log_scores)
        self.system_weights = torch.softmax(log_scores, dim=0)

        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions.

        Args:
            predictions: Predicted probabilities to calibrate, shape (n_samples,)

        Returns:
            Calibrated probabilities, shape (n_samples,)
        """
        if not self.is_fitted:
            msg = "Calibrator must be fitted before prediction"
            raise RuntimeError(msg)

        # Vectorized prediction
        calibrated = torch.zeros_like(predictions, dtype=torch.float32)

        for sys_idx, edges in enumerate(self.bin_edges):
            bin_probs = self.system_bin_probs[sys_idx]

            if self.system_weights is None:
                msg = "Calibrator must be fitted before prediction"
                raise RuntimeError(msg)
            weight = self.system_weights[sys_idx]

            # Find bin for all predictions at once
            bin_indices = torch.bucketize(predictions, edges) - 1
            bin_indices = torch.clamp(bin_indices, 0, len(bin_probs) - 1)

            # Add weighted contribution from this system
            calibrated += weight * bin_probs[bin_indices]

        return calibrated


@register_bayesian_binning_factory(nn.Module)
def _(_base: nn.Module, _device: object) -> type[BayesianBinningQuantiles]:
    return BayesianBinningQuantiles
