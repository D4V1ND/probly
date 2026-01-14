"""The BBQ Calibrator with Torch."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.special import betaln

from probly.calibration.template import CalibratorBaseTorch


class BayesianBinningQuantiles(CalibratorBaseTorch):
    """Calibrator using Bayesian Binning into Quantiles (BBQ)."""

    def __init__(self, max_bins: int = 10, device: torch.device = torch.device("cpu")) -> None:  # noqa: B008, D107
        # Pass a dummy model if you don't need one
        super().__init__(base_model=nn.Identity(), device=device)
        self.max_bins = max_bins
        self.bin_edges: list[Tensor] = []  # type: ignore  # noqa: PGH003
        self.system_bin_probs: list[Tensor] = []  # type: ignore  # noqa: PGH003
        self.system_scores: list[float] = []
        self.system_weights: list[float] = []
        self.is_fitted = False

    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> BayesianBinningQuantiles:
        """Fit the BBQ calibrator."""
        if calibration_set.shape[0] != truth_labels.shape[0]:
            msg = "calibration_set and truth_labels must have same length"
            raise ValueError(msg)
        if calibration_set.shape[0] == 0:
            msg = "calibration_set cannot be empty"
            raise ValueError(msg)

        self.system_bin_probs = []
        self.system_scores = []
        self.bin_edges = []

        for num_bins in range(2, self.max_bins + 1):
            # Quantile-based bin edges
            edges = torch.quantile(calibration_set, torch.linspace(0, 1, num_bins + 1))
            edges[0] = 0.0
            edges[-1] = 1.0
            self.bin_edges.append(edges)

            # Compute n and k for each bin
            bin_counts = torch.zeros(num_bins, dtype=torch.int64)
            bin_positives = torch.zeros(num_bins, dtype=torch.int64)

            # Assign each sample to a bin
            bin_ids = torch.bucketize(calibration_set, edges) - 1
            bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

            for idx, bin_id in enumerate(bin_ids):
                bin_counts[bin_id] += 1
                bin_positives[bin_id] += truth_labels[idx].item()

            # Bayesian smoothed probabilities
            bin_probs = torch.zeros(num_bins, dtype=torch.float32)
            for i in range(num_bins):
                if bin_counts[i] > 0:
                    bin_probs[i] = (bin_positives[i].float() + 1.0) / (bin_counts[i].float() + 2.0)
                else:
                    bin_probs[i] = 0.5  # neutral for empty bins

            self.system_bin_probs.append(bin_probs)

            # Compute bin scores
            log_bin_scores = torch.zeros(num_bins)
            for i in range(num_bins):
                k = bin_positives[i].item()
                n = bin_counts[i].item()
                log_bin_scores[i] = betaln(k + 1, n - k + 1) - betaln(1, 1)  # log-space

            # System score = product of bin scores (in log-space)
            system_log_score = log_bin_scores.sum()
            system_score = torch.exp(system_log_score).item()
            self.system_scores.append(system_score)

        # Normalize system scores to weights
        total_score = sum(self.system_scores)
        self.system_weights = [s / total_score for s in self.system_scores]
        self.is_fitted = True
        return self

    def predict(self, predictions: Tensor) -> Tensor:
        """Return calibrated probabilities for input predictions."""
        if not self.is_fitted:
            msg = "Calibrator must be fitted before prediction"
            raise RuntimeError(msg)

        calibrated = torch.zeros_like(predictions, dtype=torch.float32)

        for i, pred in enumerate(predictions):
            calibrated_prob = 0.0
            for sys_idx, edges in enumerate(self.bin_edges):
                bin_probs = self.system_bin_probs[sys_idx]
                weight = self.system_weights[sys_idx]
                bin_idx = torch.bucketize(pred, edges) - 1
                bin_idx = torch.clamp(bin_idx, 0, len(bin_probs) - 1)
                calibrated_prob += weight * bin_probs[bin_idx]
            calibrated[i] = calibrated_prob

        return calibrated
