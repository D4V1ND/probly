"""Basic Template for Calibration Methods with Torch and Flax."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from torch import Tensor, device as TorchDevice, nn


class CalibratorBaseTorch(ABC):
    """Abstract base class for calibrators with torch."""

    def __init__(self, base_model: nn.Module, device: TorchDevice) -> None:
        """Create a calibrator.

        Args:
            base_model: The base model whose outputs are to be calibrated.
            device: Torch device or a device string (e.g. 'cpu', 'cuda:0').
        """
        self.model = base_model
        self.device = device

    @abstractmethod
    def fit(self, calibration_set: Tensor, truth_labels: Tensor) -> Self:
        """Fit calibrator from calibration_set (DataLoader-like)."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        """Return calibrated probabilities for input x."""
        raise NotImplementedError


"""WILL BE IMPLEMENTED LATER"""
class CalibratorBaseFlax(ABC):
    """Abstract base class for calibrators with flax."""

    @abstractmethod
    def fit(self, calibration_set: Tensor) -> Self:
        """Fit calibrator from calibration_set (DataLoader-like)."""
        raise NotImplementedError
