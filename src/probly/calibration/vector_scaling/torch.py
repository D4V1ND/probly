"""Implementation For Vector Scaling With Torch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from probly.calibration.vector_scaling import common
from probly.utils.torch import torch_collect_outputs

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class TorchVectorScaling(nn.Module):
    """Torch Wrapper Class for Vector Scaling."""

    def __init__(self, base: nn.Module, num_classes: int, device: torch.device) -> None:
        """Set up the wrapper.

        Args:
            base: The base model that get wrapped and calibrated
            num_classes: The number of classes in the models output
            device: The torch device used for model calculations
        """
        super().__init__()
        self.base = base
        self.device = device
        self.w = nn.Parameter(torch.ones(num_classes, device=device))
        self.b = nn.Parameter(torch.zeros(num_classes, device=device))

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the model and scales the logits.

        Args:
            x: The input tensor for the prediction

        Returns:
            scaled_logits: The models output logits scaled with parameters w and b
        """
        logits = self.base(x)

        return logits * self.w + self.b

    def fit(self, calibration_set: DataLoader, lr: float = 0.01, max_iter: int = 50) -> None:
        """Optimizes the temperatures and biases.

        Args:
            calibration_set: The data set used for the optimization
            lr: The learning rate for the optimizer
            max_iter: The max iteration the optimizer can make
        """
        optim = torch.optim.LBFGS([self.w, self.b], lr=lr, max_iter=max_iter)
        loss_fn = nn.CrossEntropyLoss()
        logits, labels = torch_collect_outputs(self.base, calibration_set, self.device)
        labels = labels.long()

        def closure() -> Tensor:
            optim.zero_grad()
            scaled_logits = logits * self.w + self.b
            loss = loss_fn(scaled_logits, labels)
            loss.backward()

            return loss

        optim.step(closure=closure)

    def predict(self, x: Tensor) -> Tensor:
        """Make a prediction based on the input x.

        Args:
            x: The input tensor for the prediction

        Returns:
            probabilities: The models predicted calibrated probabilities
        """
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            scaled_logits = self.forward(x)
            return torch.softmax(scaled_logits, dim=1)


@common.register_vector_factory(nn.Module)
def _(_base: nn.Module, _num_classes: int, _device: torch.device) -> type[TorchVectorScaling]:
    return TorchVectorScaling(base=_base, num_classes=_num_classes, device=_device)
