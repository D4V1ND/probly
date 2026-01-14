"""Tests for vector scaling with torch."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from probly.calibration.vector_scaling.torch import TorchVectorScaling

SetupReturnType = tuple[nn.Module, DataLoader, Tensor, Tensor, torch.device]

@pytest.fixture
def setup(torch_custom_model: nn.Sequential) -> SetupReturnType:
    """Set up test data."""
    device = torch.device("cpu")
    base_model = torch_custom_model.to(device)

    inputs = torch.randn(20, 10)
    labels = torch.randint(0, 3, (20,))

    loader = DataLoader(TensorDataset(inputs, labels), batch_size=10)

    return base_model, loader, inputs, labels, device

def test_forward(setup) -> None:
    """Tests the forward pass of the Vector Scaling Implementation."""
    base, _, inputs, _, device = setup
    vs_model = TorchVectorScaling(base, 4, device)

    vs_model.w.data = torch.tensor([2.0, 3.0, 4.0, 1.0])
    vs_model.b.data = torch.tensor([1.0, 0.5, 2.0, 1.5])

    logits = base(inputs)
    scaled_logits = vs_model(inputs)
    expected_logits = logits * vs_model.w + vs_model.b

    assert torch.allclose(scaled_logits, expected_logits, atol=1e-5)
    assert logits.shape == scaled_logits.shape

def test_fit(setup) -> None:
    """Tests the optimization of the temperatures and biases."""
    base, dataloader, _, _, device = setup
    vs_model = TorchVectorScaling(base, 4, device)
    standard_w = vs_model.w.detach().clone()
    standard_b = vs_model.b.detach().clone()

    vs_model.fit(dataloader)
    optimized_w = vs_model.w
    optimized_b = vs_model.b

    assert not torch.equal(optimized_b, standard_b)
    assert not torch.equal(optimized_w, standard_w)

def test_predict(setup) -> None:
    """Tests the correct form of the prediction output."""
    base, _, inputs, _, device = setup
    vs_model = TorchVectorScaling(base, 4, device)
    probabilities = vs_model.predict(inputs)
    sums = probabilities.sum(dim=1)

    assert probabilities.shape == (20, 4)
    assert torch.allclose(sums, torch.ones_like(sums))
