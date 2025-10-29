from __future__ import annotations

import pytest

from probly.transformation.bayesian import torch
from probly.layers.torch import BayesLinear, BayesConv2d

from torch import nn


class TestNetworkArchitectures:

    # Test if linear layer can be transformed to bayesian linear layer and still hold the correct values
    def test_replace_linear_layer(self, torch_model_small_2d_2d: nn.Sequential) -> None:
        # dummy parameters
        first_layer = torch_model_small_2d_2d[0]
        use_base_weights = True
        posterior_std = 0.1
        prior_mean = 0.5
        prior_std = 1.0

        # transformation
        first_layer_bayesian = torch.replace_torch_bayesian_linear(first_layer, use_base_weights, posterior_std, prior_mean, prior_std)

        # checks
        assert first_layer_bayesian is not None
        assert isinstance(first_layer_bayesian, BayesLinear)
        assert first_layer_bayesian.obj == first_layer
        assert first_layer_bayesian.posterior_std == 0.1
        assert first_layer_bayesian.prior_mean == 0.5
        assert first_layer_bayesian.prior_std == 1.0

    # Test if conv2d layer can be transformed to bayesian conv2d layer and still hold the correct values
    def test_replace_conv2d_layer(self, torch_conv_linear_model: nn.Sequential) -> None:
        # dummy parameters
        first_layer = torch_conv_linear_model[0]
        use_base_weight = True
        posterior_std = 0.1
        prior_mean = 0.5
        prior_std = 1.0

        # transformation
        first_layer_bayesian = torch.replace_torch_bayesian_conv2d(first_layer, use_base_weight, posterior_std, prior_mean, prior_std)

        # checks
        assert first_layer_bayesian is not None
        assert isinstance(first_layer_bayesian, BayesConv2d)
        assert first_layer_bayesian.obj == first_layer
        assert first_layer_bayesian.posterior_std == 0.1
        assert first_layer_bayesian.prior_mean == 0.5
        assert first_layer_bayesian.prior_std == 1.0





    


