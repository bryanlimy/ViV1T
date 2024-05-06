from typing import Any, Dict, Literal, Tuple, Union

import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F

from viv1t.model.readout.readout import Readout, register


@register("factorized")
class FactorizedReadout(Readout):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: Union[np.ndarray, torch.Tensor],
        mean_responses: torch.Tensor = None,
        positive_weights: bool = False,
        normalize: bool = False,
        constrain_pos: bool = False,
        init_noise: float = 0.001,
        bias: bool = False,
    ):
        super(FactorizedReadout, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
        self.bias_mode = args.readout_bias_mode
        c, t, h, w = self.input_shape
        self.positive_weights = positive_weights
        self.constrain_pos = constrain_pos
        self.init_noise = init_noise
        self.normalize = normalize
        self.spatial_and_feature_reg_weight = 0.0

        self._original_features = True
        self.initialize_features()
        self.spatial = nn.Parameter(torch.Tensor(self.num_neurons, h, w))
        self.register_parameter(
            "bias", nn.Parameter(torch.Tensor(self.num_neurons)) if bias else None
        )
        self.dropout = nn.Dropout(p=args.readout_dropout)

        self.initialize(mean_responses=mean_responses)

    @property
    def shared_features(self):
        return self._features

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[self.feature_sharing_index, ...]
        else:
            return self._features

    @property
    def weight(self):
        if self.positive_weights:
            self.features.data.clamp_min_(0)
        n = self.num_neurons
        c, _, h, w = self.input_shape
        return self.normalized_spatial.view(n, 1, w, h) * self.features.view(n, c, 1, 1)

    @property
    def normalized_spatial(self):
        """
        Normalize the spatial mask
        """
        if self.normalize:
            norm = self.spatial.pow(2).sum(dim=1, keepdim=True)
            norm = norm.sum(dim=2, keepdim=True).sqrt().expand_as(self.spatial) + 1e-6
            weight = self.spatial / norm
        else:
            weight = self.spatial
        if self.constrain_pos:
            weight.data.clamp_min_(0)
        return weight

    def regularizer(self, reduction="sum", average=None):
        return (
            self.l1(reduction=reduction, average=average)
            * self.spatial_and_feature_reg_weight
        )

    def l1(self, reduction="sum", average=None):
        reduction = self.resolve_reduction_method(reduction=reduction, average=average)
        if reduction is None:
            raise ValueError("Reduction of None is not supported in this regularizer")

        n = self.num_neurons
        c, _, h, w = self.input_shape
        ret = (
            self.normalized_spatial.view(n, -1).abs().sum(dim=1, keepdim=True)
            * self.features.view(n, -1).abs().sum(dim=1)
        ).sum()
        if reduction == "mean":
            ret = ret / (n * c * w * h)
        return ret

    def initialize_bias(self, mean_responses: torch.Tensor = None):
        match self.bias_mode:
            case 0:
                bias = None
            case 1:
                bias = torch.zeros(self.num_neurons)
            case 2:
                if mean_responses is None:
                    bias = torch.zeros(self.num_neurons)
                else:
                    bias = mean_responses
            case _:
                raise NotImplementedError(
                    f"--bias_mode {self.bias_mode} not implemented."
                )
        self.register_parameter("bias", None if bias is None else nn.Parameter(bias))

    def initialize(self, mean_responses: torch.Tensor = None):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """
        self.spatial.data.normal_(0, self.init_noise)
        self._features.data.normal_(0, self.init_noise)
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.initialize_bias(mean_responses=mean_responses)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, _, h, w = self.input_shape
        if match_ids is not None:
            assert self.num_neurons == len(match_ids)
            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    n_match_ids,
                    c,
                ), f"shared features need to have shape ({n_match_ids}, {c})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = nn.Parameter(
                    torch.Tensor(n_match_ids, c)
                )  # feature weights for each channel of the core
            self.scales = nn.Parameter(
                torch.Tensor(self.num_neurons, 1)
            )  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            # feature weights for each channel of the core
            self._features = nn.Parameter(torch.Tensor(self.num_neurons, c))
            self._shared_features = False

    def forward(
        self,
        inputs: torch.Tensor,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        b, c, t, h, w = inputs.size()
        if self.constrain_pos:
            self.features.data.clamp_min_(0)
        outputs = rearrange(inputs, "b c t h w -> (b t) c h w")
        outputs = einsum(outputs, self.normalized_spatial, "b c h w, n h w -> b c n")
        outputs = self.dropout(outputs)
        outputs = einsum(outputs, self.features, "b c n, n c -> b n")
        if self.bias is not None:
            outputs = outputs + self.bias
        outputs = rearrange(outputs, "(b t) n -> b n t", b=b, t=t)
        return outputs
