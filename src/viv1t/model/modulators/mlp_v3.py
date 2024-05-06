from typing import Any, Tuple

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from torch import nn
from torch.nn import functional as F

from viv1t.model.modulators.modulator import Modulator, register


@register("mlp-v3")
class MLP3Modulator(Modulator):
    def __init__(self, args: Any, input_shape: Tuple[int, int]):
        super(MLP3Modulator, self).__init__(args, input_shape)
        match args.modulator_activation:
            case "identity":
                activation = nn.Identity()
            case "sigmoid":
                activation = nn.Sigmoid()
            case "tanh":
                activation = nn.Tanh()
            case _:
                raise NotImplementedError(
                    f"--modulator_activation {args.modulator_activation} not implemented."
                )
        self.modulator = nn.Sequential(
            nn.Linear(in_features=4, out_features=input_shape[0], bias=False),
            activation,
        )

    def regularizer(self):
        return 0.0

    def forward(
        self,
        responses: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, n, t = responses.shape
        states = torch.cat((behaviors, pupil_centers), dim=1)
        states = rearrange(states[:, :, -t:], "b d t -> b t d")
        states = self.modulator(states)
        outputs = responses + rearrange(states, "b t n -> b n t")
        return outputs
