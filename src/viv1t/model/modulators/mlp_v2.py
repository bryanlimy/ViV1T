from typing import Any, Tuple

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from torch import nn
from torch.nn import functional as F

from viv1t.model.modulators.modulator import Modulator, register


@register("mlp-v2")
class MLP2Modulator(Modulator):
    def __init__(self, args: Any, input_shape: Tuple[int, int]):
        super(MLP2Modulator, self).__init__(args, input_shape)
        self.h_size = args.modulator_history_size
        self.num_behaviors = 4
        in_features = (self.h_size + 1) * (self.num_behaviors + 1)
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=in_features // 2, bias=True
            ),
            nn.GELU(),
            nn.Dropout(p=args.modulator_history_dropout),
            nn.Linear(in_features=in_features // 2, out_features=1, bias=True),
        )

    def regularizer(self):
        return 0.0

    def unfold(self, inputs: torch.Tensor):
        """zero pad the input and unfold it to get the history of self.h_size"""
        b, d, t = inputs.shape
        return torch.cat(
            (inputs.new_zeros((b, d, self.h_size)), inputs), dim=-1
        ).unfold(dimension=-1, size=self.h_size + 1, step=1)

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, n, t = inputs.shape
        state = torch.cat((behaviors, pupil_centers), dim=1)
        outputs, state = self.unfold(inputs), self.unfold(state[..., -t:])
        state = rearrange(state, "b d t h -> b t (d h)")
        state = repeat(state, "b t d -> b n t d", n=n)
        outputs = torch.concat((outputs, state), dim=-1)
        outputs = self.encoder(outputs)
        outputs = inputs + F.tanh(rearrange(outputs, "b n t 1 -> b n t"))
        return outputs
