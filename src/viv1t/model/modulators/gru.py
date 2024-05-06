from typing import Any, Dict, Tuple

import numpy as np
import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F

from viv1t.model.modulators.modulator import Modulator, register


@register("gru")
class GRUModulator(Modulator):
    def __init__(self, args: Any, input_shape: Tuple[int, int]):
        super(GRUModulator, self).__init__(args, input_shape)

        self.hidden_dim = args.modulator_hidden_dim
        if self.hidden_dim == None:
            self.hidden_dim = self.num_neurons

        self.state_encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=self.num_neurons),
            nn.Tanh(),
            nn.Dropout(p=args.modulator_dropout),
        )
        self.gru_cell = nn.GRUCell(
            input_size=self.num_neurons, hidden_size=self.hidden_dim, bias=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.num_neurons),
            nn.Tanh(),
        )

    def regularizer(self):
        return 0.0

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, n, t = inputs.shape
        inputs = rearrange(inputs, "b n t -> b t n")

        states = torch.concat((behaviors, pupil_centers), dim=1)
        states = rearrange(states[..., -t:], "b d t -> b t d")

        prev_state = inputs.new_zeros(b, self.hidden_dim)
        hidden_states = []
        for i in range(t):
            frame = inputs[:, i, :] + self.state_encoder(states[:, i, :])
            prev_state = self.gru_cell(frame, prev_state)
            hidden_states.append(self.decoder(prev_state))
        hidden_states = torch.stack(hidden_states, dim=1)
        outputs = inputs + hidden_states
        return rearrange(outputs, "b t n -> b n t")
