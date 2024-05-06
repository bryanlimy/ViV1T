from typing import Any, Dict, Tuple

import torch
from einops import einsum, rearrange
from torch import nn
from torch.nn import functional as F

from viv1t.model.modulators.modulator import Modulator, register


@register("mlp")
class MLPModulator(Modulator):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int],
        history_reg: float = 0.0,
        behavior_reg: float = 0.0,
    ):
        super(MLPModulator, self).__init__(args, input_shape)

        self.register_buffer("history_reg", torch.tensor(history_reg))
        self.register_buffer("behavior_reg", torch.tensor(behavior_reg))

        self.include_history = args.modulator_include_history
        if self.include_history:
            assert args.modulator_history_size > 0
            self.h_size = args.modulator_history_size
            self.history_encoder = nn.Sequential(
                nn.Linear(in_features=self.h_size, out_features=self.h_size, bias=True),
                nn.GELU(),
                nn.Dropout(p=args.modulator_history_dropout),
                nn.Linear(in_features=self.h_size, out_features=1, bias=True),
            )

        self.include_behaviors = args.modulator_include_behaviors
        if self.include_behaviors:
            in_features = 2 if self.include_behaviors == 1 else 4
            self.behavior_encoder = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=in_features, bias=True),
                nn.GELU(),
                nn.Dropout(p=args.modulator_behaviors_dropout),
                nn.Linear(
                    in_features=in_features, out_features=self.num_neurons, bias=True
                ),
            )

        if not self.include_history and not self.include_behaviors:
            print(f"MLP Modulator: no history or behavior modulator included.")

    def regularizer(self):
        return 0.0

    def unfold(self, inputs: torch.Tensor):
        """zero pad the input and unfold it to get the history of self.h_size"""
        b, n, t = inputs.shape
        return torch.cat(
            (inputs.new_zeros((b, n, self.h_size)), inputs), dim=-1
        ).unfold(-1, size=self.h_size, step=1)[:, :, :-1]

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, n, t = inputs.shape
        outputs = inputs
        if self.include_history:
            history = self.unfold(outputs)
            history = self.history_encoder(history)
            outputs = outputs * F.sigmoid(torch.squeeze(history, dim=-1))
        if self.include_behaviors:
            state = behaviors
            if self.include_behaviors == 2:
                state = torch.cat((behaviors, pupil_centers), dim=1)
            state = state[..., -t:]
            state = rearrange(state, "b d t -> (b t) d")
            state = self.behavior_encoder(state)
            state = rearrange(state, "(b t) n -> b n t", t=t)
            outputs = outputs + F.tanh(state)
        return outputs
