_MODULATORS = dict()
from typing import Any, Dict, Tuple

import numpy as np
import torch
from einops import einsum, rearrange
from scipy import signal
from torch import nn
from torch.nn import functional as F


def register(name):
    def add_to_dict(fn):
        global _MODULATORS
        _MODULATORS[name] = fn
        return fn

    return add_to_dict


class Modulator(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, int]):
        super(Modulator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.num_neurons = input_shape[0]

    def regularizer(self):
        raise NotImplementedError

    def forward(
        self,
        inputs: torch.Tensor,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        """
        Args:
            inputs: torch.Tensor, response logits in shape (B, N, T)
            behaviors: torch.Tensor, behavior in shape (B, 2, T)
            pupil_centers: torch.Tensor, pupil center in shape (B, 2, T)
        """
        raise NotImplementedError


class Modulators(nn.ModuleDict):
    def __init__(self, args: Any, input_shapes: Dict[str, Tuple[int, int]]):
        super(Modulators, self).__init__()
        self.input_shapes = input_shapes
        self.output_shapes = input_shapes
        self.modulator_weight_decay = args.modulator_weight_decay
        match args.modulator_mode:
            case 1:
                modulator = _MODULATORS["mlp"]
            case 2:
                modulator = _MODULATORS["gru"]
            case 3:
                modulator = _MODULATORS["mlp-v2"]
            case 4:
                modulator = _MODULATORS["mlp-v3"]
            case _:
                raise NotImplementedError(
                    f"--modulator_mode {args.modulator_mode} not implemented."
                )
        for mouse_id, input_shape in input_shapes.items():
            self.add_module(mouse_id, modulator(args, input_shape=input_shape))

    def get_parameters(self):
        params = [
            {
                "params": self.parameters(),
                "name": "modulators",
                "weight_decay": self.modulator_weight_decay,
            }
        ]
        return params

    def regularizer(self, mouse_id: str):
        return self.models[mouse_id].regularizer()

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        return self[mouse_id](inputs, behaviors, pupil_centers)
