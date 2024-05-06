"""
Code reference: https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/shifters/mlp.py
"""
from typing import Any, Dict, List, Tuple

import torch
from einops import rearrange
from torch import nn


class MLPShifter(nn.Module):
    """
    Code reference: https://github.com/sinzlab/neuralpredictors/blob/9b85300ab854be1108b4bf64b0e4fa2e960760e0/neuralpredictors/layers/shifters/mlp.py
    """

    def __init__(
        self,
        args: Any,
        in_features: int,
        hidden_features: int,
        num_layers: int,
        bias: bool,
    ):
        super(MLPShifter, self).__init__()
        self.register_buffer("reg_scale", torch.tensor(0.0))
        out_features = in_features
        layers = []
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(
                        in_features=out_features,
                        out_features=hidden_features,
                        bias=bias,
                    ),
                    nn.Tanh(),
                ]
            )
            out_features = hidden_features
        layers.extend(
            [nn.Linear(in_features=out_features, out_features=2, bias=bias), nn.Tanh()]
        )
        self.mlp = nn.Sequential(*layers)

    def regularizer(self):
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def forward(self, inputs: torch.Tensor):
        return self.mlp(inputs)


class MLPShifters(nn.ModuleDict):
    def __init__(
        self,
        args: Any,
        input_shapes: Dict[str, Tuple[int, int]],
        mouse_ids: List[str],
        bias: bool = True,
    ):
        super(MLPShifters, self).__init__()
        self.shifter_mode = args.shifter_mode
        d1, t = input_shapes["behavior"]
        d2, _ = input_shapes["pupil_center"]
        in_features = d1 + d2 if self.shifter_mode == 2 else 2
        self.input_shape = (in_features, t)
        for mouse_id in mouse_ids:
            self.add_module(
                mouse_id,
                MLPShifter(
                    args,
                    in_features=in_features,
                    hidden_features=args.shifter_size,
                    num_layers=args.shifter_layers,
                    bias=bias,
                ),
            )
        self.initialize()

    def initialize(self, **kwargs):
        for layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            nn.init.xavier_normal_(layer.weight)

    def regularizer(self, mouse_id: str):
        return self[mouse_id].regularizer()

    def forward(
        self,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        mouse_id: str,
    ):
        match self.shifter_mode:
            case 1:
                outputs = pupil_centers
            case 2:
                outputs = torch.cat((behaviors, pupil_centers), dim=1)
            case _:
                raise NotImplementedError(
                    f"--shifter_mode {self.shifter_mode} not implemented."
                )
        b, _, t = outputs.shape
        outputs = rearrange(outputs, "b c t -> (b t) c")
        outputs = self[mouse_id](outputs)
        outputs = rearrange(outputs, "(b t) c -> b c t", b=b, t=t)
        return outputs
