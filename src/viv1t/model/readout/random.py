from typing import Any, Tuple

import torch
from torch import nn

from viv1t.model.readout.readout import Readout, register


@register("random")
class RandomReadout(Readout):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: torch.Tensor,
        mean_responses: torch.Tensor = None,
    ):
        super(RandomReadout, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )

        self.weight = nn.Parameter(torch.rand(1))

    def regularizer(self, reduction: str):
        return 0.0

    def forward(
        self,
        inputs: torch.Tensor,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        b, _, t, _, _ = inputs.shape
        outputs = torch.rand(*(b, self.num_neurons, t), device=inputs.device)
        return outputs + self.weight - self.weight
