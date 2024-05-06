from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


class ELU1(nn.Module):
    """ELU + 1 activation to output standardized responses"""

    def __init__(self):
        super(ELU1, self).__init__()
        self.elu = nn.ELU()
        self.register_buffer("one", torch.tensor(1.0))

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs) + self.one


class Exponential(nn.Module):
    """Exponential activation to output standardized responses"""

    def __init__(self):
        super(Exponential, self).__init__()

    def forward(self, inputs: torch.Tensor):
        return torch.exp(inputs)


class SwiGLU(nn.Module):
    """
    SwiGLU activation by Shazeer et al. 2022
    https://arxiv.org/abs/2002.05202
    """

    def forward(self, inputs: torch.Tensor):
        outputs, gate = inputs.chunk(2, dim=-1)
        return F.silu(gate) * outputs


class AdaptiveELU(nn.Module):
    """
    ELU shifted by user specified values. This helps to ensure the output to stay positive.
    """

    def __init__(self, xshift: int, yshift: int, **kwargs):
        super(AdaptiveELU, self).__init__(**kwargs)

        self.x_shift = torch.tensor(xshift, dtype=torch.float)
        self.y_shift = torch.tensor(yshift, dtype=torch.float)
        self.elu = nn.ELU()

    def forward(self, inputs: torch.Tensor):
        return self.elu(inputs - self.x_shift) + self.y_shift


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate"""

    def __init__(
        self,
        size: Union[int, Tuple[int, int], Tuple[int, int, int]],
        mode: str = "nearest",
        antialias: bool = False,
    ):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.antialias = antialias

    def forward(self, inputs: torch.Tensor):
        return F.interpolate(
            inputs,
            size=self.size,
            mode=self.mode,
            antialias=self.antialias,
        )


class DropPath(nn.Module):
    """
    Stochastic depth for regularization https://arxiv.org/abs/1603.09382
    Reference:
    - https://github.com/huggingface/pytorch-image-models/blob/7c67d6aca992f039eece0af5f7c29a43d48c00e4/timm/models/layers/drop.py#L135
    """

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        assert 0 <= drop_prob <= 1
        self.register_buffer("keep_prob", torch.tensor(1 - drop_prob))

    def forward(self, inputs: torch.Tensor, scale_by_keep: bool = True):
        if self.keep_prob == 1 or not self.training:
            return inputs
        shape = (inputs.size(0),) + (1,) * (inputs.ndim - 1)
        random_tensor = inputs.new_empty(shape).bernoulli_(self.keep_prob)
        if self.keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(self.keep_prob)
        return inputs * random_tensor
