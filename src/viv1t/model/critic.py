from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn


def get_output_shape(
    input_shape: Tuple[int, int, int, int],
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = None,
):
    """According to https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html"""
    if dilation is None:
        dilation = (1,) * 3
    if isinstance(padding, int):
        padding = (padding,) * 3
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 3
    if isinstance(stride, int):
        stride = (stride,) * 3

    output_size = (
        out_channels,
        np.floor(
            (input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            / stride[0]
            + 1
        ).astype(int),
        np.floor(
            (input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            / stride[1]
            + 1
        ).astype(int),
        np.floor(
            (input_shape[3] + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1)
            / stride[2]
            + 1
        ).astype(int),
    )
    return output_size


class Critic(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int, int]):
        super(Critic, self).__init__()
        self.input_shape = input_shape
        self.num_classes = len(args.output_shapes)
        out_channels = args.critic_hidden_dim
        kernel_size = 3
        stride = 1
        dropout = args.critic_dropout

        self.conv1 = nn.Conv3d(
            in_channels=input_shape[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.gelu1 = nn.GELU()
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm3d(num_features=out_channels)
        self.dropout1 = nn.Dropout3d(p=dropout)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.gelu2 = nn.GELU()
        self.max_pool2 = nn.MaxPool3d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm3d(num_features=out_channels)
        self.dropout2 = nn.Dropout3d(p=dropout)

        self.linear1 = nn.Linear(
            in_features=out_channels, out_features=self.num_classes
        )
        self.gelu3 = nn.GELU()
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(
            in_features=self.num_classes, out_features=self.num_classes
        )
        # self.weight = nn.Parameter(torch.rand(1))

    def predict(self, logits: torch.Tensor):
        return torch.softmax(logits, dim=-1).argmax(dim=-1)

    def forward(self, inputs: torch.Tensor):
        outputs = self.gelu1(self.conv1(inputs))
        outputs = self.max_pool1(outputs)
        outputs = self.batch_norm1(outputs)
        outputs = self.dropout1(outputs)

        outputs = self.gelu2(self.conv2(outputs))
        outputs = self.max_pool2(outputs)
        outputs = self.batch_norm2(outputs)
        outputs = self.dropout2(outputs)

        outputs = torch.mean(outputs, dim=(2, 3, 4))
        outputs = self.gelu3(self.linear1(outputs))
        outputs = self.dropout3(outputs)
        outputs = self.linear2(outputs)
        # b = inputs.size(0)
        # outputs = torch.rand(*(b, self.num_classes), device=inputs.device)
        # outputs = outputs + self.weight - self.weight
        return outputs
