from contextlib import nullcontext
from typing import Any, Tuple

import torch
from torch import nn

from viv1t.utils import utils

_CORES = dict()


def register(name):
    """Note: update __init__.py so that register works"""

    def add_to_dict(fn):
        global _CORES
        _CORES[name] = fn
        return fn

    return add_to_dict


class Core(nn.Module):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int, int]):
        super(Core, self).__init__()
        self.input_shape = input_shape
        self.core_lr = args.core_lr
        if self.core_lr is None:
            self.core_lr = args.lr
            print("Setting core learning rate to model learning rate.")
        self.core_weight_decay = args.core_weight_decay
        if self.core_weight_decay is None:
            self.core_weight_decay = args.weight_decay
            print("Setting core weight decay to model weight decay rate.")
        self.frozen = False
        self.verbose = args.verbose

        support_bf16 = utils.support_bf16(args.device)
        match args.precision:
            case "bf16":
                precision = torch.bfloat16
            case "32":
                precision = torch.float32
            case None:
                precision = torch.bfloat16 if support_bf16 else torch.float32
            case _:
                raise NotImplementedError(f"Unknown precision type: {args.precision}")
        enable_bf16 = support_bf16 and precision == torch.bfloat16
        if enable_bf16:
            self.autocast = torch.autocast(args.device, dtype=torch.bfloat16)
        else:
            self.autocast = nullcontext()

        if args.verbose and args.precision == "bf16":
            if enable_bf16:
                print("Enable bfloat16 computation.")
            else:
                print(f"Device {args.device} does not support bfloat16.")

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)
        self.frozen = True
        if self.verbose:
            print("Freeze core module.")

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)
        self.forzen = False
        if self.verbose:
            print("Unfreeze core module.")

    def initialize(self):
        raise NotImplementedError("initialize function has not been implemented")

    def autocast(self, device: str):
        """Return autocast context manager"""
        return torch.autocast(device, dtype=self.precision, enabled=True)

    def get_parameters(self):
        """Return a list of parameters for torch.optim.Optimizer"""
        return [
            {
                "params": self.parameters(),
                "name": "core",
                "lr": self.core_lr,
                "weight_decay": self.core_weight_decay,
            }
        ]

    def regularizer(self):
        raise NotImplementedError("regularizer function has not been implemented")

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        """
        Args:
            inputs: torch.Tensor, visual stimuli in (B, C, T, H, W)
            mouse_id: str, mouse ID
            behaviors: torch.Tensor, behavioral information in (B, 2, T)
            pupil_centers: torch.Tensor, pupil (x, y) coordinates in (B, 2, T)
        Return:
            outputs: torch.Tensor, core representation in (B, C', T, H', W')
        """
        raise NotImplementedError("forward function has not been implemented")


def get_core(args):
    if not args.core in _CORES.keys():
        raise NotImplementedError(f"Core {args.core} has not been implemented.")
    return _CORES[args.core]
