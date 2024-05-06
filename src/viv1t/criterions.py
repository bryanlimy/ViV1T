from math import ceil
from typing import Dict

import torch
from einops import rearrange
from torch import nn
from torch.utils.data import DataLoader

from viv1t.metrics import EPS, correlation
from viv1t.utils.bufferdict import BufferDict

_CRITERION = dict()


def register(name):
    def add_to_dict(fn):
        global _CRITERION
        _CRITERION[name] = fn
        return fn

    return add_to_dict


class Criterion(nn.Module):
    """Basic Criterion class"""

    def __init__(self, args, ds: Dict[str, DataLoader]):
        super(Criterion, self).__init__()
        self.ds_scale = args.ds_scale
        self.ds_sizes = BufferDict(
            buffers={
                mouse_id: torch.tensor(len(mouse_ds.dataset), dtype=torch.float32)
                for mouse_id, mouse_ds in ds.items()
            }
        )

    def scale_ds(self, loss: torch.Tensor, mouse_id: str, batch_size: int):
        """Scale loss based on the size of the dataset"""
        if self.ds_scale:
            loss *= torch.sqrt(self.ds_sizes[mouse_id] / batch_size)
        return loss


@register("poisson")
class PoissonLoss(Criterion):
    def __init__(self, args, ds: Dict[str, DataLoader], eps: float = EPS):
        super(PoissonLoss, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        # add eps to targets and predictions to avoid numeric instability
        y_true, y_pred = y_true + self.eps, y_pred + self.eps
        loss = torch.sum(y_pred - y_true * torch.log(y_pred))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


@register("correlation")
class Correlation(Criterion):
    """single trial correlation"""

    def __init__(self, args, ds: Dict[str, DataLoader], eps: float = EPS):
        super(Correlation, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("one", torch.tensor(1.0))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        y_true = rearrange(y_true, "b n t -> (b t) n")
        y_pred = rearrange(y_pred, "b n t -> (b t) n")
        corr = correlation(y1=y_true, y2=y_pred, dim=0, eps=self.eps)
        loss = -torch.sum(corr + self.one)
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


@register("logMSE")
class LogMSE(Criterion):
    """Log MSE loss"""

    def __init__(self, args, ds: Dict[str, DataLoader], eps: float = EPS):
        super(LogMSE, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        y_true = torch.log(y_true + self.eps)
        y_pred = torch.log(y_pred + self.eps)
        loss = torch.mean(torch.square(y_true - y_pred))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


@register("mse")
class MSE(Criterion):
    def __init__(self, args, ds: Dict[str, DataLoader]):
        super(MSE, self).__init__(args, ds=ds)

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        loss = torch.mean(torch.square(y_true - y_pred))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        loss = loss * (y_true.size(0) / batch_size)
        return loss


@register("mean_poisson")
class MeanPoissonLoss(Criterion):
    def __init__(self, args, ds: Dict[str, DataLoader], eps: float = EPS):
        super(MeanPoissonLoss, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        # add eps to targets and predictions to avoid numeric instability
        y_true, y_pred = y_true + self.eps, y_pred + self.eps
        loss = torch.mean(torch.sum(y_pred - y_true * torch.log(y_pred), dim=(1, 2)))
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        loss = loss * (y_true.size(0) / batch_size)
        return loss


@register("poisson_correlation")
class PoissonCorrelationLoss(Criterion):
    def __init__(self, args, ds: Dict[str, DataLoader], eps: float = EPS):
        super(PoissonCorrelationLoss, self).__init__(args, ds=ds)
        self.register_buffer("eps", torch.tensor(eps))
        self.register_buffer("one", torch.tensor(1.0))
        self.register_buffer("two", torch.tensor(2.0))
        self.register_buffer("alpha", torch.tensor(1000))

    def poisson_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        # add eps to targets and predictions to avoid numeric instability
        y_true, y_pred = y_true + self.eps, y_pred + self.eps
        return torch.sum(y_pred - y_true * torch.log(y_pred))

    def correlation_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        y_true = rearrange(y_true, "b n t -> (b t) n")
        y_pred = rearrange(y_pred, "b n t -> (b t) n")
        corr = correlation(y1=y_true, y2=y_pred, dim=0, eps=self.eps)
        return torch.sum(self.two - (corr + self.one))

    def forward(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mouse_id: str,
        batch_size: int,
    ):
        poisson_loss = self.poisson_loss(y_true, y_pred)
        correlation_loss = self.correlation_loss(y_true, y_pred)
        loss = poisson_loss + self.alpha * correlation_loss
        loss = self.scale_ds(loss, mouse_id=mouse_id, batch_size=batch_size)
        return loss


def get_criterion(args, ds: Dict[str, DataLoader]) -> Criterion:
    assert args.criterion in _CRITERION, f"Criterion {args.criterion} not found."
    criterion = _CRITERION[args.criterion](args, ds=ds)
    criterion.to(args.device)
    return criterion
