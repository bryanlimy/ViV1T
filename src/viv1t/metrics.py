from typing import List, Literal, Tuple, Union

import numpy as np
import torch
from einops import rearrange

REDUCTION = Literal["sum", "mean"]
EPS = torch.finfo(torch.float32).eps
TENSOR = Union[np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]


def sse(y_true: torch.Tensor, y_pred: torch.Tensor, reduction: REDUCTION = "mean"):
    """sum squared error over frames and neurons"""
    loss = torch.sum(torch.square(y_true - y_pred), dim=(1, 2))
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def poisson_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    eps: Union[float, torch.Tensor] = 1e-8,
    reduction: REDUCTION = "sum",
):
    """poisson loss over frames and neurons"""
    y_pred, y_true = y_pred + eps, y_true + eps
    loss = torch.sum(y_pred - y_true * torch.log(y_pred), dim=(1, 2))
    return torch.sum(loss) if reduction == "sum" else torch.mean(loss)


def _torch_correlation(
    y1: torch.Tensor,
    y2: torch.Tensor,
    dim: Union[None, int, Tuple[int]] = -1,
    eps: float = 1e-8,
):
    if dim is None:
        dim = tuple(range(y1.dim()))
    y1 = (y1 - y1.mean(dim=dim, keepdim=True)) / (
        y1.std(dim=dim, unbiased=False, keepdim=True) + eps
    )
    y2 = (y2 - y2.mean(dim=dim, keepdim=True)) / (
        y2.std(dim=dim, unbiased=False, keepdim=True) + eps
    )
    corr = (y1 * y2).mean(dim=dim)
    return corr


def _numpy_correlation(
    y1: np.ndarray,
    y2: np.ndarray,
    axis: Union[None, int, Tuple[int]] = -1,
    eps: float = 1e-8,
    **kwargs,
):
    y1 = (y1 - y1.mean(axis=axis, keepdims=True)) / (
        y1.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    y2 = (y2 - y2.mean(axis=axis, keepdims=True)) / (
        y2.std(axis=axis, ddof=0, keepdims=True) + eps
    )
    corr = (y1 * y2).mean(axis=axis, **kwargs)
    return corr


def correlation(
    y1: TENSOR,
    y2: TENSOR,
    dim: Union[None, int, Tuple[int]] = -1,
    eps: Union[torch.Tensor, float] = 1e-8,
    **kwargs,
):
    return (
        _torch_correlation(y1=y1, y2=y2, dim=dim, eps=eps)
        if isinstance(y1, torch.Tensor)
        else _numpy_correlation(y1=y1, y2=y2, axis=dim, eps=eps, **kwargs)
    )


def single_trial_correlation(y_true: TENSOR, y_pred: TENSOR, skip: int = 0):
    """
    Compute signal trial correlation with the first `skip` frames skipped
    Args:
        y_true: TENSOR, responses in (B, N, T)
        y_pred: TENSOR, responses in (B, N, T)
        skip: int, number of frames to skip in calculation
    """
    assert type(y_true) == type(y_pred) and type(y_true[0]) == type(y_pred[0])
    mean = torch.mean if isinstance(y_true[0], torch.Tensor) else np.mean
    vstack = torch.vstack if isinstance(y_true[0], torch.Tensor) else np.vstack
    num_frame = y_true[0].shape[-1] - skip
    if isinstance(y_true, torch.Tensor) or isinstance(y_true, np.ndarray):
        assert y_true.shape == y_pred.shape
        y_true = rearrange(y_true[..., -num_frame:], "b n t -> (b t) n")
        y_pred = rearrange(y_pred[..., -num_frame:], "b n t -> (b t) n")
    else:
        # y_true and y_pred are List[TENSOR] in shape [(N, T), (N, T), ...]
        assert len(y_true) == len(y_pred)
        flatten = lambda tensors: vstack([y[:, -num_frame:].T for y in tensors])
        y_true, y_pred = flatten(y_true), flatten(y_pred)
    corr = correlation(y1=y_true, y2=y_pred, dim=0)
    return mean(corr)


@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Metrics to compute as part of training and validation step"""
    mean_sse = sse(y_true=y_true, y_pred=y_pred, reduction="mean")
    mean_poisson = poisson_loss(y_true=y_true, y_pred=y_pred, reduction="mean")
    correlation = single_trial_correlation(y_true=y_true, y_pred=y_pred)
    return {
        "metrics/msse": mean_sse,
        "metrics/poisson_loss": mean_poisson,
        "metrics/single_trial_correlation": correlation,
    }
