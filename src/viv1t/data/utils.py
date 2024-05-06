import os
from typing import Dict, List, Tuple, Union
from zipfile import ZipFile

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from viv1t.data.constants import *


def unzip(filename: str, unzip_dir: str):
    """Extract zip file with filename to unzip_dir"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"file {filename} not found.")
    print(f"Unzipping {filename}...")
    with ZipFile(filename, mode="r") as file:
        file.extractall(unzip_dir)


def micro_batching(batch: Dict[str, torch.Tensor], batch_size: int):
    """Divide batch into micro batches"""
    indexes = np.arange(0, len(batch["video"]), step=batch_size, dtype=int)
    for i in indexes:
        yield {k: v[i : i + batch_size] for k, v in batch.items()}


def find_nan(array: np.ndarray) -> int:
    """Given a 1D array, find the first NaN value and return its index"""
    assert len(array.shape) == 1
    nans = np.where(np.isnan(array))[0]
    return nans[0] if nans.any() else len(array)


def set_shapes(args, ds: Dict[str, DataLoader]):
    """Set args.input_shapes and args.output_shapes"""
    mouse_ids = list(ds.keys())
    max_frame = ds[mouse_ids[0]].dataset.crop_frame
    args.input_shapes = {
        "video": (
            ds[mouse_ids[0]].dataset.num_channels,
            max_frame,
            *ds[mouse_ids[0]].dataset.video_shape,
        ),
        "behavior": (2, max_frame),
        "pupil_center": (2, max_frame),
    }
    args.output_shapes = {
        mouse_id: (ds[mouse_id].dataset.num_neurons, max_frame)
        for mouse_id in mouse_ids
    }


def get_dataloader_kwargs(args, device: torch.device, num_workers: int = None):
    # settings for DataLoader
    num_workers = args.num_workers if num_workers is None else num_workers
    kwargs = {"num_workers": num_workers, "pin_memory": False}
    if device.type in ("cuda", "mps"):
        kwargs |= {
            "prefetch_factor": 2 * num_workers if num_workers else None,
            "persistent_workers": num_workers > 0,
        }
    return kwargs


def estimate_mean_response(ds: Dict[str, DataLoader]):
    # estimate (preprocessed) mean responses from 3 batches
    mean_responses = {}
    for mouse_id, mouse_ds in ds.items():
        responses = torch.concat(
            [next(iter(mouse_ds))["response"] for _ in range(3)], dim=0
        )
        mean_responses[mouse_id] = torch.mean(responses, dim=(0, 2))
    return mean_responses


def num_steps(ds: Dict[str, DataLoader]):
    """Return the number of total steps to iterate all the DataLoaders"""
    return sum([len(ds[k]) for k in ds.keys()])


def load_provided_stats(mouse_dir: str):
    load_stat = lambda a, b: np.load(
        os.path.join(mouse_dir, "meta", "statistics", a, "all", f"{b}.npy")
    )
    stat_keys = ["min", "max", "mean", "std"]
    stats = {
        "video": {k: load_stat("videos", k) for k in stat_keys},
        "response": {k: load_stat("responses", k) for k in stat_keys},
        "behavior": {k: load_stat("behavior", k) for k in stat_keys},
        "pupil_center": {k: load_stat("pupil_center", k) for k in stat_keys},
    }
    return stats


def load_trial_data(mouse_dir: str, trial: str, to_tensor: bool = False):
    """
    Load data from a single trial in mouse_dir

    Data from each mouse has a fixed tensor shape, in order to find the duration
    of the trial, we first find the first NaN value in the recording and crop
    each tensor.
    Response is None for live and final test sets

    Args:
        mouse_dir: str, directory with mouse data
        trial: str, the trial index to load
        to_tensor: bool, convert np.ndarray to torch.Tensor
    Returns
        data: Dict[str, TENSOR]
            video: TENSOR, video in format (C, T, H, W)
            response: TENSOR, response in format (N, T) where N is num. of neurons
            behavior: TENSOR, pupil dilation and speed in format (2, T)
            pupil center: TENSOR, pupil center x and y coordinates in format (2, T)
            duration: int, the duration (i.e. num. of frames) of the trial
    """
    basename, data_dir = f"{trial}.npy", os.path.join(mouse_dir, "data")
    load = lambda key: np.load(os.path.join(data_dir, key, basename)).astype(np.float32)
    video = load("videos")
    # find duration of the trial by searching for the first NaN in the video
    num_frames = find_nan(video[0, 0, :])
    sample = {
        "video": rearrange(video[..., :num_frames], "h w t -> 1 t h w"),
        "response": load("responses")[:, :num_frames],
        "behavior": load("behavior")[:, :num_frames],
        "pupil_center": load("pupil_center")[:, :num_frames],
        "duration": num_frames,
    }
    if to_tensor:
        sample["video"] = torch.from_numpy(sample["video"])
        sample["response"] = torch.from_numpy(sample["response"])
        sample["behavior"] = torch.from_numpy(sample["behavior"])
        sample["pupil_center"] = torch.from_numpy(sample["pupil_center"])
    return sample


def get_neuron_coordinates(data_dir: str, mouse_ids: List[str]):
    return {
        mouse_id: np.load(
            os.path.join(
                data_dir,
                MOUSE_IDS[mouse_id],
                "meta",
                "neurons",
                "cell_motor_coordinates.npy",
            )
        ).astype(np.float32)
        for mouse_id in mouse_ids
    }
