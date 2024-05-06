import os
import pickle
import warnings
from typing import Dict, Tuple, Union

import numpy as np
from einops import rearrange

from viv1t.data import utils
from viv1t.data.constants import *


def crop(stat: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: v[..., :MAX_FRAME] for k, v in stat.items()}


def measure(array: np.ndarray, axis: Union[int, Tuple]):
    """
    Measure min, max, median, mean, std of array along dim
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = {
            "min": np.nanmin(array, axis=axis),
            "max": np.nanmax(array, axis=axis),
            "median": np.nanmedian(array, axis=axis),
            "mean": np.nanmean(array, axis=axis),
            "std": np.nanstd(array, axis=axis),
        }
    return stats


def compute_statistics(mouse_dir: str, filename: str):
    """
    Compute statistics (min, max, median, mean, std) of the training set with
    the following format:
    - video: (H, W, T)
    - responses:  (N, T)
    - behavior: (2, T)
    - pupil_center: (2, T)
    """
    print(f"Compute statistics in {os.path.basename(mouse_dir)}...")

    # read from training set
    tiers = np.load(os.path.join(mouse_dir, "meta", "trials", "tiers.npy"))
    trials = np.where(tiers == "train")[0]

    # load a sample to get shapes
    load = lambda k: np.load(os.path.join(mouse_dir, "data", k, f"{trials[0]}.npy"))
    h, w, t = load("videos").shape
    n = load("responses").shape[0]
    d1, d2 = load("behavior").shape[0], load("pupil_center").shape[0]

    empty = lambda shape: np.full(shape, fill_value=np.nan, dtype=np.float32)
    videos = empty((h, w, t, len(trials)))
    responses = empty((n, t, len(trials)))
    behavior = empty((d1, t, len(trials)))
    pupil_center = empty((d2, t, len(trials)))

    for i, trial in enumerate(trials):
        sample = utils.load_trial_data(mouse_dir, trial=trial, to_tensor=True)
        frame = sample["duration"]
        videos[:, :, :frame, i] = rearrange(sample["video"], "1 t h w -> h w t")
        responses[:, :frame, i] = sample["response"]
        behavior[:, :frame, i] = sample["behavior"]
        pupil_center[:, :frame, i] = sample["pupil_center"]

    # compute statistics over trials and over trials and time
    stats = {
        "trial": {
            "video": measure(videos, axis=-1),
            "response": measure(responses, axis=-1),
            "behavior": measure(behavior, axis=-1),
            "pupil_center": measure(pupil_center, axis=-1),
        },
        "time": {
            "video": measure(videos, axis=(-2, -1)),
            "response": measure(responses, axis=(-2, -1)),
            "behavior": measure(behavior, axis=(-2, -1)),
            "pupil_center": measure(pupil_center, axis=(-2, -1)),
        },
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(stats, file)
    return stats


def load_stats(
    mouse_dir: str, stat_mode: int, transform_mode: int
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load data statistics from a mouse directory.

    Args:
        mouse_dir: str, path to the mouse directory
        stat_mode: int, 0 for provided statistics, 1 for computed statistics
        transform_mode: int, data transformation mode
    Returns:
        stats: Dict[str, Dict[str, np.ndarray]], statistics of the data
    """
    filename = os.path.join(mouse_dir, "statistics.pkl")
    if stat_mode == 0:
        info = utils.load_provided_stats(mouse_dir)
        response_stat = info["response"]
    else:
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                info = pickle.load(file)
        else:
            info = compute_statistics(mouse_dir, filename=filename)
        response_stat = info["trial"]["response"]
        info = info["trial"] if transform_mode in (0, 1, 2) else info["time"]

    # responses are always standardized with trial statistics
    stats = {"response": crop(response_stat)}
    match transform_mode:
        case 0 | 1 | 2:
            stats["video"] = {
                k: rearrange(v, "h w t -> t h w")
                for k, v in crop(info["video"]).items()
            }
            stats["behavior"] = crop(info["behavior"])
            stats["pupil_center"] = crop(info["pupil_center"])
        case 3 | 4:
            stats["video"] = info["video"]
            stats["behavior"] = info["behavior"]
            stats["pupil_center"] = info["pupil_center"]
        case _:
            raise NotImplementedError(
                f"--transform_mode {transform_mode} not implemented."
            )
    return stats
