import os
import platform
from math import ceil
from multiprocessing import Manager
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from viv1t.data import utils
from viv1t.data.constants import *
from viv1t.data.statistics import load_stats

TENSOR = Union[np.ndarray, torch.Tensor]
SAMPLE = Dict[str, Union[int, str, TENSOR]]


def get_mouse_ids(args):
    """Retrieve the mouse IDs when args.mouse_ids is not provided"""
    match args.ds_mode:
        case 0:
            mouse_ids = OLD_MICE
        case 1:
            mouse_ids = NEW_MICE
        case 2 | 3:
            mouse_ids = OLD_MICE + NEW_MICE
        case _:
            raise NotImplementedError("--ds_mode must be in (0, 1, 2, 3).")
    if args.mouse_ids is None:
        args.mouse_ids = mouse_ids
    else:
        for mouse_id in args.mouse_ids:
            assert mouse_id in mouse_ids, f"mouse {mouse_id} is not in {mouse_ids}"


def load_mouse_metadata(mouse_dir: str, stat_mode: int, transform_mode: int):
    """
    Load the relevant metadata of a specific mouse
    Args:
        mouse_dir: str, path to folder to host mouse recordings and metadata
        stat_mode: int, 0 for provided statistics, 1 for computed statistics
        transform_mode: int, data transformation mode
    Returns:
        metadata: Dict[str, Any]
            num_neurons: int, number of neurons
            neuron_coordinates: np.ndarray, (x, y, z) coordinates of each
                neuron in the cortex
            neuron_ids: np.ndarray, neuron IDs
            tiers: np.ndarray, 'train', 'validation', 'live_test_main',
                'live_test_bonus', 'final_test_main', 'final_test_bons', 'none'
            statistics: t.Dict[str, t.Dict[str, np.ndarray]], the
                statistics (min, max, median, mean, std) of the training data
    """
    if not os.path.isdir(mouse_dir):
        utils.unzip(filename=f"{mouse_dir}.zip", unzip_dir=os.path.dirname(mouse_dir))
    meta_dir = os.path.join(mouse_dir, "meta")

    trial_dir = os.path.join(meta_dir, "trials")
    tiers = np.load(os.path.join(trial_dir, "tiers.npy"))
    tiers[tiers == "oracle"] = "validation"  # rename oracle to validation

    neuron_dir = os.path.join(meta_dir, "neurons")
    neuron_coordinates = np.load(
        os.path.join(neuron_dir, "cell_motor_coordinates.npy")
    ).astype(np.float32)
    neuron_ids = np.load(os.path.join(neuron_dir, "unit_ids.npy")).astype(np.int16)

    return {
        "num_neurons": len(neuron_coordinates),
        "neuron_coordinates": neuron_coordinates,
        "neuron_ids": neuron_ids,
        "tiers": tiers,
        "stats": load_stats(
            mouse_dir, stat_mode=stat_mode, transform_mode=transform_mode
        ),
    }


class MovieDataset(Dataset):
    """
    MoveDataset class for loading data from a single mouse

    Notable attributes:
    - tier: str, the tier of the dataset
    - video_stats: Dict[str, np.ndarray], the statistics of the video data
    - response_stats: Dict[str, np.ndarray], the statistics of the response data
    - behavior_stats: Dict[str, np.ndarray], the statistics of the behavior data
    - pupil_center_stats: Dict[str, np.ndarray], the statistics of the pupil center data
    - num_neurons: int, number of neurons
    - neuron_coordinates: np.ndarray, (x, y, z) coordinates of each neuron
    - neuron_ids: np.ndarray, neuron IDs
    - trials: np.ndarray, the trial IDs
    - max_frame: int, the maximum number of frames the dataset returns
    - hidden_response: bool, whether the recorded responses are hidden (zeros)
    """

    def __init__(
        self,
        ds_mode: str,
        tier: str,
        data_dir: str,
        mouse_id: str,
        mouse_ids: List[str],
        stat_mode: int,
        transform_mode: int,
        crop_frame: int = -1,
        center_crop: float = 1.0,
        cache_data: bool = False,
        limit_data: int = None,
        num_workers: int = 2,
        verbose: int = 0,
    ):
        """
        Construct Movie Dataset

        Args:
            ds_mode: int, dataset mode
                0: train on the 5 original mice
                1: train on the 5 new mice
                2: train on all 10 mice jointly
                3: train on all 10 mice with all tiers from the 5 original mice
            tier: str, 'train', 'validation', 'live_test_main',
                'live_test_bonus', 'final_test_main', 'final_test_bonus'
            data_dir: str, path to where all data are stored
            mouse_id: str, the mouse ID
            stat_mode: int, data statistics mode
            transform_mode: int, data transformation and preprocessing mode
            crop_frame: int, number of frames to take from each trial, set to -1
                to use all available frames
            cache_data: bool, cache data into memory to speed up data loading
            limit_data: int, limit the number of samples, set None to use all
                available trials
        """
        assert ds_mode in (0, 1, 2, 3)
        self.ds_mode = ds_mode
        assert tier in TIERS
        self.tier = tier
        assert stat_mode in (0, 1)
        self.stat_mode = stat_mode
        assert transform_mode in (0, 1, 2, 3, 4)
        assert (
            stat_mode == 1 or transform_mode <= 2
        ), "--stat_mode must be 1 if --transform_mode >= 3"
        self.transform_mode = transform_mode
        assert crop_frame == -1 or crop_frame > 50
        self.crop_frame = crop_frame
        assert center_crop > 0.0 and center_crop <= 1.0
        self.center_crop = center_crop

        self.mouse_class = mouse_ids.index(mouse_id)
        self.verbose = verbose

        self.mouse_id = mouse_id
        self.mouse_dir = os.path.join(data_dir, MOUSE_IDS[mouse_id])
        metadata = load_mouse_metadata(
            self.mouse_dir, stat_mode=self.stat_mode, transform_mode=self.transform_mode
        )
        # use multiprocessing manager to share data between processes
        self.manager = (
            Manager() if platform.system() == "Linux" and num_workers else None
        )
        self.video_stats = self.store_dict(metadata["stats"]["video"])
        self.response_stats = self.store_dict(metadata["stats"]["response"])
        self.behavior_stats = self.store_dict(metadata["stats"]["behavior"])
        self.pupil_center_stats = self.store_dict(metadata["stats"]["pupil_center"])

        self.max_frame = MAX_FRAME
        self.num_neurons = metadata["num_neurons"]
        self.neuron_ids = torch.from_numpy(metadata["neuron_ids"])
        self.neuron_coordinates = torch.from_numpy(metadata["neuron_coordinates"])

        assert self.crop_frame <= self.max_frame
        self.response_precision = self.compute_response_precision()
        self.eps = torch.finfo(torch.float32).eps

        self.tiers = metadata["tiers"].astype(np.string_)
        self.select_trials(metadata["tiers"], limit_data=limit_data)

        # get data dimensions
        sample = utils.load_trial_data(self.mouse_dir, trial=self.trial_ids[0])
        self.video_shape = sample["video"].shape[-2:]
        self.num_channels = sample["video"].shape[0]
        self.hidden_response = not np.any(sample["response"])

        # create cache dictionary to store dataset
        self.cache = None
        if cache_data:
            full = lambda size: torch.full(
                (len(self), *size), fill_value=torch.nan, dtype=torch.float32
            )
            self.cache = {
                "video": full(sample["video"].shape),
                "response": full(sample["response"].shape),
                "behavior": full(sample["behavior"].shape),
                "pupil_center": full(sample["pupil_center"].shape),
                "duration": torch.zeros((len(self)), dtype=torch.int32),
            }
            if self.manager is not None:
                self.cache = self.manager.dict(self.cache)
        del sample, metadata

        if self.center_crop < 1:
            self.prepare_center_crop()

    def prepare_center_crop(self):
        in_h, in_w = self.video_shape
        crop_h = int(in_h * self.center_crop)
        crop_w = int(in_w * self.center_crop)
        crop_scale = self.center_crop
        h_pixels = torch.linspace(-crop_scale, crop_scale, crop_h)
        w_pixels = torch.linspace(-crop_scale, crop_scale, crop_w)
        mesh_y, mesh_x = torch.meshgrid(h_pixels, w_pixels, indexing="ij")
        # grid_sample uses (x, y) coordinates
        grid = torch.stack((mesh_x, mesh_y), dim=2)
        self.grid = grid.unsqueeze(0)
        self.resize = transforms.Resize(size=(in_h, in_w), antialias=False)

    def select_trials(self, tiers: np.ndarray, limit_data: int):
        match self.ds_mode:
            case 0 | 1 | 2:
                trial_ids = np.where(tiers == self.tier)[0].astype(np.int32)
            case 3:
                # train on OLD_MICE test sets as well
                if self.mouse_id in OLD_MICE and self.tier == "train":
                    trial_ids = np.where(
                        np.in1d(
                            tiers,
                            [
                                "train",
                                "live_test_main",
                                "live_test_bonus",
                                "final_test_main",
                                "final_test_bonus",
                            ],
                        )
                    )[0].astype(np.int32)
                else:
                    trial_ids = np.where(tiers == self.tier)[0].astype(np.int32)
            case _:
                raise NotImplementedError(f"--ds_mode {self.ds_mode} not implemented.")

        if limit_data is not None:
            # randomly select limit_data number of samples in the training set
            rng = np.random.default_rng(1234)
            trial_ids = rng.choice(trial_ids, size=limit_data, replace=False)
            if self.verbose:
                print(
                    f"limit mouse {self.mouse_id} {self.tier} to {limit_data} samples."
                )

        self.trial_ids = torch.from_numpy(trial_ids)

    def store_dict(self, d: Dict[str, np.ndarray]):
        """Convert dict of np.ndarray to manager.dict of torch.tensor"""
        convert = lambda data: {k: torch.from_numpy(v) for k, v in data.items()}
        return convert(d) if self.manager is None else self.manager.dict(convert(d))

    def __len__(self):
        return len(self.trial_ids)

    def compute_response_precision(self):
        std = self.response_stats["std"][:, : self.max_frame]
        threshold = 0.01 * torch.nanmean(std)
        idx = std > threshold
        precision = torch.ones_like(std) / threshold
        precision[idx] = 1 / std[idx]
        return precision

    def crop_image(self, video: TENSOR):
        if self.center_crop < 1:
            video = F.grid_sample(video, self.grid, mode="nearest", align_corners=True)
            video = self.resize(video)
        return video

    def transform_video(self, video: TENSOR, duration: int):
        stats, video = self.video_stats, video[:, :duration]
        match self.transform_mode:
            case 1:
                v_mean = stats["mean"][:duration, ...]
                v_std = stats["std"][:duration, ...]
                video = (video - v_mean) / (v_std + self.eps)
            case 2:
                v_min = stats["mean"][:duration, ...]
                v_max = stats["std"][:duration, ...]
                video = (video - v_min) / (v_max - v_min)
            case 3:
                v_mean, v_std = stats["mean"], stats["std"]
                video = (video - v_mean) / (v_std + self.eps)
            case 4:
                v_min, v_max = stats["min"], stats["max"]
                video = (video - v_min) / (v_max - v_min)
        if self.center_crop < 1:
            video = self.crop_image(video)
        return video

    def transform_response(self, response: TENSOR, duration: int):
        return response[:, :duration] * self.response_precision[:, :duration]

    def transform_behavior(self, behavior: np.ndarray, duration: int):
        stats, behavior = self.behavior_stats, behavior[:, :duration]
        match self.transform_mode:
            case 1:
                behavior = behavior / stats["std"][:, :duration]
            case 2:
                b_min, b_max = stats["min"][:, :duration], stats["max"][:, :duration]
                behavior = (behavior - b_min) / (b_max - b_min)
            case 3:
                behavior = behavior / stats["std"][:, None]
            case 4:
                b_min, b_max = stats["min"][:, None], stats["max"][:, None]
                behavior = (behavior - b_min) / (b_max - b_min)
        return behavior

    def transform_pupil_center(self, pupil_center: np.ndarray, duration: int):
        stats, pupil_center = self.pupil_center_stats, pupil_center[:, :duration]
        match self.transform_mode:
            case 1:
                p_mean, p_std = stats["mean"][:, :duration], stats["std"][:, :duration]
                pupil_center = (pupil_center - p_mean) / (p_std + self.eps)
            case 2:
                p_min, p_max = stats["min"][:, :duration], stats["max"][:, :duration]
                pupil_center = (pupil_center - p_min) / (p_max - p_min)
            case 3:
                p_mean, p_std = stats["mean"][:, None], stats["std"][:, None]
                pupil_center = (pupil_center - p_mean) / (p_std + self.eps)
            case 4:
                p_min, p_max = stats["min"][:, None], stats["max"][:, None]
                pupil_center = (pupil_center - p_min) / (p_max - p_min)
        return pupil_center

    def transform_sample(self, data: SAMPLE):
        duration = min(data["duration"], self.max_frame)
        data["video"] = self.transform_video(data["video"], duration)
        data["response"] = self.transform_response(data["response"], duration)
        data["behavior"] = self.transform_behavior(data["behavior"], duration)
        data["pupil_center"] = self.transform_pupil_center(
            data["pupil_center"], duration
        )
        return data

    def crop_duration(self, sample: Dict[str, TENSOR], crop_frame: int):
        # randomly crop the trial to crop_frame in training set
        start, frame_diff = 0, sample["video"].shape[1] - crop_frame
        if self.tier == "train" and frame_diff > 0:
            start = np.random.randint(0, frame_diff)
        sample["video"] = sample["video"][:, start : start + crop_frame, ...]
        sample["response"] = sample["response"][:, start : start + crop_frame]
        sample["behavior"] = sample["behavior"][:, start : start + crop_frame]
        sample["pupil_center"] = sample["pupil_center"][:, start : start + crop_frame]

    def load_sample(self, idx: int, to_tensor: bool = False):
        """
        Load sample from cache if it exists, otherwise load from disk,
        apply transformation and store in cache
        """
        trial = self.trial_ids[idx]
        if self.cache is not None and self.cache["duration"][idx]:
            t = self.cache["duration"][idx]
            sample = {
                "video": self.cache["video"][idx, :, :t, :, :],
                "response": self.cache["response"][idx, :, :t],
                "behavior": self.cache["behavior"][idx, :, :t],
                "pupil_center": self.cache["pupil_center"][idx, :, :t],
                "duration": t,
            }
        else:
            sample = utils.load_trial_data(
                self.mouse_dir, trial=trial, to_tensor=to_tensor
            )
            t = sample["duration"]
            if self.transform_mode:
                sample = self.transform_sample(sample)
            if self.cache is not None:
                self.cache["video"][idx, :, :t, :, :] = sample["video"]
                self.cache["response"][idx, :, :t] = sample["response"]
                self.cache["behavior"][idx, :, :t] = sample["behavior"]
                self.cache["pupil_center"][idx, :, :t] = sample["pupil_center"]
                self.cache["duration"][idx] = t
        return sample

    def __getitem__(self, idx: Union[int, torch.Tensor], to_tensor: bool = True):
        """Return a sample

        Returns
            sample, t.Dict[str, torch.Tensor]
                video: TENSOR, the movie stimulus in (C, T, H, W)
                response: TENSOR, the corresponding response in (N, T)
                behavior: TENSOR, pupil size and locomotive speed in (2, T)
                pupil_center: TENSOR, pupil center(x, y) coordinates  in (2, T)
                mouse_id: str, the mouse ID
                trial_id: int, the trial ID
        """
        trial = self.trial_ids[idx]
        sample = self.load_sample(idx, to_tensor=to_tensor)
        if self.crop_frame != -1 and sample["duration"] > self.crop_frame:
            self.crop_duration(sample, crop_frame=self.crop_frame)
        sample["mouse_id"] = self.mouse_id
        sample["trial_id"] = trial
        sample["tier"] = str(self.tiers[trial], "utf-8")
        sample["mouse_class"] = self.mouse_class
        del sample["duration"]
        return sample


def get_training_ds(
    args,
    data_dir: str,
    mouse_ids: List[str],
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):
    """
    Get DataLoaders for training
    Args:
        args
        data_dir: str, path to directory where the zip files are stored
        mouse_ids: t.List[int], mouse IDs to extract
        batch_size: int, batch size of the DataLoaders
        device: torch.device, computing device
        num_workers: int, number of workers for DataLoader, use args.num_workers if None.
    Return:
        train_ds: t.Dict[str, DataLoader], dictionary of DataLoaders of the
            training sets where keys are the mouse IDs.
        val_ds: t.Dict[str, DataLoader], dictionary of DataLoaders of the
            validation sets where keys are the mouse IDs.
    """
    dataset_kwargs = {
        "ds_mode": args.ds_mode,
        "data_dir": data_dir,
        "mouse_ids": args.mouse_ids,
        "stat_mode": args.stat_mode,
        "transform_mode": args.transform_mode,
        "center_crop": args.center_crop if hasattr(args, "center_crop") else 1.0,
        "cache_data": args.cache_data if hasattr(args, "cache_data") else False,
        "num_workers": args.num_workers,
    }
    loader_kwargs = utils.get_dataloader_kwargs(
        args, device=device, num_workers=num_workers
    )

    train_ds, val_ds = {}, {}
    for mouse_id in mouse_ids:
        train_ds[mouse_id] = DataLoader(
            MovieDataset(
                tier="train",
                mouse_id=mouse_id,
                crop_frame=args.crop_frame,
                limit_data=args.limit_data,
                **dataset_kwargs,
            ),
            batch_size=batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_ds[mouse_id] = DataLoader(
            MovieDataset(tier="validation", mouse_id=mouse_id, **dataset_kwargs),
            batch_size=1,
            **loader_kwargs,
        )

    utils.set_shapes(args, ds=train_ds)
    args.max_frame = train_ds[mouse_ids[0]].dataset.max_frame

    return train_ds, val_ds


def get_submission_ds(
    args,
    data_dir: str,
    mouse_ids: List[str],
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
    num_workers: int = None,
):
    """
    Get DataLoaders for submission
    Args:
        args
        data_dir: str, path to directory where the zip files are stored
        mouse_ids: t.List[int], mouse IDs to extract
        batch_size: int, batch size of the DataLoaders
        device: torch.device, computing device
    Return:
        val_ds: Dict[str, DataLoader], dictionary of DataLoaders of the
            validation sets where keys are the mouse IDs.
        test_ds: Dict[str, Dict[str, DataLoader]]
            live_main: dictionary of DataLoaders of the live main test sets
                where keys are the mouse IDs.
            live_bonus: dictionary of DataLoaders of the live bonus test sets
                where keys are the mouse IDs.
            final_main: dictionary of DataLoaders of the final main test sets
                where keys are the mouse IDs.
            final_bonus: dictionary of DataLoaders of the final bonus test sets
                where keys are the mouse IDs.
    """
    assert batch_size == 1, "batch_size must be 1 for submission datasets"

    dataset_kwargs = {
        "ds_mode": args.ds_mode,
        "data_dir": data_dir,
        "mouse_ids": args.mouse_ids,
        "stat_mode": args.stat_mode,
        "transform_mode": args.transform_mode,
        "center_crop": args.center_crop if hasattr(args, "center_crop") else 1.0,
        "num_workers": args.num_workers,
    }
    loader_kwargs = utils.get_dataloader_kwargs(
        args, device=device, num_workers=num_workers
    )
    loader_kwargs["batch_size"] = 1

    val_ds = {}
    test_ds = {"live_main": {}, "live_bonus": {}, "final_main": {}, "final_bonus": {}}
    for mouse_id in mouse_ids:
        val_ds[mouse_id] = DataLoader(
            MovieDataset(tier="validation", mouse_id=mouse_id, **dataset_kwargs),
            **loader_kwargs,
        )
        test_ds["live_main"][mouse_id] = DataLoader(
            MovieDataset(tier="live_test_main", mouse_id=mouse_id, **dataset_kwargs),
            **loader_kwargs,
        )
        test_ds["live_bonus"][mouse_id] = DataLoader(
            MovieDataset(tier="live_test_bonus", mouse_id=mouse_id, **dataset_kwargs),
            **loader_kwargs,
        )
        test_ds["final_main"][mouse_id] = DataLoader(
            MovieDataset(tier="final_test_main", mouse_id=mouse_id, **dataset_kwargs),
            **loader_kwargs,
        )
        test_ds["final_bonus"][mouse_id] = DataLoader(
            MovieDataset(tier="final_test_bonus", mouse_id=mouse_id, **dataset_kwargs),
            **loader_kwargs,
        )

    utils.set_shapes(args, ds=val_ds)
    args.max_frame = val_ds[mouse_ids[0]].dataset.max_frame

    return val_ds, test_ds
