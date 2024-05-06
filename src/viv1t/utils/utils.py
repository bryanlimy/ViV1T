import copy
import gc
import os
import random
import subprocess
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from viv1t.metrics import single_trial_correlation
from viv1t.utils import yaml
from wandb.sdk.lib.runid import generate_id

import wandb


def get_timestamp() -> str:
    """Return timestamp in the format of YYYYMMDD-HHhMMm"""
    return f"{datetime.now():%Y%m%d-%Hh%Mm}"


def set_random_seed(seed: int, deterministic: bool = False):
    """Set random seed for Python, Numpy and PyTorch.
    Args:
        seed: int, the random seed to use.
        deterministic: bool, use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def get_device(device: str = None) -> torch.device:
    """return the appropriate torch.device if device is not set"""
    if not device:
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_num_threads(4)
            torch.set_num_interop_threads(8)
        elif torch.backends.mps.is_available():
            device = "mps"
    return torch.device(device)


def support_bf16(device: str):
    """Check if device supports bfloat16"""
    if isinstance(device, torch.device):
        device = device.type
    match device:
        case "cpu" | "mps":
            return False
        case "cuda":
            return torch.cuda.get_device_capability(device)[0] >= 8
        case _:
            raise KeyError(f"Unknown device type {device}.")


def clear_gpu_memory():
    """Move all tensors to CPU and clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            obj.cpu()
    gc.collect()
    torch.cuda.empty_cache()


def update_dict(target: Dict[str, Any], source: Dict[str, Any], replace: bool = False):
    """Update target dictionary with values from source dictionary"""
    for k, v in source.items():
        if replace:
            target[k] = v
        else:
            if k not in target:
                target[k] = []
            target[k].append(v)


def check_output(command: list):
    """Run command in subprocess and return output as string"""
    return subprocess.check_output(command).strip().decode()


def get_git_hash():
    # get git hash using current file directory
    return check_output(
        ["git", "-C", os.path.dirname(__file__), "describe", "--always"]
    )


def get_hostname():
    return check_output(["hostname"])


def save_args(args):
    """Save args object as dictionary to args.output_dir/args.json"""
    try:
        setattr(args, "git_hash", get_git_hash())
        setattr(args, "hostname", get_hostname())
    except subprocess.CalledProcessError as e:
        if args.verbose > 1:
            print(f"Unable to call subprocess: {e}")
    arguments = copy.deepcopy(args.__dict__)
    yaml.save(filename=os.path.join(args.output_dir, "args.yaml"), data=arguments)


def load_args(args):
    """Load args object from args.output_dir/args.yaml"""
    content = yaml.load(os.path.join(args.output_dir, "args.yaml"))
    for key, value in content.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def wandb_init(args, wandb_sweep: bool):
    """initialize wandb and strip information from args"""
    os.environ["WANDB_SILENT"] = "true"
    if not wandb_sweep:
        try:
            config = deepcopy(args.__dict__)
            config.pop("input_shapes", None)
            config.pop("output_shapes", None)
            config.pop("output_dir", None)
            config.pop("device", None)
            config.pop("format", None)
            config.pop("dpi", None)
            config.pop("save_plots", None)
            config.pop("verbose", None)
            config.pop("wandb", None)
            config.pop("trainable_params", None)
            config.pop("clear_output_dir", None)
            if args.wandb_id is None:
                wandb_id = generate_id()
                args.wandb_id = wandb_id
            else:
                wandb_id = args.wandb_id
            wandb.init(
                config=config,
                project="sensorium2023",
                entity="bryanlimy",
                group=args.wandb,
                name=os.path.basename(args.output_dir),
                resume="allow",
                id=wandb_id,
            )
            del config
        except AssertionError as e:
            print(f"wandb.init error: {e}\n")
            args.use_wandb = False
    if args.wandb is not None and hasattr(args, "trainable_params"):
        wandb.log({"trainable_params": args.trainable_params}, step=0)


def log_metrics(results: Dict[str, Dict[str, Any]]):
    """Compute the mean of the metrics in results and log to Summary

    Args:
        results: Dict[str, Dict[str, List[float]]],
            a dictionary of tensors where keys are the name of the metrics
            that represent results from of a mouse.
    """
    mouse_ids = list(results.keys())
    metrics = list(results[mouse_ids[0]].keys())
    for mouse_id in mouse_ids:
        for metric in metrics:
            value = results[mouse_id][metric]
            if isinstance(value, list):
                if torch.is_tensor(value[0]):
                    value = torch.stack(value).cpu().numpy()
                results[mouse_id][metric] = np.mean(value)
            elif torch.is_tensor(value):
                results[mouse_id][metric] = value.cpu().numpy()
    overall_result = {}
    for metric in metrics:
        value = np.mean([results[mouse_id][metric] for mouse_id in mouse_ids])
        overall_result[metric[metric.find("/") + 1 :]] = value
    return overall_result


def compile_model(
    args,
    model: nn.Module,
    mode: Literal["default", "reduce-overhead", "max-autotune"] = "default",
):
    """
    Compile model using torch.compile

    einops require additional call before compiling the model
    https://github.com/arogozhnikov/einops/wiki/Using-torch.compile-with-einops
    """
    if args.verbose:
        print(f"torch.compile with {args.backend} backend.")
    from einops._torch_specific import allow_ops_in_compiled_graph

    allow_ops_in_compiled_graph()
    model.core = torch.compile(model.core, backend=args.backend, mode=mode)


@torch.no_grad()
def inference(
    ds: DataLoader,
    model: nn.Module,
    device: torch.device = "cpu",
) -> Dict[str, List[torch.Tensor]]:
    """
    Inference data in test DataLoaders

    Given the test sets have variable frames, we therefore inference 1 sample
    at a time and return a list of (N, T) Tensor instead of a (B, N, T) Tensor.

    Returns:
        responses: Dict[str, torch.Tensor]
            - y_pred: List[torch.Tensor], list predicted responses in (N, T)
            - y_true: List[torch.Tensor], list of recorded responses in (N, T)
    """
    responses = {"y_true": [], "y_pred": []}
    mouse_id = ds.dataset.mouse_id
    model = model.to(device)
    model.train(False)
    to_batch = lambda x: torch.unsqueeze(x, dim=0).to(device)
    for i in range(len(ds.dataset.trial_ids)):
        sample = ds.dataset.__getitem__(i, to_tensor=True)
        predictions, _ = model(
            inputs=to_batch(sample["video"]),
            mouse_id=mouse_id,
            behaviors=to_batch(sample["behavior"]),
            pupil_centers=to_batch(sample["pupil_center"]),
        )
        responses["y_pred"].append(predictions.cpu()[0])
        responses["y_true"].append(sample["response"].cpu())
    return responses


def evaluate(
    args,
    ds: Dict[str, DataLoader],
    model: nn.Module,
    skip: int = 50,
    epoch: int = None,
    mode: int = 2,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate DataLoaders

    Args:
        args
        ds: Dict[str, DataLoader], dictionary of DataLoader, one for each mouse.
        model: nn.Module, the model.
        skip: int, number of frames to skip at the beginning of the trial when
            computing single trial correlation.
        epoch: int (optional), the current epoch number.
        mode: int (optional), Summary mode.
    Returns:
        results: Dict[str, Union[float, np.ndarray]], single trial correlation
            between recorded and predicted responses for each mouse and the
            average correlation across animals.
    """
    results = {}
    tier = list(ds.values())[0].dataset.tier

    for mouse_id, mouse_ds in tqdm(
        ds.items(), desc=f"Evaluate {tier}", disable=args.verbose < 2
    ):
        if mouse_ds.dataset.hidden_response:
            continue  # skip dataset with no response labels
        responses = inference(ds=mouse_ds, model=model, device=args.device)

        # crop response and prediction to the same length after skipping frames
        for i in range(len(responses["y_true"])):
            t = responses["y_true"][i].shape[1] - skip
            responses["y_true"][i] = responses["y_true"][i][:, -t:]
            responses["y_pred"][i] = responses["y_pred"][i][:, -t:]

        results[mouse_id] = single_trial_correlation(
            y_true=responses["y_true"], y_pred=responses["y_pred"]
        ).item()

        if summary is not None:
            summary.scalar(
                f"single_trial_correlation/{tier}/mouse{mouse_id}",
                value=results[mouse_id],
                step=epoch,
                mode=mode,
            )
        del responses
        torch.cuda.empty_cache()

    correlations = list(results.values())
    if len(correlations):
        results["average"] = np.mean(correlations)
        if summary is not None:
            summary.scalar(
                f"single_trial_correlation/{tier}/average",
                value=results["average"],
                step=epoch,
                mode=mode,
            )
    return results


def get_critic_accuracy(results: Dict[str, Dict[str, Any]]):
    """Get overall critic accuracy from results dictionary"""
    critic_acc = []
    for mouse_id in results.keys():
        critic_acc.append(torch.concat(results[mouse_id]["acc/critic_acc"]))
    critic_acc = torch.concat(critic_acc)
    critic_acc = torch.sum(critic_acc) / len(critic_acc)
    for mouse_id in results.keys():
        results[mouse_id]["acc/critic_acc"] = critic_acc
    return critic_acc


def restore(
    args: Any, model: nn.Module, filename: str, val_ds: Dict[str, DataLoader] = None
):
    """restore model weights from checkpoint"""
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    device = model.device
    model = model.to("cpu")
    ckpt = torch.load(filename, map_location="cpu")
    state_dict = model.state_dict()
    num_params = 0
    for k in ckpt["model"].keys():
        if k in state_dict:
            state_dict[k] = ckpt["model"][k]
            num_params += 1
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(
        f"\nLoaded {num_params} parameters from {filename} "
        f"(epoch {ckpt['epoch']}, correlation: {ckpt['value']:.04f}).\n"
    )
    del ckpt

    if val_ds is not None:
        results = evaluate(args, ds=val_ds, model=model)
        print(f"Validation correlation: {results['average']:.04f}")
