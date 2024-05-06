import argparse
import os
from argparse import RawTextHelpFormatter
from shutil import rmtree
from time import time
from typing import Dict, Tuple, Union

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from viv1t import data, metrics
from viv1t.criterions import Criterion, get_criterion
from viv1t.data import CycleDataloaders
from viv1t.model import Model, get_model
from viv1t.scheduler import Scheduler
from viv1t.utils import logger, utils, yaml
from viv1t.utils.estimate_batch_size import estimate_batch_size
 

def train_step(
    mouse_id: str,
    batch: Dict[str, torch.Tensor],
    model: Model,
    optimizer: torch.optim,
    criterion: Criterion,
    update: bool,
    micro_batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    b = batch["video"].size(0)
    result, responses = {}, {"y_true": [], "y_pred": []}
    batch_loss = torch.tensor(0.0, device=device)
    for micro_batch in data.micro_batching(batch, micro_batch_size):
        y_pred, core_outputs = model(
            inputs=micro_batch["video"].to(device),
            mouse_id=mouse_id,
            behaviors=micro_batch["behavior"].to(device),
            pupil_centers=micro_batch["pupil_center"].to(device),
        )
        y_true = micro_batch["response"][..., -y_pred.size(2) :].to(device)
        loss = criterion(y_true=y_true, y_pred=y_pred, mouse_id=mouse_id, batch_size=b)
        loss.backward()
        batch_loss += loss.detach()
        responses["y_true"].append(y_true.detach())
        responses["y_pred"].append(y_pred.detach())
    if update:
        result["grad_norm"] = model.get_grad_norm()
        model.clip_grad_norm()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    result["loss/loss"] = batch_loss.cpu()
    result["metrics/single_trial_correlation"] = metrics.single_trial_correlation(
        y_true=torch.vstack(responses["y_true"]),
        y_pred=torch.vstack(responses["y_pred"]),
    ).cpu()
    del responses
    return result


def train(
    args,
    ds: CycleDataloaders,
    model: Model,
    optimizer: torch.optim,
    criterion: Criterion,
    epoch: int,
) -> Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    results, grad_norms = {mouse_id: {} for mouse_id in args.mouse_ids}, []
    # accumulate gradients over all mouse for one batch
    update_frequency = len(args.mouse_ids)
    model = model.to(args.device)
    model.train(True)
    optimizer.zero_grad(set_to_none=True)
    for i, (mouse_id, mouse_batch) in tqdm(
        enumerate(ds), desc="Train", total=len(ds), disable=args.verbose < 2
    ):
        result = train_step(
            mouse_id=mouse_id,
            batch=mouse_batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            update=(i + 1) % update_frequency == 0,
            micro_batch_size=args.micro_batch_size,
            device=args.device,
        )
        if "grad_norm" in result:
            grad_norms.append(result.pop("grad_norm"))
        utils.update_dict(results[mouse_id], result)
    if args.wandb is not None and grad_norms:
        wandb.log({"grad_norm": torch.mean(torch.stack(grad_norms))}, step=epoch)
    return utils.log_metrics(results)


@torch.no_grad()
def validation_step(
    mouse_id: str,
    batch: Dict[str, torch.Tensor],
    model: Model,
    criterion: Criterion,
    device: torch.device,
    skip: int = 50,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    result = {}
    b, t = batch["video"].size(0), batch["video"].size(2) - skip
    y_pred, core_outputs = model(
        inputs=batch["video"].to(device),
        mouse_id=mouse_id,
        behaviors=batch["behavior"].to(device),
        pupil_centers=batch["pupil_center"].to(device),
    )
    y_pred, y_true = y_pred[..., -t:], batch["response"][..., -t:].to(device)
    result["loss/loss"] = criterion(
        y_true=y_true, y_pred=y_pred, mouse_id=mouse_id, batch_size=b
    )
    return result, {"y_true": y_true, "y_pred": y_pred}


def validate(
    args, ds: Dict[str, DataLoader], model: Model, criterion: Criterion
) -> Dict[Union[str, int], Union[torch.Tensor, Dict[str, torch.Tensor]]]:
    model = model.to(args.device)
    model.train(False)
    results = {}
    with tqdm(desc="Val", total=data.num_steps(ds), disable=args.verbose < 2) as pbar:
        for mouse_id, mouse_ds in ds.items():
            mouse_result = {}
            mouse_responses = {"y_true": [], "y_pred": []}
            for batch in mouse_ds:
                batch_result, responses = validation_step(
                    mouse_id=mouse_id,
                    batch=batch,
                    model=model,
                    criterion=criterion,
                    device=args.device,
                )
                utils.update_dict(mouse_result, batch_result)
                utils.update_dict(mouse_responses, responses)
                del responses
                pbar.update(1)
            mouse_result.update(
                metrics.compute_metrics(
                    y_true=torch.vstack(mouse_responses["y_true"]),
                    y_pred=torch.vstack(mouse_responses["y_pred"]),
                )
            )
            results[mouse_id] = mouse_result
            del mouse_result, mouse_responses
    return utils.log_metrics(results)


def main(args, wandb_sweep: bool = False) -> float:
    if args.clear_output_dir and os.path.isdir(args.output_dir):
        rmtree(args.output_dir)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    logger.Logger(os.path.join(args.output_dir, "output.log"))
    args.device = utils.get_device(args.device)
    utils.set_random_seed(args.seed, deterministic=args.deterministic)

    data.get_mouse_ids(args)
    estimate_batch_size(args)
    train_ds, val_ds = data.get_training_ds(
        args,
        data_dir=args.data,
        mouse_ids=args.mouse_ids,
        batch_size=args.batch_size,
        device=args.device,
    )

    model, _ = get_model(args, train_ds=train_ds)
    optimizer = torch.optim.AdamW(
        params=model.get_parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    criterion = get_criterion(args, ds=train_ds)
    scheduler = Scheduler(args, model=model, optimizer=optimizer, mode="max")

    utils.save_args(args)
    epoch = scheduler.restore(load_optimizer=True, load_scheduler=True)

    if args.wandb is not None:
        utils.wandb_init(args, wandb_sweep=wandb_sweep)

    if hasattr(args, "restore") and args.restore is not None:
        utils.restore(args, model=model, filename=args.restore, val_ds=val_ds)

    train_ds = CycleDataloaders(train_ds)

    while (epoch := epoch + 1) < args.epochs + 1:
        if args.verbose:
            print(f"\nEpoch {epoch:03d}/{args.epochs:03d}")

        start = time()
        train_result = train(
            args,
            ds=train_ds,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
        )
        val_result = validate(
            args,
            ds=val_ds,
            model=model,
            criterion=criterion,
            epoch=epoch,
        )
        elapse = time() - start

        if args.verbose:
            print(
                f'Train\t\tloss: {train_result["loss"]:.02f}\t'
                f'correlation: {train_result["single_trial_correlation"]:.04f}\n'
                f'Validation\tloss: {val_result["loss"]:.02f}\t'
                f'correlation: {val_result["single_trial_correlation"]:.04f}\n'
                f"Elapse: {elapse:.02f}s"
            )
        early_stop = scheduler.step(val_result["single_trial_correlation"], epoch=epoch)

        report = {
            "train_loss": train_result["loss"],
            "train_corr": train_result["single_trial_correlation"],
            "val_loss": val_result["loss"],
            "val_corr": val_result["single_trial_correlation"],
            "val_poisson": val_result["poisson_loss"],
            "best_corr": scheduler.best_value,
            "learning_rate": optimizer.defaults["lr"],
            "elapse": elapse,
            "epoch": epoch,
        }
        if args.wandb is not None:
            wandb.log(report, step=epoch)

        if np.isnan(train_result["loss"]) or np.isnan(val_result["loss"]):
            if args.wandb is not None:
                wandb.finish(exit_code=1)  # mark run as failed
            raise ValueError("\nNaN loss detected, terminate training.")
        if early_stop:
            break

    scheduler.restore()

    eval_result = utils.evaluate(args, ds=val_ds, model=model)
    print_result = lambda d: "\t".join([f"{k}: {v:.04f}" for k, v in d.items()])
    if args.verbose:
        statement = "\nValidation"
        statement += print_result(eval_result)
        print(statement)
    yaml.save(
        os.path.join(args.output_dir, "evaluation.yaml"),
        data={"validation": eval_result},
    )

    if args.verbose:
        print(f"\nResults saved to {args.output_dir}.")

    if args.wandb is not None:
        wandb.finish(exit_code=0)

    return eval_result["average"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    # dataset settings
    parser.add_argument(
        "--data",
        type=str,
        default="data/sensorium",
        help="path to directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to directory to log training performance and model checkpoint.",
    )
    parser.add_argument(
        "--ds_mode",
        type=int,
        required=True,
        choices=[0, 1, 2, 3],
        help="0: train on the 5 original mice\n"
        "1: train on the 5 new mice\n"
        "2: train on all 10 mice jointly\n"
        "3: train on all 10 mice with all tiers from the 5 original mice\n",
    )
    parser.add_argument(
        "--stat_mode",
        type=int,
        default=1,
        choices=[0, 1],
        help="data statistics to use:\n"
        "0: use the provided statistics\n"
        "1: compute statistics from the training set",
    )
    parser.add_argument(
        "--transform_mode",
        type=int,
        choices=[0, 1, 2, 3, 4],
        required=True,
        help="data transformation and preprocessing\n"
        "0: apply no transformation\n"
        "1: standard response using statistics over trial\n"
        "2: normalize response using statistics over trial\n"
        "3: standard response using statistics over trial and time\n"
        "4: normalize response using statistics over trial and time\n",
    )
    parser.add_argument(
        "--center_crop",
        type=float,
        default=1.0,
        help="center crop the video frame to center_crop percentage.",
    )
    parser.add_argument(
        "--mouse_ids",
        nargs="+",
        type=str,
        default=None,
        help="Mouse to use for training.",
    )
    parser.add_argument(
        "--limit_data",
        type=int,
        default=None,
        help="limit the number of training samples.",
    )
    parser.add_argument(
        "--cache_data",
        action="store_true",
        help="cache data in memory in MovieDataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="number of workers for DataLoader.",
    )

    # training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="maximum epochs to train the model.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="micro batch size to train the model. if the model is being "
        "trained on CUDA device and micro batch size 0 is provided, then "
        "automatically increase micro batch size until OOM.",
    )
    parser.add_argument(
        "--crop_frame",
        type=int,
        default=150,
        help="number of frames to take from each trial.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for computation. "
        "use the best available device if --device is not specified.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="use deterministic algorithms in PyTorch",
    )
    parser.add_argument("--precision", type=str, default="32", choices=["32", "bf16"])
    parser.add_argument(
        "--grad_checkpointing",
        type=int,
        default=None,
        choices=[0, 1],
        help="Enable gradient checkpointing for supported models if set to 1. "
        "If None is provided, then enable by default if CUDA is detected.",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="pretrained model to restore from before training begins.",
    )

    # optimizer settings
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument(
        "--criterion",
        type=str,
        default="poisson_correlation",
        help="criterion (loss function) to use.",
    )
    parser.add_argument(
        "--ds_scale",
        type=int,
        default=1,
        choices=[0, 1],
        help="scale loss by the size of the dataset",
    )
    parser.add_argument(
        "--grad_norm",
        type=float,
        default=None,
        help="max value for gradient norm clipping. set None to disable",
    )

    # plot settings
    parser.add_argument(
        "--save_plots", action="store_true", help="save plots to --output_dir"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="matplotlib figure DPI",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg",
        choices=["pdf", "svg", "png"],
        help="file format when --save_plots",
    )

    # misc
    parser.add_argument(
        "--wandb",
        type=str,
        default=None,
        help="wandb group name, disable wandb logging if not provided.",
    )
    parser.add_argument(
        "--wandb_id", type=str, default=None, help="wandb run ID to resume from."
    )
    parser.add_argument(
        "--clear_output_dir",
        action="store_true",
        help="overwrite content in --output_dir",
    )
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2, 3])

    # model settings
    parser.add_argument(
        "--core",
        type=str,
        required=True,
        help="The core module to use.",
    )
    parser.add_argument(
        "--core_compile",
        action="store_true",
        help="compile core module with inductor backend via torch.compile",
    )
    parser.add_argument(
        "--readout",
        type=str,
        required=True,
        help="The readout module to use.",
    )
    parser.add_argument(
        "--shifter_mode",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0: disable shifter\n"
        "1: learn shift from pupil center\n"
        "2: learn shift from pupil center and behavior variables",
    )
    parser.add_argument(
        "--modulator_mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="0: disable modulator\n"
        "1: MLP Modulator\n"
        "2: GRU Modulator\n"
        "3: MLP-v2 Modulator\n"
        "4: MLP-v3 Modulator",
    )
    parser.add_argument("--critic_mode", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--output_mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Output activation:\n"
        "0: ELU + 1 activation\n"
        "1: Exponential activation\n"
        "2: SoftPlus activation",
    )

    temp_args = parser.parse_known_args()[0]

    # hyper-parameters for core module
    match temp_args.core:
        case "factorized_baseline":
            parser.add_argument("--lr", type=float, default=0.005)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=1,
                choices=[0, 1, 2, 6, 7, 8, 9],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input\n"
                "6: --behavior_mode 2 and their first-order derivative\n"
                "7: --behavior_mode 2 and pixel positional encoding\n"
                "8: --behavior_mode 2 with GRU modulator\n"
                "9: --behavior_mode 7 and 8",
            )
            # factorized 3D CNN settings
            parser.add_argument("--core_spatial_input_kernel", type=int, default=11)
            parser.add_argument("--core_temporal_input_kernel", type=int, default=11)
            parser.add_argument("--core_spatial_hidden_kernel", type=int, default=5)
            parser.add_argument("--core_temporal_hidden_kernel", type=int, default=5)
            parser.add_argument("--core_num_layers", type=int, default=4)
            parser.add_argument("--core_hidden_dim", type=int, default=16)
            parser.add_argument("--core_dropout", type=float, default=0.0)
        case "vivit":
            parser.add_argument("--lr", type=float, default=0.0036)
            parser.add_argument("--core_lr", type=float, default=0.0048)
            parser.add_argument("--weight_decay", type=float, default=0.3939)
            parser.add_argument("--core_weight_decay", type=float, default=0.1789)
            # ViViT settings
            parser.add_argument(
                "--core_behavior_mode",
                type=int,
                default=2,
                choices=[0, 1, 2, 3, 4, 5],
                help="0: do not include behavior\n"
                "1: concat behavior with visual input\n"
                "2: concat behavior and pupil center with visual input\n"
                "3: feed behavior to B-MLP\n"
                "4: feed behavior and pupil center to B-MLP\n"
                "5: feed behavior and pupil center to per-animal B-MLP\n",
            )
            parser.add_argument(
                "--core_patch_mode",
                type=int,
                default=0,
                choices=[0, 1, 2],
                help="3D patch extraction via:\n"
                "0: tensor.unfold followed linear projection\n"
                "1: F.conv3d with identity weight followed by linear projection\n"
                "2: nn.Conv3d layer",
            )
            parser.add_argument(
                "--core_pos_encoding",
                type=int,
                default=3,
                choices=[0, 1, 2, 3, 4, 5],
                help="Positional encoding:\n"
                "0: no positional encoding\n"
                "1: learnable positional encoding\n"
                "2: PositionalEncodingGenerator (PEG, Chu et al. 2023)\n"
                "3: separate learnable spatial and temporal positional encoding\n"
                "4: learnable spatial positional encoding and sinusoidal temporal positional encoding\n"
                "5: sinusoidal spatial and temporal positional encoding\n",
            )
            parser.add_argument("--core_spatial_patch_size", type=int, default=7)
            parser.add_argument(
                "--core_spatial_patch_stride",
                type=int,
                default=2,
                help="stride size to extract spatial patches",
            )
            parser.add_argument("--core_spatial_depth", type=int, default=3)
            parser.add_argument("--core_temporal_patch_size", type=int, default=25)
            parser.add_argument(
                "--core_temporal_patch_stride",
                type=int,
                default=1,
                help="stride size to extract temporal patches",
            )
            parser.add_argument("--core_temporal_depth", type=int, default=5)
            parser.add_argument("--core_num_heads", type=int, default=11)
            parser.add_argument("--core_emb_dim", type=int, default=112)
            parser.add_argument("--core_head_dim", type=int, default=48)
            parser.add_argument("--core_mlp_dim", type=int, default=136)
            parser.add_argument(
                "--core_ff_activation",
                type=str,
                default="gelu",
                choices=["gelu", "swiglu"],
                help="Transformer block FF activation function",
            )
            parser.add_argument(
                "--core_p_dropout",
                type=float,
                default=0.1338,
                help="patch embeddings dropout",
            )
            parser.add_argument(
                "--core_mha_dropout",
                type=float,
                default=0.3580,
                help="Transformer block MHA dropout",
            )
            parser.add_argument(
                "--core_ff_dropout",
                type=float,
                default=0.0592,
                help="Transformer block FF dropout",
            )
            parser.add_argument(
                "--core_drop_path",
                type=float,
                default=0.0505,
                help="stochastic depth dropout rate",
            )
            parser.add_argument("--core_parallel_attention", action="store_true")
            parser.add_argument(
                "--core_flash_attention", type=int, default=1, choices=[0, 1]
            )
            parser.add_argument("--core_norm_qk", action="store_true")
            parser.add_argument("--pretrain_core", type=str, default=None)
        case _:
            parser.add_argument("--lr", type=float, default=0.001)
            parser.add_argument("--core_lr", type=float, default=None)
            parser.add_argument("--weight_decay", type=float, default=0.0)
            parser.add_argument("--core_weight_decay", type=float, default=0.0)

    # hyper-parameters for readout modules
    match temp_args.readout:
        case "gaussian2d":
            parser.add_argument(
                "--readout_grid_mode",
                type=int,
                default=1,
                choices=[0, 1, 2],
                help="Readout grid predictor mode:\n"
                "0: disable grid predictor\n"
                "1: grid predictor using (x, y) neuron coordinates\n"
                "2: grid predictor using (x, y, z) neuron coordinates",
            )
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
            parser.add_argument("--readout_dropout", type=float, default=0.0)
        case "factorized":
            parser.add_argument(
                "--readout_bias_mode",
                type=int,
                default=2,
                choices=[0, 1, 2],
                help="Gaussian2d readout bias mode:\n"
                "0: disable bias term\n"
                "1: initialize bias with zeros\n"
                "2: initialize bias with the mean responses",
            )
            parser.add_argument("--readout_dropout", type=float, default=0.0)

    # hyper-parameters for shifter module
    if temp_args.shifter_mode > 0:
        parser.add_argument("--shifter_layers", type=int, default=3)
        parser.add_argument("--shifter_size", type=int, default=5)

    # hyper-parameters for modulator module
    match temp_args.modulator_mode:
        case 1:  # MLP Modulator
            parser.add_argument(
                "--modulator_include_history", type=int, default=1, choices=[0, 1]
            )
            parser.add_argument("--modulator_history_size", type=int, default=5)
            parser.add_argument("--modulator_history_dropout", type=float, default=0.0)
            parser.add_argument(
                "--modulator_include_behaviors", type=int, default=2, choices=[0, 1, 2]
            )
            parser.add_argument(
                "--modulator_behaviors_dropout", type=float, default=0.0
            )
            parser.add_argument("--modulator_weight_decay", type=float, default=0.0)
        case 2:  # GRU Modulator
            parser.add_argument("--modulator_weight_decay", type=float, default=0.0)
            parser.add_argument("--modulator_hidden_dim", type=int, default=16)
            parser.add_argument("--modulator_dropout", type=float, default=0.0)
        case 3:  # MLP-v2 modulator
            parser.add_argument("--modulator_history_size", type=int, default=5)
            parser.add_argument("--modulator_history_dropout", type=float, default=0.0)
            parser.add_argument("--modulator_weight_decay", type=float, default=0.0)
        case 4:  # MLP-v3 modulator
            parser.add_argument(
                "--modulator_activation",
                type=str,
                default="tanh",
                choices=["identity", "sigmoid", "tanh"],
            )
            parser.add_argument("--modulator_dropout", type=float, default=0.0)
            parser.add_argument("--modulator_weight_decay", type=float, default=0.0)

    match temp_args.critic_mode:
        case 1:
            parser.add_argument("--critic_hidden_dim", type=int, default=64)
            parser.add_argument("--critic_dropout", type=float, default=0.2)
            parser.add_argument("--critic_weight_decay", type=float, default=0.01)

    del temp_args

    main(parser.parse_args())
