import os
import warnings
from typing import Any, Dict, Mapping

import torch
import torchinfo
from einops._torch_specific import allow_ops_in_compiled_graph
from torch import nn
from torch.utils.data import DataLoader

from viv1t.data import estimate_mean_response
from viv1t.model.core import get_core
from viv1t.model.critic import Critic
from viv1t.model.helper import ELU1, Exponential
from viv1t.model.modulators.modulator import Modulators
from viv1t.model.readout import Readouts
from viv1t.model.shifter import MLPShifters


def get_model_info(
    model: nn.Module,
    input_data: Mapping[str, Any],
    mouse_id: str = None,
    filename: str = None,
    device: torch.device = "cpu",
):
    args = {
        "model": model,
        "input_data": input_data,
        "col_names": ["input_size", "output_size", "num_params"],
        "depth": 8,
        "device": device,
        "verbose": 0,
    }
    if mouse_id is not None:
        args["mouse_id"] = mouse_id

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model_info = torchinfo.summary(**args)

    if filename is not None:
        with open(filename, "w") as file:
            file.write(str(model_info))
    del input_data
    return model_info


class Model(nn.Module):
    def __init__(
        self,
        args: Any,
        neuron_coordinates: Dict[str, torch.Tensor],
        mean_responses: Dict[str, torch.Tensor] = None,
    ):
        super(Model, self).__init__()
        self.input_shapes = args.input_shapes
        self.output_shapes = args.output_shapes
        self.grad_norm = args.grad_norm if hasattr(args, "grad_norm") else None
        self.register_buffer("verbose", torch.tensor(args.verbose), persistent=False)

        self.add_module(
            name="core",
            module=get_core(args)(args, input_shape=self.input_shapes["video"]),
        )

        if args.shifter_mode:
            self.add_module(
                "shifters",
                module=MLPShifters(
                    args,
                    input_shapes=self.input_shapes,
                    mouse_ids=list(self.output_shapes.keys()),
                ),
            )
        else:
            self.shifters = None

        self.add_module(
            name="readouts",
            module=Readouts(
                args,
                model=args.readout,
                input_shape=self.core.output_shape,
                neuron_coordinates=neuron_coordinates,
                mean_responses=mean_responses,
            ),
        )

        if args.modulator_mode:
            self.add_module(
                "modulators", module=Modulators(args, input_shapes=self.output_shapes)
            )
        else:
            self.modulators = None

        match args.output_mode:
            case 0:
                self.output_activation = ELU1()
            case 1:
                self.output_activation = Exponential()
            case 2:
                self.output_activation = nn.Softplus()
            case _:
                raise NotImplementedError(
                    f"output_mode {args.output_mode} not implemented."
                )

    @property
    def device(self) -> torch.device:
        """return the device that the model parameters is on"""
        return next(self.parameters()).device

    def get_parameters(self):
        """Return a list of parameters for torch.optim.Optimizer"""
        params = []
        params.extend(self.core.get_parameters())
        params.append({"params": self.readouts.parameters(), "name": "readouts"})
        if self.shifters is not None:
            params.append({"params": self.shifters.parameters(), "name": "shifters"})
        if self.modulators is not None:
            params.extend(self.modulators.get_parameters())
        return params

    def compile_core(self, backend: str = "inductor"):
        if self.verbose:
            print(f"Compile core module with {backend} backend.")
        allow_ops_in_compiled_graph()
        self.core = torch.compile(self.core, fullgraph=True, backend=backend)

    def get_grad_norm(self):
        """return the gradient norm of the model parameters"""
        grads = [
            p.grad.detach().flatten() for p in self.parameters() if p.grad is not None
        ]
        return torch.cat(grads).norm()

    def clip_grad_norm(self):
        if self.grad_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_norm)

    def regularizer(self, mouse_id: str):
        reg = 0
        if not self.core.frozen:
            reg += self.core.regularizer()
        reg += self.readouts.regularizer(mouse_id=mouse_id)
        if self.shifters is not None:
            reg += self.shifters.regularizer(mouse_id=mouse_id)
        if self.modulators is not None:
            reg += self.modulators.regularizer(mouse_id=mouse_id)
        return reg

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
        activate: bool = True,
    ):
        core_outputs = self.core(
            inputs=inputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        shifts, t = None, core_outputs.size(2)
        if self.shifters is not None:
            shifts = self.shifters(
                behaviors=behaviors[..., -t:],
                pupil_centers=pupil_centers[..., -t:],
                mouse_id=mouse_id,
            )
        outputs = self.readouts(
            core_outputs,
            mouse_id=mouse_id,
            shifts=shifts,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        if self.modulators is not None:
            outputs = self.modulators(
                outputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
            )
        if activate:
            outputs = self.output_activation(outputs)
        return outputs, core_outputs.detach()


def get_model(args, train_ds: Dict[str, DataLoader]) -> (Model, Critic):
    model = Model(
        args,
        neuron_coordinates={
            mouse_id: mouse_ds.dataset.neuron_coordinates
            for mouse_id, mouse_ds in train_ds.items()
        },
        mean_responses=estimate_mean_response(train_ds),
    )

    # get model info
    mouse_id = args.mouse_ids[0]
    random_tensor = lambda size: torch.rand((1, *size))
    model_info = get_model_info(
        model=model,
        input_data={
            "inputs": random_tensor(args.input_shapes["video"]),
            "behaviors": random_tensor(args.input_shapes["behavior"]),
            "pupil_centers": random_tensor(args.input_shapes["pupil_center"]),
        },
        mouse_id=mouse_id,
        filename=os.path.join(args.output_dir, "model.txt"),
    )
    args.trainable_params = model_info.trainable_params
    if args.verbose > 2:
        print(str(model_info))
    del model_info

    critic = None
    if args.critic_mode:
        critic = Critic(args, input_shape=model.core.output_shape)

        critic_info = get_model_info(
            model=critic,
            input_data={"inputs": random_tensor(critic.input_shape)},
            filename=os.path.join(args.output_dir, "critic.txt"),
        )
        if args.verbose > 2:
            print(str(critic_info))
        del critic_info
        critic = critic.to(args.device)

    if hasattr(args, "core_compile") and args.core_compile:
        model.compile_core()

    return model.to(args.device), critic
