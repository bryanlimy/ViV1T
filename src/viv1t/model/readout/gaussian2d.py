"""
Code reference: https://github.com/sinzlab/neuralpredictors/blob/9b85300ab854be1108b4bf64b0e4fa2e960760e0/neuralpredictors/layers/readouts/gaussian.py
"""

from typing import Any, Literal, Tuple, Union

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from torch import nn
from torch.nn import functional as F

from viv1t.model.readout.readout import Readout, register

REDUCTIONS = Literal["sum", "mean", None]


@register("gaussian2d")
class Gaussian2DReadout(Readout):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        mouse_id: str,
        neuron_coordinates: Union[np.ndarray, torch.Tensor],
        mean_responses: torch.Tensor = None,
        init_mu_range: float = 0.2,
        init_sigma: float = 1.0,
        gaussian_type: str = "full",
    ):
        """
        Grid predictor mode (--readout_grid_mode):
            0: disable grid predictor
            1: grid predictor using (x, y) neuron coordinates
            2: grid predictor using (x, y, z) neuron coordinates
        """
        super(Gaussian2DReadout, self).__init__(
            args,
            input_shape=input_shape,
            mouse_id=mouse_id,
            neuron_coordinates=neuron_coordinates,
            mean_responses=mean_responses,
        )
        self.bias_mode = args.readout_bias_mode

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or "
                "init_sigma_range is non-positive"
            )
        self.init_mu_range = init_mu_range

        # position grid shape
        self.grid_shape = (1, self.num_neurons, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False
        self._original_grid = not self._predicted_grid

        match args.readout_grid_mode:
            case 0:
                # mean location of gaussian for each neuron
                self._mu = nn.Parameter(torch.Tensor(*self.grid_shape))
            case 1:
                # input neuron (x, y) coordinates
                self.init_grid_predictor(
                    source_grid=neuron_coordinates,
                    input_dimensions=2,
                )
            case 2:
                # input neuron (x, y, z) coordinates
                self.init_grid_predictor(
                    source_grid=neuron_coordinates,
                    input_dimensions=3,
                )
            case _:
                raise NotImplementedError(
                    f"--readout_grid_mode {args.readout_grid_mode} not implemented."
                )

        self.gaussian_type = gaussian_type
        match self.gaussian_type:
            case "full":
                sigma_shape = (1, self.num_neurons, 2, 2)
            case "uncorrelated":
                sigma_shape = (1, self.num_neurons, 1, 2)
            case "isotropic":
                sigma_shape = (1, self.num_neurons, 1, 1)
            case _:
                raise NotImplementedError(
                    f"Gaussian type {gaussian_type} not implemented."
                )

        self.init_sigma = init_sigma
        # standard deviation for gaussian for each neuron
        self.sigma = nn.Parameter(torch.Tensor(*sigma_shape))

        state_dim = input_shape[0]
        self.state_encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=state_dim, bias=True),
            nn.LayerNorm(normalized_shape=state_dim),
            nn.GELU(),
            nn.Dropout(p=args.readout_dropout),
            nn.Linear(in_features=state_dim, out_features=state_dim, bias=True),
            nn.Tanh(),
        )

        self.initialize_features()
        self.initialize(mean_responses=mean_responses)

    def feature_l1(self, reduction: REDUCTIONS = "sum"):
        """
        Returns l1 regularization term for features.
        Args:
            reduction: str, Specifies the reduction to apply to the output:
                            'none' | 'mean' | 'sum'
        """
        l1 = 0
        if self._original_features:
            l1 = self.features.abs()
            if reduction == "sum":
                l1 = l1.sum()
            elif reduction == "mean":
                l1 = l1.mean()
        return l1

    def regularizer(self, reduction: REDUCTIONS = "sum"):
        return self.reg_scale * self.feature_l1(reduction=reduction)

    def init_grid_predictor(
        self,
        source_grid: torch.Tensor,
        hidden_features: int = 30,
        hidden_layers: int = 1,
        tanh_output: bool = True,
        input_dimensions: int = 2,
    ):
        self._original_grid = False
        source_grid = source_grid[:, :input_dimensions]

        layers = [
            nn.Linear(
                in_features=source_grid.shape[1],
                out_features=hidden_features if hidden_layers > 0 else 2,
            )
        ]
        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),
                    nn.Linear(
                        in_features=hidden_features,
                        out_features=hidden_features if i < hidden_layers - 1 else 2,
                    ),
                ]
            )
        if tanh_output:
            layers.append(nn.Tanh())
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(dim=0, keepdims=True)
        source_grid = source_grid / source_grid.abs().max()
        self.register_buffer("source_grid", source_grid)
        self._predicted_grid = True

    def initialize_features(self):
        """
        The internal attribute `_original_features` in this function denotes
        whether this instance of the FullGaussian2d learns the original
        features (True) or if it uses a copy of the features from another
        instance of FullGaussian2d via the `shared_features` (False). If it
        uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, t, h, w = self.input_shape
        self._original_features = True
        # feature weights for each channel of the core
        self.features = nn.Parameter(torch.Tensor(1, c, self.num_neurons))
        self._shared_features = False

    def initialize_bias(self, mean_responses: torch.Tensor = None):
        match self.bias_mode:
            case 0:
                bias = None
            case 1:
                bias = torch.zeros(self.num_neurons)
            case 2:
                if mean_responses is None:
                    bias = torch.zeros(self.num_neurons)
                else:
                    bias = mean_responses
            case _:
                raise NotImplementedError(
                    f"--bias_mode {self.bias_mode} not implemented."
                )
        self.register_parameter("bias", None if bias is None else nn.Parameter(bias))

    def initialize(self, mean_responses: torch.Tensor):
        """
        Initializes the mean, and sigma of the Gaussian readout along with
        the features weights
        """
        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gaussian_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self.features.data.fill_(1 / self.input_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)
        self.initialize_bias(mean_responses=mean_responses)

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size: Union[int, torch.Tensor], sample: bool = None):
        """
        Returns the grid locations from the core by sampling from a Gaussian
        distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample
                                from Gaussian distribution, N(mu,sigma),
                                defined per neuron or use the mean, mu, of the
                                Gaussian distribution without sampling.
                                If sample is None (default), samples from the
                                N(mu,sigma) during training phase and fixes to
                                the mean, mu, during evaluation phase.
                                If sample is True/False, overrides the
                                model_state (i.e. training or eval) and does as
                                instructed
        """
        with torch.no_grad():
            # at eval time, only self.mu is used, so it must belong to [-1,1]
            # sigma/variance is always a positive quantity
            self.mu.clamp_(min=-1, max=1)

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            # for consistency and CUDA capability
            norm = self.mu.new(*grid_shape).zero_()

        if self.gaussian_type != "full":
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(norm * self.sigma + self.mu, min=-1, max=1)
        else:
            # grid locations in feature space sampled randomly around the mean self.mu
            return torch.clamp(
                einsum(self.sigma, norm, "a n c d, b n i d->b n i c") + self.mu,
                min=-1,
                max=1,
            )

    def forward(
        self,
        inputs: torch.Tensor,
        sample: bool = None,
        shifts: torch.Tensor = None,
        behaviors: torch.Tensor = None,
        pupil_centers: torch.Tensor = None,
    ):
        """
        Propagates the input forwards through the readout
        Args:
            inputs: torch.Tensor, visual representation in (B, C, T, H, W)
            sample: bool, sample determines whether we draw a sample from
                Gaussian distribution, N(mu,sigma), defined per neuron or use
                the mean, mu, of the Gaussian distribution without sampling.
                If sample is None (default), samples from the N(mu,sigma)
                during training phase and fixes to the mean, mu, during
                evaluation phase.
                If sample is True/False, overrides the model_state (i.e.
                training or eval) and does as instructed
            shifts: torch.Tensor, shifts the location of the grid from
                eye-tracking data
        """
        b, c, t, h, w = inputs.size()
        outputs = rearrange(inputs, "b c t h w -> (b t) c h w")
        # sample the grid_locations separately per image per batch
        grid = self.sample_grid(batch_size=b * t, sample=sample)
        if shifts is not None:
            grid = grid + rearrange(shifts, "b c t -> (b t) 1 1 c")
        outputs = F.grid_sample(outputs, grid=grid, align_corners=True)

        states = torch.concat((behaviors, pupil_centers), dim=1)
        states = rearrange(states[..., -t:], "b d t -> (b t) d")
        states = self.state_encoder(states)
        states = repeat(states, "b d -> b d n 1", n=self.num_neurons)
        outputs = outputs + states

        outputs = einsum(outputs, self.features, "b c n d, d c n -> b n")
        if self.bias is not None:
            outputs = outputs + self.bias
        outputs = rearrange(outputs, "(b t) n -> b n t", b=b, t=t)
        return outputs
