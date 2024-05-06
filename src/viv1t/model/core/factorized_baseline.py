"""
Code reference
- https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/layers/cores/conv3d.py
- https://github.com/sinzlab/neuralpredictors/blob/main/neuralpredictors/regularizers.py
"""

from collections import OrderedDict
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from viv1t.model.core.core import Core, register
from viv1t.model.helper import AdaptiveELU


def check_hyperparam_for_layers(hyperparameter, layers):
    if isinstance(hyperparameter, (list, tuple)):
        assert len(hyperparameter) == layers, (
            f"Hyperparameter list should have same length "
            f"{len(hyperparameter)} as layers {layers}"
        )
        return hyperparameter
    elif isinstance(hyperparameter, int):
        return (hyperparameter,) * layers


class Bias3DLayer(nn.Module):
    def __init__(self, channels, initial=0, **kwargs):
        super().__init__(**kwargs)

        self.bias = torch.nn.Parameter(
            torch.empty((1, channels, 1, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        return x + self.bias


class Scale3DLayer(nn.Module):
    def __init__(self, channels, initial=1, **kwargs):
        super().__init__(**kwargs)

        self.scale = torch.nn.Parameter(
            torch.empty((1, channels, 1, 1, 1)).fill_(initial)
        )

    def forward(self, x):
        return x * self.scale


def laplace():
    """
    Returns a 3x3 laplace filter.

    """
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)[
        None, None, ...
    ]


def laplace5x5():
    """
    Returns a 5x5 LaplacianOfGaussians (LoG) filter.

    """
    return np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    ).astype(np.float32)[None, None, ...]


def laplace7x7():
    """
    Returns a 7x7 LaplacianOfGaussians (LoG) filter.

    """
    return np.array(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 3, 3, 3, 1, 0],
            [1, 3, 0, -7, 0, 3, 1],
            [1, 3, -7, -24, -7, 3, 1],
            [1, 3, 0, -7, 0, 3, 1],
            [0, 1, 3, 3, 3, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ]
    ).astype(np.float32)[None, None, ...]


def laplace1d():
    return np.array([-1, 4, -1]).astype(np.float32)[None, None, ...]


class Laplace(nn.Module):
    """
    Laplace filter for a stack of data. Utilized as the input weight regularizer.
    """

    def __init__(self, padding=None, filter_size=3):
        """
        Laplace filter for a stack of data.

        Args:
            padding (int): Controls the amount of zero-padding for the convolution operation
                            Default is half of the kernel size (recommended)

        Attributes:
            filter (2D Numpy array): 3x3 Laplace filter.
            padding_size (int): Number of zeros added to each side of the input image
                before convolution operation.
        """
        super().__init__()
        if filter_size == 3:
            kernel = laplace()
        elif filter_size == 5:
            kernel = laplace5x5()
        elif filter_size == 7:
            kernel = laplace7x7()

        self.register_buffer("filter", torch.from_numpy(kernel))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv2d(x, self.filter, bias=None, padding=self.padding_size)


class LaplaceL2norm(nn.Module):
    """
    Normalized Laplace regularizer for a 2D convolutional layer.
        returns |laplace(filters)| / |filters|
    """

    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace(padding=padding)

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        return agg_fn(self.laplace(x.reshape(oc * ic, 1, k1, k2)).pow(2)) / agg_fn(
            x.reshape(oc * ic, 1, k1, k2).pow(2)
        )


def gaussian2d(size, sigma=5, gamma=1, theta=0, center=(0, 0), normalize=True):
    """
    Returns a 2D Gaussian filter.

    Args:
        size (tuple of int, or int): Image height and width.
        sigma (float): std deviation of the Gaussian along x-axis. Default is 5..
        gamma (float): ratio between std devidation along x-axis and y-axis. Default is 1.
        theta (float): Orientation of the Gaussian (in ratian). Default is 0.
        center (tuple): The position of the filter. Default is center (0, 0).
        normalize (bool): Whether to normalize the entries. This is computed by
            subtracting the minimum value and then dividing by the max. Default is True.

    Returns:
        2D Numpy array: A 2D Gaussian filter.

    """

    sigma_x = sigma
    sigma_y = sigma / gamma

    xmax, ymax = (size, size) if isinstance(size, int) else size
    xmax, ymax = (xmax - 1) / 2, (ymax - 1) / 2
    xmin, ymin = -xmax, -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # shift the position
    y -= center[0]
    x -= center[1]

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))

    if normalize:
        gaussian -= gaussian.min()
        gaussian /= gaussian.max()

    return gaussian.astype(np.float32)


class GaussianLaplaceL2(nn.Module):
    """
    Laplace regularizer, with a Gaussian mask, for a single 2D convolutional layer.

    """

    def __init__(self, kernel, padding=None):
        """
        Args:
            kernel: Size of the convolutional kernel of the filter that is getting regularized
            padding (int): Controls the amount of zero-padding for the convolution operation.
        """
        super().__init__()

        self.laplace = Laplace(padding=padding)
        self.kernel = (kernel, kernel) if isinstance(kernel, int) else kernel
        sigma = min(*self.kernel) / 4
        self.gaussian2d = torch.from_numpy(
            gaussian2d(size=(*self.kernel,), sigma=sigma)
        )

    def forward(self, x, avg=False):
        agg_fn = torch.mean if avg else torch.sum

        oc, ic, k1, k2 = x.size()
        out = self.laplace(x.reshape(oc * ic, 1, k1, k2))
        out = out * (1 - self.gaussian2d.expand(1, 1, k1, k2).to(x.device))

        return agg_fn(out.pow(2)) / agg_fn(x.reshape(oc * ic, 1, k1, k2).pow(2))


class Laplace1d(nn.Module):
    def __init__(self, padding):
        super().__init__()
        filter = laplace1d()
        self.register_buffer("filter", torch.from_numpy(filter))
        self.padding_size = self.filter.shape[-1] // 2 if padding is None else padding

    def forward(self, x):
        return F.conv1d(x, self.filter, bias=None, padding=self.padding_size)


class DepthLaplaceL21d(nn.Module):
    def __init__(self, padding=None):
        super().__init__()
        self.laplace = Laplace1d(padding=padding)

    def forward(self, x, avg=False):
        oc, ic, t = x.size()
        if avg:
            return torch.mean(
                self.laplace(x.reshape(oc * ic, 1, t)).pow(2)
            ) / torch.mean(x.reshape(oc * ic, 1, t).pow(2))
        else:
            return torch.sum(self.laplace(x.reshape(oc * ic, 1, t)).pow(2)) / torch.sum(
                x.reshape(oc * ic, 1, t).pow(2)
            )


class Factorized3d(nn.Module):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
        final_nonlin: bool = True,
        stride: int = 1,
        x_shift: float = 0.0,
        y_shift: float = 0.0,
        hidden_nonlinearities: str = "elu",
        bias: bool = True,
        batch_norm: bool = True,
        padding: bool = True,
        batch_norm_scale: bool = True,
        independent_bn_bias: bool = True,
        momentum: float = 0.7,
        laplace_padding: int = None,
        input_regularizer: str = "LaplaceL2norm",
        spatial_dilation: int = 1,
        temporal_dilation: int = 1,
        hidden_spatial_dilation: int = 1,
        hidden_temporal_dilation: int = 1,
    ):
        """
        Core3d, similar to Basic3dCore but the convolution is separated into first spatial and then temporal.

        :param input_channels: integer, number of input channels as in
        :param hidden_channels: number of hidden channels (i.e feature maps) in each hidden layer
        :param spatial_input_kernel: kernel size of the first spatial layer (i.e. the input layer)
        :param temporal_input_kernel: kernel size of the temporal layer
        :param spatial_hidden_kernel:  kernel size of each hidden layer's spatial kernel
        :param temporal_hidden_kernel:  kernel size of each hidden layer's temporal kernel
        :param spatial_dilation: dilation of ONLY the first spatial kernel (both width and height)
        :param temporal dilation: dilation of ONLY the first temporal kernel
        :param final_nonlin: bool specifiyng whether to include a nonlinearity after last core convolution
        :param layers: number of layers
        :param stride: the stride of the convolutions.
        :param x_shift: shift in x axis in case ELU is the nonlinearity
        :param y_shift: shift in y axis in case ELU is the nonlinearity
        :param gamma_input_spatial: regularizer factor for spatial smoothing
        :param gamma_input_temporal: regularizer factor for temporal smoothing
        :param hidden_nonlinearities:
        :param bias: adds a bias layer - TODO: actually now does not do anything I think
        :param batch_norm: bool specifying whether to include batch norm after convolution in core
        :param padding: whether to pad convolutions. Defaults to False.
        :param batch_norm_scale: bool, if True, a scaling factor after BN will be learned.
        :param independent_bn_bias: If False, will allow for scaling the batch norm, so that batchnorm
                                    and bias can both be true. Defaults to True.
        :param momentum: momentum for batch norm
        :param laplace_padding: padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
        :param input_regularizer: specifies what kind of spatial regularized is applied. Must match one of the
                                  regularizers in neuralpredictors.regularizers
        """
        super(Factorized3d, self).__init__()

        self.input_shape = input_shape
        layers = args.core_num_layers if hasattr(args, "core_num_layers") else 4
        hidden_dim = args.core_hidden_dim if hasattr(args, "core_hidden_dim") else 16
        hidden_channels = [hidden_dim * 2**i for i in range(layers)]

        if hasattr(args, "core_spatial_input_kernel"):
            spatial_input_kernel = (
                args.core_spatial_input_kernel,
                args.core_spatial_input_kernel,
            )
        else:
            spatial_input_kernel = (11, 11)
        if hasattr(args, "core_temporal_input_kernel"):
            temporal_input_kernel = args.core_temporal_input_kernel
        else:
            temporal_input_kernel = 11

        if hasattr(args, "core_spatial_hidden_kernel"):
            spatial_hidden_kernel = (
                args.core_spatial_hidden_kernel,
                args.core_spatial_hidden_kernel,
            )
        else:
            spatial_hidden_kernel = (5, 5)
        if hasattr(args, "core_temporal_hidden_kernel"):
            temporal_hidden_kernel = args.core_temporal_hidden_kernel
        else:
            temporal_hidden_kernel = 5

        regularizer_config = (
            dict(padding=laplace_padding, kernel=spatial_input_kernel)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weight_regularizer = LaplaceL2norm(**regularizer_config)
        self.temporal_regularizer = DepthLaplaceL21d()
        self.layers = layers
        self.input_channels = self.input_shape[0]
        self.spatial_input_kernel = spatial_input_kernel
        self.temporal_input_kernel = temporal_input_kernel
        self.hidden_channels = hidden_channels
        self.spatial_hidden_kernel = spatial_hidden_kernel
        self.temporal_hidden_kernel = temporal_hidden_kernel
        self.bias = bias
        self.batch_norm = batch_norm
        self.batch_norm_scale = batch_norm_scale
        self.independent_bn_bias = independent_bn_bias
        self.momentum = momentum
        self.stride = stride
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.hidden_spatial_dilation = (hidden_spatial_dilation,)
        self.hidden_temporal_dilation = (hidden_temporal_dilation,)
        self.padding = padding
        self.register_buffer("gamma_input_spatial", torch.tensor(0.0))
        self.register_buffer("gamma_input_temporal", torch.tensor(0.0))
        self.nonlinearities = {
            "elu": torch.nn.ELU,
            "softplus": torch.nn.Softplus,
            "relu": torch.nn.ReLU,
            "adaptive_elu": AdaptiveELU,
        }

        self.hidden_channels = check_hyperparam_for_layers(hidden_channels, self.layers)
        self.hidden_temporal_dilation = check_hyperparam_for_layers(
            hidden_temporal_dilation, self.layers - 1
        )
        self.hidden_spatial_dilation = check_hyperparam_for_layers(
            hidden_spatial_dilation, self.layers - 1
        )

        if isinstance(self.spatial_input_kernel, int):
            self.spatial_input_kernel = (self.spatial_input_kernel,) * 2

        if isinstance(self.spatial_hidden_kernel, int):
            self.spatial_hidden_kernel = (self.spatial_hidden_kernel,) * 2

        if isinstance(self.spatial_hidden_kernel, (tuple, list)):
            if self.layers > 1:
                self.spatial_hidden_kernel = [self.spatial_hidden_kernel] * (
                    self.layers - 1
                )
                self.temporal_hidden_kernel = [self.temporal_hidden_kernel] * (
                    self.layers - 1
                )
            else:
                self.spatial_hidden_kernel = []
                self.temporal_hidden_kernel = []

        if isinstance(self.stride, int):
            self.stride = [self.stride] * self.layers

        self.features = nn.Sequential()

        # input layer
        layer = OrderedDict()
        layer["conv_spatial"] = nn.Conv3d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels[0],
            kernel_size=(1,) + self.spatial_input_kernel,
            stride=(1, self.stride[0], self.stride[0]),
            bias=self.bias,
            dilation=(1, self.spatial_dilation, self.spatial_dilation),
            padding=(
                (
                    0,
                    self.spatial_input_kernel[0] // 2,
                    self.spatial_input_kernel[1] // 2,
                )
                if self.padding
                else 0
            ),
        )
        layer["conv_temporal"] = nn.Conv3d(
            self.hidden_channels[0],
            self.hidden_channels[0],
            kernel_size=(temporal_input_kernel, 1, 1),
            bias=self.bias,
            dilation=(self.temporal_dilation, 1, 1),
        )
        self.add_bn_layer(layer=layer, hidden_channels=hidden_channels[0])
        if layers > 1 or final_nonlin:
            if hidden_nonlinearities == "adaptive_elu":
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                    xshift=x_shift, yshift=y_shift
                )
            else:
                layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()
        layer["dropout"] = nn.Dropout3d(p=args.core_dropout)
        self.features.add_module("layer0", nn.Sequential(layer))

        # hidden layers
        for l in range(0, self.layers - 1):
            layer = OrderedDict()
            layer[f"conv_spatial_{l+1}"] = nn.Conv3d(
                in_channels=self.hidden_channels[l],
                out_channels=self.hidden_channels[l + 1],
                kernel_size=(1,) + (self.spatial_hidden_kernel[l]),
                stride=(1, self.stride[l], self.stride[l]),
                bias=self.bias,
                dilation=(
                    1,
                    self.hidden_spatial_dilation[l],
                    self.hidden_spatial_dilation[l],
                ),
                padding=(
                    (
                        0,
                        self.spatial_hidden_kernel[l][0] // 2,
                        self.spatial_hidden_kernel[l][1] // 2,
                    )
                    if self.padding
                    else 0
                ),
            )
            layer[f"conv_temporal_{l+1}"] = nn.Conv3d(
                self.hidden_channels[l + 1],
                self.hidden_channels[l + 1],
                kernel_size=(self.temporal_hidden_kernel[l], 1, 1),
                bias=self.bias,
                dilation=(self.hidden_temporal_dilation[l], 1, 1),
            )
            self.add_bn_layer(layer=layer, hidden_channels=hidden_channels[l + 1])
            if final_nonlin or l < self.layers:
                if hidden_nonlinearities == "adaptive_elu":
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities](
                        x_shift=x_shift, y_shift=y_shift
                    )
                else:
                    layer["nonlin"] = self.nonlinearities[hidden_nonlinearities]()
            if l < self.layers - 2:
                layer[f"dropout{l+1}"] = nn.Dropout3d(p=args.core_dropout)
            self.features.add_module(f"layer{l + 1}", nn.Sequential(layer))

        self.apply(self.init_conv)

        self.output_shape = (
            self.hidden_channels[-1],
            self.input_shape[1] - 10 - (layers - 1) * 4,
            input_shape[2],
            input_shape[3],
        )

    @staticmethod
    def init_conv(m: nn.Module):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def laplace_spatial(self):
        laplace = 0
        laplace += self._input_weight_regularizer(
            self.features[0].conv_spatial.weight[:, :, 0, :, :]
        )
        return laplace

    def laplace_temporal(self):
        laplace = self.temporal_regularizer(
            self.features[0].conv_temporal.weight[:, :, :, 0, 0]
        )
        return laplace

    def regularizer(self):
        reg_spatial = self.gamma_input_spatial * self.laplace_spatial()
        reg_temporal = self.gamma_input_temporal * self.laplace_temporal()
        return reg_spatial + reg_temporal

    def get_kernels(self):
        return [(self.temporal_input_kernel,) + self.spatial_input_kernel] + [
            (temporal_kernel,) + spatial_kernel
            for temporal_kernel, spatial_kernel in zip(
                self.temporal_hidden_kernel, self.spatial_hidden_kernel
            )
        ]

    def add_bn_layer(self, layer, hidden_channels):
        if self.batch_norm:
            if self.independent_bn_bias:
                layer["norm"] = nn.BatchNorm3d(hidden_channels, momentum=self.momentum)
            else:
                layer["norm"] = nn.BatchNorm3d(
                    hidden_channels,
                    momentum=self.momentum,
                    affine=self.bias and self.batch_norm_scale,
                )
                if self.bias and not self.batch_norm_scale:
                    layer["bias"] = Bias3DLayer(hidden_channels)
                elif self.batch_norm_scale:
                    layer["scale"] = Scale3DLayer(hidden_channels)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for features in self.features:
            outputs = features(outputs)
        return outputs


@register("factorized_baseline")
class FactorizedCore(Core):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int, int]):
        """
        Behavior mode (--core_behavior_mode)
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
        """
        super(FactorizedCore, self).__init__(args, input_shape=input_shape)
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode
        input_shape = list(input_shape)
        match self.behavior_mode:
            case 0:
                pass
            case 1:
                input_shape[0] += args.input_shapes["behavior"][0]
            case 2:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                )
            case 6:
                input_shape[0] += (
                    2 * args.input_shapes["behavior"][0]
                    + 2 * args.input_shapes["pupil_center"][0]
                )
            case 7:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                    + 2
                )
                self.create_positional_encoding()
            case 8:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                )
                self.modulator = nn.GRU(
                    input_size=4,
                    hidden_size=4,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                )
            case 9:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                    + 2
                )
                self.create_positional_encoding()
                self.modulator = nn.GRU(
                    input_size=4,
                    hidden_size=4,
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0,
                    bidirectional=False,
                )
            case _:
                raise NotImplementedError(f"--behavior_mode {self.behavior_mode}")
        self.cnn = Factorized3d(args, input_shape=tuple(input_shape))
        self.output_shape = self.cnn.output_shape

    def regularizer(self):
        return self.cnn.regularizer()

    def create_positional_encoding(self):
        _, _, h, w = self.input_shape
        vertical = torch.linspace(start=-1, end=1, steps=h)
        vertical = repeat(vertical, "h -> 1 h w", w=w)
        horizontal = torch.linspace(start=-1, end=1, steps=w)
        horizontal = repeat(horizontal, "w -> 1 h w", h=h)
        pos_encoding = torch.cat((vertical, horizontal), dim=0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        b, _, t, h, w = inputs.shape
        outputs = inputs
        match self.behavior_mode:
            case 1:
                behaviors = repeat(behaviors, "b d t -> b d t h w", h=h, w=w)
                outputs = torch.concat((outputs, behaviors), dim=1)
            case 2:
                behaviors = repeat(behaviors, "b d t -> b d t h w", h=h, w=w)
                pupil_centers = repeat(pupil_centers, "b d t -> b d t h w", h=h, w=w)
                outputs = torch.concat((outputs, behaviors, pupil_centers), dim=1)
            case 6:
                time = 30  # behavioral variables are recorded at 30Hz
                d_dilation = torch.diff(behaviors[:, 0, :], dim=1) / time
                d_speed = torch.diff(behaviors[:, 1, :], dim=1) / time
                d_pupil_x = torch.diff(pupil_centers[:, 0, :], dim=1) / time
                d_pupil_y = torch.diff(pupil_centers[:, 1, :], dim=1) / time

                reshape = lambda tensor: repeat(
                    F.pad(tensor, pad=(1, 0, 0, 0)), "b t -> b 1 t h w", h=h, w=w
                )
                d_dilation = reshape(d_dilation)
                d_speed = reshape(d_speed)
                d_pupil_x = reshape(d_pupil_x)
                d_pupil_y = reshape(d_pupil_y)

                behaviors = repeat(behaviors, "b d t -> b d t h w", h=h, w=w)
                pupil_centers = repeat(pupil_centers, "b d t -> b d t h w", h=h, w=w)
                outputs = torch.concat(
                    (
                        outputs,
                        behaviors,
                        d_dilation,
                        d_speed,
                        pupil_centers,
                        d_pupil_x,
                        d_pupil_y,
                    ),
                    dim=1,
                )
            case 7:
                behaviors = repeat(behaviors, "b d t -> b d t h w", h=h, w=w)
                pupil_centers = repeat(pupil_centers, "b d t -> b d t h w", h=h, w=w)
                pos_encoding = repeat(self.pos_encoding, "d h w -> b d t h w", b=b, t=t)
                outputs = torch.concat(
                    (outputs, behaviors, pupil_centers, pos_encoding), dim=1
                )
            case 8:
                states = torch.concat((behaviors, pupil_centers), dim=1)
                states, _ = self.modulator(rearrange(states, "b d t -> b t d"))
                states = repeat(states, "b t d -> b d t h w", h=h, w=w)
                outputs = torch.concat((outputs, states), dim=1)
            case 9:
                states = torch.concat((behaviors, pupil_centers), dim=1)
                states, _ = self.modulator(rearrange(states, "b d t -> b t d"))
                states = repeat(states, "b t d -> b d t h w", h=h, w=w)
                pos_encoding = repeat(self.pos_encoding, "d h w -> b d t h w", b=b, t=t)
                outputs = torch.concat((outputs, states, pos_encoding), dim=1)
        with self.autocast:
            outputs = self.cnn(outputs)
        return outputs.to(torch.float32)
