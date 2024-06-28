import math
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.backends import cuda
from torch.utils.checkpoint import checkpoint

from viv1t.model.core.core import Core, register
from viv1t.model.helper import DropPath, SwiGLU
from viv1t.utils.utils import support_bf16

FF_ACTIVATIONS = Literal["gelu", "swiglu"]

class SAdapter2(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.75, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()
        self.skip_connect = skip_connect  # 添加 skip_connect 参数

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class TAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.75, act_layer=nn.GELU):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.act = act_layer()
        

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        x = xs
        # x = rearrange(x, 'B N T D -> (B T) N D')
        return x

def find_shape(num_patches: int):
    dim1 = math.ceil(math.sqrt(num_patches))
    while num_patches % dim1 != 0 and dim1 > 0:
        dim1 -= 1
    dim2 = num_patches // dim1
    return dim1, dim2

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int,
        dimension: Literal["spatial", "temporal"],
        dropout: float = 0.0,
    ):
        super(SinusoidalPositionalEncoding, self).__init__()
        # input has shape (B, T, P, D)
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        match self.dimension:
            case "temporal":
                pos_encoding = torch.zeros(1, max_length, 1, d_model)
                pos_encoding[0, :, 0, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, :, 0, 1::2] = torch.cos(position * div_term)
            case "spatial":
                pos_encoding = torch.zeros(1, 1, max_length, d_model)
                pos_encoding[0, 0, :, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, 0, :, 1::2] = torch.cos(position * div_term)
            case _:
                raise NotImplementedError(
                    f"invalid dimension {self.dimension} in "
                    f"SinusoidalPositionalEncoding"
                )

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        match self.dimension:
            case "temporal":
                outputs += self.pos_encoding[:, : inputs.size(1), :, :]
            case "spatial":
                outputs += self.pos_encoding[:, :, : inputs.size(2), :]
        return self.dropout(outputs)

class Unfold3d(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        spatial_patch_size: int,
        spatial_patch_stride: int,
        temporal_patch_size: int,
        temporal_patch_stride: int,
    ):
        super(Unfold3d, self).__init__()
        self.input_shape = input_shape
        self.unfold = (
            lambda tensor: tensor.unfold(2, temporal_patch_size, temporal_patch_stride)
            .unfold(3, spatial_patch_size, spatial_patch_stride)
            .unfold(4, spatial_patch_size, spatial_patch_stride)
        )

    def forward(self, inputs: torch.Tensor):
        if len(inputs.shape) == 4:
            inputs = inputs[None, ...]
        outputs = self.unfold(inputs)
        outputs = rearrange(
            outputs, "b c nt nh nw pt ph pw -> b nt (nh nw) (c pt ph pw)"
        )
        return outputs

class UnfoldConv3d(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        spatial_patch_size: int,
        spatial_patch_stride: int,
        temporal_patch_size: int,
        temporal_patch_stride: int,
        dilation: int = 1,
        padding: int = 0,
    ):
        super(UnfoldConv3d, self).__init__()
        self.input_shape = input_shape
        c, t, h, w = input_shape
        self.in_channels = c
        self.dilation = dilation
        self.padding = padding
        self.kernel_size = (temporal_patch_size, spatial_patch_size, spatial_patch_size)
        self.stride = (
            temporal_patch_stride,
            spatial_patch_stride,
            spatial_patch_stride,
        )
        # prepare one-hot convolution kernel
        kernel_dim = int(np.prod(self.kernel_size))
        repeat = [c, 1] + [1] * len(self.kernel_size)
        self.register_buffer(
            "weight",
            torch.eye(kernel_dim)
            .reshape((kernel_dim, 1, *self.kernel_size))
            .repeat(*repeat),
        )

    def forward(self, inputs: torch.Tensor):
        outputs = F.conv3d(
            inputs,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        outputs = rearrange(outputs, "b c t h w -> b t (h w) c")
        return outputs


class Tokenizer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        patch_mode: int,
        spatial_patch_size: int,
        spatial_patch_stride: int,
        temporal_patch_size: int,
        temporal_patch_stride: int,
        pos_encoding: int,
        emb_dim: int,
        dropout: float = 0.0,
        max_frame: int = None,
    ):
        """
        Patch mode (--core_patch_mode)
            0: extract 3D patches via tensor.unfold followed by linear projection
            1: extract 3D patches via F.conv3d with identity weight followed by linear projection
            2: extract 3D patches via a 3D convolution layer
        """
        super(Tokenizer, self).__init__()
        self.input_shape = input_shape
        c, t, h, w = input_shape

        new_t = self.unfold_size(
            t if max_frame is None else max_frame,
            temporal_patch_size,
            stride=temporal_patch_stride,
        )
        new_h = self.unfold_size(h, spatial_patch_size, stride=spatial_patch_stride)
        new_w = self.unfold_size(w, spatial_patch_size, stride=spatial_patch_stride)
        patch_dim = c * spatial_patch_size * spatial_patch_size * temporal_patch_size

        match patch_mode:
            case 0:
                self.tokenizer = nn.Sequential(
                    Unfold3d(
                        input_shape=input_shape,
                        spatial_patch_size=spatial_patch_size,
                        spatial_patch_stride=spatial_patch_stride,
                        temporal_patch_size=temporal_patch_size,
                        temporal_patch_stride=temporal_patch_stride,
                    ),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                    nn.LayerNorm(normalized_shape=emb_dim),
                    nn.Dropout(p=dropout),
                )
            case 1:
                self.tokenizer2 = nn.Sequential(
                    UnfoldConv3d(
                        input_shape=input_shape,
                        spatial_patch_size=spatial_patch_size,
                        spatial_patch_stride=spatial_patch_stride,
                        temporal_patch_size=temporal_patch_size,
                        temporal_patch_stride=temporal_patch_stride,
                    ),
                    nn.LayerNorm(normalized_shape=patch_dim),
                    nn.Linear(in_features=patch_dim, out_features=emb_dim),
                    nn.LayerNorm(normalized_shape=emb_dim),
                    nn.Dropout(p=dropout),
                )
            case 2:
                self.tokenizer = nn.Sequential(
                    nn.Conv3d(
                        in_channels=c,
                        out_channels=emb_dim,
                        kernel_size=(
                            temporal_patch_size,
                            spatial_patch_size,
                            spatial_patch_size,
                        ),
                        stride=(
                            temporal_patch_stride,
                            spatial_patch_stride,
                            spatial_patch_stride,
                        ),
                    ),
                    Rearrange("b c t h w -> b t (h w) c"),
                    nn.LayerNorm(normalized_shape=emb_dim),
                    nn.Dropout(p=dropout),
                )
            case _:
                raise NotImplementedError(
                    f"--core_patch_mode {patch_mode} not implemented."
                )

        self.pos_encoding = pos_encoding
        match self.pos_encoding:
            case 1:
                self.pos_embedding = nn.Parameter(
                    torch.randn(1, new_t, new_h * new_w, emb_dim)
                )
            case 3:
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, 1, new_h * new_w, emb_dim)
                )
                self.temporal_pos_embedding = nn.Parameter(
                    torch.randn(1, new_t, 1, emb_dim)
                )
            case 4:
                self.spatial_pos_embedding = nn.Parameter(
                    torch.randn(1, 1, new_h * new_w, emb_dim)
                )
                self.temporal_pos_encoding = SinusoidalPositionalEncoding(
                    d_model=emb_dim,
                    max_length=300,
                    dimension="temporal",
                    dropout=dropout,
                )
            case 5:
                self.spatial_pos_embedding = SinusoidalPositionalEncoding(
                    d_model=emb_dim,
                    max_length=new_h * new_w,
                    dimension="spatial",
                    dropout=dropout,
                )
                self.temporal_pos_encoding = SinusoidalPositionalEncoding(
                    d_model=emb_dim,
                    max_length=300,
                    dimension="temporal",
                    dropout=dropout,
                )

        self.output_shape = (new_t, (new_h * new_w), emb_dim)

    @staticmethod
    def unfold_size(dim: int, patch_size: int, padding: int = 0, stride: int = 1):
        return math.floor(((dim + 2 * padding - patch_size) / stride) + 1)

    def forward(self, inputs: torch.Tensor):
        # inputs = inputs.permute(0, 1, 3, 2) # edit input
        outputs = self.tokenizer(inputs)
        _, t, p, _ = outputs.shape
        match self.pos_encoding:
            case 1:
                outputs = outputs + self.pos_embedding[:, :t, :p, :]
            case 3:
                outputs = (
                    outputs
                    + self.spatial_pos_embedding[:, :, :p, :]
                    + self.temporal_pos_embedding[:, :t, :, :]
                )
            case 4:
                outputs = outputs + self.spatial_pos_embedding[:, :, :p, :]
                outputs = self.temporal_pos_encoding(outputs)
            case 5:
                outputs = self.spatial_pos_embedding(outputs)
                outputs = self.temporal_pos_encoding(outputs)
        return outputs

class PositionalEncodingGenerator(nn.Module):
    """Position Encoding Generator from https://arxiv.org/abs/2102.10882"""

    def __init__(
        self,
        dimension: Literal["spatial", "temporal"],
        input_shape: Tuple[int, int],
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(PositionalEncodingGenerator, self).__init__()
        self.dimension = dimension
        match self.dimension:
            case "spatial":
                new_h, new_w = find_shape(input_shape[0])
                self.rearrange = Rearrange("b (h w) c -> b c h w", h=new_h, w=new_w)
                self.pos_embedding = nn.Conv2d(
                    in_channels=input_shape[-1],
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    groups=out_channels,
                    padding_mode="zeros",
                )
            case "temporal":
                self.pos_embedding = nn.Conv1d(
                    in_channels=input_shape[-1],
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1,
                    groups=out_channels,
                    padding_mode="zeros",
                )
            case _:
                raise ValueError(f"invalid dimension option {self.dimension}")

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        match self.dimension:
            case "spatial":
                outputs = self.rearrange(outputs)
                outputs = outputs + self.pos_embedding(outputs)
                outputs = rearrange(outputs, "b c h w -> b (h w) c")
            case "temporal":
                # 假设新的输入格式为 (B, N, T, C)
                b, n, t, d = inputs.shape
                outputs = rearrange(outputs, "b n t d -> b d (n t)")
                outputs = outputs + self.pos_embedding(outputs)
                outputs = rearrange(outputs, "b d (n t) -> b n t d", n=n)
        return outputs

class Attention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        flash_attention: bool = True,
        normalize_qk: bool = True,
        grad_checkpointing: bool = False,
    ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.flash_attention = flash_attention
        self.grad_checkpointing = grad_checkpointing
        self.dropout = dropout
        inner_dim = head_dim * num_heads

        self.norm = nn.LayerNorm(normalized_shape=emb_dim)
        self.to_qkv = nn.Linear(
            in_features=emb_dim, out_features=inner_dim * 3, bias=False
        )
        self.register_buffer("scale", torch.tensor(emb_dim**-0.5))

        self.attn_out = nn.Sequential(
            nn.Linear(in_features=inner_dim, out_features=emb_dim),
            nn.Dropout(p=dropout),
        )

        self.normalize_qk = normalize_qk
        if self.normalize_qk:
            self.norm_q = nn.LayerNorm(normalized_shape=inner_dim)
            self.norm_k = nn.LayerNorm(normalized_shape=inner_dim)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.flash_attention:
            # Adding dropout to Flash attention layer significantly increase memory usage
            outputs = F.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = F.softmax(dots, dim=-1)
            outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.dropout, training=self.training)
        return outputs

    def mha(self, inputs: torch.Tensor):
        outputs = self.norm(inputs)
        q, k, v = torch.chunk(self.to_qkv(outputs), chunks=3, dim=-1)
        if self.normalize_qk:
            q, k = self.norm_q(q), self.norm_k(k)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        outputs = self.scaled_dot_product_attention(q, k, v)
        outputs = rearrange(outputs, "b h n d -> b n (h d)")
        outputs = self.attn_out(outputs)
        return outputs

    def forward(self, inputs: torch.Tensor):
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.mha, inputs, preserve_rng_state=True, use_reentrant=False
            )
        else:
            outputs = self.mha(inputs)
        return outputs


class BehaviorMLP(nn.Module):
    def __init__(
        self,
        behavior_mode: int,
        temporal_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        mouse_ids: List[str] = None,
        use_bias: bool = True,
    ):
        """
        behavior mode:
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
            3: feed behavior to B-MLP
            4: feed behavior and pupil center to B-MLP
            5: feed behavior and pupil center to per-animal B-MLP
        """
        super(BehaviorMLP, self).__init__()
        assert behavior_mode in (3, 4, 5)
        self.behavior_mode = behavior_mode
        in_dim = 2 if behavior_mode == 3 else 4
        in_dim *= temporal_dim
        mouse_ids = mouse_ids if behavior_mode == 5 else ["share"]
        self.models = nn.ModuleDict(
            {
                mouse_id: nn.Sequential(
                    nn.Linear(
                        in_features=in_dim,
                        out_features=out_dim // 2,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                    nn.Dropout(p=dropout),
                    nn.Linear(
                        in_features=out_dim // 2,
                        out_features=out_dim,
                        bias=use_bias,
                    ),
                    nn.Tanh(),
                )
                for mouse_id in mouse_ids
            }
        )

    def forward(
        self,
        behaviors: torch.Tensor,
        pupil_center: torch.Tensor,
        mouse_id: str,
    ):
        mouse_id = mouse_id if self.behavior_mode == 5 else "share"
        match self.behavior_mode:
            case 3:
                outputs = behaviors
            case 4 | 5:
                outputs = torch.cat((behaviors, pupil_center), dim=-1)
        outputs = self.models[mouse_id](outputs)
        return outputs
        
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: Literal["spatial", "temporal"],
        input_shape: Tuple[int, int],
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        pos_encoding: bool,
        behavior_mode: int,
        mouse_ids: List[str],
        temporal_dim: int,
        spatial_dim: int,
        mha_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        drop_path: float = 0.0,
        flash_attention: bool = True,
        normalize_qk: bool = False,
        grad_checkpointing: bool = False,
        ff_activation: str = "gelu",
        mlp_ratio: float = 4.0  # 添加 mlp_ratio 参数，设置默认值为 4.0
    ):
        super(TransformerBlock, self).__init__()
        assert dimension in ("spatial", "temporal")
        self.dimension = dimension
        emb_dim = input_shape[-1]
        self.attention = Attention(
            emb_dim=emb_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=mha_dropout,
            flash_attention=flash_attention,
            normalize_qk=normalize_qk,
            grad_checkpointing=grad_checkpointing,
        )
        if dimension == "spatial":
            self.adapter = SAdapter2(D_features=emb_dim, mlp_ratio=0.5)
        else:
            self.adapter = TAdapter(D_features=emb_dim, mlp_ratio=0.5)
        # self.sadapter2 = SAdapter2(D_features=emb_dim, mlp_ratio=0.25)
        # self.tadapter2 = TAdapter(D_features=emb_dim, mlp_ratio=0.45)
        match ff_activation:
            case "gelu":
                mlp_out = mlp_dim
                activation = nn.GELU()
            case "swiglu":
                mlp_out = mlp_dim * 2
                activation = SwiGLU()
            case _:
                raise NotImplementedError(
                    f"activation for FF {ff_activation} not implemented."
                )
        self.ff = nn.Sequential(
            nn.LayerNorm(normalized_shape=emb_dim),
            nn.Linear(in_features=emb_dim, out_features=mlp_out),
            activation,
            nn.Dropout(p=ff_dropout),
            nn.Linear(in_features=mlp_out, out_features=emb_dim),  # 修正这里的 in_features
            nn.Dropout(p=ff_dropout),
        )
        self.pos_encoding = None
        if pos_encoding:
            self.pos_encoding = PositionalEncodingGenerator(
                dimension=dimension,
                input_shape=input_shape,
                out_channels=emb_dim,
            )
        self.state_encoder = None
        if behavior_mode in (3, 4, 5):
            self.state_encoder = BehaviorMLP(
                behavior_mode=behavior_mode,
                temporal_dim=temporal_dim,
                out_dim=emb_dim,
                mouse_ids=mouse_ids,
            )
            self.spatial_dim = spatial_dim
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.drop_path2 = DropPath(drop_prob=drop_path)
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        if self.state_encoder is not None:
            states = self.state_encoder(behaviors, pupil_centers, mouse_id=mouse_id)
            match self.dimension:
                case "spatial":
                    states = repeat(states, "b d -> b 1 d")
                case "temporal":
                    states = repeat(states, "b d -> b t d", t=inputs.size(1))  # 修正这里的维度
            outputs = outputs + states
        # LayerNorm --> S-MSA ---> Adapter ---> + ---> LayerNorm ---> MLP
        #    |                                  |
        #    |--------------------------------->|
        # outputs = self.drop_path1(self.attention(outputs)) + outputs
        # outputs = self.drop_path1(adapter_result) + outputs
        # outputs = self.drop_path2(self.ff(outputs)) + outputs
        attn_output = self.attention(outputs)
        adapter_output = self.adapter(attn_output)
        # sadapter_output = self.sadapter2(attn_output) # Applying SAdapter2 right after attention
        # tadapter_output = self.tadapter2(attn_output)
        # adapter_output = sadapter_output + tadapter_output
        # adapter_output = torch.cat((sadapter_output, tadapter_output), dim=0)
        outputs = self.drop_path1(adapter_output) + outputs
        outputs = self.drop_path2(self.ff(outputs)) + outputs
        
        # outputs = self.drop_path1(self.attention(outputs)) + outputs
        # outputs = self.sadapter2(outputs)  # Applying SAdapter2 after attention
        # outputs = self.drop_path2(self.ff(outputs)) + outputs
        if self.pos_encoding is not None:
            outputs = self.pos_encoding(outputs)
        return outputs

    def freeze_except_adapter(self):
        for name, param in self.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                print(f"{name} is frozen\n");
                param.requires_grad = False

class ParallelTransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: Literal["spatial", "temporal"],
        input_shape: Tuple[int, int],
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        pos_encoding: bool,
        behavior_mode: int,
        mouse_ids: List[str],
        temporal_dim: int,
        spatial_dim: int,
        mha_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        drop_path: float = 0.0,
        flash_attention: bool = True,
        normalize_qk: bool = False,
        grad_checkpointing: bool = False,
        ff_activation: FF_ACTIVATIONS = "gelu",
    ):
        super(ParallelTransformerBlock, self).__init__()
        emb_dim = input_shape[-1]
        self.dimension = dimension
        self.num_heads = num_heads
        self.mha_dropout = mha_dropout
        self.flash_attention = flash_attention
        self.grad_checkpointing = grad_checkpointing
        self.norm = nn.LayerNorm(emb_dim)
        inner_dim = head_dim * num_heads
        self.register_buffer("scale", torch.tensor(head_dim**-0.5))
        match ff_activation:
            case "gelu":
                mlp_out = mlp_dim
                activation = nn.GELU()
            case "swiglu":
                mlp_out = mlp_dim * 2
                activation = SwiGLU()
            case _:
                raise NotImplementedError(
                    f"activation for FF {ff_activation} not implemented."
                )
        self.fused_dims = (inner_dim, inner_dim, inner_dim, mlp_out)
        self.fused_linear = nn.Linear(
            in_features=emb_dim, out_features=sum(self.fused_dims), bias=False
        )
        self.attn_out = nn.Linear(
            in_features=inner_dim, out_features=emb_dim, bias=False
        )
        self.ff_out = nn.Sequential(
            activation,
            nn.Dropout(p=ff_dropout),
            nn.Linear(in_features=mlp_dim, out_features=emb_dim, bias=False),
        )
        self.drop_path1 = DropPath(drop_prob=drop_path)
        self.drop_path2 = DropPath(drop_prob=drop_path)

        self.normalize_qk = normalize_qk
        if self.normalize_qk:
            self.norm_q = nn.LayerNorm(normalized_shape=inner_dim)
            self.norm_k = nn.LayerNorm(normalized_shape=inner_dim)

        self.state_encoder = None
        if behavior_mode in (3, 4, 5):
            self.state_encoder = BehaviorMLP(
                behavior_mode=behavior_mode,
                temporal_dim=temporal_dim,
                out_dim=emb_dim,
                mouse_ids=mouse_ids,
            )
            self.spatial_dim = spatial_dim

        self.apply(self.init_weight)
        self.sadapter2 = SAdapter2(D_features=emb_dim, mlp_ratio=0.75)

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        if self.flash_attention:
            # Adding dropout to Flash attention layer significantly increase memory usage
            outputs = F.scaled_dot_product_attention(q, k, v)
        else:
            dots = torch.matmul(q * self.scale, k.transpose(-2, -1))
            attn = F.softmax(dots, dim=-1)
            outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.mha_dropout, training=self.training)
        return outputs

    def parallel_attention(self, inputs: torch.Tensor):
        outputs = self.norm(inputs)
        q, k, v, ff = self.fused_linear(outputs).split(self.fused_dims, dim=-1)
        if self.normalize_qk:
            q, k = self.norm_q(q), self.norm_k(k)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)
        attn = self.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")
        outputs = (
            inputs
            + self.drop_path1(self.attn_out(attn))
            + self.drop_path2(self.ff_out(ff))
        )
        return self.sadapter2(outputs)
        # return outputs

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        if self.state_encoder is not None:
            states = self.state_encoder(behaviors, pupil_centers, mouse_id=mouse_id)
            match self.dimension:
                case "spatial":
                    states = repeat(states, "b d -> b 1 d")
                case "temporal":
                    states = repeat(states, "b t d -> b p t d", p=self.spatial_dim)
                    states = rearrange(states, "b p t d -> (b p) t d")
            outputs = outputs + states
        if self.grad_checkpointing:
            outputs = checkpoint(
                self.parallel_attention,
                outputs,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        else:
            outputs = self.parallel_attention(outputs)
        return outputs
    
    def freeze_except_adapter(self):
        for name, param in self.named_parameters():
            if 'sadapter2' in name:
                param.requires_grad = True
            else:
                print(f"{name} is frozen\n");
                param.requires_grad = False

class Transformer(nn.Module):
    def __init__(
        self,
        dimension: Literal["spatial", "temporal"],
        input_shape: Tuple[int, int],
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        pos_encoding: int,
        behavior_mode: int,
        mouse_ids: List[str],
        temporal_dim: int,
        spatial_dim: int,
        mha_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        drop_path: float = 0.0,
        parallel_attention: bool = False,
        flash_attention: bool = True,
        normalize_qk: bool = False,
        grad_checkpointing: bool = False,
        ff_activation: FF_ACTIVATIONS = "gelu",
    ):
        super(Transformer, self).__init__()
        assert dimension in ("spatial", "temporal")
        block = ParallelTransformerBlock if parallel_attention else TransformerBlock
        self.blocks = nn.ModuleList(
            [
                block(
                    dimension=dimension,
                    input_shape=input_shape,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    mlp_dim=mlp_dim,
                    pos_encoding=pos_encoding == 2 and i == 0,
                    behavior_mode=behavior_mode,
                    mouse_ids=mouse_ids,
                    temporal_dim=temporal_dim,
                    spatial_dim=spatial_dim,
                    mha_dropout=mha_dropout,
                    ff_dropout=ff_dropout,
                    drop_path=drop_path,
                    flash_attention=flash_attention,
                    normalize_qk=normalize_qk,
                    grad_checkpointing=grad_checkpointing,
                    ff_activation=ff_activation,
                )
                for i in range(depth)
            ]
        )
    def freeze_blocks(self):
        for blk in self.blocks:
            blk.freeze_except_adapter()

    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = inputs
        for i in range(len(self.blocks)):
            outputs = self.blocks[i](
                outputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
            )
        return outputs

class ViViT(nn.Module):
    def __init__(
        self,
        args: Any,
        input_shape: Tuple[int, int, int, int],
    ):
        super(ViViT, self).__init__()
        self.register_buffer("reg_scale", torch.tensor(0.0))
        self.behavior_mode = args.core_behavior_mode
        self.reshape_behavior = self.behavior_mode in (3, 4, 5)
        emb_dim, num_heads = args.core_emb_dim, args.core_num_heads

        if args.grad_checkpointing is None:
            args.grad_checkpointing = "cuda" in args.device.type
        if args.grad_checkpointing and args.verbose:
            print(f"Enable gradient checkpointing in ViViT")

        if (
            args.core_flash_attention
            and support_bf16(args.device)
            and num_heads % 8 == 0
        ):
            cuda.enable_flash_sdp(True)
            cuda.enable_mem_efficient_sdp(True)

        if not hasattr(args, "core_parallel_attention"):
            args.core_parallel_attention = False

        self.temporal_patch_size = args.core_temporal_patch_size
        self.temporal_patch_stride = args.core_temporal_patch_stride
        self.tokenizer = Tokenizer(
            input_shape=input_shape,
            max_frame=args.max_frame,
            patch_mode=args.core_patch_mode,
            spatial_patch_size=args.core_spatial_patch_size,
            spatial_patch_stride=args.core_spatial_patch_stride,
            temporal_patch_size=args.core_temporal_patch_size,
            temporal_patch_stride=args.core_temporal_patch_stride,
            pos_encoding=args.core_pos_encoding,
            emb_dim=emb_dim,
            dropout=args.core_p_dropout,
        )
        spatial_dim = self.tokenizer.output_shape[1]

        # self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.spatial_cls_token = None
        self.temporal_cls_token = None

        normalize_qk = hasattr(args, "core_norm_qk") and args.core_norm_qk
        if not hasattr(args, "core_ff_dropout") or args.core_ff_dropout is None:
            args.core_ff_dropout = args.core_mha_dropout
        if not hasattr(args, "core_ff_activation"):
            args.core_ff_activation = "gelu"

        self.spatial_transformer = Transformer(
            dimension="spatial",
            input_shape=(self.tokenizer.output_shape[1], emb_dim),
            depth=args.core_spatial_depth,
            num_heads=num_heads,
            head_dim=args.core_head_dim,
            mlp_dim=args.core_mlp_dim,
            pos_encoding=args.core_pos_encoding,
            behavior_mode=args.core_behavior_mode,
            mouse_ids=args.mouse_ids,
            temporal_dim=args.core_temporal_patch_size,
            spatial_dim=spatial_dim,
            mha_dropout=args.core_mha_dropout,
            ff_dropout=args.core_ff_dropout,
            drop_path=args.core_drop_path,
            parallel_attention=args.core_parallel_attention,
            flash_attention=args.core_flash_attention == 1,
            normalize_qk=normalize_qk,
            grad_checkpointing=args.grad_checkpointing,
            ff_activation=args.core_ff_activation,
        )
        self.temporal_transformer = Transformer(
            dimension="temporal",
            input_shape=(self.tokenizer.output_shape[0], emb_dim),
            depth=args.core_temporal_depth,
            num_heads=num_heads,
            head_dim=args.core_head_dim,
            mlp_dim=args.core_mlp_dim,
            pos_encoding=args.core_pos_encoding,
            behavior_mode=args.core_behavior_mode,
            mouse_ids=args.mouse_ids,
            temporal_dim=args.core_temporal_patch_size,
            spatial_dim=spatial_dim,
            mha_dropout=args.core_mha_dropout,
            ff_dropout=args.core_ff_dropout,
            drop_path=args.core_drop_path,
            parallel_attention=args.core_parallel_attention,
            flash_attention=args.core_flash_attention == 1,
            normalize_qk=normalize_qk,
            grad_checkpointing=args.grad_checkpointing,
            ff_activation=args.core_ff_activation,
        )
        self.spatial_transformer.freeze_blocks()
        self.temporal_transformer.freeze_blocks()
        # calculate latent height and width based on num_patches
        new_h, new_w = find_shape(self.tokenizer.output_shape[1])
        self.rearrange = Rearrange("b t (h w) c -> b c t h w", h=new_h, w=new_w)
        self.output_shape = (emb_dim, self.tokenizer.output_shape[0], new_h, new_w)

    def regularizer(self):
        """L1 regularization"""
        return self.reg_scale * sum(p.abs().sum() for p in self.parameters())

    def temporal_unfold(self, inputs: torch.Tensor):
        outputs = inputs.unfold(2, self.temporal_patch_size, self.temporal_patch_stride)
        outputs = rearrange(outputs, "b d t pt -> b t (pt d)")
        return outputs

    def freeze_non_transformer_layers(self):
        for name, param in self.named_parameters():
            if 'transformer' in name:
                param.requires_grad = True
            else:
                print(f"{name} is frozen in Vivit")
                param.requires_grad = False
    def forward(
        self,
        inputs: torch.Tensor,
        mouse_id: str,
        behaviors: torch.Tensor,
        pupil_centers: torch.Tensor,
    ):
        outputs = self.tokenizer(inputs)
        b, t, p, _ = outputs.shape

        if self.reshape_behavior:
            behaviors = self.temporal_unfold(behaviors)
            pupil_centers = self.temporal_unfold(pupil_centers)

        if self.spatial_cls_token is not None:
            spatial_cls_tokens = repeat(
                self.spatial_cls_token, "1 1 c -> b t 1 c", b=b, t=t
            )
            outputs = torch.cat((spatial_cls_tokens, outputs), dim=2)

        # attend across space
        outputs = rearrange(outputs, "b t p c -> (b t) p c")
        outputs = self.spatial_transformer(
            outputs,
            mouse_id=mouse_id,
            behaviors=(
                rearrange(behaviors, "b t d -> (b t) d")
                if self.reshape_behavior
                else behaviors
            ),
            pupil_centers=(
                rearrange(pupil_centers, "b t d -> (b t) d")
                if self.reshape_behavior
                else pupil_centers
            ),
        )
        outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

        # append temporal CLS tokens
        if self.temporal_cls_token is not None:
            temporal_cls_tokens = repeat(
                self.temporal_cls_token, "1 1 c -> b 1 p c", b=b, p=p + 1
            )
            outputs = torch.cat((temporal_cls_tokens, outputs), dim=1)

        # attend across time
        outputs = rearrange(outputs, "b t p c -> (b p) t c")
        outputs = self.temporal_transformer(
            outputs,
            mouse_id=mouse_id,
            behaviors=behaviors,
            pupil_centers=pupil_centers,
        )
        outputs = rearrange(outputs, "(b p) t c -> b t p c", b=b)

        # remove CLS tokens
        if self.spatial_cls_token is not None:
            outputs = outputs[:, :, 1:, :]
        if self.temporal_cls_token is not None:
            outputs = outputs[:, 1:, :, :]

        outputs = self.rearrange(outputs)

        return outputs

@register("vit")
class ViViTCore(Core):
    def __init__(self, args: Any, input_shape: Tuple[int, int, int, int]):
        """
        Behavior mode (--core_behavior_mode)
            0: do not include behavior
            1: concat behavior with visual input
            2: concat behavior and pupil center with visual input
            3: feed behavior to B-MLP
            4: feed behavior and pupil center to B-MLP
            5: feed behavior and pupil center to per-animal B-MLP
        """
        super(ViViTCore, self).__init__(args, input_shape=input_shape)
        self.input_shape = input_shape
        self.behavior_mode = args.core_behavior_mode
        input_shape = list(input_shape)
        match self.behavior_mode:
            case 0 | 3 | 4 | 5:
                pass
            case 1:
                input_shape[0] += args.input_shapes["behavior"][0]
            case 2:
                input_shape[0] += (
                    args.input_shapes["behavior"][0]
                    + args.input_shapes["pupil_center"][0]
                )
            case _:
                raise NotImplementedError(f"--behavior_mode {self.behavior_mode}")
        self.vivit = ViViT(args, input_shape=tuple(input_shape))
        self.vivit.freeze_non_transformer_layers()
        self.output_shape = self.vivit.output_shape

    def regularizer(self):
        return self.vivit.regularizer()

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
                outputs = torch.concat(
                    (outputs, repeat(behaviors, "b d t -> b d t h w", h=h, w=w)), dim=1
                )
            case 2:
                outputs = torch.concat(
                    (
                        outputs,
                        repeat(behaviors, "b d t -> b d t h w", h=h, w=w),
                        repeat(pupil_centers, "b d t -> b d t h w", h=h, w=w),
                    ),
                    dim=1,
                )
        with self.autocast:
            outputs = self.vivit(
                outputs,
                mouse_id=mouse_id,
                behaviors=behaviors,
                pupil_centers=pupil_centers,
            )
        outputs = outputs.to(torch.float32)
        return outputs

