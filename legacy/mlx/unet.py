# Copyright © 2023 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)

    return x


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, x):
        x = self.linear_1(x)
        x = nn.silu(x)
        x = self.linear_2(x)

        return x


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        groups: int = 32,
        temb_channels: Optional[int] = None,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(groups, in_channels, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Linear(in_channels, out_channels)

    def __call__(self, x, temb=None):
        dtype = x.dtype

        if temb is not None:
            temb = self.time_emb_proj(nn.silu(temb))

        y = self.norm1(x.astype(mx.float32)).astype(dtype)
        y = nn.silu(y)
        y = self.conv1(y)
        if temb is not None:
            y = y + temb[:, None, None, :]
        y = self.norm2(y.astype(mx.float32)).astype(dtype)
        y = nn.silu(y)
        y = self.conv2(y)

        x = y + (x if "conv_shortcut" not in self else self.conv_shortcut(x))

        return x


class UNetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        prev_out_channels: Optional[int] = None,
        num_layers: int = 1,
        resnet_groups: int = 32,
        add_downsample=True,
        add_upsample=True,
    ):
        super().__init__()

        # Prepare the in channels list for the resnets
        if prev_out_channels is None:
            in_channels_list = [in_channels] + [out_channels] * (num_layers - 1)
        else:
            in_channels_list = [prev_out_channels] + [out_channels] * (num_layers - 1)
            res_channels_list = [out_channels] * (num_layers - 1) + [in_channels]
            in_channels_list = [
                a + b for a, b in zip(in_channels_list, res_channels_list)
            ]

        # Add resnet blocks that also process the time embedding
        self.resnets = [
            ResnetBlock2D(
                in_channels=ic,
                out_channels=out_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
            )
            for ic in in_channels_list
        ]

        # Add an optional downsampling layer
        if add_downsample:
            self.downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

        # or upsampling layer
        if add_upsample:
            self.upsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def __call__(
        self,
        x,
        temb=None,
        residual_hidden_states=None,
    ):
        output_states = []

        for i in range(len(self.resnets)):
            if residual_hidden_states is not None:
                x = mx.concatenate([x, residual_hidden_states.pop()], axis=-1)

            x = self.resnets[i](x, temb)

            output_states.append(x)

        if "downsample" in self:
            x = self.downsample(x)
            output_states.append(x)

        if "upsample" in self:
            x = self.upsample(upsample_nearest(x))
            output_states.append(x)

        return x, output_states


class UNetModel(nn.Module):
    """The conditional 2D UNet model that actually performs the denoising."""

    def __init__(self, in_channels=1, 
                 out_channels=1, 
                 conv_in_kernel=3, 
                 conv_out_kernel=3,
                 block_out_channels=(8, 16, 32, 64), 
                 layers_per_block=(1,1,1,1),
                 mid_block_layers=1,
                 norm_num_groups=8
                ):
        super().__init__()
        # You can now use these arguments to set up your model layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_in_kernel = conv_in_kernel
        self.conv_out_kernel = conv_out_kernel
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.mid_block_layers = mid_block_layers
        self.norm_num_groups = norm_num_groups

        self.conv_in = nn.Conv2d(
            self.in_channels,
            self.block_out_channels[0],
            self.conv_in_kernel,
            padding=(self.conv_in_kernel - 1) // 2,
        )

        self.timesteps = nn.SinusoidalPositionalEncoding(
            self.block_out_channels[0],
            max_freq=1,
            min_freq=math.exp(
                -math.log(10000) + 2 * math.log(10000) / self.block_out_channels[0]
            ),
            scale=1.0,
            cos_first=True,
            full_turns=False,
        )
        self.time_embedding = TimestepEmbedding(
            self.block_out_channels[0],
            self.block_out_channels[0] * 4,
        )

        # Make the downsampling blocks
        block_channels = [self.block_out_channels[0]] + list(
            self.block_out_channels
        )
        self.down_blocks = [
            UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=self.block_out_channels[0] * 4,
                num_layers=self.layers_per_block[i],
                resnet_groups=self.norm_num_groups,
                add_downsample=(i < len(self.block_out_channels) - 1),
                add_upsample=False,
            )
            for i, (in_channels, out_channels) in enumerate(
                zip(block_channels, block_channels[1:])
            )
        ]

        # Make the middle block
        self.mid_blocks = [
            ResnetBlock2D(
                in_channels=self.block_out_channels[-1],
                out_channels=self.block_out_channels[-1],
                temb_channels=self.block_out_channels[0] * 4,
                groups=self.norm_num_groups,
            ),
            ResnetBlock2D(
                in_channels=self.block_out_channels[-1],
                out_channels=self.block_out_channels[-1],
                temb_channels=self.block_out_channels[0] * 4,
                groups=self.norm_num_groups,
            ),
        ]

        # Make the upsampling blocks
        block_channels = (
            [self.block_out_channels[0]]
            + list(self.block_out_channels)
            + [self.block_out_channels[-1]]
        )
        self.up_blocks = [
            UNetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=self.block_out_channels[0] * 4,
                prev_out_channels=prev_out_channels,
                num_layers=self.layers_per_block[i] + 1,
                resnet_groups=self.norm_num_groups,
                add_downsample=False,
                add_upsample=(i > 0),
            )
            for i, (in_channels, out_channels, prev_out_channels) in reversed(
                list(
                    enumerate(
                        zip(block_channels, block_channels[1:], block_channels[2:])
                    )
                )
            )
        ]

        self.conv_norm_out = nn.GroupNorm(
            self.norm_num_groups,
            self.block_out_channels[0],
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(
            self.block_out_channels[0],
            self.out_channels,
            self.conv_out_kernel,
            padding=(self.conv_out_kernel - 1) // 2,
        )

    def __call__(
        self,
        x,
        timestep
    ):
        # Compute the time embeddings
        temb = self.timesteps(timestep).astype(x.dtype)
        temb = self.time_embedding(temb)

        # Preprocess the input
        x = self.conv_in(x)

        # Run the downsampling part of the unet
        residuals = [x]
        for block in self.down_blocks:
            x, res = block(
                x,
                temb=temb,
            )
            residuals.extend(res)

        # Run the middle part of the unet
        x = self.mid_blocks[0](x, temb)
        x = self.mid_blocks[1](x, temb)

        # Run the upsampling part of the unet
        for block in self.up_blocks:
            x, _ = block(
                x,
                temb=temb,
                residual_hidden_states=residuals,
            )

        # Postprocess the output
        dtype = x.dtype
        x = self.conv_norm_out(x.astype(mx.float32)).astype(dtype)
        x = nn.silu(x)
        x = self.conv_out(x)

        return x