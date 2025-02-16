# Taken from FLUX and edited for variational rectified flow
import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from dit import TimestepEmbedder  # <-- NEW: Import time embedder from dit.py


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.head_dim = 64
        self.num_heads = in_channels // self.head_dim
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        nn.init.normal_(self.proj_out.weight, std=0.2 / math.sqrt(in_channels))

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        qkv = self.qkv(h_)
        q, k, v = qkv.chunk(3, dim=1)
        b, c, h, w = q.shape
        q = rearrange(
            q, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        k = rearrange(
            k, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        v = rearrange(
            v, "b (h d) x y -> b h (x y) d", h=self.num_heads, d=self.head_dim
        )
        h_ = F.scaled_dot_product_attention(q, k, v)
        h_ = rearrange(h_, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return h_

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = nn.GroupNorm(
            num_groups=32, num_channels=out_channels, eps=1e-6, affine=True
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )
        self.time_embed = TimestepEmbedder(hidden_size=self.ch, frequency_embedding_size=256)
        
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_in, eps=1e-6, affine=True
        )
        self.conv_out = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        h = self.conv_in(x)
        t_emb = self.time_embed(t)
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h
    

class VariationalDiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
        self.kl = None  # Will hold the KL divergence computed in forward

    def forward(self, z: Tensor) -> Tensor:
        # Split the input tensor into mean and log-variance along the specified dimension.
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        # Compute the KL divergence for a diagonal Gaussian:
        # KL = -0.5 * sum(1 + logvar - mean^2 - exp(logvar)) over all latent dimensions (excluding batch)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - torch.exp(logvar), dim=list(range(1, mean.dim())))
        self.kl = kl_div.mean()  # Average over the batch to produce a scalar loss

        if self.sample:
            # Compute the standard deviation from log-variance.
            std = torch.exp(0.5 * logvar)
            # Reparameterization trick: sample epsilon and generate latent sample.
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
class VariationalEncoder(nn.Module):
    def __init__(self, resolution, in_channels, ch, ch_mult, num_res_blocks, z_channels):
        super().__init__()
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.reg = VariationalDiagonalGaussian()

    def forward(self, x0: Tensor, x1: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        x = torch.cat([x0, x1, xt], dim=1)
        z = self.encoder(x, t)
        z_s = self.reg(z)
        return z_s
    
    
if __name__ == "__main__":
    # vae = VAE(
    #     resolution=28,
    #     in_channels=3,
    #     ch=64,
    #     out_ch=3,
    #     ch_mult=[1, 2, 4],
    #     num_res_blocks=2,
    #     z_channels=2,
    # )
    # vae.eval().to("cuda")
    # print("VAE parameter count: ", sum(p.numel() for p in vae.parameters())//1e6, "M")
    # x = torch.randn(1, 3, 28, 28).to("cuda")
    # x_hat, z = vae(x)
    # print(x_hat.shape, z.shape)
    
    variational_encoder = VariationalEncoder(
        resolution=28,
        in_channels=3,
        ch=64,
        ch_mult=[1],
        num_res_blocks=2,
        z_channels=2,
    )
    variational_encoder.eval().to("cuda")
    x0 = torch.randn(16, 1, 28, 28).to("cuda")
    x1 = torch.randn(16, 1, 28, 28).to("cuda")  
    xt = torch.randn(16, 1, 28, 28).to("cuda")
    t = torch.randn(16,).to("cuda")
    z_s = variational_encoder(x0, x1, xt, t)
    print(f'num parameters: {sum(p.numel() for p in variational_encoder.parameters())}')
    print(z_s.shape)