from typing import Tuple

import torch
import math
from torch import Tensor, nn
from typing import *


class TimeEmbedding(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        assert (n_channels & (-n_channels)) >= 8
        self.n_channels = n_channels
        self.model = nn.Sequential(
            nn.Linear(n_channels // 4, n_channels),
            nn.LeakyReLU(),
            nn.Linear(n_channels, n_channels)
        )

    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        return self.model(emb)


class ResidualBlock(nn.Module):

    @staticmethod
    def _conv_block(in_channel: int, out_channel: int, group: int):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.GroupNorm(group, out_channel)
        )

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        super().__init__()
        self.conv1 = ResidualBlock._conv_block(in_channels, out_channels, n_groups)

        self.conv2 = ResidualBlock._conv_block(out_channels, out_channels, n_groups)

        self.out_channel = out_channels

        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels,
                                                                                    kernel_size=3, padding=1)

        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        h = self.conv1(x)

        h += self.time_emb(t).reshape(-1, self.out_channel, *((1,) * (len(x.shape) - 2)))

        h = self.conv2(h)

        return h + self.identity(x)


class AttentionBlock(nn.Module):

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):
        super().__init__()

        if d_k is None:
            d_k = n_channels

        self.attention = nn.MultiheadAttention(d_k * n_heads, d_k, batch_first=True)

        self.qkv_linear = nn.Linear(n_channels, n_heads * d_k * 3)

        self.out_linear = nn.Linear(n_heads * d_k, n_channels)

        self.n_channels = n_channels
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        _ = t
        shape = x.shape
        batch = shape[0]

        p = x.view(batch, self.n_channels, -1) \
            .permute(0, 2, 1)  # transform x from (f, d1, d2, ...) into (d1, d2, ..., f)
        q, k, v = torch.chunk(self.qkv_linear(p), chunks=3, dim=-1)  # get q, k, v

        result, attn = self.attention(q, k, v)
        result = self.out_linear(result)
        result += p  # residual

        result = result.permute(0, 2, 1) \
            .view(batch, -1, *shape[2:])  # transform x from (d1, d2, ..., f) into (f, d1, d2, ...)

        return result


class DownBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, attention: bool):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, time_channels)
        self.attention = AttentionBlock(out_channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.attention(self.block(x, t))


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, attention: bool):
        super().__init__()
        self.block = ResidualBlock(in_channels, out_channels, time_channels)
        self.attention = AttentionBlock(out_channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.attention(self.block(x, t))


class MiddleBlock(nn.Module):

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attention = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        return self.res2(self.attention(self.res1(x, t)), t)


class Upsample(nn.Module):

    def __init__(self, n_channels: int):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2)
        self.conv = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class Downsample(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self,
                 input_channel: int, output_channel: int,
                 n_channel: int = 64,
                 resolute_multiplication: Tuple[int, ...] = (1, 2, 2, 4),
                 is_attention: Tuple[int, ...] = (False, False, True, True),
                 num_blocks: int = 2,
                 time_channel: int = 256):
        super().__init__()

        assert len(resolute_multiplication) == len(is_attention)

        n_resolution = len(resolute_multiplication)

        self.time_embed = TimeEmbedding(time_channel)

        self.start = nn.Conv2d(input_channel, n_channel, kernel_size=3, padding=1)

        down_sampling = []

        in_channel = n_channel
        out_channel = n_channel

        for i in range(n_resolution):
            out_channel = in_channel * resolute_multiplication[i]
            for _ in range(num_blocks):
                down_sampling.append(DownBlock(in_channel, out_channel, time_channel, is_attention[i]))
                in_channel = out_channel
            if i < n_resolution - 1:
                down_sampling.append(Downsample(in_channel))

        self.down = nn.ModuleList(down_sampling)

        self.middle = MiddleBlock(in_channel, time_channel)

        up_sampling = []

        for i in reversed(range(n_resolution)):
            out_channel = in_channel
            for _ in range(num_blocks):
                up_sampling.append(UpBlock(in_channel + out_channel, out_channel, time_channel, is_attention[i]))

            out_channel = in_channel // resolute_multiplication[i]
            up_sampling.append(UpBlock(in_channel + out_channel, out_channel, time_channel, is_attention[i]))
            in_channel = out_channel
            if i > 0:
                up_sampling.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_sampling)

        assert in_channel == n_channel

        self.final = nn.Sequential(
            nn.GroupNorm(8, in_channel),
            nn.LeakyReLU(),
            nn.Conv2d(in_channel, output_channel, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        t = self.time_embed(t)
        x = self.start(x)

        past = [x]

        for layer in self.down:
            x = layer(x, t)
            past.append(x)

        x = self.middle(x, t)

        for layer in self.up:
            if not isinstance(layer, Upsample):
                x = torch.concat([x, past.pop()], dim=1)
            x = layer(x, t)

        return self.final(x)
