"""
Task 5: GAN improved with self-attention and cross-attention.
The generator can focus on relevant parts of the text embedding (cross-attn)
and on long-range spatial relationships (self-attn).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Simple self-attention over spatial feature maps (SAGAN-style)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B, N, C'
        k = self.key(x).view(B, -1, H * W)  # B, C', N
        attn = self.softmax(torch.bmm(q, k))  # B, N, N
        v = self.value(x).view(B, -1, H * W)  # B, C, N
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class CrossAttention(nn.Module):
    """
    Cross-attention: spatial features attend to text embedding.
    text_emb is projected to key/value; spatial features produce queries.
    """

    def __init__(self, feat_channels: int, text_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feat_channels // num_heads
        assert feat_channels % num_heads == 0

        self.q_proj = nn.Conv2d(feat_channels, feat_channels, 1)
        self.k_proj = nn.Linear(text_dim, feat_channels)
        self.v_proj = nn.Linear(text_dim, feat_channels)
        self.out_proj = nn.Conv2d(feat_channels, feat_channels, 1)
        self.scale = self.head_dim ** -0.5

    def forward(self, feat: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)
        text_emb: (B, text_dim)  – single vector per sample
        """
        B, C, H, W = feat.shape
        N = H * W

        q = self.q_proj(feat).view(B, self.num_heads, self.head_dim, N)  # B, h, d, N
        q = q.permute(0, 1, 3, 2)  # B, h, N, d

        # expand text to a short sequence of length 1 (or keep as is)
        k = self.k_proj(text_emb).view(B, self.num_heads, self.head_dim, 1)  # B, h, d, 1
        v = self.v_proj(text_emb).view(B, self.num_heads, self.head_dim, 1)

        k = k.permute(0, 1, 3, 2)  # B, h, 1, d
        v = v.permute(0, 1, 3, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # B, h, N, 1
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # B, h, N, d
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return self.out_proj(out) + feat


class AttnGenerator(nn.Module):
    """
    Generator that accepts continuous text embeddings (from HF encoder)
    and uses cross-attention + self-attention.
    Output: (B, 3, 64, 64) in [-1, 1]
    """

    def __init__(
        self,
        z_dim: int = 100,
        text_dim: int = 768,
        base_channels: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.text_dim = text_dim

        # project text embedding
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
        )

        # initial projection of noise + text
        self.fc = nn.Sequential(
            nn.Linear(z_dim + 256, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True),
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.cross1 = CrossAttention(256, 256)
        self.self1 = SelfAttention(256)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.cross2 = CrossAttention(128, 256)
        self.self2 = SelfAttention(128)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.cross3 = CrossAttention(64, 256)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)  # (B, 256)
        x = torch.cat([z, t], dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)

        x = self.up1(x)
        x = self.cross1(x, t)
        x = self.self1(x)

        x = self.up2(x)
        x = self.cross2(x, t)
        x = self.self2(x)

        x = self.up3(x)
        x = self.cross3(x, t)

        x = self.up4(x)
        return x


class AttnDiscriminator(nn.Module):
    """Discriminator with optional self-attention."""

    def __init__(self, text_dim: int = 768):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 256)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SelfAttention(128),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4 + 256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        feat = self.conv(img)
        x = torch.cat([feat, t], dim=1)
        return self.fc(x)


def build_attn_gan(z_dim: int = 100, text_dim: int = 768, device: str = "cpu"):
    G = AttnGenerator(z_dim=z_dim, text_dim=text_dim).to(device)
    D = AttnDiscriminator(text_dim=text_dim).to(device)
    return G, D
