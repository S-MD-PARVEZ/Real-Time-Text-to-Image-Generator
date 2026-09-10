"""
Task 2: Conditional GAN (CGAN) that generates basic shapes
from textual / categorical labels such as "square", "circle", "triangle".
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Simple DCGAN-style generator conditioned on a discrete label embedding.
    Input: noise z (B, z_dim) + label embedding (B, embed_dim)
    Output: image (B, 3, 64, 64) in [-1, 1]
    """

    def __init__(self, z_dim: int = 100, embed_dim: int = 50, n_classes: int = 5, img_channels: int = 3):
        super().__init__()
        self.z_dim = z_dim
        self.label_emb = nn.Embedding(n_classes, embed_dim)

        self.net = nn.Sequential(
            # input: (z_dim + embed_dim) → 4x4
            nn.Linear(z_dim + embed_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4)),
            # 4x4 → 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 8x8 → 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 16x16 → 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 32x32 → 64x64
            nn.ConvTranspose2d(32, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: (B,) long
        emb = self.label_emb(labels)  # (B, embed_dim)
        x = torch.cat([z, emb], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    Discriminator that also receives the class label.
    """

    def __init__(self, embed_dim: int = 50, n_classes: int = 5, img_channels: int = 3):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, embed_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        # after 4 downsamples: 64 → 4, so 256*4*4 = 4096
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 + embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_emb(labels)
        feat = self.conv(img)
        x = torch.cat([feat, emb], dim=1)
        return self.fc(x)


# Label mapping used throughout the project
SHAPE_LABELS = {
    0: "circle",
    1: "square",
    2: "triangle",
    3: "star",
    4: "hexagon",
}
LABEL_TO_IDX = {v: k for k, v in SHAPE_LABELS.items()}


def build_cgan(z_dim: int = 100, n_classes: int = 5, device: str = "cpu"):
    G = Generator(z_dim=z_dim, n_classes=n_classes).to(device)
    D = Discriminator(n_classes=n_classes).to(device)
    return G, D
