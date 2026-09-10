"""
Task 5 – Attention-enhanced GAN
Adds self-attention and cross-attention so the generator focuses on
relevant parts of the text embedding and long-range spatial structure.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.text_encoder import get_default_encoder
from src.models.attn_gan import build_attn_gan
from src.models.cgan import SHAPE_LABELS
from PIL import Image, ImageDraw


def generate_shape_image(label_idx: int, size: int = 64) -> np.ndarray:
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    color = (255, 255, 255)
    if label_idx == 0:
        draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    elif label_idx == 1:
        draw.rectangle([margin, margin, size - margin, size - margin], fill=color)
    elif label_idx == 2:
        draw.polygon(
            [(size // 2, margin), (size - margin, size - margin), (margin, size - margin)],
            fill=color,
        )
    elif label_idx == 3:
        cx, cy = size // 2, size // 2
        r_out, r_in = size // 2 - margin, size // 5
        pts = []
        for i in range(5):
            a = np.deg2rad(90 + i * 72)
            pts.append((cx + r_out * np.cos(a), cy - r_out * np.sin(a)))
            a2 = np.deg2rad(90 + i * 72 + 36)
            pts.append((cx + r_in * np.cos(a2), cy - r_in * np.sin(a2)))
        draw.polygon(pts, fill=color)
    else:
        cx, cy = size // 2, size // 2
        r = size // 2 - margin
        pts = [
            (cx + r * np.cos(np.deg2rad(60 * i)), cy + r * np.sin(np.deg2rad(60 * i)))
            for i in range(6)
        ]
        draw.polygon(pts, fill=color)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)


def make_text_conditioned_dataset(n_per_class: int = 150, size: int = 64):
    """
    Procedural shapes + text embeddings derived from class name captions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_default_encoder(prefer_clip=False, device=device)

    images, embeddings, labels = [], [], []
    for idx, name in SHAPE_LABELS.items():
        captions = [
            f"a {name}",
            f"simple {name} shape",
            f"geometric {name}",
            f"white {name} on black background",
        ]
        for i in range(n_per_class):
            img = generate_shape_image(idx, size)
            noise = np.random.normal(0, 0.04, img.shape).astype(np.float32)
            img = np.clip(img + noise, -1, 1)
            images.append(img)
            labels.append(idx)
            emb = encoder.encode(captions[i % len(captions)])
            embeddings.append(emb.cpu().squeeze(0))

    images = torch.tensor(np.stack(images), dtype=torch.float32)
    embeddings = torch.stack(embeddings)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(images, embeddings, labels), encoder.hidden_size


def train_attn_gan(
    epochs: int = 12,
    batch_size: int = 32,
    z_dim: int = 100,
    lr: float = 2e-4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Attention-GAN on {device}")

    dataset, text_dim = make_text_conditioned_dataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G, D = build_attn_gan(z_dim=z_dim, text_dim=text_dim, device=device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, real_embs, _ in pbar:
            real_imgs = real_imgs.to(device)
            real_embs = real_embs.to(device)
            B = real_imgs.size(0)

            # Discriminator
            opt_D.zero_grad()
            real_pred = D(real_imgs, real_embs)
            real_loss = criterion(real_pred, torch.ones_like(real_pred))

            z = torch.randn(B, z_dim, device=device)
            # use the same embeddings for fake (matching condition)
            fake_imgs = G(z, real_embs).detach()
            fake_pred = D(fake_imgs, real_embs)
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # Generator
            opt_G.zero_grad()
            z = torch.randn(B, z_dim, device=device)
            gen_imgs = G(z, real_embs)
            gen_pred = D(gen_imgs, real_embs)
            g_loss = criterion(gen_pred, torch.ones_like(gen_pred))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            pbar.set_postfix(d=f"{d_loss.item():.3f}", g=f"{g_loss.item():.3f}")

        print(f"  epoch {epoch}: D={np.mean(d_losses):.3f}  G={np.mean(g_losses):.3f}")

    out = Path("outputs/attn_gan")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), out / "generator.pt")
    torch.save(D.state_dict(), out / "discriminator.pt")
    print(f"Saved → {out}")
    return G, D


if __name__ == "__main__":
    train_attn_gan(epochs=8)
    print("Task 5 complete.")
