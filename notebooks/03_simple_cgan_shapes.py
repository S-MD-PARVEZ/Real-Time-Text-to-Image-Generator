"""
Task 2 – Conditional GAN for basic shapes
Generates circle / square / triangle / star / hexagon from discrete labels.
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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.cgan import build_cgan, SHAPE_LABELS, LABEL_TO_IDX


def generate_shape_image(label_idx: int, size: int = 64) -> np.ndarray:
    """Procedurally create a clean shape image (white on black)."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = size // 8
    color = (255, 255, 255)

    if label_idx == 0:  # circle
        draw.ellipse([margin, margin, size - margin, size - margin], fill=color)
    elif label_idx == 1:  # square
        draw.rectangle([margin, margin, size - margin, size - margin], fill=color)
    elif label_idx == 2:  # triangle
        draw.polygon(
            [(size // 2, margin), (size - margin, size - margin), (margin, size - margin)],
            fill=color,
        )
    elif label_idx == 3:  # star (simple 5-point approximation)
        cx, cy = size // 2, size // 2
        r_out, r_in = size // 2 - margin, size // 5
        pts = []
        for i in range(5):
            a = np.deg2rad(90 + i * 72)
            pts.append((cx + r_out * np.cos(a), cy - r_out * np.sin(a)))
            a2 = np.deg2rad(90 + i * 72 + 36)
            pts.append((cx + r_in * np.cos(a2), cy - r_in * np.sin(a2)))
        draw.polygon(pts, fill=color)
    else:  # hexagon
        cx, cy = size // 2, size // 2
        r = size // 2 - margin
        pts = [
            (cx + r * np.cos(np.deg2rad(60 * i)), cy + r * np.sin(np.deg2rad(60 * i)))
            for i in range(6)
        ]
        draw.polygon(pts, fill=color)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    return arr.transpose(2, 0, 1)  # C,H,W


def make_dataset(n_per_class: int = 200, size: int = 64):
    images, labels = [], []
    for idx in range(len(SHAPE_LABELS)):
        for _ in range(n_per_class):
            # slight random shift / noise for variety
            img = generate_shape_image(idx, size)
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, -1, 1)
            images.append(img)
            labels.append(idx)
    images = torch.tensor(np.stack(images), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return TensorDataset(images, labels)


def train_cgan(
    epochs: int = 15,
    batch_size: int = 64,
    z_dim: int = 100,
    lr: float = 2e-4,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CGAN on {device}")

    dataset = make_dataset(n_per_class=300)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    G, D = build_cgan(z_dim=z_dim, n_classes=len(SHAPE_LABELS), device=device)
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(len(SHAPE_LABELS) * 4, z_dim, device=device)
    fixed_labels = torch.arange(len(SHAPE_LABELS), device=device).repeat_interleave(4)

    history = {"d_loss": [], "g_loss": []}

    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, real_labels in pbar:
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            B = real_imgs.size(0)

            # ----- Discriminator -----
            opt_D.zero_grad()
            real_pred = D(real_imgs, real_labels)
            real_loss = criterion(real_pred, torch.ones_like(real_pred))

            z = torch.randn(B, z_dim, device=device)
            fake_labels = torch.randint(0, len(SHAPE_LABELS), (B,), device=device)
            fake_imgs = G(z, fake_labels).detach()
            fake_pred = D(fake_imgs, fake_labels)
            fake_loss = criterion(fake_pred, torch.zeros_like(fake_pred))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

            # ----- Generator -----
            opt_G.zero_grad()
            z = torch.randn(B, z_dim, device=device)
            gen_labels = torch.randint(0, len(SHAPE_LABELS), (B,), device=device)
            gen_imgs = G(z, gen_labels)
            gen_pred = D(gen_imgs, gen_labels)
            g_loss = criterion(gen_pred, torch.ones_like(gen_pred))
            g_loss.backward()
            opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            pbar.set_postfix(d=f"{d_loss.item():.3f}", g=f"{g_loss.item():.3f}")

        history["d_loss"].append(np.mean(d_losses))
        history["g_loss"].append(np.mean(g_losses))

        # save sample grid
        if epoch % 5 == 0 or epoch == epochs:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z, fixed_labels)
            G.train()
            save_grid(samples, fixed_labels, epoch)

    # save final weights
    out_dir = Path("outputs/cgan")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), out_dir / "generator.pt")
    torch.save(D.state_dict(), out_dir / "discriminator.pt")
    print(f"Saved checkpoints → {out_dir}")
    return G, D, history


def save_grid(imgs: torch.Tensor, labels: torch.Tensor, epoch: int):
    imgs = (imgs.clamp(-1, 1) + 1) / 2
    n = len(SHAPE_LABELS)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i in range(n):
        for j in range(4):
            idx = i * 4 + j
            arr = imgs[idx].cpu().permute(1, 2, 0).numpy()
            axes[i, j].imshow(arr)
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_ylabel(SHAPE_LABELS[i], rotation=0, labelpad=40)
    plt.suptitle(f"CGAN samples – epoch {epoch}")
    plt.tight_layout()
    out = Path("outputs/cgan") / f"samples_epoch{epoch:03d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  saved {out}")


if __name__ == "__main__":
    train_cgan(epochs=10, batch_size=64)
    print("Task 2 complete.")
