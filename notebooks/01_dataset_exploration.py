"""
Task 4 – Dataset Exploration
Load and examine Oxford-102 Flowers (or COCO).
Analyse: number of classes, image resolution, description statistics,
and visualise image + text pairs.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from collections import Counter
from PIL import Image


def load_flowers(root="data/flowers"):
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
    ])
    train = datasets.Flowers102(root=root, split="train", download=True, transform=transform)
    val = datasets.Flowers102(root=root, split="val", download=True, transform=transform)
    test = datasets.Flowers102(root=root, split="test", download=True, transform=transform)
    return train, val, test


def analyse(train, val, test):
    print("=" * 60)
    print("OXFORD-102 FLOWERS – DATASET STATISTICS")
    print("=" * 60)
    print(f"Train images : {len(train)}")
    print(f"Val images   : {len(val)}")
    print(f"Test images  : {len(test)}")
    print(f"Total        : {len(train) + len(val) + len(test)}")
    print(f"Number of classes : 102")

    # class distribution (train)
    labels = [y for _, y in train]
    counts = Counter(labels)
    print(f"\nClass distribution (train) – min/max/mean samples per class:")
    vals = list(counts.values())
    print(f"  min={min(vals)}, max={max(vals)}, mean={np.mean(vals):.1f}")

    # image resolution (after transform all are 128x128)
    img, _ = train[0]
    print(f"\nImage tensor shape after transform: {tuple(img.shape)}  (C,H,W)")
    print("Original images vary; we resize + center-crop to 128×128 for training.")

    # Note: torchvision Flowers102 does not ship with free-text captions.
    # For caption-style experiments we either:
    #   a) use class names as weak text labels, or
    #   b) load a captioned subset / COCO.
    print("\nText / description note:")
    print("  Oxford-102 provides class labels (0-101).")
    print("  Class names can be used as simple textual categories.")
    print("  For richer captions, switch to COCO or add your own.")

    return counts


def visualise_samples(dataset, n=6, title="Train samples"):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    for ax, idx in zip(axes, range(n)):
        img, label = dataset[idx]
        # img is tensor CxHxW in [0,1]
        arr = img.permute(1, 2, 0).numpy()
        ax.imshow(arr)
        ax.set_title(f"class {label}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    out = Path("outputs") / "dataset_samples.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"Saved visualisation → {out}")
    plt.close()


def main():
    train, val, test = load_flowers()
    analyse(train, val, test)
    visualise_samples(train)
    print("\nTask 4 complete. Proceed to text preprocessing notebook.")


if __name__ == "__main__":
    main()
