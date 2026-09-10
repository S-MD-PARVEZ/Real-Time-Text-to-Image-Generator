"""
Download and prepare public datasets used by the project.
Oxford-102 Flowers is the primary dataset for exploration (Task 4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive


def prepare_flowers(root: str = "data/flowers"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ]
    )

    print("Downloading / loading Oxford-102 Flowers (train split)...")
    train_set = datasets.Flowers102(
        root=str(root),
        split="train",
        download=True,
        transform=transform,
    )
    val_set = datasets.Flowers102(
        root=str(root),
        split="val",
        download=True,
        transform=transform,
    )
    test_set = datasets.Flowers102(
        root=str(root),
        split="test",
        download=True,
        transform=transform,
    )

    print(f"Train: {len(train_set)} images")
    print(f"Val  : {len(val_set)} images")
    print(f"Test : {len(test_set)} images")
    print(f"Number of classes: 102")
    print(f"Data root: {root.resolve()}")
    return train_set, val_set, test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/flowers")
    args = parser.parse_args()
    prepare_flowers(args.root)


if __name__ == "__main__":
    main()
