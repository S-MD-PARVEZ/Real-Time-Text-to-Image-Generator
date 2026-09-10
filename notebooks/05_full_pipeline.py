"""
Task 6 – Full text-to-image pipeline
Integrates: text preprocessing (HF) → embedding → attention-GAN generation.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.pipeline import Text2ImagePipeline, ShapeCGANPipeline
from src.models.cgan import SHAPE_LABELS
from src.models.attn_gan import build_attn_gan
from src.text_encoder import get_default_encoder


def demo_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Shape CGAN path (Task 2) ---
    print("\n=== Shape CGAN (discrete labels) ===")
    shape_pipe = ShapeCGANPipeline(device=device)
    for name in list(SHAPE_LABELS.values())[:3]:
        imgs = shape_pipe(name, num_samples=2, seed=42)
        print(f"  Generated {len(imgs)} images for label '{name}' – size {imgs[0].size}")

    # --- Full text pipeline (Task 6) ---
    print("\n=== Full Text → Embedding → Attn-GAN pipeline ===")
    # Build with randomly initialised generator (replace with trained weights for real results)
    encoder = get_default_encoder(prefer_clip=False, device=device)
    G, _ = build_attn_gan(z_dim=100, text_dim=encoder.hidden_size, device=device)

    # Optional: load checkpoint if it exists
    ckpt = Path("outputs/attn_gan/generator.pt")
    if ckpt.exists():
        G.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint {ckpt}")
    else:
        print("No checkpoint found – outputs will be random (train notebook 04 first).")

    pipe = Text2ImagePipeline(
        text_encoder=encoder,
        generator=G,
        device=device,
        use_attention=True,
    )

    prompts = [
        "a red circle",
        "simple geometric square",
        "white triangle on black background",
    ]
    for p in prompts:
        imgs = pipe(p, num_samples=1, seed=123)
        out = Path("outputs/pipeline") / f"{p.replace(' ', '_')[:40]}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        imgs[0].save(out)
        print(f"  '{p}' → {out}")

    print("\nTask 6 complete. Pipeline is ready for Gradio demo (scripts/app.py).")


if __name__ == "__main__":
    demo_pipeline()
