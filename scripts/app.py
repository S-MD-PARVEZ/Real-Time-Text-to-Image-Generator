"""
Gradio demo for the text-to-image pipeline.
Provides a live URL suitable for the internship submission.
"""

from __future__ import annotations

import gradio as gr
import torch
from pathlib import Path
import sys

# add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import Text2ImagePipeline, ShapeCGANPipeline
from src.models.cgan import SHAPE_LABELS


def create_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # Note: generators are randomly initialised unless checkpoints are loaded.
    # For a polished demo, load trained weights from outputs/.
    text_pipe = Text2ImagePipeline(device=device, use_attention=True)
    shape_pipe = ShapeCGANPipeline(device=device)

    def generate_from_text(prompt: str, num_samples: int, seed: int):
        if not prompt.strip():
            return []
        imgs = text_pipe(prompt, num_samples=num_samples, seed=seed if seed >= 0 else None)
        return imgs

    def generate_shape(label: str, num_samples: int, seed: int):
        imgs = shape_pipe(label, num_samples=num_samples, seed=seed if seed >= 0 else None)
        return imgs

    with gr.Blocks(title="Text-to-Image GAN Pipeline") as demo:
        gr.Markdown(
            """
            # Real-Time Text-to-Image Generator – Gen AI (v2)
            Internship project (Elevanceskills).  
            **Note:** Without trained checkpoints the outputs are random noise.
            Train the models (notebooks 03-05) and load weights for meaningful results.
            """
        )

        with gr.Tab("Free-form Text (Attn-GAN)"):
            txt = gr.Textbox(label="Prompt", placeholder="a red flower with green leaves")
            n1 = gr.Slider(1, 4, value=2, step=1, label="Number of samples")
            seed1 = gr.Number(value=-1, label="Seed (-1 = random)")
            btn1 = gr.Button("Generate")
            gallery1 = gr.Gallery(label="Generated images")
            btn1.click(generate_from_text, [txt, n1, seed1], gallery1)

        with gr.Tab("Shape CGAN (discrete labels)"):
            label = gr.Dropdown(
                choices=list(SHAPE_LABELS.values()),
                value="circle",
                label="Shape label",
            )
            n2 = gr.Slider(1, 8, value=4, step=1, label="Number of samples")
            seed2 = gr.Number(value=-1, label="Seed (-1 = random)")
            btn2 = gr.Button("Generate shape")
            gallery2 = gr.Gallery(label="Generated shapes")
            btn2.click(generate_shape, [label, n2, seed2], gallery2)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
