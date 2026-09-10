"""
Task 6: Comprehensive text-to-image generating pipeline.
Combines text preprocessing (HF), embedding creation, and GAN-based generation.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from PIL import Image
import numpy as np

from .text_encoder import TextEncoder, get_default_encoder
from .models.attn_gan import AttnGenerator, build_attn_gan
from .models.cgan import Generator as CganGenerator, LABEL_TO_IDX, SHAPE_LABELS


class Text2ImagePipeline:
    """
    End-to-end pipeline:
        text prompt → HF encoder → embedding → (Attn)Generator → image
    """

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        generator: Optional[torch.nn.Module] = None,
        z_dim: int = 100,
        device: Optional[str] = None,
        use_attention: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = z_dim
        self.use_attention = use_attention

        self.text_encoder = text_encoder or get_default_encoder(
            prefer_clip=True, device=self.device
        )
        text_dim = self.text_encoder.hidden_size

        if generator is not None:
            self.generator = generator.to(self.device)
        else:
            if use_attention:
                self.generator, _ = build_attn_gan(
                    z_dim=z_dim, text_dim=text_dim, device=self.device
                )
            else:
                # fallback to simple CGAN (needs discrete labels)
                raise ValueError("Non-attention path expects a pre-built CGAN generator")

        self.generator.eval()

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        num_samples: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        if isinstance(prompt, str):
            prompts = [prompt] * num_samples
        else:
            prompts = prompt
            num_samples = len(prompts)

        if seed is not None:
            torch.manual_seed(seed)

        emb = self.text_encoder.encode(prompts)  # (B, text_dim)
        z = torch.randn(num_samples, self.z_dim, device=self.device)

        imgs = self.generator(z, emb)  # (B, 3, H, W) in [-1, 1]
        return self._tensor_to_pil(imgs)

    def _tensor_to_pil(self, tensor: torch.Tensor) -> List[Image.Image]:
        # tensor: (B, 3, H, W) in [-1, 1]
        tensor = (tensor.clamp(-1, 1) + 1) / 2  # [0, 1]
        tensor = tensor.cpu().permute(0, 2, 3, 1).numpy()
        images = []
        for arr in tensor:
            arr = (arr * 255).astype(np.uint8)
            images.append(Image.fromarray(arr))
        return images


class ShapeCGANPipeline:
    """
    Convenience pipeline for the discrete-label CGAN (Task 2).
    Accepts label names such as "square", "circle".
    """

    def __init__(
        self,
        generator: Optional[CganGenerator] = None,
        z_dim: int = 100,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.z_dim = z_dim
        if generator is None:
            from .models.cgan import build_cgan
            generator, _ = build_cgan(z_dim=z_dim, device=self.device)
        self.generator = generator.to(self.device).eval()

    @torch.no_grad()
    def __call__(
        self,
        label: str,
        num_samples: int = 4,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        if label not in LABEL_TO_IDX:
            raise ValueError(f"Unknown label '{label}'. Choose from {list(LABEL_TO_IDX.keys())}")
        idx = LABEL_TO_IDX[label]
        labels = torch.full((num_samples,), idx, dtype=torch.long, device=self.device)

        if seed is not None:
            torch.manual_seed(seed)
        z = torch.randn(num_samples, self.z_dim, device=self.device)
        imgs = self.generator(z, labels)
        return Text2ImagePipeline._tensor_to_pil(None, imgs)  # reuse helper
