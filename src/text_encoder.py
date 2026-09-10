"""
Task 3: Text preprocessing with Hugging Face Transformers.
Converts free-form text descriptions into tokenized and encoded embeddings
that can be fed to the text-to-image models.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    """
    Wrapper around a Hugging Face text encoder (BERT or CLIP).
    Produces fixed-size embeddings usable by a Generator.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 64,
        pooling: str = "cls",  # "cls" | "mean"
        device: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if "clip" in model_name.lower():
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.encoder = CLIPTextModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)

        self.encoder.to(self.device)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.hidden_size = self.encoder.config.hidden_size

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        return_tokens: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Encode a single string or a list of strings into embeddings.

        Returns:
            embeddings: (B, hidden_size)
            optionally (input_ids, attention_mask)
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.encoder(**inputs)
        last_hidden = outputs.last_hidden_state  # (B, L, H)

        if self.pooling == "cls":
            embeddings = last_hidden[:, 0, :]
        else:  # mean pooling (respect attention mask)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            embeddings = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        if return_tokens:
            return embeddings, inputs["input_ids"], inputs["attention_mask"]
        return embeddings

    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        return self.encode(texts)


def get_default_encoder(prefer_clip: bool = True, device: Optional[str] = None) -> TextEncoder:
    """Convenience factory. CLIP is usually better for vision-language tasks."""
    if prefer_clip:
        name = "openai/clip-vit-base-patch32"
    else:
        name = "bert-base-uncased"
    return TextEncoder(model_name=name, device=device, freeze=True)


if __name__ == "__main__":
    # quick smoke test
    enc = get_default_encoder(prefer_clip=False)
    emb = enc.encode(["a red rose", "yellow sunflower with green leaves"])
    print("Embedding shape:", emb.shape)
    print("Hidden size:", enc.hidden_size)
