"""
Task 3 – Hugging Face Transformers text preprocessing
Tokenise and encode text descriptions into embeddings that the
text-to-image model will consume.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from src.text_encoder import TextEncoder, get_default_encoder


def demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. BERT-based encoder
    print("\n--- BERT encoder ---")
    bert_enc = TextEncoder(
        model_name="bert-base-uncased",
        max_length=32,
        pooling="cls",
        device=device,
        freeze=True,
    )
    texts = [
        "a bright red rose",
        "yellow sunflower in a field",
        "purple orchid with green leaves",
        "square",
        "circle",
    ]
    emb = bert_enc.encode(texts)
    print(f"Input texts : {texts}")
    print(f"Embedding shape : {emb.shape}")  # (5, 768)
    print(f"Hidden size     : {bert_enc.hidden_size}")

    # 2. CLIP text encoder (better for vision-language)
    print("\n--- CLIP text encoder ---")
    try:
        clip_enc = get_default_encoder(prefer_clip=True, device=device)
        emb_clip = clip_enc.encode(texts)
        print(f"CLIP embedding shape : {emb_clip.shape}")
        print(f"CLIP hidden size     : {clip_enc.hidden_size}")
    except Exception as e:
        print(f"CLIP load skipped (network or missing): {e}")
        print("BERT embeddings are sufficient for the rest of the pipeline.")

    # 3. Show tokenisation details
    print("\n--- Tokenisation example ---")
    tok = bert_enc.tokenizer(
        texts[0],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    print(f"Text          : {texts[0]}")
    print(f"Input IDs     : {tok['input_ids'].tolist()}")
    print(f"Attention mask: {tok['attention_mask'].tolist()}")
    print(f"Tokens        : {bert_enc.tokenizer.convert_ids_to_tokens(tok['input_ids'][0])}")

    print("\nTask 3 complete. Embeddings are ready to condition the Generator.")


if __name__ == "__main__":
    demo()
