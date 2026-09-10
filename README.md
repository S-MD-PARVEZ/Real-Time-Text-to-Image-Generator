# Real-Time Text-to-Image Generator – Gen AI (v2)

**Internship Project – Elevanceskills**  
**Student:** S Md Parvez (`smdparvez@gmail.com`)

This repository extends the original training project into a complete **text-to-image generation system**. All six internship tasks are implemented as additional features / modules on the **same codebase**.

---

## Project Overview

We build a modular pipeline that:

1. Loads & analyses a public image-caption dataset
2. Pre-processes free-form text with Hugging Face Transformers
3. Trains a Conditional GAN (CGAN) on simple shapes using discrete labels
4. Improves the GAN with self-attention / cross-attention
5. Assembles a full end-to-end text → embedding → image pipeline
6. Fine-tunes a pre-trained Stable Diffusion model (LoRA) on a domain-specific / custom dataset

The result is a working system that can generate images from text prompts, with both a lightweight custom GAN path and a higher-quality diffusion path.

---

## Task Mapping (Internship Requirements)

| # | Internship Task | Implementation |
|---|-----------------|----------------|
| 1 | Refine a pre-trained text-to-image model (Stable Diffusion / DALL-E style) on a custom/domain dataset | `notebooks/06_finetune_stable_diffusion_lora.ipynb` + `src/finetune_sd.py` |
| 2 | Create a CGAN conditioned on textual labels (“square”, “circle”, …) | `notebooks/03_simple_cgan_shapes.ipynb` + `src/models/cgan.py` |
| 3 | Hugging Face Transformers text preprocessing → tokenized & encoded embeddings | `src/text_encoder.py` + `notebooks/02_text_preprocessing_hf.ipynb` |
| 4 | Load & examine public dataset (Oxford-102 Flowers / COCO), statistics, image+text visualisation | `notebooks/01_dataset_exploration.ipynb` |
| 5 | Improve GAN with self-attention / cross-attention | `src/models/attn_gan.py` + `notebooks/04_attention_gan.ipynb` |
| 6 | Comprehensive pipeline (text preprocess + embedding + GAN generation) | `src/pipeline.py` + `notebooks/05_full_pipeline.ipynb` + Gradio demo |

All tasks live inside **one** project. No unrelated datasets or separate repositories.

---

## Quick Start

```bash
git clone <your-repo-url>
cd text2image-gan-pipeline
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Recommended order to run notebooks

1. `01_dataset_exploration.ipynb` – understand the data
2. `02_text_preprocessing_hf.ipynb` – text → embeddings
3. `03_simple_cgan_shapes.ipynb` – basic conditional generation
4. `04_attention_gan.ipynb` – attention-enhanced generator
5. `05_full_pipeline.ipynb` – end-to-end inference
6. `06_finetune_stable_diffusion_lora.ipynb` – domain adaptation (optional, heavier)

### Live Demo (Gradio)

```bash
python scripts/app.py
```

A public Gradio link can be generated with `share=True` for the “live URL” submission requirement.

---

## Dataset

- **Primary public dataset:** Oxford-102 Flowers (via `torchvision` or Hugging Face `datasets`)
- **Alternative / captions:** COCO captions subset
- **Custom / domain set** (for Task 1): a small collection of domain images + captions (artwork style, medical sketches, etc.). Place under `data/custom/`.

Scripts to download and prepare data are in `scripts/prepare_data.py`.

---

## Architecture Summary

```
Text Prompt
    │
    ▼
Hugging Face Tokenizer + Encoder (BERT / CLIP)
    │
    ▼
Text Embedding  ──────────────────────────────┐
    │                                         │
    ▼                                         │
Noise z  →  Generator (CGAN / AttnGAN)  ←─────┘
    │
    ▼
Generated Image
```

For higher quality the same text embedding can condition a fine-tuned Stable Diffusion UNet (LoRA).

---

## Repository Structure

```
text2image-gan-pipeline/
├── README.md
├── requirements.txt
├── data/                    # datasets & processed files
├── notebooks/               # one notebook per major task
├── src/
│   ├── text_encoder.py      # Task 3
│   ├── dataset.py
│   ├── pipeline.py          # Task 6
│   ├── finetune_sd.py       # Task 1
│   └── models/
│       ├── cgan.py          # Task 2
│       └── attn_gan.py      # Task 5
├── scripts/
│   ├── prepare_data.py
│   └── app.py               # Gradio live demo
└── outputs/                 # samples, checkpoints, logs
```

---

## Submission Checklist

- [x] All 6 tasks implemented as features of the **same** project
- [ ] GitHub repository (public or private with access for evaluators)
- [ ] Live URL (Gradio share link or deployed demo)
- [ ] Project report (short PDF/Markdown describing architecture, experiments, results)
- [ ] Daily work reports logged on the Elevanceskills dashboard

---

## Daily Progress Log (example for dashboard)

> Today I set up the complete project skeleton, mapped all 6 internship tasks to modules/notebooks, implemented the Hugging Face text encoder, the simple CGAN for shapes, and the dataset exploration notebook. Next: attention layers + full pipeline + SD LoRA fine-tuning.

---

## Licence & Acknowledgements

Educational project for Elevanceskills internship.  
Built with PyTorch, Hugging Face Transformers & Diffusers, Gradio.
