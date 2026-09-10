# Project Report – Real-Time Text-to-Image Generator (Gen AI v2)

**Intern:** S Md Parvez (`smdparvez@gmail.com`)  
**Programme:** Elevanceskills – Learn to Build Real Time Text To Image Generator – Gen AI (v2)  
**Date:** 10 September 2026  

---

## 1. Objective

Build on the completed training project and implement all six internship tasks as **additional features / modules of the same codebase**. The result is a modular, reproducible text-to-image system that covers:

- Public dataset analysis
- Hugging Face text preprocessing
- Conditional GAN for discrete labels
- Attention-enhanced generation
- End-to-end pipeline
- Domain adaptation of a pre-trained diffusion model (Stable Diffusion + LoRA)

All work follows the portal rules: single project, code quality, originality, and full integration for the ElevanceSkills Rubric.

---

## 2. Task Mapping (Evaluation Alignment)

| # | Internship Requirement | Implementation | Status |
|---|------------------------|----------------|--------|
| 4 | Load & examine public dataset (Oxford-102 / COCO), statistics, image+text visualisation | `notebooks/01_dataset_exploration.py` | Done |
| 3 | HF Transformers → tokenised & encoded text embeddings | `src/text_encoder.py`, `notebooks/02_text_preprocessing_hf.py` | Done |
| 2 | CGAN conditioned on textual labels (“square”, “circle”, …) | `src/models/cgan.py`, `notebooks/03_simple_cgan_shapes.py` | Done + trained |
| 5 | Self-attention / cross-attention to improve GAN | `src/models/attn_gan.py`, `notebooks/04_attention_gan.py` | Done |
| 6 | Comprehensive pipeline (preprocess + embedding + GAN) | `src/pipeline.py`, `notebooks/05_full_pipeline.py`, Gradio app | Done |
| 1 | Refine pre-trained text-to-image model (SD / DALL-E style) on custom/domain data | `notebooks/06_finetune_stable_diffusion_lora.py` | Done (pipeline + recipe) |

---

## 3. Architecture

```
┌─────────────────┐
│  Text Prompt    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ HF Tokenizer +  │  (BERT or CLIP)
│ Text Encoder    │
└────────┬────────┘
         │ embedding (B, H)
         ▼
┌─────────────────┐     ┌──────────┐
│ Noise z         │────▶│ Generator│ (CGAN or Attn-GAN)
└─────────────────┘     │ + Attn   │
                        └────┬─────┘
                             │
                             ▼
                        Generated Image
```

- **Lightweight path:** Custom CGAN / Attn-GAN (64×64, educational, fully trainable on CPU/GPU).
- **High-quality path:** Same text conditioning applied to Stable Diffusion UNet via LoRA adapters.

---

## 4. Experiments & Results

### 4.1 Dataset (Task 4)
- Primary public dataset: **Oxford-102 Flowers**.
- Statistics (standard): 102 classes, ~2 040 train / 1 020 val / 6 149 test images.
- Images resized & centre-cropped to 128×128 for consistency.
- Class names used as weak textual labels; richer captions can be added for free-text experiments.
- Visualisation of image–label pairs saved under `outputs/`.

### 4.2 Text Preprocessing (Task 3)
- Implemented reusable `TextEncoder` supporting BERT and CLIP.
- Output: fixed-size embeddings `(B, 768)` (or CLIP dimension) ready for the Generator.
- Tokenisation details (input_ids, attention_mask, special tokens) demonstrated in notebook 02.

### 4.3 Conditional GAN – Shapes (Task 2)
- Labels: `circle`, `square`, `triangle`, `star`, `hexagon`.
- Procedural dataset generated on-the-fly (clean shapes + light noise).
- Trained 10 epochs on CPU (Adam, lr=2e-4, batch=64).
- Checkpoints: `outputs/cgan/generator.pt`, `discriminator.pt`.
- Sample grids: `outputs/cgan/samples_epoch005.png`, `samples_epoch010.png`.
- Final per-class samples: `outputs/cgan/final_samples/{circle,square,...}_*.png`.

After training the generator clearly responds to the discrete label (different shapes appear for different class indices).

### 4.4 Attention Improvements (Task 5)
- Added **SelfAttention** (spatial long-range) and **CrossAttention** (spatial features attend to text embedding).
- Generator and Discriminator both incorporate attention blocks.
- Training script conditions on continuous text embeddings produced by the HF encoder.
- Architecture is modular so attention can be ablated.

### 4.5 Full Pipeline (Task 6)
- `Text2ImagePipeline`: prompt → HF encode → noise + embedding → AttnGenerator → PIL image.
- `ShapeCGANPipeline`: discrete label name → CGAN → images.
- Gradio demo (`scripts/app.py`) provides a browser UI and can generate a public share link (live URL for submission).

### 4.6 Stable Diffusion LoRA (Task 1)
- Notebook demonstrates the standard open-source workflow:
  - Load `runwayml/stable-diffusion-v1-5` (or similar).
  - Prepare a small custom/domain dataset (`data/custom/` – artwork, medical illustrations, etc.).
  - Inject LoRA adapters (PEFT / Diffusers).
  - Train only the low-rank matrices.
- Full quality training requires a GPU; the notebook is structured for easy transfer to Colab/Kaggle with increased steps.
- Baseline generations before fine-tuning are saved for comparison.

---

## 5. Code Quality & Best Practices

- Modular package layout (`src/`, `src/models/`).
- Clear docstrings and type hints.
- Reproducible training scripts with fixed seeds where useful.
- Separation of concerns: data, text encoding, models, pipeline, demo.
- Requirements pinned in `requirements.txt`.
- README maps every internship bullet to concrete files.
- No unrelated datasets or brand-new projects; everything extends the same text-to-image codebase.

---

## 6. How to Reproduce

```bash
git clone <your-repo-url>
cd text2image-gan-pipeline
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Core flow
python notebooks/01_dataset_exploration.py
python notebooks/02_text_preprocessing_hf.py
python notebooks/03_simple_cgan_shapes.py      # trains CGAN, produces samples
python notebooks/04_attention_gan.py
python notebooks/05_full_pipeline.py
python notebooks/06_finetune_stable_diffusion_lora.py

# Live demo (for submission live URL)
python scripts/app.py          # set share=True inside for public link
```

---

## 7. Submission Artefacts

- **GitHub repository** – this codebase (public or shared with evaluators).
- **Live URL** – Gradio share link from `scripts/app.py`.
- **Report** – this document (`REPORT.md` or exported PDF).
- **Daily updates** – logged on the Elevanceskills dashboard.

---

## 8. Conclusion

All six internship tasks have been implemented as integrated features of a single, well-structured text-to-image project. The system demonstrates:

- Conditional generation from discrete labels (CGAN),
- Continuous text embeddings via Hugging Face,
- Attention mechanisms for better conditioning,
- A ready-to-extend path for high-quality domain-specific generation with Stable Diffusion + LoRA.

The project is ready for GitHub push, live demo, and final evaluation against the ElevanceSkills Rubric.
