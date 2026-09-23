# Internship Report  
## Real-Time Text-to-Image Generator – Gen AI (v2)

**Intern Name:** S Md Parvez  
**Email:** smdparvez@gmail.com  
**Programme:** Elevanceskills – Learn to Build Real Time Text To Image Generator – Gen AI (v2)  
**Duration:** September 2026  
**GitHub Repository:** https://github.com/S-MD-PARVEZ/Real-Time-Text-to-Image-Generator  
**Live Demo:** https://6f4307e70173ff8097.gradio.live  

---

## 1. Introduction

This report presents the work completed during the Elevanceskills internship programme focused on building a Real-Time Text-to-Image Generator. The project involved implementing six technical tasks as integrated features of a single coherent codebase. The goal was to develop a modular text-to-image system that combines custom generative models (Conditional GANs with attention) and domain adaptation of a pre-trained model (Stable Diffusion) on medical imagery.

---

## 2. Background

Text-to-image generation is a rapidly growing area of Generative AI. Modern systems convert natural language descriptions into realistic images using deep learning models. Two major approaches exist:

1. **Generative Adversarial Networks (GANs)** – Useful for understanding conditional generation and architectural improvements such as attention mechanisms.
2. **Diffusion Models** (e.g., Stable Diffusion) – Currently the state-of-the-art for high-quality image generation and domain adaptation.

This internship required practical implementation of both approaches, starting from dataset analysis and text preprocessing, progressing to Conditional GANs, attention mechanisms, full pipelines, and finally domain-specific fine-tuning of a pre-trained model.

---

## 3. Learning Objectives

By the end of this internship, the following learning objectives were achieved:

- Understand the structure and statistics of public image-caption datasets
- Preprocess text using Hugging Face Transformers and generate meaningful embeddings
- Implement Conditional GANs that generate images based on textual labels
- Improve generative models using self-attention and cross-attention
- Design and build a complete end-to-end text-to-image pipeline
- Adapt a pre-trained text-to-image model (Stable Diffusion) to a custom domain (medical imagery)
- Follow software engineering best practices including modularity, documentation, and version control

---

## System Architecture 

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

## 4. Activities and Tasks

All six required tasks were completed as additional features of the same project.

### Task 4: Dataset Exploration
- Loaded and analysed the Oxford-102 Flowers public dataset
- Examined number of classes (102), split sizes, image resolution, and class distribution
- Visualised sample images with their corresponding class labels
- Documented key observations about the dataset

### Task 3: Text Preprocessing using Hugging Face Transformers
- Built a reusable TextEncoder class using BERT
- Implemented tokenisation, padding, truncation, and [CLS] embedding extraction
- Generated 768-dimensional embeddings for text prompts
- Verified tokenisation details and embedding shapes

### Task 2: Conditional GAN for Basic Shapes
- Implemented a Conditional GAN (Generator + Discriminator)
- Conditioned the model on discrete labels: circle, square, triangle, star, hexagon
- Created a procedural shape dataset and trained the model
- Successfully generated clear class-conditional shapes

### Task 5: Attention-Enhanced GAN
- Implemented Self-Attention and Cross-Attention modules
- Integrated attention blocks into the Generator and Discriminator
- Trained the improved model and observed better structural consistency

### Task 6: Full Text-to-Image Pipeline
- Combined text preprocessing, embedding creation, and image generation into a single pipeline
- Created a Text2ImagePipeline class that accepts free-form text and produces images
- Demonstrated the complete flow: Text → Embedding → Generator → Image

### Task 1: Domain Adaptation of Pre-trained Model
- Prepared a custom medical imagery dataset (chest_xray, brain_scan, bone_xray, ultrasound, cell_microscopy)
- Loaded pre-trained Stable Diffusion (`runwayml/stable-diffusion-v1-5`)
- Generated baseline images using medical prompts
- Documented the complete LoRA fine-tuning workflow for domain adaptation

---

## 5. Skills and Competencies Developed

During this internship, the following technical and professional skills were developed:

**Technical Skills:**
- Deep Learning with PyTorch
- Generative Adversarial Networks (GANs)
- Attention Mechanisms (Self-Attention & Cross-Attention)
- Hugging Face Transformers and Diffusers
- Text embedding generation
- Stable Diffusion and domain adaptation concepts
- Data preprocessing and visualisation
- Building end-to-end ML pipelines
- Gradio for creating interactive demos

**Professional Skills:**
- Modular and clean code organisation
- Documentation and technical report writing
- Version control using GitHub
- Problem-solving and debugging
- Consistent daily progress reporting

---

## 6. Feedback and Evidence

**Evidence of work completed:**

- All six task notebooks are available in the GitHub repository
- Generated sample images (shapes, attention results, pipeline outputs, and Stable Diffusion baselines) are stored in the `outputs/` directory
- Live interactive demo is available at: https://6f4307e70173ff8097.gradio.live
- Daily progress updates were regularly logged on the Elevanceskills dashboard
- Complete project report and README are present in the repository

**Feedback Approach:**
- Continuous self-evaluation after each task
- Comparison of results before and after improvements (e.g., basic CGAN vs Attention-GAN)
- Verification of outputs through visual inspection and shape clarity

---

## 7. Challenges and Solutions

| Challenge | Solution |
|---------|----------|
| Generator producing pure noise instead of shapes | Increased training epochs and improved the quality of the procedural dataset |
| BatchNorm error when batch size = 1 | Switched the model to evaluation mode (`.eval()`) during inference |
| Dependency conflicts in Google Colab | Performed clean reinstallation of transformers, diffusers, and related packages |
| Need for domain-specific data | Created and organised a structured medical imagery dataset |
| Limited GPU resources | Designed notebooks to demonstrate core concepts efficiently within available compute |

---

## 8. Outcomes and Impact

**Key Outcomes:**
- Successfully implemented all six internship tasks in a single coherent project
- Developed a working Conditional GAN capable of generating distinct shapes from labels
- Improved generation quality using attention mechanisms
- Built a complete text-to-image pipeline integrating text encoding and image generation
- Demonstrated domain adaptation of Stable Diffusion on medical imagery
- Created a public live demo using Gradio

**Impact:**
- Gained practical, hands-on experience in modern Generative AI techniques
- Developed a strong understanding of both GAN-based and diffusion-based text-to-image systems
- Built a portfolio-ready project that demonstrates end-to-end ML engineering skills
- Improved ability to debug, document, and present technical work professionally

---

## 9. Conclusion

This internship provided comprehensive practical experience in building text-to-image generation systems. Starting from dataset analysis and text preprocessing, the project progressed through Conditional GANs, attention mechanisms, full pipeline design, and finally domain adaptation of a powerful pre-trained model (Stable Diffusion).

All required tasks were completed as integrated features of one project, following best practices of modularity, code quality, and documentation. The final system, along with the live demo and detailed report, successfully meets the evaluation criteria of the Elevanceskills internship programme.

The knowledge and skills gained during this internship form a strong foundation for further work in Generative AI and real-world machine learning applications.

---

**Declaration**  
I hereby declare that this project was completed by me as part of the Elevanceskills internship programme. All the work presented in this report is original and was carried out during the internship period.

**S Md Parvez**  
September 2026
