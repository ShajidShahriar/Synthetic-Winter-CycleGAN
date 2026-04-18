![Sample Result](/image.png)

#  Synthetic Winter: CycleGAN for Summer-to-Winter Translation

This project explores the use of CycleGAN for generating synthetic winter driving data from summer scenes, with the goal of improving robustness in autonomous driving systems under adverse weather conditions.

---

##  Problem Statement

Autonomous driving models suffer from **domain shift** when exposed to conditions not present in training data, particularly **snowy and winter environments**.

However:
- Winter driving datasets are **limited**
- Data collection is **expensive and risky**

 This project investigates whether **synthetic winter data** can be generated from summer images using **unpaired image-to-image translation (CycleGAN)**.

---

##  Approach

We use **CycleGAN**, which enables translation between domains without requiring paired data.

### Model Components:
- 2 Generators (Summer → Winter, Winter → Summer)
- 2 Discriminators
- Loss Functions:
  - Adversarial Loss
  - Cycle Consistency Loss
  - Identity Loss

---

## Dataset

Since no suitable dataset was found, data was **manually collected from YouTube driving videos**.

### Versions:

| Version | Summer | Winter | Notes |
|--------|--------|--------|------|
| V1 | ~700 | ~700 | Imbalanced (mostly scenic roads) |
| V2 | 414 | 406 | Curated, balanced dataset |

### Improvements in V2:
- Balanced scene distribution (urban, highway, residential)
- Inclusion of vehicles, buildings, traffic signs
- Removal of redundant frames
- Higher resolution (512×512)

---

## ⚙️ Training Setup

| Parameter | V1 | V2 |
|----------|----|----|
| Resolution | 256×256 | 512×512 |
| Dataset Size | ~700 | ~400 |
| Hardware | RTX 3060 | Google Colab (T4 GPU) |
| Epochs | 100 | 100 |
| Batch Size | 1 | 1 |

---

##  Results

### Quantitative Metrics

| Metric | V1 (256) | V2 (512) |
|--------|---------|----------|
| FID ↓ | 165.40 | **150.00** |
| SSIM ↑ | 0.4058 | **0.5118** |
| PSNR | 13.06 | 12.88 |

 V2 shows:
- **~9% improvement in FID**
- **~26% improvement in SSIM (structure preservation)**

---

##  Downstream Validation

We evaluate the usefulness of generated images using a pretrained **Faster R-CNN detector**.

| Metric | V1 | V2 |
|-------|----|----|
| Mean Count Difference ↓ | 2.08 | **1.75** |
| Retention Ratio ↑ | 0.21 | **0.39** |
| Confidence Drop ↓ | -0.28 | **-0.20** |

 V2 preserves objects (cars, signs, etc.) significantly better.

---

##  Qualitative Results

Example comparison:

| Summer Input | V1 Output | V2 Output |
|-------------|----------|----------|
| (see report / drive link) |

 V2 shows:
- better structure preservation
- less blur
- improved traffic sign retention

---

##  Full Results & Models

Due to size limitations, full outputs and trained models are hosted on Google Drive:

 **Drive Link (All Outputs & Models):**  
https://drive.google.com/drive/u/0/folders/1byolCKUKlCEtXKfnLoEDpXfkeWtv0-7V

---

##  Notebooks

- `Colab_Training.ipynb` → Training pipeline
- `Colab_Evaluate.ipynb` → Evaluation + metrics

---

##  Limitations

- Small dataset (~400 images per domain)
- Limited photorealism
- No semantic understanding (GAN learns patterns, not objects)
- Some artifacts in generated textures

---

##  Future Work

- Larger curated dataset
- Semantic-aware GANs
- Diffusion-based models
- Video consistency (temporal GANs)

---






