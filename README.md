# 🔬 SkinAI — Skin Lesion Classification with Deep Learning Ensemble

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-HAM10000-green.svg)](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

An end-to-end deep learning pipeline for classifying 7 types of skin lesions from dermatoscopic images. The system combines **U-Net segmentation** with an **ensemble of 3 CNN classifiers** and is deployed as an interactive **Streamlit web application**.


---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training](#training)
- [Web Application](#web-application)
- [Future Work](#future-work)

---

## Overview

Skin cancer is one of the most common cancers worldwide. Early detection of melanoma and other malignant lesions dramatically improves survival rates. This project builds an AI-assisted screening tool that:

1. **Segments** the lesion from the background using a U-Net model
2. **Classifies** the lesion into one of 7 categories using 3 diverse CNN models
3. **Combines** predictions via an optimized weighted average ensemble
4. **Deploys** the full pipeline as an interactive web app with risk assessment

---

## Dataset

**HAM10000** (*Human Against Machine with 10000 training images*) — a large collection of dermatoscopic images from different populations, acquired and stored by different modalities.

| Class | Full Name | Samples | Type |
|-------|-----------|---------|------|
| NV | Melanocytic Nevi | 6,705 | Benign |
| MEL | Melanoma | 1,113 | **Malignant** |
| BKL | Benign Keratosis | 1,099 | Benign |
| BCC | Basal Cell Carcinoma | 514 | **Malignant** |
| AKIEC | Actinic Keratoses | 327 | Precancerous |
| VASC | Vascular Lesions | 142 | Benign |
| DF | Dermatofibroma | 115 | Benign |

**Total: 10,015 images** with ground truth confirmed by histopathology, expert consensus, or follow-up examination.

### Handling Class Imbalance

- NV (majority class) capped at 1,500 samples during training
- **Focal Loss** (γ=2, label smoothing=0.1) to focus on hard/rare examples
- **GroupShuffleSplit** by `lesion_id` to prevent data leakage (same lesion never in both train and test)
- Split: 70% train / 15% validation / 15% test

---

## Architecture

The pipeline consists of two stages: segmentation and classification.

```
                     ┌───────────────────────────┐
                     │    Input Image (RGB)      │
                     └──────┬─────────┬──────────┘
                            │         │
                            │  ┌──────▼────────────────────┐
                            │  │   U-Net Segmentation      │
                            │  │   (EfficientNetV2B2)      │
                            │  │   Dice: 0.943 | IoU: 0.897│
                            │  └──────┬──────────┬─────────┘
                            │         │          │
               ┌────────────▼────┐  ┌───▼─────────┐ │ ┌────────────────┐
               │  Model 1        │  │  Model 2    │ └►│  Model 3       │
               │ EfficientNetV2B2│  │ ConvNeXtTiny│   │ DenseNet121    │
               │ Raw RGB         │  │ Segmented   │   │ 4ch RGB+Mask   │
               │ (260×260)       │  │ (224×224)   │   │ (224×224×4)    │
               │ F1: 0.699       │  │ F1: 0.681   │   │ F1: 0.635      │
               └───────┬─────────┘  └─────┬───────┘   └───────┬────────┘
                       │                  │                   │
                       └──────────────────┼───────────────────┘
                                          │
                          ┌───────────────▼────────────┐
                          │  Weighted Average Ensemble │
                          │  F1 Macro: 0.749           │
                          │  Accuracy: 84.0%           │
                          │  AUC: 0.965                │
                          └────────────────────────────┘
```

### Segmentation: U-Net

- **Encoder:** EfficientNetV2B2 (ImageNet pretrained)
- **Decoder:** Transposed convolutions with skip connections
- **Loss:** BCE + Dice combined
- **Training:** Two-stage (freeze encoder → fine-tune all)
- **Augmentations:** Synced transforms on image + mask (flip, rotate, elastic, brightness)

### Model 1: EfficientNetV2B2

- **Input:** 260×260 raw RGB
- **Augmentations:** Flip, rotate, brightness, contrast, HSV shift, CoarseDropout
- **Training:** Two-stage + CosineDecay LR schedule
- **Inference:** Test-Time Augmentation (TTA, 8 views)

### Model 2: ConvNeXtTiny + Segmentation

- **Input:** 224×224 segmented image (background removed via U-Net mask)
- **Preprocessing:** U-Net mask → Gaussian blur → multiply with RGB
- **Idea:** Force the model to focus exclusively on the lesion

### Model 3: DenseNet121 (4-Channel)

- **Input:** 224×224×4 (R, G, B, Segmentation Mask)
- **Preprocessing:** Modified first conv layer to accept 4 channels
- **Idea:** Let the model learn how to use the mask information itself

### Ensemble: Weighted Average

Optimal weights found via Nelder-Mead optimization on the validation set:

```
Final prediction = w₁·Model1 + w₂·Model2 + w₃·Model3
```

Compared approaches: Logistic Regression, XGBoost, Neural Network, Simple Average, Weighted Average. **Weighted Average won** with the best F1 Macro score.

---

## Results

### Individual Models vs Ensemble

| Method | Accuracy | F1 Macro | AUC Macro |
|--------|----------|----------|-----------|
| Model 1 — EfficientNetV2B2 | 82.2% | 0.699 | 0.953 |
| Model 2 — ConvNeXtTiny + Seg | 80.8% | 0.681 | 0.948 |
| Model 3 — DenseNet121 4ch | 79.4% | 0.635 | 0.943 |
| Meta: Logistic Regression | 83.6% | 0.709 | 0.959 |
| Meta: XGBoost | 81.9% | 0.703 | 0.959 |
| Meta: Neural Network | 82.6% | 0.724 | 0.957 |
| Simple Average | 83.9% | 0.748 | 0.965 |
| **Weighted Average** | **84.0%** | **0.749** | **0.965** |

### Per-Class Performance (Weighted Average Ensemble)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| AKIEC | 0.63 | 0.50 | 0.56 | 48 |
| BCC | 0.66 | 0.89 | 0.76 | 66 |
| BKL | 0.69 | 0.70 | 0.70 | 172 |
| DF | 0.89 | 0.80 | 0.84 | 10 |
| MEL | 0.70 | 0.54 | 0.61 | 186 |
| NV | 0.90 | 0.93 | 0.92 | 1,016 |
| VASC | 0.92 | 0.79 | 0.85 | 29 |

### Segmentation Performance

| Metric | Score |
|--------|-------|
| Dice Coefficient | 0.943 ± 0.064 |
| IoU (Jaccard) | 0.897 ± 0.095 |

---

## Installation

### Requirements

- Python 3.10+
- TensorFlow 2.x
- Streamlit

```bash
git clone https://github.com/KonchakivskyiDanylo/HAM10000.git
cd HAM10000
pip install -r requirements.txt
```

### Dependencies

```
tensorflow>=2.15
streamlit>=1.30
numpy
pandas
opencv-python-headless
Pillow
scikit-learn
albumentations
matplotlib
seaborn
```

---

## Usage

### Web Application

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Upload a dermatoscopic image and get:
- Segmentation overlay visualization
- Classification with risk level (benign / precancerous / malignant)
- Per-class probability distribution
- Per-model agreement check

### Quick Inference (Python)

```python
import numpy as np
import cv2
import tensorflow as tf

# Load models
seg_model = tf.keras.models.load_model("models/segmentation_unet.keras", custom_objects={...})
clf_v4 = tf.keras.models.load_model("models/v4_efficientnet.keras", custom_objects={...})
clf_v5 = tf.keras.models.load_model("models/v5_convnext_seg.keras", custom_objects={...})
clf_v6 = tf.keras.models.load_model("models/v6_densenet_4ch.keras", custom_objects={...})

# Predict
img = cv2.cvtColor(cv2.imread("lesion.jpg"), cv2.COLOR_BGR2RGB)
mask = predict_mask(seg_model, img)
p1 = predict_v4(clf_v4, img)
p2 = predict_v5(clf_v5, img, mask)
p3 = predict_v6(clf_v6, img, mask)

# Ensemble
ensemble = 0.4 * p1 + 0.3 * p2 + 0.3 * p3  # approximate weights
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
print(f"Prediction: {classes[np.argmax(ensemble)]}, confidence: {np.max(ensemble):.2%}")
```

---

## Project Structure

```
SkinAI/
├── app.py                          # Streamlit web application
├── models/                         # Trained model weights
│   ├── segmentation_unet.keras     # U-Net segmentation model
│   ├── v4_efficientnet.keras       # Model 1: EfficientNetV2B2
│   ├── v5_convnext_seg.keras       # Model 2: ConvNeXtTiny
│   └── v6_densenet_4ch.keras       # Model 3: DenseNet121 4-channel
├── notebooks/
│   ├── model1.ipynb                # EfficientNetV2B2 training
│   ├── model1_prep.ipynb           # Model 1 prediction extraction
│   ├── model2.ipynb                # ConvNeXtTiny + segmentation training
│   ├── model3.ipynb                # DenseNet121 4-channel training
│   ├── segmentation.ipynb          # U-Net segmentation training
│   └── ensemble.ipynb              # Meta-learner comparison & final ensemble
├── requirements.txt
└── README.md
```

---

## Training

All models were trained in **Google Colab** (T4 / A100 GPU). Training notebooks are in the `notebooks/` directory.

### Reproduction Steps

1. **Segmentation:** Run `segmentation.ipynb` to train U-Net and save the model
2. **Model 1:** Run `model1.ipynb` — trains EfficientNetV2B2 on raw RGB images
3. **Model 2:** Run `model2.ipynb` — uses U-Net masks to segment images before classification
4. **Model 3:** Run `model3.ipynb` — trains DenseNet121 with 4-channel (RGB + mask) input
5. **Ensemble:** Run `ensemble.ipynb` — loads predictions from all 3 models, compares meta-learners, selects best

### Key Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss | Focal Loss (γ=2, label smoothing=0.1) |
| LR Schedule | CosineDecay |
| Training Strategy | Two-stage (freeze → fine-tune) |
| Augmentation Library | Albumentations |
| Early Stopping | Patience 7 (monitor: val_accuracy) |
| TTA | 8 augmented views at inference |

---

## Web Application

The Streamlit app (`app.py`) provides a complete diagnostic interface:

| Feature | Description |
|---------|-------------|
| Image Upload | JPG, PNG, BMP support |
| Segmentation | Real-time U-Net overlay with lesion boundary |
| Classification | 3-model ensemble with per-class probabilities |
| Risk Assessment | Color-coded: 🟢 Benign, ⚠️ Precancerous, 🔴 Malignant |
| Model Agreement | Shows if all 3 models agree on the diagnosis |
| Bilingual | Ukrainian + English disease names |

> ⚠️ **Disclaimer:** This tool is for educational purposes only and is NOT a medical diagnosis. Always consult a dermatologist.

---

## Future Work

- Fine-tune each model with extended training and learning rate search
- Add more architectures to the ensemble (ViT, Swin Transformer)
- Improve segmentation with higher resolution and attention mechanisms
- Implement CutMix / MixUp augmentation strategies
- Use external data sources for rare classes (DF, VASC, AKIEC)
- Improve melanoma recall (currently 0.54) with class-specific loss weighting
- Deploy as a mobile-friendly PWA for field use by dermatologists
- Add input validation to reject non-dermatoscopic images (e.g., photos of everyday objects), 
  as the current pipeline will produce a confident but meaningless diagnosis for any image

---

## Acknowledgments

- [HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) by Philipp Tschandl et al.
- [HAM10000 Segmentation Masks](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
- TensorFlow / Keras for model development
- Streamlit for web application framework

---

## License

This project is for educational and research purposes. The HAM10000 dataset has its own [license terms](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
