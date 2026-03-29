# 🔬 HAM10000 — Skin Lesion Classification with MobileNetV2

A deep learning project for multi-class classification of dermoscopic skin lesion images using transfer learning with MobileNetV2. Built with TensorFlow/Keras on the HAM10000 dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Preprocessing & Balancing](#data-preprocessing--balancing)
  - [Augmentation](#augmentation)
  - [Model Architecture](#model-architecture)
  - [Training Configuration](#training-configuration)
- [Results](#results)
- [Visualizations](#visualizations)
- [Inference](#inference)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [Disclaimer](#disclaimer)

---

## Overview

Skin cancer is among the most common cancers globally, and early detection is critical for successful treatment. This project aims to classify dermoscopic images into **7 diagnostic categories** using a convolutional neural network based on **MobileNetV2** with transfer learning from ImageNet.

| Code | Diagnosis | Type |
|------|-----------|------|
| **MEL** | Melanoma | Malignant |
| **NV** | Melanocytic nevi | Benign |
| **BCC** | Basal cell carcinoma | Malignant |
| **AKIEC** | Actinic keratoses | Pre-malignant |
| **BKL** | Benign keratosis | Benign |
| **DF** | Dermatofibroma | Benign |
| **VASC** | Vascular lesions | Benign |

---

## Dataset

**HAM10000** (*Human Against Machine with 10,000 training images*) — a large-scale benchmark dataset for dermatoscopic image analysis.

- **Source:** [ISIC Archive](https://www.isic-archive.com/) via [Kaggle](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
- **Total images:** 10,015
- **Image resolution:** Variable (~600×450), resized to 224×224
- **Labels:** One-hot encoded across 7 classes in `GroundTruth.csv`

### Class Distribution (Original)

| Class | Count | Share |
|-------|-------|-------|
| NV | 6,705 | 66.9% |
| MEL | 1,113 | 11.1% |
| BKL | 1,099 | 11.0% |
| BCC | 514 | 5.1% |
| AKIEC | 327 | 3.3% |
| VASC | 142 | 1.4% |
| DF | 115 | 1.1% |

**Imbalance ratio:** 58.3× (NV vs DF) — severe class imbalance that required dedicated handling.

---

## Methodology

### Data Preprocessing & Balancing

The severe class imbalance was addressed with a combined strategy:

1. **Downsampling** the dominant class: NV reduced from 6,705 → 800 (random sample)
2. **Oversampling** rare classes with replacement: DF, VASC, AKIEC, BCC increased to 800
3. **Keeping** medium-sized classes as-is: MEL (1,113), BKL (1,099)

**Balanced dataset:** 6,212 samples

**Data split** (stratified):

| Split | Samples | Share |
|-------|---------|-------|
| Train | 4,348 | 70% |
| Validation | 932 | 15% |
| Test | 932 | 15% |

Additionally, **class weights** (inversely proportional to class frequency) were applied during training to further mitigate residual imbalance.

### Augmentation

Real-time augmentation was applied to training data only via `ImageDataGenerator`:

| Augmentation | Parameter |
|-------------|-----------|
| Rotation | ±30° |
| Width/Height shift | ±10% |
| Shear | 0.1 |
| Zoom | ±10% |
| Horizontal flip | Yes |
| Vertical flip | Yes |
| Brightness | [0.8, 1.2] |
| Fill mode | Nearest |

### Model Architecture

**Transfer learning** with MobileNetV2 pretrained on ImageNet:

```
Input (224×224×3)
    │
    ▼
MobileNetV2 (all layers trainable, GlobalAveragePooling)
    │
    ▼
BatchNormalization (momentum=0.99, epsilon=0.001)
    │
    ▼
Dense (256 units, ReLU activation)
    │
    ▼
Dropout (rate=0.45)
    │
    ▼
Dense (7 units, Softmax activation)
    │
    ▼
Output: 7-class probability distribution
```

**Total parameters:** 2,592,839

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adamax |
| Initial learning rate | 0.001 |
| LR schedule | Custom LRA callback (factor=0.5 on plateau) |
| Loss function | Categorical cross-entropy |
| Class weights | Yes (inversely proportional) |
| Batch size | 40 |
| Max epochs | 40 |
| Patience (LR reduction) | 2 epochs |
| Patience (early stopping) | 5 epochs |
| Preprocessing | `mobilenet_v2.preprocess_input` (scales pixels to [-1, 1]) |

The **LRA (Learning Rate Adjustment)** callback monitors both training and validation metrics, reduces the learning rate when progress stalls, and implements early stopping.

---

## Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **79.08%** |
| **F1-macro** | **0.8053** |
| **F1-weighted** | **0.7888** |

### Key Findings

- The model performs best on visually distinctive classes (VASC, NV)
- Hardest classes are those sharing visual features: AKIEC vs BCC, MEL vs BKL
- Confusion patterns are clinically plausible — even dermatologists find these distinctions challenging
- Grad-CAM confirms the model focuses on lesion regions, not on background artifacts or skin texture

---

## Visualizations

The notebook produces the following visualizations:

| Visualization | Description |
|--------------|-------------|
| Class distribution | Bar chart + pie chart of original dataset |
| Sample images | One example per class |
| Augmented samples | Examples of augmented training images |
| Training curves | Loss and accuracy over epochs (train vs validation) |
| Confusion matrix | Absolute values + normalized (per-class recall) |
| Per-class metrics | Grouped bar chart: Precision, Recall, F1 per class |
| Grad-CAM heatmaps | Attention maps for each class showing what the model "sees" |
| Error analysis | 12 most confident misclassifications |

---

## Inference

The notebook includes a `predict_skin_lesion()` function for single-image inference:

```python
predict_skin_lesion("path/to/image.jpg", top_k=3)
```

**Output:**
- Top-K predicted classes with confidence scores
- Original image alongside a bar chart of class probabilities
- Grad-CAM overlay showing the attention region

---

## Requirements

```
tensorflow >= 2.19
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
kagglehub
```

### Installation

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow kagglehub
```

---

## How to Run

1. **Clone / download** this repository
2. **Open** `HAM10000_Classification_v3.ipynb` in Jupyter Notebook or Google Colab
3. **Run all cells** sequentially — the dataset will be automatically downloaded via `kagglehub`
4. The trained model will be saved as `ham10000_mobilenetv2.keras` and `ham10000_mobilenetv2.h5`
5. Use the inference cell at the end to classify new images

> **Note:** Training was performed on CPU. For faster training, use a GPU runtime (Colab GPU or local CUDA-enabled GPU).

---

## Future Work

- [ ] Compare with other architectures (EfficientNetB0, ResNet50)
- [ ] Implement focal loss for hard example mining
- [ ] Add advanced augmentation techniques (Cutout, MixUp, CutMix)
- [ ] Compute per-class ROC curves and AUC scores
- [ ] Build an interactive demo with Gradio or Streamlit
- [ ] Explore SMOTE-style augmentation in feature space
- [ ] Optimize for mobile deployment (TFLite conversion)

---

## Disclaimer

⚠️ **This project is for educational and research purposes only.** The model is **NOT** a medical diagnostic tool. Always consult a qualified dermatologist for skin lesion evaluation. Do not use this model to make clinical decisions.

---

## Author

**Artem** — Applied Mathematics, NaUKMA
**Danylo** — Applied Mathematics, NaUKMA

Machine Learning course project, 2026
