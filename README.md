# CIFAKE: Real vs AI-Generated Image Detector

**Live Demo:** [https://four-clg-hack.streamlit.app/](https://four-clg-hack.streamlit.app/)

Detect whether an image is **real** or **AI-generated** using four trained deep learning models, with Grad-CAM visual explanations and a full model comparison dashboard.

---

## Overview

With the rapid rise of AI-generated imagery, distinguishing real photos from synthetic ones has become increasingly important. This project trains and compares four CNN architectures on the **CIFAKE** dataset — a benchmark dataset of real images (from CIFAR-10) and AI-generated counterparts — and deploys them as an interactive web application.

---

## 🌐 Deployed App

👉 [Click here to try the live app](https://four-clg-hack.streamlit.app/)

**What you can do:**
- Upload any image and get an instant REAL / FAKE prediction
- Choose from 4 different models and compare their confidence
- View Grad-CAM heatmaps showing what regions the model focused on
- Explore the full comparison dashboard with metrics, ROC curves and confusion matrices

---

## Models

| Model | Architecture | Input Size | Normalization |
|-------|-------------|------------|---------------|
| ResNet18 | Pretrained + fine-tuned | 224×224 | ImageNet stats |
| AlexNet | Pretrained + fine-tuned | 224×224 | ImageNet stats |
| CustomCNN | Built from scratch | 64×64 | [0.5, 0.5, 0.5] |
| EfficientNet-B0 | Pretrained + fine-tuned | 224×224 | ImageNet stats |

---

## Features

- **Single Model Prediction** — Upload an image, pick a model, get REAL/FAKE prediction with confidence score
- **Grad-CAM Heatmaps** — Visual explanation of what the model focused on
- **4-Model Comparison** — Side-by-side live prediction from all 4 models with majority vote
- **Metrics Dashboard** — Accuracy, Precision, Recall, F1-Score, Specificity, ROC-AUC
- **Confusion Matrices** — Per-model confusion matrix visualization
- **ROC Curves** — Comparative ROC curves for all models

---

## Tech Stack

- **Framework:** PyTorch, torchvision
- **UI:** Streamlit
- **Explainability:** Grad-CAM
- **Evaluation:** scikit-learn
- **Image Processing:** OpenCV, Pillow
- **Model Hosting:** Google Drive + gdown

---

## Dataset

[CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

- 60,000 REAL images (from CIFAR-10)
- 60,000 FAKE images (AI-generated)
- 2 classes: `REAL` and `FAKE`

---
