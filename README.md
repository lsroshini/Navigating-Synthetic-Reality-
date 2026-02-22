CIFAKE: Real vs AI-Generated Image Detector
A deep learning web application that detects whether an image is real or AI-generated, built using the CIFAKE dataset.
Models
Four CNN architectures trained and compared:

ResNet18 — pretrained transfer learning
AlexNet — pretrained transfer learning
CustomCNN — lightweight 3-block CNN built from scratch
EfficientNet-B0 — pretrained transfer learning

Features

Single model prediction with confidence scores
Side-by-side 4-model comparison dashboard
Grad-CAM visual explainability heatmaps
Performance metrics — Accuracy, Precision, Recall, F1, ROC-AUC
Confusion matrix visualization for all models

Tech Stack
Python PyTorch Streamlit torchvision OpenCV scikit-learn
