# 🏍️ Automated Helmet Compliance & License Plate Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end computer vision pipeline and web application designed to automatically detect two-wheeler riders, verify helmet compliance, and localize license plates in dense, real-world traffic surveillance imagery.

---

## 📌 Problem Overview
Road traffic accidents involving two-wheeler riders remain a leading cause of severe preventable injuries. Manual enforcement of helmet regulations across dense intersections and highways is labor-intensive and difficult to scale. 

This project provides an automated deep-learning solution capable of:
1. Identifying motorcycle riders in complex traffic scenes.
2. Accurately determining helmet compliance (`with helmet` vs. `without helmet`).
3. Detecting vehicle `number plate` regions for violation logging and downstream OCR.
4. Providing an intuitive **Streamlit** dashboard for instant inference and automated violation summary statistics.

---

## 🚀 Key Features

- **Multi-Class Detection:** Trained to identify 4 critical classes simultaneously:
  - `rider`
  - `with helmet`
  - `without helmet`
  - `number plate`
- **Fast & Lightweight:** Built upon **YOLOv8 Nano** (`yolov8n.pt`), optimized for high inference speed and edge deployment on traffic monitoring cameras.
- **High Accuracy on Small Objects:** Fine-tuned over 50 epochs on GPU accelerators to ensure reliable detection of smaller targets (helmets and distant license plates).
- **Interactive Web App:** Includes a clean Streamlit interface with drag-and-drop image analysis and live detection metrics.
- **Out-of-Distribution (OOD) Validated:** Tested and verified against diverse, unseen web traffic photos to confirm generalization without overfitting.

---

## 📊 Model Performance & Metrics

Trained on **Nvidia Tesla T4x2 GPUs** via Kaggle using the [`aneesarom/rider-with-helmet-without-helmet-number-plate`](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate) dataset.

| Metric | 10 Epochs (Baseline) | 50 Epochs (Final Model) | Improvement |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Overall)** | 67.4% | **95.2%** | **+27.8%** |
| **Recall - With Helmet** | 43.1% | **88.3%** | **+45.2%** |
| **Recall - Without Helmet** | 18.7% | **93.3%** | **+74.6%** |
| **Recall - Number Plate** | 0.0% | **99.2%** | **+99.2%** |

### Training Progression & Loss Curves
![Training Results](assets/results.png)

### Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### Validation Detections Sample
![Validation Batch Predictions](assets/val_batch0_pred.jpg)

---

## 📁 Repository Structure

```text
traffic-helmet-detection/
├── assets/                          # Evaluation curves, confusion matrix, sample predictions
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── confusion_matrix.png
│   ├── results.png
│   └── val_batch0_pred.jpg
├── weights/
│   └── best.pt                      # Pre-trained fine-tuned YOLOv8 weights (5.96 MB)
├── app.py                           # Streamlit Web Application
├── TrafficHElmet.ipynb              # Kaggle Training & Fine-tuning Notebook
├── kernel-metadata.json             # Kaggle API deployment metadata
├── coco128.yaml                     # Reference configuration
├── check_dataset.py                 # Dataset validation and label distribution check
├── run_test.py                      # Batch test evaluation script
├── print_detections.py              # CLI detection output utility
├── test_uploaded.py                 # Single-image inference test script
├── crop_tests.py                    # Bounding box crop and inspection helper
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ignored cache, temp, and environment files
└── README.md                        # Project documentation
```

---

## 🛠️ Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AutonomousUI/traffic-helmet-detection.git
cd traffic-helmet-detection
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test custom traffic images.

---

## 🧠 Model Training on Kaggle

To retrain or run the notebook on Kaggle GPU:
1. Ensure the Kaggle API CLI is installed: `pip install kaggle`
2. Push and execute the notebook directly using:
   ```bash
   kaggle kernels push
   ```
3. Or open `TrafficHElmet.ipynb` inside Kaggle with the dataset [`aneesarom/rider-with-helmet-without-helmet-number-plate`](https://www.kaggle.com/datasets/aneesarom/rider-with-helmet-without-helmet-number-plate) attached.

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
