# EcoVision AI: Real-Time AI Waste Classification System
**College AI Mini Project • Computer Science & Engineering**

![Python](https://img.shields.io/badge/Python-3.11.9-3776AB?logo=python)
![YOLO](https://img.shields.io/badge/Model-YOLO11n-00FFFF?logo=ultralytics)
![Framework](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0+cpu-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/Dataset_License-CC_BY_4.0-blue.svg)

---

## 📌 1. Project Overview

**EcoVision AI** is a real-time computer vision and deep learning application designed to classify solid waste into **five essential segregation categories** using standard laptop hardware.

Built specifically for local execution on Windows 11, the system utilizes the cutting-edge **YOLO11n (Nano)** architecture developed by Ultralytics. The system operates via both **live webcam video streaming** and **static image uploads**, providing instant bounding-box localization, confidence scoring, and actionable waste disposal recommendations.

---

## 🏷️ 2. The Five Target Waste Categories

| Index | Category | Emoji | Bin Type | Recommended Disposal Action |
| :---: | :--- | :---: | :--- | :--- |
| **0** | **Organic** | 🌱 | Green Bin (Wet Waste) | Backyard compost pile or municipal wet waste collection. Keep free of plastics. |
| **1** | **Plastic** | 🧴 | Blue Bin (Dry Recyclables) | Empty and rinse thoroughly. Flatten bottles. Check resin codes (#1, #2). |
| **2** | **Glass** | 🍾 | Yellow / Teal Bin | Rinse bottles/jars. Separate metal/plastic caps. Wrap broken glass safely. |
| **3** | **Paper** | 📄 | Blue Bin (Paper Stream) | Keep clean and dry. Flatten cardboard boxes. Avoid food-greasy paper. |
| **4** | **Metal** | 🥫 | Grey / Silver Bin | Empty and rinse aluminum cans and tin food tins. Crush to optimize space. |

> [!NOTE]  
> **Municipal Disclaimer**: Waste segregation categories and processing standards may vary across different cities and recycling authorities. Always verify local municipal protocols.

---

## 💻 3. System Hardware & Compatibility Analysis

* **Host Machine**: HP Laptop running **Windows 11 (64-bit AMD64)**
* **Processor (CPU)**: **Intel(R) Core(TM) Ultra 7 255U** (14 logical cores, Meteor Lake architecture)
* **RAM**: **32 GB Total** (~16.7 GB available headroom)
* **Execution Mode**: **CPU Native** (No NVIDIA GPU or Jetson Nano required)

---

## 📊 4. Dataset Attribution & Academic Integrity (CC BY 4.0)

This project strictly adheres to academic integrity and Creative Commons licensing standards:

* **Dataset Source**: *Garbage Classification 3* hosted on **Roboflow Universe**
* **Project Reference**: [https://universe.roboflow.com/material-identification/garbage-classification-3](https://universe.roboflow.com/material-identification/garbage-classification-3)
* **License**: **Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))**
* **Dataset Transformations & Mapping**:
  1. `BIODEGRADABLE` is mapped directly to **Organic (Class 0)**.
  2. `PLASTIC` is mapped to **Plastic (Class 1)**.
  3. `GLASS` is mapped to **Glass (Class 2)**.
  4. `PAPER` and `CARDBOARD` are consolidated into **Paper (Class 3)**.
  5. `METAL` is mapped to **Metal (Class 4)**.
  6. Any non-standard `CLOTH` annotations are excluded and empty frames purged.
  7. **Zero Data Leakage Assertion**: Cryptographic SHA-256 hashes are computed for every image file to guarantee that no training images overlap with validation or test sets ($\text{Train} \cap \text{Val} = \emptyset, \text{Train} \cap \text{Test} = \emptyset$).

---

## 🚀 5. Quick-Start Guide (Step-by-Step)

### Step 1: Open Terminal & Activate Virtual Environment
Open PowerShell inside the project directory:
```powershell
# Navigate to project folder
cd C:\Users\chand\Desktop\AI_waste_classification

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Train or Fine-Tune YOLO11n Model
Train the model on your Intel Core Ultra 7 CPU using the verified 5-class dataset:
```powershell
python train.py --epochs 25 --imgsz 416 --batch 16
```
This script will:
* Train YOLO11n using transfer learning from `yolo11n.pt`.
* Export `best.pt` to `model/waste_yolo11n.pt`.
* Generate the authentic **Confusion Matrix** (`model/confusion_matrix.png`) and **Training Curves** (`model/results.png`).
* Run an independent evaluation on the unseen test set.

### Step 4: Run Offline Inference on Test Images
Test the trained model on static images without launching the full web dashboard:
```powershell
python detect.py --source dataset/images/test --conf 0.40
```
Annotated detection results are saved to `runs/detect/predictions/`.

### Step 5: Benchmark Actual Hardware Performance
Empirically measure inference latency, FPS, and RAM footprint on your CPU:
```powershell
python benchmark.py
```

### Step 6: Launch the EcoVision AI Web Application
Start the multi-page Streamlit dashboard:
```powershell
streamlit run app.py
```
Your default web browser will open at `http://localhost:8501`.

---

## 📁 6. Project Directory Structure

```
AI_waste_classification/
│
├── .venv/                          # Isolated Python 3.11 virtual environment
├── app.py                          # Streamlit Multi-Page Web Dashboard
├── train.py                        # Automated YOLO11n model training pipeline
├── detect.py                       # Standalone offline inference testing script
├── benchmark.py                    # Empirical latency, FPS, and RAM benchmark
├── download_dataset.py             # Dataset downloader, remapper & verification runner
├── requirements.txt                # Pinned dependencies
├── README.md                       # Project documentation & report guide
│
├── dataset/                        # 5-class YOLO format dataset
│   ├── data.yaml                   # YOLO configuration defining 5 classes
│   ├── images/                     # train/, val/, test/ images
│   └── labels/                     # train/, val/, test/ normalized annotations
│
├── model/                          # Trained model weights and genuine metrics
│   ├── waste_yolo11n.pt            # Best fine-tuned model weights
│   ├── confusion_matrix.png        # Validation confusion matrix
│   └── results.png                 # Loss and mAP training graphs
│
├── utils/                          # Modular helper libraries
│   ├── __init__.py
│   ├── detector.py                 # YOLO11 inference wrapper & HUD bounding-box drawer
│   ├── disposal_guide.py           # Disposal rules & environmental advice per category
│   ├── prepare_dataset.py          # Class remapping, cloth filtering & leakage checks
│   └── dataset_checker.py          # Annotation validator & visual bounding-box inspector
│
└── assets/                         # UI styling assets
    └── style.css                   # Custom green-and-white theme stylesheet
```

---

## 🛡️ 7. Troubleshooting & Common Questions

1. **Inference Latency Optimization**:
   - The default resolution is set to $416 \times 416$ for high responsiveness on CPU.
   - Adjust the confidence threshold slider in the dashboard to fine-tune sensitivity.
2. **Model Weights Missing Warning**:
   - Ensure you run `python train.py` once to generate `model/waste_yolo11n.pt`.

---
*Developed as a College AI Mini Project in Computer Science and Engineering.*
