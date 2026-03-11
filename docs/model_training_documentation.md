# Aether-Eye: Model Training & Intelligence Pipelines

This document provides a comprehensive overview of the model architectures, intelligence pipelines, and geospatial data management for the Aether-Eye platform.

## 1. Project Vision

Aether-Eye is a professional-grade satellite intelligence platform designed to transform raw imagery into actionable insights. It mirrors the capabilities of industry leaders like Maxar and Palantir by integrating detection, classification, and change analysis into a single geospatial pipeline.

This project is moving toward a global monitoring and scanning intelligence system by unifying aircraft detection/classification with multi-temporal change detection, and exposing the pipeline through production API services.

### Core Intelligence Capabilities:
- **Aircraft Intelligence**: A multi-stage pipeline that detects aircraft and classifies their specific models (variants).
- **Change Intelligence**: Temporal analysis of building and infrastructure changes between two satellite captures.
- **Geospatial & Timeline Awareness**: Detections are mapped to GPS coordinates and tracked over time to identify events (arrivals, departures, construction).

---

## 2. Integrated Intelligence Pipelines

### 2.1 Aircraft Intelligence Pipeline (Stage 1 & 2)
The system uses a sequential flow to provide high-precision identification:

1.  **Detection (YOLOv8)**: Scans the satellite imagery for aircraft bounding boxes.
2.  **Cropping**: Extracts the region of interest (ROI) around the aircraft.
3.  **Classification (ViT)**: Refines the detection by identifying the specific aircraft type (e.g., F-16, Boeing 737).

**Example Output:**
- `Detected Aircraft: 2`
- `[1] F-16 Fighting Falcon (Conf: 0.92)`
- `[2] Boeing 737 (Conf: 0.87)`

### 2.2 Change Intelligence Pipeline
A dual-stream approach for monitoring structural evolution:

1.  **Dual Input**: Accepts "Before" and "After" status imagery.
2.  **Siamese Analysis**: Extracts features from both and identifies pixel-level differences.
3.  **Impact Quantification**: Calculates changed area (%) and identifies affected structures.

### 2.3 Intelligence Orchestration (Operational)
The backend now includes an asynchronous intelligence job runner that executes:
- **Change Detection (Siamese U-Net ONNX)** on before/after imagery.
- **Aircraft Detection (YOLOv8 ONNX)** on latest imagery.
- **Optional Classification (ViT)** on detected aircraft crops.
Results are stored as persistent intelligence events for the operations dashboard.

---

## 3. Model Architectures & Mathematical Formulations

### 3.1 Detection: YOLOv8
- **Purpose**: Object detection (Aircraft, Small Objects).
- **Datasets**: xView, DOTA.
- **Key Formula**: **CIoU Loss** for bounding box regression.
    $$L_{CIoU} = 1 - IoU + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$

### 3.2 Classification: Vision Transformer (ViT) / ResNet
- **Purpose**: Fine-grained variant identification.
- **Datasets**: FGVC Aircraft, Stanford Aircraft, Military Aircraft.
- **Key Formula**: **Softmax Cross-Entropy** with Label Smoothing.
    $$L = -\sum_{i=1}^{C} [y_i(1-\alpha) + \frac{\alpha}{C}] \log(\hat{y}_i)$$

### 3.3 Change Detection: Siamese U-Net
- **Purpose**: Pixel-level change mapping.
- **Datasets**: LEVIR-CD, WHU-CD, SpaceNet-7.
- **Key Formula**: **Combined BCE-Dice Loss**.
    $$L_{Total} = 0.5 \cdot L_{BCE} + 0.5 \cdot L_{Dice}$$

---

## 4. Dataset Management & Repository Structure

### 4.1 Professional Dataset Mapping
| Model | Dataset | Primary Usage |
| :--- | :--- | :--- |
| **YOLOv8** | xView / DOTA | Aircraft & Small Object Detection |
| **ViT** | FGVC / Military Aircraft | Fine-grained Classification |
| **Siamese CNN** | LEVIR-CD / WHU-CD | Urban Change Detection |
| **SegFormer** | SpaceNet | Infrastructure / Building Mapping |

### 4.2 Recommended Repository Structure
```text
aether-eye/
├── ml_core/
│   ├── detection/          # YOLOv8 (Aircraft, Ships)
│   ├── classification/     # ViT (Aircraft variants)
│   ├── change_detection/   # Siamese Models
│   └── preprocessing/      # Tiling & Cropping Logic
├── data/
│   ├── raw/                # Original downloads (unmodified)
│   └── processed/          # YOLO/ImageFolder formatted data
├── models/                 # Production weights (.pt, .onnx)
├── scripts/                # Data conversion & training entry points
└── web-app/                # Next.js Dashboard
```

---

## 5. Operations & Geospatial Intelligence

### 5.1 Geospatial Layering
Real satellite intelligence requires mapping pixel detections to coordinates:
- **Tiling**: Large satellite images (e.g., 12k x 12k) are processed in 512x512 tiles for inference.
- **Coordinate Mapping**: Detections are projected from image space to Latitude/Longitude for mapping in the **Operations Dashboard**.

### 5.2 Event Tracking
The system moves beyond raw detections to **Intelligence Events**:
- **Arrivals/Departures**: Tracked via temporal analysis.
- **Structural Construction**: Quantified as `Change %` on the Intelligence Map.

---

## 6. Training Pipeline Workflow

1.  **Acquisition**: `download_kaggle_satellite_datasets.py` (SpaceNet, DOTA).
2.  **Preparation**: `prepare_stanford_classification_dataset.py` (Organizes image folders).
3.  **Conversion**: `convert_stanford_to_yolov8.py` (Creates detection labels).
4.  **Training**: Run in order: **Detection** → **Classification** → **Change Detection**.
5.  **Export**: `export_aircraft_detector_onnx.py` for edge/web deployment.

---

## 7. Current Status & Implementation Roadmap

This section tracks the progress toward the full **Aether-Eye Product Vision**.

### 7.1 Implemented Features (Completed)
- **Model Training Pipelines**:
    - [x] **Aircraft Classification**: Full pipeline for FGVC/Stanford Aircraft via ViT/ResNet.
    - [x] **Aircraft Detection**: Training scripts for YOLOv8 (xView subset).
    - [x] **Change Detection**: Core Siamese U-Net implementation and trainer.
- **Dataset Utilities**:
    - [x] **Automated Data Acquisition**: Kaggle download script for SpaceNet and DOTA.
    - [x] **Data Conversion**: xView/Stanford to YOLOv8 label converters.
- **Deployment**:
    - [x] **Model Export**: ONNX export for edge/web integration.
    - [x] **Backend Intelligence Job**: Asynchronous pipeline that runs change detection + aircraft detection and persists events.

### 7.2 Model Training Status (In-Progress/Pending)
- **[PENDING] Multi-Dataset Training**:
    - Training YOLOv8 on the combined **xView + DOTA** large-scale datasets.
    - Fine-tuning ViT on the **Military Aircraft Kaggle** dataset for defense-specific variants.
- **[PENDING] High-Res Change Models**:
    - Training Siamese CNN on **WHU-CD** for increased precision in structural analysis.
- **[NEW] Infrastructure Mapping**:
    - Training **SegFormer / UNet** on the **SpaceNet** dataset for runways and hangars.

### 7.3 Implementation Gaps & Product Roadmap
To reach "Intelligence Platform" status, the following components are currently pending:

1.  **Multi-Stage Inference Pipeline**: Integrating `Detection -> Crop -> Classification` into a single automated service.
2.  **Geospatial Tiling Service**: Logic for breaking down massive satellite imagery (e.g., 12k x 12k) into 512x512 tiles.
3.  **Coordinate Projection Layer**: Mathematical logic to convert image $(x, y)$ coordinates to **Latitude/Longitude (GPS)**.
4.  **Temporal Event Analysis**: Logic to correlate detections over time to identify "Arrival" or "Departure" events.
5.  **Operations Dashboard**: Full Next.js implementation of the **Interactive Intelligence Map** with MapLibre.
