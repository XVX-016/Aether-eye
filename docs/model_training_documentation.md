# Aether-Eye: Model Training & Intelligence Pipelines

This document provides a technical overview of the model architectures, training workflows, and inference pipelines implemented in the Aether-Eye platform.

## 1. Aircraft Classification

The aircraft classification module provides fine-grained identification of aircraft variants from image chips.

- **Architecture**: `convnext_small` (via `timm`). This architecture was selected over ViT for its superior performance on smaller datasets and better convergence during fine-tuning.
- **Dataset**: `FGVC Aircraft` (Fine-Grained Visual Classification). Contains 10,000 images of 100 aircraft variants.
- **Training Metrics**: 
    - **Top-1 Accuracy**: 72.5%
    - **Macro F1-Score**: 72.0%
    - (Verified from `experiments/aircraft/run_04_convnext_small/metrics.json`)
- **Loss Function**: `CrossEntropyLoss` with Label Smoothing (0.1). During training, a custom `_weighted_soft_ce` is used to support Mixup and CutMix augmentations.
- **Optimizer & Learning Rate**: `AdamW` with a base learning rate of `3e-4` (production) and weight decay of `0.05`.
- **Classes**: 100 classes representing specific aircraft variants (e.g., `707-320`, `F-16A/B`, `Eurofighter Typhoon`).
- **Explainability**: Implemented via **Grad-CAM** (`ml_core/aether_ml/explainability/grad_cam.py`). It generates heatmaps showing which regions of the aircraft (e.g., tail, wing configuration) the model focused on for its prediction.
- **Friend/Foe Classification**: A rule-based system (`app/services/geopolitics.py`) that matches the aircraft's predicted origin country against a user-selected country. Relations are categorized as `FRIEND`, `FOE`, or `NEUTRAL` based on a predefined geopolitical alliance matrix.

---

## 2. Change Detection

The change detection module identifies structural and environmental changes between two temporal satellite captures.

- **Architecture**: `SiameseUNet` with a **ResNet34** backbone. It uses a dual-branch weight-sharing encoder to extract features from "Before" and "After" images. Feature maps are fused via absolute differencing at multiple scales before being passed to a U-Net decoder.
- **Dataset**: `Building-change (WHU-style)`. Focused on structural evolution and building footprints.
- **Metrics**: **0.7936 Val IoU**. (Verified from `ml_core/artifacts/change_model_v2/model_card.json`).
- **Loss Function**: **Hybrid Tversky Loss** (`0.4 * BCE + 0.6 * FocalTversky`). This combination handles the significant class imbalance (change vs. no-change) better than standard BCE-Dice.
- **Training Split**: 
    - Train: 1,134 samples
    - Validation: 126 samples
    - Test: 690 samples
- **Ablation Findings**: Data augmentation including `ColorJitter` was found to be detrimental to change detection performance as it introduces artificial radiometric differences that the model confuses with actual physical changes. Production training uses only geometric transforms (Rotate, Flip, Crop).
- **ONNX Export**: The model is exported to ONNX (`change_model_v2.onnx`, ~110MB) for production inference, adapted via a `ConcatInputWrapper` to accept a single 6-channel input (concatenated Before/After images).

---

## 3. What Is Not Yet Built (Implementation Gaps)

- **YOLO Detector Artifact**: While training scripts for YOLOv8 exist (`ml_core/aether_ml/training/yolov8_aircraft.py`), a production-ready weights artifact for wide-area satellite detection has not yet been packaged.
- **Infrastructure Segmentation**: Models for segmenting runways, hangars, or taxiways (e.g., SegFormer) are not currently implemented.
- **End-to-End Pipeline**: The multi-stage `Detect -> Crop -> Classify` workflow is not yet fully automated in the backend; the classifier currently operates on manually provided or uploaded image chips.
- **Military Specifics**: Fine-grained military variants are limited to what is available in the FGVC dataset; coverage for specialized ISR or electronic warfare platforms is currently missing.

---

## 4. Pipeline Architecture

- **Satellite Integration**: The `stac_watcher` identifies new Sentinel-2 scenes, which are processed by the `scene_processor`. If a "Before" scene exists for the same AOI, the `change_service` triggers the `ChangeDetectionOnnxPipeline`.
- **API Inference**: The backend exposes endpoints like `/predict/aircraft` which accept image uploads and route them to the appropriate ML service.
- **Inference Engine**: All production inference is performed via **ONNX Runtime** (`onnxruntime-gpu` or `onnxruntime`).
- **Device Detection**: The system automatically detects CUDA availability. If a GPU is present, it uses `CUDAExecutionProvider`; otherwise, it falls back to `CPUExecutionProvider`.

---

## 5. Training Commands

### Change Detection
To retrain the change detection model using the production configuration:
```bash
python run_training_production.py
```
Checkpoints and logs are saved to `runs/siamese_unet_change/`.

### Aircraft Classification
To retrain the aircraft classifier:
```bash
python ml_core/classification/vit_aircraft/train_aircraft_classification.py --data-root ml_core/DATASET/Aircraft
```
Checkpoints are saved to `experiments/aircraft/`.

### Export to ONNX
After training, models can be exported for production:
```bash
python ml_core/change_detection/siamese_unet/export_change_onnx.py --checkpoint ml_core/artifacts/change_model_v2/change_model_v2.pt --output change_model_v2.onnx
```
