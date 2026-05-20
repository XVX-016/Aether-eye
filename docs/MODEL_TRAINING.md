# Model Training & Engineering Guide

This document is a comprehensive technical guide covering the architecture, training methodology, loss functions, ablation results, known weaknesses, and production deployment protocols for the machine learning models in the Aether-Eye platform.

---

## 1. Change Detection Model

### 1.1 Architecture
The change detection system is built around a multi-temporal **Siamese U-Net** architecture. 

*   **Encoder Weight Sharing**: A dual-branch encoder shares weights to process the multi-temporal "Before" and "After" images independently. This weight sharing ensures that the same spatial and spectral features are extracted identically from both inputs. This prevents artificial feature differences that would emerge if the branches diverged, forcing the model to focus strictly on relative variations (radiometric/structural shifts) rather than absolute semantic scenes.
*   **Feature Differencing**: Features extracted at each encoder stage are merged at the bottleneck and skip connections by computing the absolute element-wise difference:
    $$\mathbf{D}_i = \left| \mathbf{F}_i^{\text{before}} - \mathbf{F}_i^{\text{after}} \right|$$
    This differencing is done across multiple scales (H, H/2, H/4, H/8) and forms the skip connections routed to the decoder.
*   **Decoder and Bottleneck**: The bottleneck operates directly on the deepest absolute-difference features ($\mathbf{D}_4$). The decoder progressively upsamples and fuses these spatial differences via transpose convolutions and skip connections, culminating in a single-channel logit output representing the pixel-level probability of structural change.
*   **Concrete PyTorch Classes**:
    *   `ConvBlock` (in `siamese_unet.py`): Double convolutions ($3\times3$ Conv + BatchNorm + ReLU) forming the foundational building blocks of the encoder/decoder.
    *   `SiameseUNetChangeDetector` (in `siamese_unet.py`): The baseline Siamese U-Net network using custom convolutional blocks.
    *   `ConvBNReLU` & `UpBlock` (in `siamese_unet_resnet34.py`): Helper modules for the ResNet backbone implementation.
    *   `SiameseUNetResNet34` (in `siamese_unet_resnet34.py`): The production-grade Siamese U-Net encoder utilizing an ImageNet-pretrained ResNet-34 backbone for highly robust feature extraction.

### 1.2 Training Dataset
*   **Dataset Name**: `Building-change (WHU-style)`
*   **Dataset Split**: 
    *   **Train**: $1,134$ samples
    *   **Validation**: $126$ samples
    *   **Test**: $690$ samples
*   **Image Properties**: $256 \times 256$ pixels, 3-channel RGB format, paired with binary pixel-level masks.
*   **Target Label Format**: Binary mask where $1.0$ represents structural change (new construction, demolition, extension) and $0.0$ represents unchanged background.
*   **Change Pixel Ratio (~9.4%)**: Across the dataset, change pixels account for roughly $9.4\%$ of the total pixels. This highly sparse ratio illustrates the severe class imbalance inherent to change detection, which makes standard loss functions like standard Binary Cross-Entropy prone to predicting all-zero background masks.

### 1.3 Loss Function
To combat the $9.4\%$ sparse change ratio, Aether-Eye uses a **Hybrid Tversky Loss** (`hybrid_tversky`), defined as:
$$\mathcal{L}_{\text{hybrid\_tversky}} = 0.4 \cdot \mathcal{L}_{\text{BCE}} + 0.6 \cdot \mathcal{L}_{\text{FocalTversky}}$$

*   **BCE Loss**: Serves as a global anchor, pulling the massive background region towards $0.0$ and stabilizing early convergence.
*   **Focal Tversky Loss**: An advanced formulation optimized for highly sparse segmentations. It introduces $\alpha$ and $\beta$ weighting parameters alongside a focusing parameter $\gamma$:
    $$\text{Tversky} = \frac{\text{TP} + \epsilon}{\text{TP} + \alpha \cdot \text{FP} + \beta \cdot \text{FN} + \epsilon}$$
    $$\mathcal{L}_{\text{FocalTversky}} = (1 - \text{Tversky})^\gamma$$
    Using $\alpha=0.3$, $\beta=0.7$, and $\gamma=0.75$, the loss heavily penalizes False Negatives ($\beta=0.7$) over False Positives ($\alpha=0.3$), forcing the model to detect tiny structural alterations that would otherwise be ignored.

#### Ablation Study Results (5 Epochs, 3 Configurations)
To optimize training transformations, an ablation study was conducted over $5$ epochs:

| Config | Loss Function | Data Augmentation | Val IoU | Observation |
| :---: | :---: | :--- | :---: | :--- |
| **1** | `BCE-Dice` | `ColorJitter` + `RandomResizedCrop` | $0.5132$ | High false positive rate. Unaligned crop boundaries and color fluctuations created artificial changes. |
| **2** | `Hybrid Tversky` | `RandomResizedCrop` (No ColorJitter) | $0.6284$ | Segmentations aligned better, but crop scaling still caused alignment mismatches. |
| **3** | `Hybrid Tversky` | **Geometric Only** (No ColorJitter, No Resize Crop) | **$0.7410$** | **Optimal boundary precision**. Disabling unaligned resizing preserved spatial registration. |

*   **Removal of RandomResizedCrop**: `RandomResizedCrop` was removed because it introduces random scaling and cropping offsets between Before and After images. For change detection, pixel-level spatial registration is critical. Scaling variations disrupt this registration, causing massive false-positive edges.
*   **Removal of ColorJitter**: Color jitter was disabled because it creates synthetic radiometric differences. The Siamese encoder interprets these artificial color fluctuations as physical changes, degrading overall validation performance. Production training relies exclusively on rigid geometric transforms (Random Horizontal/Vertical Flips and $90^\circ$ Rotations) applied identically to both images.

### 1.4 Training Commands
To retrain the Siamese U-Net from scratch using the production configuration:

```bash
# Execute the production change training script
python run_training_production.py
```

*   **Automatic Resume**: The trainer is configured to auto-resume by default (`cfg.resume = True`). If the process is interrupted, it searches the output directory for `siamese_unet_change_latest.pt` (and falls back to `siamese_unet_change_best.pt`), reloading the optimizer parameters and faithfully resuming at the interrupted epoch.
*   **Checkpoint Destination**: Output weights are saved locally to `runs/siamese_unet_change/`.
*   **Exporting to ONNX**:
    ```bash
    python ml_core/change_detection/siamese_unet/export_change_onnx.py \
      --checkpoint runs/siamese_unet_change/siamese_unet_change_best.pt \
      --output ml_core/artifacts/change_model_v2/change_model_v2.onnx
    ```

### 1.5 Evaluation
To evaluate a trained checkpoint on the validation/test datasets and output IoU metrics:

```bash
python ml_core/evaluation/evaluate_change_detection.py \
  --checkpoint ml_core/artifacts/change_model_v2/change_model_v2.pt \
  --data-root data/processed/building_change
```

*   **Intersection over Union (IoU)**: Defined as the area of overlap between the predicted change mask and the ground truth divided by the area of union.
*   **Current Validation Score**: **$0.7936$ Val IoU** (production model `v2` checkpoint).

### 1.6 Known Weaknesses & Mitigation
*   **Structural Focus**: The dataset consists primarily of urban buildings, which translates poorly to specific military-geospatial features (e.g., runway surface deterioration or concrete hangar updates).
*   **Sensor Domain Shift**: The model is trained on sharp aerial imagery chips, which can cause false alarms when evaluated on medium-resolution Sentinel-2 multispectral chips due to cloud cover or atmospheric scattering.
*   **Mitigation Strategy**: Fine-tune the weights on the full `LEVIR-CD` dataset once available, integrating a wider variety of structural classes and sensor profiles to broaden the model's domain.

---

## 2. Aircraft Classification Model

### 2.1 Architecture
The fine-grained aircraft identification system is powered by a **ConvNeXt-Small** convolutional neural network.

*   **Why ConvNeXt-Small?**: ConvNeXt-Small incorporates modern Vision Transformer (ViT) design principles—such as patchify layers, depthwise convolutions, inverted bottlenecks, and GeLU activations—while retaining the inductive bias, robust spatial hierarchy, and scaling simplicity of standard CNNs. It was chosen over ViTs because it converges significantly faster on highly class-imbalanced datasets and does not suffer from data-hungry attention instabilities.
*   **Classification Depth**: Configured with **100 classes** representing specific commercial, military, and general aviation airframes.
*   **Preprocessing Pipeline**: 
    1.  Resize input to $1.1\times$ target size ($246 \times 246$).
    2.  Apply `RandomResizedCrop` to crop back to $224 \times 224$ pixels.
    3.  Apply `RandomHorizontalFlip` ($p=0.5$).
    4.  Apply light `ColorJitter` (brightness/contrast/saturation=0.2) to match variations in airfield illumination (train only).
    5.  Normalize with standard ImageNet channels: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.

### 2.2 Training Dataset
*   **Dataset**: `FGVC Aircraft` (Fine-Grained Visual Classification) benchmark containing 10,000 images of 100 distinct aircraft variants.
*   **Example Classes**:
    *   *Commercial Airliners*: `737-800`, `747-400`, `A320`, `A380`, `MD-11`, `ATR-72`
    *   *Tactical Military Assets*: `F-16A/B`, `F/A-18`, `Eurofighter Typhoon`, `C-130`, `Il-76`, `An-12`
    *   *Business/Civil Jets*: `Gulfstream V`, `Falcon 900`, `Cessna 172`, `Cessna 208`
*   **Class Imbalance**: Certain rare variants contain fewer training samples. Aether-Eye handles this by enabling `imbalance_mode="weighted_loss"` in the config. This scales the cross-entropy loss weights inversely with class frequency:
    $$w_c = \frac{1}{\text{count}_c}$$

### 2.3 Training Commands
To fine-tune the ConvNeXt classifier on the aircraft dataset:

```bash
python ml_core/classification/vit_aircraft/train_aircraft_classification.py
```

*   **Key Hyperparameters**:
    *   `model_name`: `convnext_small`
    *   `epochs`: `40`
    *   `batch_size`: `64`
    *   `learning_rate`: `1e-4`
    *   `weight_decay`: `0.05`
    *   `label_smoothing`: `0.1` (mitigates overconfident predictions)
    *   `mixup_alpha`: `0.2` (regularizes model via soft convex combination of inputs)
*   **Checkpoint Destination**: Epoch weights are saved to `experiments/aircraft/run_04_convnext_small/best.pt`.

### 2.4 Explainability (Grad-CAM)
To build operational trust, the classification pipeline implements **Grad-CAM** (Gradient-weighted Class Activation Mapping).

*   **Mechanism**: Implemented in `vit_gradcam.py`. For the ConvNeXt network, it computes the gradients of the target class score with respect to the final convolutional layer feature maps. These gradients are globally pooled to weight the activation maps, yielding a 2D localization heatmap showing where the model focused (e.g., wing structure, nose cone, tail configuration) to make its classification.
*   **REST Endpoint**: The backend exposes this explainability module via `/v1/aircraft-gradcam`. It accepts an image upload and an optional target class ID, returning a base64-encoded grayscale PNG overlay indicating the model's focus.

### 2.5 Known Weaknesses & Mitigation
*   **Perspective Domain Shift (Oblique vs. Nadir)**: The FGVC benchmark consists of ground-level, side-view oblique photography. Satellite imagery, however, is strictly top-down (nadir). This perspective shift causes a drop in classification confidence when classifying high-altitude imagery.
*   **Mitigation Strategy**: Fine-tune the classifier's stem and initial layers using the `xView` dataset, which contains high-resolution top-down airfield bounding boxes, to bridge the domain gap.

---

## 3. Deploying Updated Models

### 3.1 Exporting to ONNX
Production environments run all inferences on ONNX Runtime for optimal execution speed.

1.  **Export Change Model**:
    ```bash
    python ml_core/change_detection/siamese_unet/export_change_onnx.py \
      --checkpoint experiments/change/run_01/best.pt \
      --output ml_core/artifacts/change_model_v2/change_model_v2.onnx
    ```
2.  **Export Aircraft Classifier**:
    ```bash
    python ml_core/classification/vit_aircraft/export_aircraft_onnx.py \
      --checkpoint experiments/aircraft/run_04_convnext_small/best.pt \
      --output ml_core/artifacts/aircraft_classifier_v1/aircraft_classifier_v1.onnx
    ```

3.  **Place Output Files**: Move the exported `.onnx` models to their respective directories:
    *   `ml_core/artifacts/change_model_v2/change_model_v2.onnx`
    *   `ml_core/artifacts/aircraft_classifier_v1/aircraft_classifier_v1.onnx`

4.  **Update Inference Configurations**: Ensure that the YAML configuration files point to the correct paths:
    *   **Change Detection Config** (`backend/configs/inference/change_detector.yaml`):
        ```yaml
        onnx_path: "ml_core/artifacts/change_model_v2/change_model_v2.onnx"
        metrics_path: "ml_core/artifacts/change_model_v2/model_card.json"
        ```
    *   **Aircraft Classifier Config** (`backend/configs/inference/aircraft_classifier.yaml`):
        ```yaml
        onnx_path: "ml_core/artifacts/aircraft_classifier_v1/aircraft_classifier_v1.onnx"
        model_path: "ml_core/artifacts/aircraft_classifier_v1/aircraft_classifier_v1.pt"
        ```

### 3.2 Validating the New Model
1.  **Run Startup Diagnostics Check**: Launch the FastAPI server. Upon startup, Aether-Eye runs an automatic health check to confirm that the paths resolve and the models load correctly. You can trigger this diagnostic check manually:
    ```bash
    curl -X GET http://localhost:8000/health/models
    ```
    A successful check returns a list of active models:
    ```json
    {
      "status": "healthy",
      "change_detection": "loaded",
      "aircraft_classification": "loaded"
    }
    ```
2.  **Verify Endpoints**: Test classification and explanation endpoints using mock imagery:
    ```bash
    curl -X POST http://localhost:8000/v1/aircraft-classify \
      -F "image=@tests/data/sample_aircraft.png" \
      -F "country=USA"
    ```

---

## 4. Datasets Available in Repository

The `ml_core/DATASET/` directory contains the raw data used for model development and evaluation:

*   **`Aircraft/fgvc-aircraft-2013b`**
    *   **File Count**: $10,000$ images (split across train, val, and test partitions).
    *   **Status**: **Active**. This is the primary dataset used for fine-tuning and evaluating the fine-grained airframe classifier.
*   **`Aircraft/Military Aircraft Dataset`**
    *   **File Count**: $2,450$ images.
    *   **Status**: **Unused/Partial**. This dataset contains oblique military aircraft crops under investigation for future model fine-tuning.
*   **`Satellite-Change/Building-change`**
    *   **File Count**: $1,950$ paired temporal chips ($3,900$ total images + $1,950$ ground-truth masks).
    *   **Status**: **Active**. This is the production dataset used to train and validate the Siamese change detection network.
*   **`Satellite-Change/LEVIR CD`**
    *   **File Count**: Path initialized (empty placeholders).
    *   **Status**: **Unused**. Configured as a target for future model expansion.
*   **`Satellite-Change/spacenet-7-multitemporal-urban-development`**
    *   **File Count**: Path initialized.
    *   **Status**: **Unused**. Intended for future deep multi-temporal tracking research.
*   **`Satellite-Change/train-vehicle`**
    *   **File Count**: Path initialized.
    *   **Status**: **Unused**. Holds vehicle detection footprints for future wide-area monitoring expansions.
