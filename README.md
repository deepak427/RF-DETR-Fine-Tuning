# RF-DETR Wildlife Detection: AWS SageMaker Fine-Tuning Pipeline

[![AWS SageMaker](https://img.shields.io/badge/AWS-SageMaker%20GPU-FF9900?logo=amazoneks&logoColor=white)](https://aws.amazon.com/sagemaker/)
[![Architecture](https://img.shields.io/badge/Model-RF--DETR%20Medium%20(DINOv2)-blue)](https://github.com/roboflow/rf-detr)
[![Dataset](https://img.shields.io/badge/Dataset-NTLNP%20(25k%20Images)-green)](#dataset-details)
[![mAP@50](https://img.shields.io/badge/Val%20mAP@50-98.5%25-brightgreen)](#training-results)
[![Target](https://img.shields.io/badge/Edge%20Target-OpenCV%205%20DNN-red)](#edge-deployment)

Complete reproducibility package and documentation for fine-tuning **RF-DETR (Real-Time Detection Transformer)** on the **NTLNP (Northeast Tiger and Leopard National Park)** wildlife dataset using **AWS SageMaker GPU infrastructure**.

---

## 1. Project Overview

This repository demonstrates the fine-tuning of `RFDETRMedium` with a **DINOv2 windowed vision transformer backbone** to detect 17 distinct wildlife species in camera-trap environments. 

The resulting fine-tuned weights were exported to ONNX format and integrated directly into the edge inference service powered by **OpenCV 5 DNN** on **AWS EC2 Graviton2 (`t4g.medium`)**, delivering real-time edge tracking with pan-tilt servo tracking and deterrent actions.

```
       +-----------------------------------------------------------+
       |                  AWS SageMaker (Cloud)                    |
       |  - GPU Instance: ml.g4dn.xlarge / ml.g5.xlarge            |
       |  - Dataset: NTLNP (25,297 camera-trap images)             |
       |  - Model: RF-DETR Medium (DINOv2 Windowed Backbone)       |
       |  - Training: 60 epochs, AdamW, effective batch size 16    |
       |  - Validation mAP@50: 98.5% | mAP@50-95: 87.9%            |
       +-----------------------------+-----------------------------+
                                     |
                         Export PyTorch -> ONNX
                                     v
       +-----------------------------------------------------------+
       |              AWS EC2 Graviton (t4g.medium)                |
       |  - OpenCV 5 DNN C++ Engine (Zero PyTorch runtime overhead)|
       |  - Slew-rate limiter & pan/tilt servo tracking loop       |
       |  - Deterrent dispatcher (strobe / ultrasound)             |
       +-----------------------------------------------------------+
```

---

## 2. AWS SageMaker Infrastructure

The training workload was orchestrated on **Amazon SageMaker**:

- **Compute**: AWS SageMaker Notebook Instance (`ml.g4dn.xlarge` NVIDIA T4 16GB / `ml.g5.xlarge` NVIDIA A10G 24GB).
- **Storage**: AWS EBS GP3 volume with 125 MB/s baseline throughput.
- **Environment**: Amazon Linux 2, PyTorch 2.1+, CUDA 12.x, PyTorch Lightning.
- **Acceleration**: Automatic Mixed Precision (AMP FP16) with fused AdamW optimizer.

---

## 3. Dataset Details: NTLNP (Northeast Tiger and Leopard National Park)

The **NTLNP** dataset consists of camera-trap captures monitoring protected wildlife across diverse seasons, lighting conditions, and camouflage:

- **Total Images**: **25,297**
- **Train Set**: 17,708 images (70%)
- **Validation Set**: 5,059 images (20%)
- **Test Set**: 2,530 images (10%)
- **Target Resolution**: 576 &times; 576 (multi-scale batch jittering enabled)

### 17 Wildlife Classes
Alphabetical class indexing preserved for ONNX export compatibility:
| ID | Class Name | ID | Class Name | ID | Class Name |
|:--:|:-----------|:--:|:-----------|:--:|:-----------|
| 0 | `AmurTiger` | 6 | `Leopard` | 12 | `Sable` |
| 1 | `Badger` | 7 | `LeopardCat` | 13 | `SikaDeer` |
| 2 | `BlackBear` | 8 | `MuskDeer` | 14 | `Weasel` |
| 3 | `Cow` | 9 | `RaccoonDog` | 15 | `WildBoar` |
| 4 | `Dog` | 10 | `RedFox` | 16 | `Y-T-Marten` |
| 5 | `Hare` | 11 | `RoeDeer` | | |

---

## 4. Model Architecture & Hyperparameters

| Parameter | Specification | Details |
|---|---|---|
| **Base Architecture** | `RFDETRMedium` | Real-Time Detection Transformer |
| **Vision Backbone** | `dinov2_windowed_small` | Self-supervised Vision Transformer with windowed attention |
| **Resolution** | `576 x 576` | Divisible by 64 for patch embedding alignment |
| **Decoder Layers** | 4 | Two-stage transformer decoder |
| **Object Queries** | 300 | Maximum candidate detections per frame |
| **Loss Formulation** | IA-BCE + GIoU + L1 | Task-aligned bounding box and classification loss |
| **Optimizer** | AdamW | `lr=1e-4`, `lr_encoder=1.5e-4`, `weight_decay=1e-4` |
| **Batching** | Batch Size 8 &times; 2 Grad Accum | Effective Batch Size: **16** |
| **Epochs** | 60 | Checkpoints saved every 10 epochs |
| **EMA** | Exponential Moving Avg | Decay rate = `0.993`, tau = `100` |

---

## 5. Training Results & Metrics

Training logged to `output/metrics.csv` over 60 epochs. Final epoch metrics on validation set:

| Metric | Score |
|---|---|
| **mAP@50 (EMA)** | **98.54%** (`0.9854`) |
| **mAP@50:95 (EMA)** | **87.87%** (`0.8787`) |
| **mAP@75 (EMA)** | **95.97%** (`0.9597`) |
| **F1 Score** | **97.92%** (`0.9792`) |
| **Precision** | **98.16%** (`0.9816`) |
| **Recall** | **97.69%** (`0.9769`) |

Visual test predictions demonstrate crisp bounding boxes and high confidence scores across challenging night-vision and daylight captures (see `prediction_result.jpg`).

---

## 6. Repository Structure

```
rfdetr-sagemaker-finetuning/
├── how_to_finetune_rf_detr_on_detection_dataset.ipynb  # Primary AWS SageMaker training notebook
├── predict.py                                          # Local inference validation script
├── upload_to_roboflow.py                               # Model registry upload script
├── requirements.txt                                    # Python dependencies
├── .env.example                                        # Template for Roboflow API keys
├── .gitignore                                          # Excludes heavy checkpoints (>100MB) & venv
├── images (8).jpg                                      # Sample validation test image
├── prediction_result.jpg                               # Visual inference output
└── output/
    ├── class_names.txt                                 # List of 17 target classes
    ├── training_config.json                            # Complete hyperparameter audit trail
    ├── metrics.csv                                     # Epoch-by-epoch loss & validation metrics
    └── hparams.yaml                                    # Lightning configuration
```

---

## 7. How to Reproduce on AWS SageMaker

### Step 1: Launch SageMaker GPU Notebook
1. Open AWS SageMaker Console &rarr; **Notebook Instances** &rarr; **Create notebook instance**.
2. Select Instance Type: `ml.g4dn.xlarge` or `ml.g5.xlarge`.
3. Set Volume Size: **50 GB** minimum.
4. Clone or upload this repository.

### Step 2: Configure Secrets
Copy the environment template:
```bash
cp .env.example .env
```
Fill in your `ROBOFLOW_API_KEY` in `.env`.

### Step 3: Run Training Notebook
Execute `how_to_finetune_rf_detr_on_detection_dataset.ipynb` cell-by-cell:
1. Verify GPU via `!nvidia-smi`.
2. Download NTLNP dataset via Roboflow API.
3. Instantiate `RFDETRMedium` with pretrained weights.
4. Run `model.train()` for 60 epochs.
5. Export best checkpoint to ONNX format.

### Step 4: Verify Local Prediction
```bash
python predict.py
```
Outputs bounding boxes and saves annotated image to `prediction_result.jpg`.

---

## 8. Edge Deployment Integration

The exported ONNX model is deployed into our edge system:
1. **Model Path**: `inference_service/models/inference_model.onnx`
2. **Runtime Engine**: OpenCV 5 DNN (`cv2.dnn.readNetFromONNX`)
3. **Hardware**: AWS EC2 `t4g.medium` (Graviton2 ARM64)
4. **Latency**: ~85ms per frame on Graviton2 CPU without PyTorch dependency.
