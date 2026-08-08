# LiquidSpike

A continuous-discrete hybrid spiking framework for event-based object detection.

Implementation of the proposed:

**LiquidSpike: Liquid Spatiotemporal Representation and Spike-Aware Alignment for Event-Based Object Detection**

---

## Overview

LiquidSpike is a continuous-discrete hybrid spiking object detector designed for efficient event-based perception.

Different from fully spike-driven architectures, LiquidSpike selectively introduces continuous latent-state processing into information-critical modules while preserving spike-driven information transmission.

The framework contains four main components:

- **LTC-IPLIF**: Input-adaptive liquid time-constant spiking neuron for dynamic temporal modeling.
- **LiquidSSM**: Spike-compatible state-space module for efficient long-range spatiotemporal dependency modeling.
- **Spiking Dynamic Routing (SDR)**: Activity-aware cross-scale spike feature fusion.
- **SpikingEMA**: Lightweight localization refinement module for bounding-box regression.

---

# Installation

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8


Create environment:

```bash
conda create -n liquidspike python=3.10
conda activate liquidspike

pip install -r requirements.txt
```

---

# Dataset Preparation

## Gen1 Automotive Dataset

Download:

https://www.prophesee.ai/event-based-datasets/


Recommended structure:

```text
datasets/
└── Gen1/
    ├── train/
    ├── val/
    └── test/
```


## MS COCO 2017

Download:

https://cocodataset.org/


Recommended structure:

```text
datasets/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
```

---

# Repository Structure

```text
LiquidSpike-main/
│
├── ultralytics/
│   ├── cfg/
│   ├── data/
│   ├── engine/
│   ├── models/
│   ├── nn/
│   └── utils/
│
├── train.py
├── val.py
├── predict.py
├── requirements.txt
└── README.md
```

---

# Training

Example:

```bash
python train.py \
--model cfg/models/liquidspike.yaml \
--data dataset.yaml
```

---

# Evaluation

Example:

```bash
python val.py \
--model weights/liquidspike.pt \
--data dataset.yaml
```


Evaluation metrics:

- mAP@50
- mAP@50:95
- Number of parameters
- Spike-operation energy proxy

---

# Temporal Configuration

LiquidSpike adopts the **T×D evaluation strategy**.

- **T**: physical simulation timesteps during SNN inference.
- **D**: maximum integer activation level used during training.

Different T×D configurations provide different accuracy-efficiency trade-offs.

---

# Results

## Gen1 Automotive Dataset

| Method | Params | T×D | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| SpikeYOLO | 23.1M | 4×2 | 67.2 | 40.4 |
| Spiking Trans-YOLO | 26.3M | 4×2 | 68.6 | 42.1 |
| **LiquidSpike** | **18.9M** | **4×2** | **69.4** | **41.8** |


## MS COCO 2017

| Method | Params | T×D | mAP@50 | mAP@50:95 |
|---|---|---|---|---|
| SpikeYOLO | 23.1M | 1×1 | 52.7 | 36.1 |
| Spiking Trans-YOLO | 26.3M | 1×1 | 54.0 | 37.8 |
| **LiquidSpike** | **18.9M** | **1×1** | **56.2** | **38.2** |

---

# Pretrained Models

Pretrained weights will be released after paper acceptance.

---

# Citation

The citation will be updated after publication.

---

# License

This project is released for academic research purposes only.

The datasets used in this project are subject to their respective licenses.

