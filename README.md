# Object Detection Project (Ultralytics YOLO)

This repository contains a **training / inference / submission generation** workflow implemented in **`train.ipynb`**.

> Notes on terminology:  
> - **YOLO (You Only Look Once)** is an object detection model family.  
> - **GPU (Graphics Processing Unit)** / **CPU (Central Processing Unit)** are compute devices.  
> - **CUDA (Compute Unified Device Architecture)** is NVIDIA’s GPU computing platform (optional, for GPU training).

---

## 1) What this repo does

- Prepare dataset folders (images + labels)
- Train an object detection model with **Ultralytics YOLO (Ultralytics You Only Look Once)** APIs
- Run inference on test images
- Export a competition-style **`submission.txt`** (image id + bounding boxes)

The notebook includes installation hints for **YOLOv12 (You Only Look Once version 12)** from a GitHub source, and utilities for output formatting.

---

## 2) Repository structure

```
.
├─ train.ipynb                 # Main notebook: training + inference + submission output
├─ submission/                 # (Optional) output directory for submission.txt
└─ README.md
```

---

## 3) Environment setup

### Option A — Conda (recommended)
```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
python -m pip install -U pip setuptools wheel
```

### Install dependencies
```bash
# Ultralytics YOLO
pip install ultralytics

# YOLOv12 (from GitHub, if you are using it in the notebook)
pip install git+https://github.com/sunsmarterjie/yolov12.git

# (Optional) Hugging Face tools used in some environments
pip install huggingface_hub transformers safetensors accelerate
```

> If you plan to train on GPU, install **PyTorch (PyTorch Deep Learning Framework)** with the correct CUDA build for your NVIDIA driver.

---

## 4) Dataset format

The notebook assumes two main folders:

- `TRAINING_IMAGE`: training images folder
- `TRAINING_LABEL`: training labels folder

Typical expected format:

```
training_image/
  000001.jpg
  000002.jpg
training_label/
  000001.txt
  000002.txt
```

Label text files are commonly in **YOLO format (You Only Look Once format)**:
```
<class_id> <x_center> <y_center> <width> <height>
```
where coordinates are normalized by image width/height.

> If your competition uses a different label format, update the parsing logic in `train.ipynb`.

---

## 5) How to run

### Step 1 — Open the notebook
Run with:
- **Jupyter Notebook (Project Jupyter Notebook)**
- or **Visual Studio Code (VS Code) + Jupyter extension**

### Step 2 — Update local paths
In `train.ipynb`, edit the path variables like:

```python
TRAINING_IMAGE = r"C:\path\to\training_image"
TRAINING_LABEL = r"C:\path\to\training_label"
```

> Windows paths should use raw strings `r"..."` to avoid escape issues.

### Step 3 — Train
Run the training cells in the notebook.  
(Training code and settings are inside `train.ipynb`.)

### Step 4 — Inference + submission generation
The notebook will run prediction and write a file like:

- `submission/submission.txt`  
or
- `submission.txt` (depending on your path settings)

---

## 6) Submission file format

The notebook exports detections in the following row format:

```
<image_stem> <class_id> <confidence_score> <x1> <y1> <x2> <y2>
```

- `image_stem`: image filename without extension (e.g., `000001`)
- `class_id`: integer class index  
  - (In some tasks, class is fixed to `0` for a single target category.)
- `confidence_score`: floating point score
- `x1 y1 x2 y2`: bounding box pixel coordinates (integers)

Important behavior (as implemented in the notebook):
- **Only output images that have detections**
- **No detections → no line written for that image**

---

## 7) Reproducibility tips

To improve reproducibility:

- Fix random seeds (Python / NumPy / PyTorch)
- Record:
  - model weights name
  - image size (resolution)
  - number of epochs
  - batch size
  - augmentation settings

---

## 8) Common issues

- **`ModuleNotFoundError: ultralytics`**  
  → `pip install ultralytics`

- **CUDA out of memory (OOM)** on GPU  
  → reduce batch size, reduce image size, or use gradient accumulation

- **Paths not found on Windows**  
  → verify folder names, and use raw strings `r"..."`

---


