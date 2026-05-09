# FR-01 Implementation Guide
## Energy-Efficient Edge-Based Deep Learning Framework for Smart Agriculture

---

## What You Are Implementing

**FR-01 — Plant Disease & Pest Detection**
> The system shall detect and classify plant diseases and pests from RGB camera images using a lightweight deep learning model with ≥85% mAP on standard agricultural datasets.

**Model:** YOLOv5n (nano) — lightest YOLO variant, runs on CPU
**Dataset:** PlantVillage — 10 disease/pest classes from Kaggle
**Framework:** PyTorch + Ultralytics
**Hardware needed:** Just your laptop (CPU-only)

---

## Folder Structure After Setup

```
fr01_implementation/
├── step1_setup.sh          ← Run first (install dependencies)
├── step2_dataset.py        ← Download + prepare PlantVillage
├── step3_train.py          ← Train YOLOv5n
├── step4_evaluate.py       ← Evaluate + generate report
├── README.md               ← This file
│
├── data/
│   ├── raw/plantvillage/   ← Downloaded dataset
│   └── yolo_dataset/       ← YOLO-formatted splits
│       ├── images/train|val|test/
│       ├── labels/train|val|test/
│       └── plantvillage.yaml
│
├── runs/train/fr01_yolov5n/
│   └── weights/
│       ├── best.pt         ← Best model checkpoint
│       └── last.pt         ← Last epoch checkpoint
│
└── reports/
    ├── eda_class_distribution.png
    ├── sample_grid.png
    ├── training_summary.json
    ├── confusion_matrix.png
    ├── per_class_metrics.png
    ├── fr01_evaluation_report.txt  ← Final verdict
    └── fr01_metrics.json
```

---

## Prerequisites (Do These Before Day 1)

### 1. Python
Make sure Python 3.9+ is installed:
```bash
python --version   # should show 3.9 or higher
```

### 2. Kaggle API Credentials (one-time setup)
1. Go to https://www.kaggle.com/settings
2. Scroll to **API** section → click **Create New Token**
3. This downloads `kaggle.json`
4. Place it in:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
5. **Linux/Mac only** — set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## Day 1 — Dataset + Start Training

### Step 1: Install Dependencies (~10 minutes)

**Linux/Mac:**
```bash
cd fr01_implementation
bash step1_setup.sh
source fr01_env/bin/activate
```

**Windows:**
```cmd
cd fr01_implementation
python -m venv fr01_env
fr01_env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.2.0 kaggle Pillow matplotlib seaborn pandas numpy scikit-learn pyyaml tqdm
```

### Step 2: Download and Prepare Dataset (~15 minutes)
```bash
python step2_dataset.py
```

What this does:
- Downloads PlantVillage from Kaggle (~1.5 GB)
- Selects 10 representative disease classes
- Creates a stratified 20% subset (for CPU-feasible training)
- Splits into 70% train / 15% val / 15% test
- Converts to YOLO annotation format
- Generates EDA charts in `reports/`

**Expected output:**
```
Dataset preparation COMPLETE.
Classes  : 10
Train    : ~XXXX  |  Val: ~XXX  |  Test: ~XXX
YAML     : data/yolo_dataset/plantvillage.yaml
```

### Step 3: Start Training (run overnight)
```bash
python step3_train.py
```

**Expected console output per epoch:**
```
Epoch 1/10: loss=X.XX, box_loss=X.XX, cls_loss=X.XX
           P=0.XX, R=0.XX, mAP50=0.XX
```

**Time estimate on CPU:**
| CPU Type        | Per Epoch | 10 Epochs |
|----------------|-----------|-----------|
| Modern i7/i9   | ~1–2 hrs  | ~10–20 hrs|
| Older i5       | ~2–4 hrs  | ~20–40 hrs|
| Ryzen 5/7      | ~1–2 hrs  | ~10–20 hrs|

> **Tip:** Start training before you sleep on Day 1. It will run overnight.

---

## Day 2 — Resume Training + Evaluate

### If 10 epochs finished overnight, resume for 20 more:
```bash
# Activate environment first (if new terminal)
source fr01_env/bin/activate   # Linux/Mac
# OR
fr01_env\Scripts\activate      # Windows

python step3_train.py --resume
```

### Once training is done, run evaluation:
```bash
python step4_evaluate.py
```

What this generates:
- Confusion matrix (raw + normalised heatmaps)
- Per-class Precision / Recall / F1 bar chart
- Inference speed benchmark (ms/image on CPU)
- FR-01 pass/fail verdict in `reports/fr01_evaluation_report.txt`

---

## Reading Your Results

### FR-01 Pass Criteria:
| Criterion | Required | Where to Check |
|-----------|----------|----------------|
| mAP@0.5   | ≥ 85%    | `fr01_evaluation_report.txt` |
| Disease classes detected | ≥ 5 | `fr01_evaluation_report.txt` |

### If mAP < 85% after 30 epochs:
This is normal on a first run. Document your achieved mAP and write:
> "The baseline model achieved X% mAP@0.5 after 30 epochs on CPU hardware. Planned Phase 2 optimisation (model compression, edge deployment with TensorRT) is expected to refine performance further."

This is academically valid — many papers report progress toward a threshold, not just pass/fail.

---

## Common Issues and Fixes

| Problem | Fix |
|---------|-----|
| `kaggle: command not found` | Run `pip install kaggle` and restart terminal |
| `401 Unauthorized` from Kaggle | Check `kaggle.json` path and permissions |
| `Out of memory` during training | Reduce `batch` from 8 to 4 in `step3_train.py` |
| `No module named ultralytics` | Make sure virtual environment is activated |
| Training is very slow | Normal on CPU. Reduce `epochs` to 5 for a quick test run first. |
| Dataset folder not found | Check that `data/raw/plantvillage/` has subfolders |

---

## Files for Your Presentation

After running all 4 steps, include these in your presentation:

1. **`reports/eda_class_distribution.png`** — shows dataset analysis
2. **`reports/sample_grid.png`** — shows sample disease images
3. **`reports/confusion_matrix.png`** — model predictions vs. ground truth
4. **`reports/per_class_metrics.png`** — per-class P/R/F1 breakdown
5. **`reports/fr01_evaluation_report.txt`** — FR-01 pass/fail verdict
6. **`runs/train/fr01_yolov5n/results.png`** — training curves (auto-generated by Ultralytics)

---

## Connecting FR-01 to Your Research Documents

| Document | Connection |
|----------|-----------|
| Gap Analysis — "Absence of energy-aware design" | FR-01 is Step 1: establish baseline model before energy profiling (FR-03) |
| Gap Analysis — "Lack of edge-deployable implementations" | FR-01 uses YOLOv5n, the lightest YOLO — directly addresses this gap |
| Problem Statement — "accuracy vs energy trade-off" | FR-01 establishes the accuracy baseline for FR-05 energy-accuracy trade-off |
| Feasibility Study — Technical feasibility | FR-01 validates technical feasibility using TinyYOLO as cited in [9],[16] |

---

## What Comes Next (Phases 2–4)

| Phase | FR | What to do |
|-------|----|------------|
| Phase 2 | FR-02 | Deploy `best.pt` on Jetson Nano, measure latency |
| Phase 2 | FR-03 | Profile per-inference energy using INA219 or tegrastats |
| Phase 3 | FR-04 | Apply INT8 quantization with TensorRT or TFLite |
| Phase 3 | FR-05 | Compare mAP vs energy across base/pruned/quantized models |
| Phase 4 | FR-06 | Test model on augmented low-light/rainy/occluded images |
