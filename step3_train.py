"""
FR-01 IMPLEMENTATION — STEP 3: MODEL TRAINING
==============================================
Trains YOLOv5n (nano) on the prepared PlantVillage subset.
Designed for CPU-only machines — uses conservative batch size
and img size settings to stay within RAM limits.

Usage:
    python step3_train.py

Expected time on CPU-only laptop:
    ~2–4 hours per epoch depending on CPU cores and RAM.
    Default: 10 epochs = feasible in Day 1 overnight run.
    Extend to 30 epochs on Day 2 using --resume flag.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────
YAML_PATH     = Path("data/yolo_dataset/plantvillage.yaml")
REPORTS_DIR   = Path("reports")
RUNS_DIR      = Path("runs/train")

# CPU-optimised hyperparameters
TRAIN_CONFIG = {
    "model"     : "yolov5nu.pt",   # YOLOv5 nano — smallest, fastest on CPU
    "data"      : str(YAML_PATH),
    "epochs"    : 10,              # Day 1: 10 epochs. Day 2: resume for 20 more.
    "imgsz"     : 320,             # Smaller than default 640 — saves ~4x compute on CPU
    "batch"     : 8,               # Low batch size for CPU RAM safety
    "workers"   : 2,               # CPU dataloader workers
    "device"    : "cpu",           # Explicit CPU
    "project"   : str(RUNS_DIR),
    "name"      : "fr01_yolov5n",
    "exist_ok"  : True,
    "patience"  : 20,              # Early stopping patience
    "cache"     : False,           # Don't cache images (saves RAM on laptop)
    "optimizer" : "Adam",
    "lr0"       : 0.001,           # Initial learning rate
    "lrf"       : 0.01,            # Final lr = lr0 * lrf
    "momentum"  : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "hsv_h"     : 0.015,           # Hue augmentation
    "hsv_s"     : 0.7,             # Saturation augmentation
    "hsv_v"     : 0.4,             # Brightness augmentation
    "flipud"    : 0.0,
    "fliplr"    : 0.5,
    "translate" : 0.1,
    "scale"     : 0.5,
    "save_period": 5,              # Save checkpoint every 5 epochs
    "verbose"   : True,
}
# ────────────────────────────────────────────────────────────

def check_prerequisites():
    if not YAML_PATH.exists():
        print("[ERROR] Dataset YAML not found.")
        print("        Run step2_dataset.py first.")
        sys.exit(1)

    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    print(f"       Dataset: {cfg['nc']} classes")
    print(f"       Classes: {cfg['names']}")

    # Check image counts
    for split in ["train", "val", "test"]:
        img_dir = Path(cfg["path"]) / "images" / split
        count = len(list(img_dir.glob("*")))
        print(f"       {split:6s}: {count} images")

    print(f"       CPU threads available: {torch.get_num_threads()}")
    return cfg

def estimate_time(n_train: int, epochs: int):
    """Rough CPU time estimate: ~0.5s per image per epoch at imgsz=320."""
    secs_per_epoch = n_train * 0.5
    total_mins = (secs_per_epoch * epochs) / 60
    print(f"\n  ⏱  Estimated training time: ~{total_mins:.0f} minutes ({total_mins/60:.1f} hours)")
    print(f"     Based on {n_train} training images × {epochs} epochs")
    print(f"     Actual time varies with CPU speed — faster machines will be quicker.\n")

def train(resume: bool = False):
    print("==============================================")
    print(" FR-01: Training YOLOv5n on PlantVillage")
    print("==============================================\n")

    cfg = check_prerequisites()

    # Count training images
    train_dir = Path(cfg["path"]) / "images" / "train"
    n_train = len(list(train_dir.glob("*")))
    estimate_time(n_train, TRAIN_CONFIG["epochs"])

    # Load model
    if resume:
        last_ckpt = RUNS_DIR / "fr01_yolov5n" / "weights" / "last.pt"
        if not last_ckpt.exists():
            print("[ERROR] No checkpoint found to resume from.")
            print(f"        Expected: {last_ckpt}")
            sys.exit(1)
        print(f"  Resuming from: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        # Extend training by 20 more epochs
        config = {**TRAIN_CONFIG, "epochs": TRAIN_CONFIG["epochs"] + 20, "resume": True}
    else:
        print(f"  Model: {TRAIN_CONFIG['model']} (YOLOv5 nano)")
        model = YOLO(TRAIN_CONFIG["model"])
        config = TRAIN_CONFIG.copy()

    print("  Starting training...\n")
    start_time = time.time()

    results = model.train(**config)

    elapsed = time.time() - start_time
    elapsed_mins = elapsed / 60

    print(f"\n  Training complete in {elapsed_mins:.1f} minutes.")

    # Save training summary
    summary = {
        "model"         : TRAIN_CONFIG["model"],
        "epochs_run"    : TRAIN_CONFIG["epochs"],
        "imgsz"         : TRAIN_CONFIG["imgsz"],
        "batch_size"    : TRAIN_CONFIG["batch"],
        "device"        : "cpu",
        "training_time_minutes": round(elapsed_mins, 1),
        "best_weights"  : str(RUNS_DIR / "fr01_yolov5n" / "weights" / "best.pt"),
    }
    with open(REPORTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Best weights: {summary['best_weights']}")
    print("  Next step: python step4_evaluate.py")

# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FR-01 YOLOv5n model")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint (use on Day 2)")
    args = parser.parse_args()
    train(resume=args.resume)
