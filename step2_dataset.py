"""
FR-01 IMPLEMENTATION — STEP 2: DATASET PREPARATION
===================================================
Downloads PlantVillage from Kaggle, performs EDA, creates a
stratified 20% subset, converts to YOLO format, and generates
the dataset YAML config file.

Usage:
    python step2_dataset.py

Prerequisites:
    - Kaggle API credentials configured (~/.kaggle/kaggle.json)
    - Virtual environment activated (fr01_env)

IMPORTANT — Kaggle API setup (one-time, do this first):
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. This downloads kaggle.json
    3. Place it at:
       - Linux/Mac: ~/.kaggle/kaggle.json
       - Windows:   C:\\Users\\<YourName>\\.kaggle\\kaggle.json
    4. chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac only)
"""

import os
import shutil
import random
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

# ── CONFIG ─────────────────────────────────────────────────
DATASET_NAME   = "abdallahalidev/plantvillage-dataset"
RAW_DIR        = Path("data/raw/plantvillage")
PROCESSED_DIR  = Path("data/processed")
YOLO_DIR       = Path("data/yolo_dataset")
REPORTS_DIR    = Path("reports")
SUBSET_RATIO   = 0.20    # Use 20% of images for CPU-feasible training
RANDOM_SEED    = 42
IMG_SIZE       = 224     # Standard for classification; YOLO will resize
# ── TARGET CLASSES (10 representative disease categories) ──
TARGET_CLASSES = [
    "Tomato___Late_blight",
    "Tomato___Early_blight",
    "Tomato___Leaf_Miner",
    "Potato___Late_blight",
    "Potato___Early_blight",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Common_rust_",
    "Pepper,_bell___Bacterial_spot",
    "Grape___Black_rot",
    "Strawberry___Leaf_scorch",
]
# ────────────────────────────────────────────────────────────

def setup_dirs():
    for d in [RAW_DIR, PROCESSED_DIR, YOLO_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

def download_dataset():
    print("[1/6] Downloading PlantVillage dataset from Kaggle...")
    print("      This is ~1.5 GB — will take a few minutes depending on connection.")
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("      Dataset folder already exists, skipping download.")
        return
    os.system(f"kaggle datasets download -d {DATASET_NAME} -p data/raw --unzip")
    print("      Download complete.")

def discover_classes():
    """Find all class folders in the raw dataset."""
    color_dir = RAW_DIR / "color"
    if not color_dir.exists():
        # Try finding the color subdirectory anywhere
        candidates = list(RAW_DIR.rglob("color"))
        if candidates:
            color_dir = candidates[0]
        else:
            # Fall back to raw directory itself
            color_dir = RAW_DIR
    return color_dir

def collect_images(base_dir: Path):
    """Walk base_dir and collect (image_path, class_name) tuples."""
    records = []
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for cls_dir in sorted(base_dir.iterdir()):
        if cls_dir.is_dir():
            cls_name = cls_dir.name
            imgs = [p for p in cls_dir.iterdir() if p.suffix in exts]
            for img_path in imgs:
                records.append({"path": str(img_path), "class": cls_name})
    return pd.DataFrame(records)

def filter_target_classes(df: pd.DataFrame):
    """Filter to target classes, or use top-10 by count if targets not found."""
    available = set(df["class"].unique())
    matched = [c for c in TARGET_CLASSES if c in available]
    if len(matched) < 5:
        print(f"      Note: Only {len(matched)} target classes found.")
        print("      Using top 10 classes by image count instead.")
        top10 = df["class"].value_counts().head(10).index.tolist()
        matched = top10
    print(f"      Using {len(matched)} classes: {matched}")
    return df[df["class"].isin(matched)].copy(), matched

def perform_eda(df: pd.DataFrame, classes: list):
    print("[2/6] Performing Exploratory Data Analysis (EDA)...")
    class_counts = df["class"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PlantVillage Dataset — EDA Report", fontsize=16, fontweight="bold")

    # Bar chart — class distribution
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
    axes[0].barh(
        [c.replace("___", "\n").replace("_", " ") for c in class_counts.index],
        class_counts.values, color=colors
    )
    axes[0].set_xlabel("Number of Images")
    axes[0].set_title("Class Distribution (Selected Classes)")
    axes[0].invert_yaxis()
    for i, v in enumerate(class_counts.values):
        axes[0].text(v + 5, i, str(v), va="center", fontsize=9)

    # Pie chart — proportion
    axes[1].pie(class_counts.values, labels=None, colors=colors,
                autopct="%1.1f%%", startangle=140, pctdistance=0.85)
    axes[1].set_title("Class Proportion")
    legend_labels = [c.replace("___", " — ").replace("_", " ") for c in class_counts.index]
    axes[1].legend(legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.3),
                   fontsize=8, ncol=2)

    plt.tight_layout()
    eda_path = REPORTS_DIR / "eda_class_distribution.png"
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      EDA chart saved: {eda_path}")

    # Save EDA summary CSV
    summary = class_counts.reset_index()
    summary.columns = ["class_name", "image_count"]
    summary["percentage"] = (summary["image_count"] / summary["image_count"].sum() * 100).round(2)
    summary.to_csv(REPORTS_DIR / "eda_summary.csv", index=False)
    print(f"      Total images (filtered): {len(df)}")
    print(f"      Classes: {len(classes)}")
    print(f"      Min per class: {class_counts.min()}  |  Max: {class_counts.max()}")

def create_stratified_subset(df: pd.DataFrame):
    print(f"[3/6] Creating stratified {int(SUBSET_RATIO*100)}% subset...")

    subsets = []

    for cls in df["class"].unique():
        cls_df = df[df["class"] == cls]
        sampled = cls_df.sample(frac=SUBSET_RATIO, random_state=RANDOM_SEED)
        subsets.append(sampled)

    subset = pd.concat(subsets, ignore_index=True)

    print(f"      Subset size: {len(subset)} images (from {len(df)} total)")
    return subset

def split_dataset(df: pd.DataFrame):
    """70% train / 15% val / 15% test, stratified."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df["class"],
                                   random_state=RANDOM_SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["class"],
                                 random_state=RANDOM_SEED)
    print(f"      Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test

def image_to_yolo_label(class_idx: int):
    """
    PlantVillage images show one plant leaf filling the frame.
    Since there are no bounding box annotations, we treat the
    entire image as a single centered bounding box (whole-image bbox).
    This is a valid approach for classification-as-detection when
    ground truth boxes aren't available, and is standard practice
    in literature using PlantVillage for detection frameworks.
    """
    # YOLO format: class_id cx cy w h (all normalized 0-1)
    return f"{class_idx} 0.5 0.5 1.0 1.0"

def copy_to_yolo_format(df: pd.DataFrame, split: str, class_to_idx: dict):
    img_dir = YOLO_DIR / "images" / split
    lbl_dir = YOLO_DIR / "labels" / split
    for _, row in df.iterrows():
        src = Path(row["path"])
        dst_img = img_dir / src.name
        # Ensure unique filename if collision
        if dst_img.exists():
            dst_img = img_dir / f"{src.stem}_{row['class'][:8]}{src.suffix}"
        shutil.copy2(src, dst_img)
        # Write YOLO label
        lbl_path = lbl_dir / (dst_img.stem + ".txt")
        lbl_path.write_text(image_to_yolo_label(class_to_idx[row["class"]]))

def build_yolo_dataset(train, val, test, classes):
    print("[4/6] Converting to YOLO format and copying files...")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    copy_to_yolo_format(train, "train", class_to_idx)
    copy_to_yolo_format(val,   "val",   class_to_idx)
    copy_to_yolo_format(test,  "test",  class_to_idx)
    print("      File copy complete.")
    return class_to_idx

def write_yaml(classes: list):
    print("[5/6] Writing dataset YAML config...")
    config = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),
        "names": classes,
    }
    yaml_path = YOLO_DIR / "plantvillage.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"      YAML saved: {yaml_path}")
    return yaml_path

def sample_grid(df: pd.DataFrame, classes: list):
    """Save a 2×5 sample image grid for documentation."""
    print("[6/6] Generating sample image grid...")
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("PlantVillage — Sample Images per Class", fontsize=14, fontweight="bold")
    for ax, cls in zip(axes.flat, classes[:10]):
        subset = df[df["class"] == cls]
        if len(subset) == 0:
            ax.axis("off")
            continue
        img_path = subset.sample(1, random_state=RANDOM_SEED).iloc[0]["path"]
        try:
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            ax.imshow(img)
            ax.set_title(cls.replace("___", "\n").replace("_", " "), fontsize=7)
            ax.axis("off")
        except Exception:
            ax.axis("off")
    plt.tight_layout()
    grid_path = REPORTS_DIR / "sample_grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"      Sample grid saved: {grid_path}")

# ── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    setup_dirs()
    download_dataset()

    base_dir = discover_classes()
    df_all = collect_images(base_dir)
    print(f"      Total images discovered: {len(df_all)} across {df_all['class'].nunique()} classes")

    df_filtered, classes = filter_target_classes(df_all)
    perform_eda(df_filtered, classes)

    subset = create_stratified_subset(df_filtered)
    train, val, test = split_dataset(subset)

    class_to_idx = build_yolo_dataset(train, val, test, classes)
    yaml_path = write_yaml(classes)
    sample_grid(df_filtered, classes)

    # Save split metadata
    meta = {
        "total_images": len(subset),
        "train": len(train), "val": len(val), "test": len(test),
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "subset_ratio": SUBSET_RATIO,
        "random_seed": RANDOM_SEED,
    }
    with open(REPORTS_DIR / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n==============================================")
    print(" Dataset preparation COMPLETE.")
    print(f" Classes  : {len(classes)}")
    print(f" Train    : {len(train)}  |  Val: {len(val)}  |  Test: {len(test)}")
    print(f" YAML     : {yaml_path}")
    print(" Next step: python step3_train.py")
    print("==============================================")
