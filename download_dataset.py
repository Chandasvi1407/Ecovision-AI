"""
EcoVision AI - Dataset Organizer & Remapper
Author: College AI Mini Project Team

Dataset: GARBAGE CLASSIFICATION 3 (Version 2)
Source: Roboflow Universe (CC BY 4.0)
Original Classes: ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

Target Classes (0-4):
  0: Organic   (from BIODEGRADABLE)
  1: Plastic   (from PLASTIC)
  2: Glass     (from GLASS)
  3: Paper     (from PAPER, CARDBOARD)
  4: Metal     (from METAL)
"""

import sys
import shutil
from pathlib import Path
import yaml

from utils.prepare_dataset import process_and_remap_labels, generate_data_yaml, TARGET_CLASSES
from utils.dataset_checker import verify_dataset


def organize_and_remap(source_folder: str = "GARBAGE-CLASSIFICATION-3-2"):
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir / source_folder
    dataset_dir = base_dir / "dataset"

    if not src_dir.exists():
        print(f"[Error] Source folder '{src_dir}' not found.")
        sys.exit(1)

    print("=" * 70)
    print("      EcoVision AI - Dataset Processing & Label Remapping")
    print("=" * 70)
    print(f"Reading downloaded dataset from: {src_dir.name}")

    raw_yaml_path = src_dir / "data.yaml"
    with open(raw_yaml_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    original_classes = raw_config.get("names", [])
    print(f"Original Classes ({len(original_classes)}): {original_classes}")
    print(f"License: {raw_config.get('roboflow', {}).get('license', 'CC BY 4.0')}")
    print(f"Project URL: {raw_config.get('roboflow', {}).get('url', '')}\n")

    # Ensure destination directories exist
    for split in ["train", "val", "test"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Roboflow export folder mapping: valid -> val
    split_map = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }

    print("Transferring images and labels into standard YOLO folder structure...")
    for src_split, dst_split in split_map.items():
        split_src = src_dir / src_split
        if not split_src.exists():
            continue

        src_imgs = split_src / "images"
        src_lbls = split_src / "labels"

        dst_imgs = dataset_dir / "images" / dst_split
        dst_lbls = dataset_dir / "labels" / dst_split

        # Copy images
        copied_imgs = 0
        if src_imgs.exists():
            for f in src_imgs.iterdir():
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy2(f, dst_imgs / f.name)
                    copied_imgs += 1

        # Copy labels
        copied_lbls = 0
        if src_lbls.exists():
            for f in src_lbls.iterdir():
                if f.is_file() and f.suffix.lower() == ".txt":
                    shutil.copy2(f, dst_lbls / f.name)
                    copied_lbls += 1

        print(f"  Split '{dst_split}': {copied_imgs} images, {copied_lbls} labels transferred.")

    # Execute label remapping and data leakage check
    print("\n" + "=" * 70)
    print("      Remapping Class IDs & Computing Cryptographic Hashes")
    print("=" * 70)
    print("Remapping schema:")
    print("  0: BIODEGRADABLE     -> Organic [0]")
    print("  1: CARDBOARD         -> Paper   [3]")
    print("  2: GLASS             -> Glass   [2]")
    print("  3: METAL             -> Metal   [4]")
    print("  4: PAPER             -> Paper   [3]")
    print("  5: PLASTIC           -> Plastic [1]")
    print("  *  CLOTH (if any)    -> EXCLUDED")

    stats = process_and_remap_labels(dataset_dir, original_classes)

    # Generate data.yaml for YOLO11
    generate_data_yaml(dataset_dir)

    # Verify dataset integrity and sample drawing
    samples_dir = base_dir / "runs" / "dataset_verification"
    verify_dataset(dataset_dir, samples_dir)

    print("\n" + "=" * 70)
    print("[SUCCESS] Stage 2 Complete! Dataset is verified and ready for training.")
    print("=" * 70)


if __name__ == "__main__":
    organize_and_remap()
