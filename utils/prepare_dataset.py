"""
EcoVision AI - Dataset Preparation, Remapping & Integrity Verifier
Author: College AI Mini Project Team

Functions:
1. Remaps labels from original waste categories to the 5 target categories:
     0: Organic (from BIODEGRADABLE, organic, food)
     1: Plastic (from PLASTIC)
     2: Glass (from GLASS)
     3: Paper (from PAPER, CARDBOARD)
     4: Metal (from METAL)
2. Excludes CLOTH annotations and prunes empty annotation files.
3. Calculates SHA-256 hashes of all images to ensure 0 data leakage between splits.
4. Generates data.yaml for YOLO11 training.
5. Prints exact, genuine image and bounding-box counts.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any
import yaml

# Canonical Project Classes (Strict 0-4 ordering)
TARGET_CLASSES = ["Organic", "Plastic", "Glass", "Paper", "Metal"]

# Mapping dictionary for known labels (lowercased)
LABEL_NAME_TO_TARGET = {
    # Biodegradable / Organic -> Class 0
    "biodegradable": 0,
    "organic": 0,
    "food": 0,
    "food waste": 0,
    "leaves": 0,

    # Plastic -> Class 1
    "plastic": 1,
    "plastic bottle": 1,
    "plastic bag": 1,
    "plastic cup": 1,

    # Glass -> Class 2
    "glass": 2,
    "glass bottle": 2,
    "glass jar": 2,

    # Paper & Cardboard -> Class 3
    "paper": 3,
    "cardboard": 3,
    "paper cup": 3,
    "carton": 3,

    # Metal -> Class 4
    "metal": 4,
    "can": 4,
    "tin": 4,
    "aluminium": 4,
    "aluminum": 4,
}

# Explicitly excluded classes
EXCLUDED_CLASSES = {"cloth", "clothes", "clothing", "textile"}


def calculate_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file to check for exact duplicates."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def process_and_remap_labels(
    dataset_dir: Path,
    original_classes: List[str]
) -> Dict[str, Any]:
    """
    Process YOLO labels across train, val, and test splits:
    - Remap IDs to 0:Organic, 1:Plastic, 2:Glass, 3:Paper, 4:Metal
    - Filter out CLOTH annotations
    - Return exact counts and verify data leakage
    """
    stats = {
        "splits": {},
        "target_box_counts": {name: 0 for name in TARGET_CLASSES},
        "dropped_cloth_boxes": 0,
        "pruned_empty_images": 0,
        "leakage_detected": False,
        "duplicate_hash_count": 0
    }

    # Track file hashes per split for leakage detection
    split_hashes: Dict[str, Dict[str, Path]] = {"train": {}, "val": {}, "test": {}}

    # Build mapping from original integer class_id to target class_id (0..4) or -1 (excluded)
    id_mapping: Dict[int, int] = {}
    for orig_id, raw_name in enumerate(original_classes):
        clean_name = raw_name.strip().lower()
        if clean_name in EXCLUDED_CLASSES:
            id_mapping[orig_id] = -1  # Exclude
        elif clean_name in LABEL_NAME_TO_TARGET:
            id_mapping[orig_id] = LABEL_NAME_TO_TARGET[clean_name]
        else:
            print(f"[Warning] Unknown class '{raw_name}' (ID: {orig_id}). Marked for exclusion.")
            id_mapping[orig_id] = -1

    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        split_image_count = 0
        split_box_count = 0

        # Scan images and calculate hashes
        for img_path in img_dir.glob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            h = calculate_file_hash(img_path)
            split_hashes[split][h] = img_path
            split_image_count += 1

            # Corresponding label file
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Read lines and remap
            new_lines = []
            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        orig_cls_id = int(parts[0])
                        mapped_cls_id = id_mapping.get(orig_cls_id, -1)
                        if mapped_cls_id == -1:
                            stats["dropped_cloth_boxes"] += 1
                            continue

                        # Update class index
                        target_name = TARGET_CLASSES[mapped_cls_id]
                        stats["target_box_counts"][target_name] += 1
                        split_box_count += 1

                        new_lines.append(f"{mapped_cls_id} {' '.join(parts[1:])}\n")
                    except ValueError:
                        continue

            # Write back remapped annotations
            if new_lines:
                with open(lbl_path, "w", encoding="utf-8") as lf:
                    lf.writelines(new_lines)
            else:
                # Annotation became empty after dropping excluded classes
                lbl_path.unlink(missing_ok=True)
                img_path.unlink(missing_ok=True)
                stats["pruned_empty_images"] += 1
                split_image_count -= 1

        stats["splits"][split] = {
            "images": split_image_count,
            "boxes": split_box_count
        }

    # Cross-Split Data Leakage Check
    train_h = set(split_hashes["train"].keys())
    val_h = set(split_hashes["val"].keys())
    test_h = set(split_hashes["test"].keys())

    train_val_leak = train_h.intersection(val_h)
    train_test_leak = train_h.intersection(test_h)
    val_test_leak = val_h.intersection(test_h)

    total_leaks = len(train_val_leak) + len(train_test_leak) + len(val_test_leak)
    if total_leaks > 0:
        stats["leakage_detected"] = True
        stats["duplicate_hash_count"] = total_leaks
        print(f"[ALERT] Data leakage detected! Found {total_leaks} identical image hashes across splits.")
    else:
        print("[Security Check] Zero data leakage verified across train, val, and test splits!")

    return stats


def generate_data_yaml(dataset_dir: Path) -> Path:
    """Generate YOLO data.yaml configuration."""
    yaml_path = dataset_dir / "data.yaml"
    # YOLO paths relative to data.yaml or absolute
    content = {
        "path": str(dataset_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False)
    print(f"[Config] Generated {yaml_path}")
    return yaml_path
