"""
EcoVision AI - Dataset Verification & Sample Visualizer
Author: College AI Mini Project Team

Functions:
- Inspects dataset/images and dataset/labels for train, val, and test splits.
- Validates bounding-box coordinates (must be normalized between 0.0 and 1.0).
- Confirms class indices are strictly 0, 1, 2, 3, or 4.
- Generates sample annotated images with bounding boxes in runs/dataset_verification/
  so you can visually inspect the ground truth annotations.
"""

import sys
from pathlib import Path
from typing import Dict, List
import yaml
import cv2

TARGET_CLASSES = ["Organic", "Plastic", "Glass", "Paper", "Metal"]
CLASS_COLORS_BGR = {
    0: (50, 125, 46),    # Organic (Green)
    1: (209, 136, 2),    # Plastic (Blue)
    2: (123, 137, 0),    # Glass (Teal)
    3: (0, 124, 245),    # Paper (Orange)
    4: (156, 144, 120),  # Metal (Slate)
}


def verify_dataset(dataset_dir: Path, output_samples_dir: Path = None) -> bool:
    print("=" * 65)
    print("      EcoVision AI - Dataset Integrity & Annotation Checker")
    print("=" * 65)

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"[Error] data.yaml not found at {data_yaml}")
        return False

    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config: {config.get('names')}")
    print(f"Number of classes: {config.get('nc')}\n")

    splits = ["train", "val", "test"]
    overall_valid = True
    class_counts = {i: 0 for i in range(5)}
    total_images = 0
    total_boxes = 0

    if output_samples_dir:
        output_samples_dir.mkdir(parents=True, exist_ok=True)

    sample_drawn_count = 0

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        if not img_dir.exists():
            print(f"[Warning] Image directory for split '{split}' does not exist: {img_dir}")
            continue

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpeg"))
        split_boxes = 0
        split_invalid_boxes = 0

        for img_path in images:
            total_images += 1
            lbl_path = lbl_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                continue

            # Read annotations
            img_bgr = None
            if output_samples_dir and sample_drawn_count < 15:
                img_bgr = cv2.imread(str(img_path))

            with open(lbl_path, "r", encoding="utf-8") as lf:
                for line_idx, line in enumerate(lf):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # Validate ranges
                        if cls_id not in class_counts:
                            print(f"[Invalid Class] Split {split}, file {lbl_path.name}: class ID {cls_id} out of bounds (0-4).")
                            split_invalid_boxes += 1
                            overall_valid = False
                            continue

                        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            print(f"[Coordinate Error] Split {split}, file {lbl_path.name}: box coordinates not normalized (0-1).")
                            split_invalid_boxes += 1
                            overall_valid = False
                            continue

                        class_counts[cls_id] += 1
                        split_boxes += 1
                        total_boxes += 1

                        # Draw sample verification box
                        if img_bgr is not None:
                            ih, iw = img_bgr.shape[:2]
                            x1 = int((xc - w / 2) * iw)
                            y1 = int((yc - h / 2) * ih)
                            x2 = int((xc + w / 2) * iw)
                            y2 = int((yc + h / 2) * ih)

                            color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
                            label = TARGET_CLASSES[cls_id]

                            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(
                                img_bgr,
                                label,
                                (x1, max(y1 - 6, 15)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                    except ValueError:
                        split_invalid_boxes += 1
                        overall_valid = False

            if img_bgr is not None and split_boxes > 0 and sample_drawn_count < 15:
                sample_out = output_samples_dir / f"sample_{split}_{sample_drawn_count + 1}.jpg"
                cv2.imwrite(str(sample_out), img_bgr)
                sample_drawn_count += 1

        print(f"  Split: {split:<6} | Images: {len(images):<5} | Bounding Boxes: {split_boxes:<5} | Invalid: {split_invalid_boxes}")

    print("\n" + "-" * 65)
    print("  Class Distribution Across Entire Dataset:")
    print("-" * 65)
    for cls_id in range(5):
        name = TARGET_CLASSES[cls_id]
        count = class_counts[cls_id]
        pct = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"  [{cls_id}] {name:<10}: {count:>5} boxes ({pct:>5.1f}%)")

    print("-" * 65)
    print(f"  Total Images: {total_images} | Total Boxes: {total_boxes}")

    if overall_valid and total_boxes > 0:
        print("\n[SUCCESS] Dataset integrity check PASSED! Ready for YOLO11n training.")
    else:
        print("\n[FAILED] Dataset contains errors or missing annotations.")

    return overall_valid


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    d_dir = base_dir / "dataset"
    out_dir = base_dir / "runs" / "dataset_verification"
    verify_dataset(d_dir, out_dir)
