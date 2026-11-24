"""
EcoVision AI - Standalone Image Inference Script
Author: College AI Mini Project Team

Purpose:
Runs offline detection on an image file or a directory of test images.
Draws bounding boxes, prints confidence scores, and displays waste disposal advice.

Usage:
  python detect.py --source dataset/images/test/sample1.jpg
  python detect.py --source dataset/images/test --conf 0.40
"""

import argparse
import sys
from pathlib import Path
import cv2
from PIL import Image

from utils.detector import WasteDetector
from utils.disposal_guide import get_category_info, get_disclaimer


def run_detection(
    model_path: str,
    source_path: str,
    conf_threshold: float = 0.40,
    output_dir: str = "runs/detect/predictions"
):
    print("=" * 65)
    print("       EcoVision AI - Offline Waste Detection Engine")
    print("=" * 65)

    source = Path(source_path)
    if not source.exists():
        print(f"[Error] Source path '{source}' does not exist.")
        sys.exit(1)

    # Initialize detector
    detector = WasteDetector(model_path=model_path, conf_threshold=conf_threshold)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect images
    if source.is_file():
        image_files = [source]
    else:
        image_files = (
            list(source.glob("*.jpg")) +
            list(source.glob("*.jpeg")) +
            list(source.glob("*.png"))
        )

    if not image_files:
        print(f"[Warning] No image files (.jpg, .jpeg, .png) found at {source}")
        return

    print(f"Processing {len(image_files)} image(s)...")
    print(f"Results will be saved to: {out_path.resolve()}\n")

    for idx, img_p in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_p.name}")
        frame_bgr = cv2.imread(str(img_p))
        if frame_bgr is None:
            print(f"  [Warning] Failed to read {img_p}. Skipping.")
            continue

        annotated_img, detections = detector.predict(frame_bgr, draw_hud=True)

        if not detections:
            print("  No waste objects detected with confidence >= {:.0f}%".format(conf_threshold * 100))
        else:
            print(f"  Detected {len(detections)} object(s):")
            for det in detections:
                cname = det["class_name"]
                conf = det["confidence"] * 100.0
                info = det["info"]
                print(f"    - {info['icon']} {cname} ({conf:.1f}%)")
                print(f"      Target Bin: {info['bin_type']}")
                print(f"      Disposal: {info['disposal_suggestion']}")

        # Save annotated image
        save_file = out_path / f"detected_{img_p.name}"
        cv2.imwrite(str(save_file), annotated_img)
        print(f"  Saved detection image to: {save_file.name}\n")

    print("=" * 65)
    print(f"[Complete] All predictions saved to {out_path.resolve()}")
    print(get_disclaimer())
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EcoVision AI Offline Detector")
    parser.add_argument(
        "--model",
        type=str,
        default="model/waste_yolo11n.pt",
        help="Path to trained YOLO weights (.pt)"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image file or a directory of test images"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold (0.0 to 1.0, default 0.40)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect/predictions",
        help="Directory to save output annotated images"
    )

    args = parser.parse_args()
    run_detection(
        model_path=args.model,
        source_path=args.source,
        conf_threshold=args.conf,
        output_dir=args.output
    )
