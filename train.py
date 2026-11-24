"""
EcoVision AI - YOLO11n Model Training & Evaluation Pipeline
Author: College AI Mini Project Team

Hardware Profile:
- Optimized for Intel Multi-Core CPU (14 logical cores)
- Safe batch size (16) and memory footprint for 32 GB RAM

Usage:
  python train.py
  python train.py --epochs 25 --imgsz 416 --batch 16
"""

import argparse
import sys
import shutil
import time
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
except ImportError:
    print("[Error] Ultralytics is not installed. Please run: pip install ultralytics")
    sys.exit(1)


def train_waste_model(
    data_yaml_path: str = "dataset/data.yaml",
    epochs: int = 25,
    imgsz: int = 416,
    batch: int = 16,
    workers: int = 4,
    pretrained_weights: str = "yolo11n.pt"
):
    print("=" * 70)
    print("     EcoVision AI - Genuine YOLO11n Model Training Pipeline")
    print("=" * 70)

    data_yaml = Path(data_yaml_path).resolve()
    if not data_yaml.exists():
        print(f"[Error] Dataset configuration not found at '{data_yaml}'.")
        print("Please ensure you have run 'utils/prepare_dataset.py' first.")
        sys.exit(1)

    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    target_classes = config.get("names", [])
    print(f"Target Categories ({len(target_classes)}): {target_classes}")
    print(f"Training Hyperparameters:")
    print(f"  Base Model Architecture : YOLO11n (Nano)")
    print(f"  Input Image Resolution  : {imgsz} x {imgsz}")
    print(f"  Batch Size              : {batch}")
    print(f"  Training Epochs         : {epochs}")
    print(f"  Device                  : CPU (Intel Multi-Core)")
    print(f"  DataLoader Workers      : {workers}")
    print("=" * 70)

    # Initialize YOLO11n model
    print(f"\n[1/4] Initializing YOLO11n from pre-trained weights '{pretrained_weights}'...")
    model = YOLO(pretrained_weights)

    # Train model
    start_time = time.time()
    print("\n[2/4] Commencing model training on your CPU...")
    print("Ultralytics will display real-time loss and epoch progress below:\n")

    train_results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=workers,
        project="runs/detect",
        name="waste_train",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    total_training_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[Training Complete] Total elapsed time: {total_training_time / 60:.2f} minutes")
    print("=" * 70)

    # Export best weights and plots to model/ directory
    print("\n[3/4] Exporting model weights and genuine validation artifacts...")
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)

    train_dir = Path("runs/detect/waste_train")
    best_weights = train_dir / "weights" / "best.pt"
    cm_plot = train_dir / "confusion_matrix.png"
    results_plot = train_dir / "results.png"

    dest_weights = model_dir / "waste_yolo11n.pt"
    if best_weights.exists():
        shutil.copy2(best_weights, dest_weights)
        print(f"  -> Saved best model weights to: {dest_weights.resolve()}")
    else:
        print(f"  [Warning] best.pt not found at {best_weights}. Checking last.pt...")
        last_weights = train_dir / "weights" / "last.pt"
        if last_weights.exists():
            shutil.copy2(last_weights, dest_weights)
            print(f"  -> Saved last model weights to: {dest_weights.resolve()}")

    if cm_plot.exists():
        shutil.copy2(cm_plot, model_dir / "confusion_matrix.png")
        print(f"  -> Saved Confusion Matrix to: {(model_dir / 'confusion_matrix.png').resolve()}")

    if results_plot.exists():
        shutil.copy2(results_plot, model_dir / "results.png")
        print(f"  -> Saved Training Metric Curves to: {(model_dir / 'results.png').resolve()}")

    # Run independent evaluation on held-out test set
    print("\n[4/4] Running independent evaluation on held-out TEST set...")
    val_model = YOLO(str(dest_weights))
    val_results = val_model.val(
        data=str(data_yaml),
        split="test",
        device="cpu",
        imgsz=imgsz,
        batch=batch,
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("          FINAL EVALUATION METRICS ON UNSEEN TEST SET")
    print("=" * 70)
    metrics = val_results.results_dict
    print(f"  mAP@50     (IoU 0.50)      : {metrics.get('metrics/mAP50(B)', 0.0) * 100:.2f}%")
    print(f"  mAP@50-95  (IoU 0.50-0.95) : {metrics.get('metrics/mAP50-95(B)', 0.0) * 100:.2f}%")
    print(f"  Precision  (P)             : {metrics.get('metrics/precision(B)', 0.0) * 100:.2f}%")
    print(f"  Recall     (R)             : {metrics.get('metrics/recall(B)', 0.0) * 100:.2f}%")
    print("=" * 70)
    print("[SUCCESS] All artifacts ready for deployment in EcoVision AI dashboard!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11n Waste Classifier")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image resolution (e.g. 416 or 640)")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Pretrained weights")

    args = parser.parse_args()
    train_waste_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        pretrained_weights=args.weights
    )
