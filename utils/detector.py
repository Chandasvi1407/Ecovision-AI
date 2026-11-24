"""
EcoVision AI - YOLO Object Detection Engine
Author: College AI Mini Project Team

Purpose:
Encapsulates YOLO11 model loading, image inference, Non-Maximum Suppression (NMS),
and HUD bounding-box annotation for webcam frames and uploaded images.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from utils.disposal_guide import get_category_info, WASTE_CATEGORIES


class WasteDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.40, iou_threshold: float = 0.45):
        """
        Initialize the YOLO Waste Detector.
        :param model_path: Path to trained .pt weights file.
        :param conf_threshold: Minimum confidence score (0.0 to 1.0).
        :param iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"[Error] Model weights not found at '{self.model_path}'.\n"
                f"Please ensure the model has been trained by running 'train.py' "
                f"or place 'waste_yolo11n.pt' inside the 'model/' directory."
            )

        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Please run:\n"
                "  pip install ultralytics"
            )

        # Load YOLO model
        print(f"[Detector] Loading YOLO model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        print("[Detector] Model loaded successfully.")

    def predict(
        self,
        image_input: Any,
        draw_hud: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run inference on an image input (NumPy array BGR/RGB or PIL Image).
        :param image_input: np.ndarray or PIL.Image
        :param draw_hud: Whether to draw bounding boxes and labels on returned image
        :return: (annotated_bgr_image, list_of_detection_dicts)
        """
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        # Convert PIL to OpenCV BGR if necessary
        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        # Run inference on CPU
        results = self.model.predict(
            source=img_bgr,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device="cpu",
            verbose=False
        )

        detections: List[Dict[str, Any]] = []
        annotated_img = img_bgr.copy()

        if not results or len(results) == 0:
            return annotated_img, detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return annotated_img, detections

        ih, iw = img_bgr.shape[:2]

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            x1, y1, x2, y2 = map(int, xyxy)

            # Clamp coordinates to image boundaries
            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(0, min(iw - 1, x2))
            y2 = max(0, min(ih - 1, y2))

            category_info = get_category_info(cls_id)
            if category_info is None:
                raw_name = self.model.names.get(cls_id, f"Class_{cls_id}") if hasattr(self.model, "names") else f"Class_{cls_id}"
                category_info = {
                    "id": cls_id,
                    "name": raw_name,
                    "icon": "📦",
                    "color_hex": "#607D8B",
                    "color_bgr": (139, 125, 96),
                    "bin_type": "General Waste / Unclassified",
                    "disposal_suggestion": "Consult local disposal guidelines for non-standard waste items.",
                    "handling_tips": ["Dispose in general non-recyclable waste if material cannot be identified."],
                    "environmental_impact": "Proper disposal prevents contamination of recycling streams."
                }

            class_name = category_info["name"]
            color_bgr = category_info["color_bgr"]

            det_entry = {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "box": [x1, y1, x2, y2],
                "info": category_info
            }
            detections.append(det_entry)

            if draw_hud:
                # 1. Draw Bounding Box with rounded-corner appearance
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 2)

                # 2. Draw Top Label Tag
                label_text = f"{class_name} {conf * 100:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Label background rectangle
                tag_y1 = max(0, y1 - th - 8)
                tag_y2 = y1
                tag_x2 = min(iw, x1 + tw + 10)

                cv2.rectangle(
                    annotated_img,
                    (x1, tag_y1),
                    (tag_x2, tag_y2),
                    color_bgr,
                    -1  # Filled
                )

                # Label white text
                cv2.putText(
                    annotated_img,
                    label_text,
                    (x1 + 5, tag_y2 - 4),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )

        return annotated_img, detections
