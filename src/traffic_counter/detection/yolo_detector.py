from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import cv2
import torch


class YOLODetector:
    """
    Wrapper đơn giản cho Ultralytics YOLO.

    Output chuẩn của detector là list[dict]:
    {
        "bbox": [x1, y1, x2, y2],
        "confidence": 0.86,
        "class_id": 3,
        "class_name": "motorcycle"
    }
    """

    def __init__(
        self,
        model_path: str | Path,
        device: Optional[str] = None,
        allowed_classes: Optional[Iterable[str]] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Bạn cần cài ultralytics: pip install ultralytics"
            ) from exc

        self.model_path = str(model_path)
        self.model = YOLO(self.model_path)
        self.device = self._resolve_device(device)
        self.allowed_classes = set(allowed_classes or [])

    def _resolve_device(self, device: Optional[str]):
        if device in (None, "", "cpu"):
            return device or "cpu"
        if device == "auto":
            return 0 if torch.cuda.is_available() else "cpu"
        return device

    def detect(
        self,
        image,
        imgsz: int = 1280,
        conf: float = 0.15,
        iou: float = 0.5,
        max_det: int = 1000,
        verbose: bool = False,
    ) -> list[dict]:
        """
        Detect trên ảnh BGR hoặc đường dẫn ảnh.
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise FileNotFoundError(f"Không đọc được ảnh: {image}")

        results = self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=self.device,
            verbose=verbose,
        )

        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[dict] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes, scores, classes):
            class_name = names.get(int(cls_id), str(cls_id))

            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = box.tolist()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score),
                    "class_id": int(cls_id),
                    "class_name": class_name,
                }
            )

        return detections
