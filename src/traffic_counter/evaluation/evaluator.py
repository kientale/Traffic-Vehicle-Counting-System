from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

import cv2

from traffic_counter.evaluation.metrics import match_detections, precision_recall_f1


def load_yolo_labels(label_path: str | Path, image_width: int, image_height: int, class_names: Sequence[str]) -> list[dict]:
    """
    Đọc label YOLO format:
    class_id x_center y_center width height, tất cả đã normalize từ 0-1.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])

            x1 = (xc - w / 2) * image_width
            y1 = (yc - h / 2) * image_height
            x2 = (xc + w / 2) * image_width
            y2 = (yc + h / 2) * image_height

            class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            labels.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

    return labels


class DetectionEvaluator:
    def __init__(
        self,
        detector,
        class_names: Sequence[str],
        iou_threshold: float = 0.5,
    ) -> None:
        self.detector = detector
        self.class_names = list(class_names)
        self.iou_threshold = iou_threshold

    def evaluate_image(
        self,
        image_path: str | Path,
        label_path: str | Path,
        imgsz: int = 1280,
        conf: float = 0.15,
        iou: float = 0.5,
        max_det: int = 1000,
    ) -> dict:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")

        h, w = image.shape[:2]
        ground_truths = load_yolo_labels(label_path, w, h, self.class_names)
        predictions = self.detector.detect(image, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det)

        result = match_detections(
            predictions=predictions,
            ground_truths=ground_truths,
            iou_threshold=self.iou_threshold,
            class_aware=True,
        )
        summary = precision_recall_f1(result["tp"], result["fp"], result["fn"])

        return {
            "image": str(image_path),
            "label": str(label_path),
            "num_predictions": len(predictions),
            "num_ground_truths": len(ground_truths),
            **result,
            **summary,
        }

    def evaluate_folder(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        output_json: str | Path | None = None,
        output_csv: str | Path | None = None,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
        **predict_kwargs,
    ) -> dict:
        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        image_paths = sorted(
            p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions
        )

        per_image = []
        total_tp = total_fp = total_fn = 0

        for image_path in image_paths:
            label_path = labels_dir / f"{image_path.stem}.txt"
            result = self.evaluate_image(image_path, label_path, **predict_kwargs)
            per_image.append(result)
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]

        overall = {
            "num_images": len(image_paths),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            **precision_recall_f1(total_tp, total_fp, total_fn),
        }

        final_result = {
            "overall": overall,
            "per_image": per_image,
        }

        if output_json:
            output_json = Path(output_json)
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as file:
                json.dump(final_result, file, ensure_ascii=False, indent=2)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "image",
                        "num_predictions",
                        "num_ground_truths",
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "f1",
                    ],
                )
                writer.writeheader()
                for item in per_image:
                    writer.writerow({key: item.get(key) for key in writer.fieldnames})

        return final_result
