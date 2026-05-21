from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from traffic_counter.detection.yolo_detector import YOLODetector
from traffic_counter.evaluation.evaluator import DetectionEvaluator


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá detection bằng label YOLO format")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images", required=True, help="Thư mục ảnh validation")
    parser.add_argument("--labels", required=True, help="Thư mục label YOLO .txt")
    parser.add_argument("--output-json", default="outputs/evaluation.json")
    parser.add_argument("--output-csv", default="outputs/evaluation.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    class_names = config.get("dataset", {}).get("class_names") or config.get("classes")
    if not class_names:
        raise ValueError("Cần khai báo dataset.class_names hoặc classes trong config.yaml")

    detector = YOLODetector(
        model_path=PROJECT_ROOT / config["model"]["yolo_path"],
        device=config["model"].get("device"),
        allowed_classes=config.get("classes"),
    )

    evaluator = DetectionEvaluator(
        detector=detector,
        class_names=class_names,
        iou_threshold=config.get("evaluation", {}).get("iou_threshold", 0.5),
    )

    result = evaluator.evaluate_folder(
        images_dir=args.images,
        labels_dir=args.labels,
        output_json=args.output_json,
        output_csv=args.output_csv,
        **config.get("predict", {}),
    )

    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))
    print(f"Đã lưu JSON: {args.output_json}")
    print(f"Đã lưu CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
