from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from traffic_counter.detection.yolo_detector import YOLODetector
from traffic_counter.pipeline import TrafficPipeline


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect phương tiện trong ảnh")
    parser.add_argument("--config", default="configs/detect.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/detected.jpg")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    detector = YOLODetector(
        model_path=PROJECT_ROOT / config["model"]["path"],
        device=config["model"].get("device"),
        allowed_classes=config.get("classes"),
    )

    pipeline = TrafficPipeline(
        detector=detector,
        predict_config=config.get("predict", {}),
    )

    detections = pipeline.process_image(
        image_path=args.image,
        output_path=args.output,
        use_segmentation=False,
    )

    print(f"Đã lưu ảnh kết quả: {args.output}")
    print(f"Số object detect được: {len(detections)}")


if __name__ == "__main__":
    main()
