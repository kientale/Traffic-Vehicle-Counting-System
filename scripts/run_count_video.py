from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from traffic_counter.counting.line_counter import LineCounter
from traffic_counter.counting.region_counter import RegionCounter
from traffic_counter.detection.yolo_detector import YOLODetector
from traffic_counter.pipeline import TrafficPipeline
from traffic_counter.tracking.tracker import SimpleIoUTracker


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_enabled_detect_classes(detect_config: dict) -> list[str]:
    classes = detect_config.get("classes", {})
    if isinstance(classes, dict):
        return [
            class_name
            for class_name, class_config in classes.items()
            if class_config.get("enabled", True)
        ]
    if isinstance(classes, list):
        return classes
    return []


def build_runtime_config(config_path: str | None) -> dict:
    if config_path:
        return load_config(PROJECT_ROOT / config_path)

    detect_config = load_config(PROJECT_ROOT / "configs/detect.yaml")
    tracking_config = load_config(PROJECT_ROOT / "configs/tracking.yaml")
    counting_config = load_config(PROJECT_ROOT / "configs/counting.yaml")

    allowed_classes = counting_config.get("allowed_classes") or get_enabled_detect_classes(detect_config)
    line_config = counting_config.get("line_counting", {}).get("lines", [{}])[0]
    region_config = counting_config.get("region_counting", {}).get("regions", [{}])[0]

    return {
        "model": {
            "yolo_path": detect_config["model"]["path"],
            "device": detect_config.get("device"),
        },
        "classes": allowed_classes,
        "predict": {
            "imgsz": detect_config.get("image_size", 1280),
            "conf": detect_config.get("confidence_threshold", 0.25),
            "iou": detect_config.get("iou_threshold", 0.45),
        },
        "tracker": {
            "iou_threshold": tracking_config.get("match_thresh", 0.3),
            "max_lost": tracking_config.get("track_buffer", 10),
            "min_hits": 1,
        },
        "counting": {
            "counting_type": counting_config.get("counting_type", "line"),
            "anchor_point": counting_config.get("anchor_point", "bottom_center"),
            "line": [
                line_config.get("start_point", [100, 500]),
                line_config.get("end_point", [1000, 500]),
            ],
            "polygon": region_config.get("polygon", []),
            "count_when": region_config.get("count_when", "enter"),
            "count_once_per_track": counting_config.get("count_once_per_track", True),
        },
        "video": {
            "frame_skip": 0,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Detect + Track + Count phuong tien trong video")
    parser.add_argument("--config", default=None, help="File config tong hop. Neu bo qua se doc detect/tracking/counting yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="outputs/counted_video.mp4")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = build_runtime_config(args.config)

    detector = YOLODetector(
        model_path=PROJECT_ROOT / config["model"]["yolo_path"],
        device=config["model"].get("device"),
        allowed_classes=config.get("classes"),
    )

    tracker_cfg = config.get("tracker", {})
    tracker = SimpleIoUTracker(
        iou_threshold=tracker_cfg.get("iou_threshold", 0.3),
        max_lost=tracker_cfg.get("max_lost", 10),
        min_hits=tracker_cfg.get("min_hits", 1),
    )

    count_cfg = config.get("counting", {})
    allowed_classes = set(config.get("classes", []))
    counting_type = count_cfg.get("counting_type", "line")
    if counting_type == "region":
        counter = RegionCounter(
            polygon=count_cfg.get("polygon", []),
            allowed_classes=allowed_classes,
            anchor_point=count_cfg.get("anchor_point", "bottom_center"),
            count_once_per_track=count_cfg.get("count_once_per_track", True),
            count_when=count_cfg.get("count_when", "enter"),
        )
    else:
        counter = LineCounter(
            p1=tuple(count_cfg.get("line", [[100, 500], [1000, 500]])[0]),
            p2=tuple(count_cfg.get("line", [[100, 500], [1000, 500]])[1]),
            allowed_classes=allowed_classes,
            count_once_per_track=count_cfg.get("count_once_per_track", True),
        )

    pipeline = TrafficPipeline(
        detector=detector,
        tracker=tracker,
        counter=counter,
        predict_config=config.get("predict", {}),
    )

    result = pipeline.process_video(
        video_path=args.video,
        output_path=args.output,
        display=args.display,
        max_frames=args.max_frames,
        frame_skip=config.get("video", {}).get("frame_skip", 0),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
