from __future__ import annotations

from pathlib import Path

import cv2


def open_video(video_path: str | Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Không mở được video: {video_path}")
    return cap


def get_video_info(cap) -> dict:
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25,
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def create_video_writer(output_path: str | Path, width: int, height: int, fps: float):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được video writer: {output_path}")
    return writer
