import argparse
from pathlib import Path

import cv2


def extract_frame(video_path: str, output_path: str, frame_index: int = 0):
    video_path = Path(video_path)
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Không đọc được frame số {frame_index}")

    cv2.imwrite(str(output_path), frame)
    print(f"Đã lưu frame tại: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frame(args.video, args.output, args.frame)