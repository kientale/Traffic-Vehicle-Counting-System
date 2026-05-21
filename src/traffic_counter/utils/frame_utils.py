from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2


def read_image(image_path: str | Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {image_path}")
    return image


def save_image(image_path: str | Path, image) -> None:
    output_path = Path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise RuntimeError(f"Không lưu được ảnh: {output_path}")


def resize_keep_aspect(image, max_width: Optional[int] = None, max_height: Optional[int] = None):
    h, w = image.shape[:2]
    scale = 1.0

    if max_width and w > max_width:
        scale = min(scale, max_width / w)
    if max_height and h > max_height:
        scale = min(scale, max_height / h)

    if scale >= 1.0:
        return image

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_roi(image, roi: Optional[Sequence[int]]):
    """
    roi = [x1, y1, x2, y2]
    """
    if roi is None:
        return image
    x1, y1, x2, y2 = map(int, roi)
    return image[y1:y2, x1:x2]


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_bottom_center(bbox: Sequence[float]) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def bbox_anchor_point(bbox: Sequence[float], anchor_point: str = "center") -> tuple[float, float]:
    if anchor_point == "bottom_center":
        return bbox_bottom_center(bbox)
    return bbox_center(bbox)


def bbox_area(bbox: Sequence[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
