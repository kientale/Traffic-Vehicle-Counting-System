from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def _to_int_box(bbox):
    x1, y1, x2, y2 = bbox
    return int(x1), int(y1), int(x2), int(y2)


def draw_detections(
    image,
    detections: list[dict],
    show_conf: bool = True,
    show_track_id: bool = True,
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (0, 0, 0),
):
    out = image.copy()

    for det in detections:
        x1, y1, x2, y2 = _to_int_box(det["bbox"])
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

        label_parts = []
        if show_track_id and det.get("track_id") is not None:
            label_parts.append(f"ID {det['track_id']}")
        label_parts.append(det.get("class_name", "object"))
        if show_conf and det.get("confidence") is not None:
            label_parts.append(f"{det['confidence']:.2f}")

        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 6, y_text + th + 6), box_color, -1)
        cv2.putText(out, label, (x1 + 3, y_text + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)

    return out


def draw_masks(image, detections: list[dict], alpha: float = 0.35):
    out = image.copy()
    overlay = image.copy()

    for det in detections:
        mask = det.get("mask")
        if mask is None:
            continue
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0, 255, 255)

    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def draw_counting_line(
    image,
    p1: Sequence[int],
    p2: Sequence[int],
    color: tuple[int, int, int] = (255, 0, 0),
):
    out = image.copy()
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.line(out, p1, p2, color, 3)
    cv2.circle(out, p1, 5, color, -1)
    cv2.circle(out, p2, 5, color, -1)
    return out


def draw_counting_region(
    image,
    polygon: Sequence[Sequence[int]],
    color: tuple[int, int, int] = (255, 128, 0),
):
    out = image.copy()
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    overlay = out.copy()
    cv2.fillPoly(overlay, [pts], color)
    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=3)
    return out


def draw_count_panel(image, counts: dict, origin: tuple[int, int] = (20, 35)):
    out = image.copy()
    x, y = origin
    lines = [f"Total: {counts.get('total', 0)}"]

    by_class = counts.get("by_class", {})
    for class_name, value in by_class.items():
        lines.append(f"{class_name}: {value}")

    by_direction = counts.get("by_direction", {})
    for direction, value in by_direction.items():
        lines.append(f"{direction}: {value}")

    by_event = counts.get("by_event", {})
    for event_type, value in by_event.items():
        lines.append(f"{event_type}: {value}")

    width = 280
    height = 35 + 28 * len(lines)
    cv2.rectangle(out, (x - 10, y - 25), (x + width, y + height), (0, 0, 0), -1)

    for idx, text in enumerate(lines):
        yy = y + idx * 28
        cv2.putText(out, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return out
