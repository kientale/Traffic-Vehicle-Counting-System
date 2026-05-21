from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from traffic_counter.utils.frame_utils import bbox_anchor_point


@dataclass
class RegionCountEvent:
    track_id: int
    class_name: str
    event_type: str
    anchor_point: tuple[float, float]


@dataclass
class RegionCounter:
    polygon: list[tuple[int, int]]
    allowed_classes: Optional[set[str]] = None
    anchor_point: str = "bottom_center"
    count_once_per_track: bool = True
    count_when: str = "enter"
    previous_inside: dict[int, bool] = field(default_factory=dict)
    counted_track_ids: set[int] = field(default_factory=set)
    total_count: int = 0
    by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_event: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self) -> None:
        self.polygon = [tuple(map(int, point)) for point in self.polygon]
        self._polygon_np = np.array(self.polygon, dtype=np.int32)

    def _is_inside(self, point: Sequence[float]) -> bool:
        return cv2.pointPolygonTest(self._polygon_np, (float(point[0]), float(point[1])), False) >= 0

    def update(self, tracked_detections: list[dict]) -> list[RegionCountEvent]:
        events: list[RegionCountEvent] = []

        for det in tracked_detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            class_name = det.get("class_name", "object")
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            point = bbox_anchor_point(det["bbox"], self.anchor_point)
            current_inside = self._is_inside(point)

            if track_id not in self.previous_inside:
                self.previous_inside[track_id] = current_inside
                continue

            previous_inside = self.previous_inside[track_id]
            self.previous_inside[track_id] = current_inside

            event_type = None
            if not previous_inside and current_inside:
                event_type = "enter"
            elif previous_inside and not current_inside:
                event_type = "exit"

            if event_type is None:
                continue

            if self.count_when != "both" and event_type != self.count_when:
                continue

            if self.count_once_per_track and track_id in self.counted_track_ids:
                continue

            self.counted_track_ids.add(track_id)
            self.total_count += 1
            self.by_class[class_name] += 1
            self.by_event[event_type] += 1

            events.append(
                RegionCountEvent(
                    track_id=track_id,
                    class_name=class_name,
                    event_type=event_type,
                    anchor_point=point,
                )
            )

        return events

    def get_counts(self) -> dict:
        return {
            "total": self.total_count,
            "by_class": dict(self.by_class),
            "by_event": dict(self.by_event),
        }

    def reset(self) -> None:
        self.previous_inside.clear()
        self.counted_track_ids.clear()
        self.total_count = 0
        self.by_class.clear()
        self.by_event.clear()
