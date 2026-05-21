from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from traffic_counter.utils.frame_utils import bbox_center


def _signed_side(point: tuple[float, float], p1: Sequence[float], p2: Sequence[float]) -> float:
    """
    Tính điểm nằm phía nào so với đường thẳng p1 -> p2.
    Giá trị đổi dấu khi object đi qua line.
    """
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


@dataclass
class CountEvent:
    track_id: int
    class_name: str
    direction: str
    center: tuple[float, float]


@dataclass
class LineCounter:
    p1: tuple[int, int]
    p2: tuple[int, int]
    allowed_classes: Optional[set[str]] = None
    count_once_per_track: bool = True
    previous_side: dict[int, float] = field(default_factory=dict)
    counted_track_ids: set[int] = field(default_factory=set)
    total_count: int = 0
    by_class: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_direction: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, tracked_detections: list[dict]) -> list[CountEvent]:
        events: list[CountEvent] = []

        for det in tracked_detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            class_name = det.get("class_name", "object")
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            center = bbox_center(det["bbox"])
            current_side = _signed_side(center, self.p1, self.p2)

            if track_id not in self.previous_side:
                self.previous_side[track_id] = current_side
                continue

            previous = self.previous_side[track_id]
            self.previous_side[track_id] = current_side

            # Nếu chưa đổi phía thì chưa qua line.
            if previous == 0 or current_side == 0 or previous * current_side > 0:
                continue

            if self.count_once_per_track and track_id in self.counted_track_ids:
                continue

            direction = "A_to_B" if previous < current_side else "B_to_A"
            self.counted_track_ids.add(track_id)
            self.total_count += 1
            self.by_class[class_name] += 1
            self.by_direction[direction] += 1

            events.append(
                CountEvent(
                    track_id=track_id,
                    class_name=class_name,
                    direction=direction,
                    center=center,
                )
            )

        return events

    def get_counts(self) -> dict:
        return {
            "total": self.total_count,
            "by_class": dict(self.by_class),
            "by_direction": dict(self.by_direction),
        }

    def reset(self) -> None:
        self.previous_side.clear()
        self.counted_track_ids.clear()
        self.total_count = 0
        self.by_class.clear()
        self.by_direction.clear()
