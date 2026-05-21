from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from traffic_counter.evaluation.metrics import bbox_iou


@dataclass
class Track:
    track_id: int
    bbox: list[float]
    class_name: str
    confidence: float
    age: int = 0
    lost: int = 0
    hits: int = 1
    last_detection: dict = field(default_factory=dict)


class SimpleIoUTracker:
    """
    Tracker đơn giản dựa trên IoU.

    Mục tiêu: đủ dùng cho bài demo/NCKH ở mức cơ bản.
    Nếu muốn production hơn, sau này thay bằng ByteTrack hoặc BoT-SORT.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 10,
        min_hits: int = 1,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Gán track_id cho detections hiện tại.
        """
        for track in self.tracks:
            track.age += 1
            track.lost += 1

        unmatched_detection_indices = set(range(len(detections)))
        unmatched_track_indices = set(range(len(self.tracks)))
        matches: list[tuple[int, int, float]] = []

        # Tạo toàn bộ cặp IoU hợp lệ, ưu tiên IoU cao nhất.
        candidates: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self.tracks):
            for d_idx, det in enumerate(detections):
                if det.get("class_name") != track.class_name:
                    continue
                iou = bbox_iou(track.bbox, det["bbox"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, t_idx, d_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])

        for iou, t_idx, d_idx in candidates:
            if t_idx not in unmatched_track_indices or d_idx not in unmatched_detection_indices:
                continue
            matches.append((t_idx, d_idx, iou))
            unmatched_track_indices.remove(t_idx)
            unmatched_detection_indices.remove(d_idx)

        # Update track đã match.
        for t_idx, d_idx, _ in matches:
            det = detections[d_idx]
            track = self.tracks[t_idx]
            track.bbox = list(det["bbox"])
            track.class_name = det.get("class_name", track.class_name)
            track.confidence = float(det.get("confidence", track.confidence))
            track.lost = 0
            track.hits += 1
            track.last_detection = det.copy()
            det["track_id"] = track.track_id

        # Tạo track mới cho detection chưa match.
        for d_idx in unmatched_detection_indices:
            det = detections[d_idx]
            track = Track(
                track_id=self.next_id,
                bbox=list(det["bbox"]),
                class_name=det.get("class_name", "object"),
                confidence=float(det.get("confidence", 0.0)),
                last_detection=det.copy(),
            )
            self.tracks.append(track)
            det["track_id"] = track.track_id
            self.next_id += 1

        # Xóa track bị mất quá lâu.
        self.tracks = [track for track in self.tracks if track.lost <= self.max_lost]

        # Chỉ trả về detection của frame hiện tại.
        return detections
