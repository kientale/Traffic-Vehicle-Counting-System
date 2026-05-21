from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2

from traffic_counter.counting.line_counter import LineCounter
from traffic_counter.counting.region_counter import RegionCounter
from traffic_counter.tracking.tracker import SimpleIoUTracker
from traffic_counter.utils.frame_utils import read_image, save_image
from traffic_counter.utils.video_utils import create_video_writer, get_video_info, open_video
from traffic_counter.utils.visualization import (
    draw_count_panel,
    draw_counting_line,
    draw_counting_region,
    draw_detections,
    draw_masks,
)


class TrafficPipeline:
    """
    Pipeline tổng quát:
    Detection -> optional Segmentation -> optional Tracking -> optional Counting -> Visualization.
    """

    def __init__(
        self,
        detector,
        tracker: Optional[SimpleIoUTracker] = None,
        counter: Optional[LineCounter | RegionCounter] = None,
        segmenter=None,
        predict_config: Optional[dict] = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.segmenter = segmenter
        self.predict_config = predict_config or {}
        self.should_stop = False

    def process_image(
        self,
        image_path: str | Path,
        output_path: str | Path,
        use_segmentation: bool = False,
        frame_callback=None,
    ) -> list[dict]:
        image = read_image(image_path)
        detections = self.detector.detect(image, **self.predict_config)

        if use_segmentation and self.segmenter is not None:
            detections = self.segmenter.segment(image, detections)
            vis = draw_masks(image, detections)
        else:
            vis = image.copy()

        vis = draw_detections(vis, detections)
        save_image(output_path, vis)
        
        if frame_callback:
            frame_callback(vis)
            
        return detections

    def process_video(
        self,
        video_path: str | Path,
        output_path: str | Path,
        use_segmentation: bool = False,
        display: bool = False,
        frame_skip: int = 0,
        max_frames: Optional[int] = None,
        frame_callback=None,
    ) -> dict:
        cap = open_video(video_path)
        info = get_video_info(cap)
        writer = create_video_writer(output_path, info["width"], info["height"], info["fps"])

        frame_index = 0
        processed_frames = 0
        latest_detections: list[dict] = []

        try:
            while True:
                if self.should_stop:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                if max_frames is not None and processed_frames >= max_frames:
                    break

                should_detect = frame_skip <= 0 or frame_index % (frame_skip + 1) == 0

                if should_detect:
                    detections = self.detector.detect(frame, **self.predict_config)

                    if use_segmentation and self.segmenter is not None:
                        detections = self.segmenter.segment(frame, detections)

                    if self.tracker is not None:
                        detections = self.tracker.update(detections)

                    if self.counter is not None:
                        self.counter.update(detections)

                    latest_detections = detections
                else:
                    detections = latest_detections

                vis = frame.copy()
                if use_segmentation:
                    vis = draw_masks(vis, detections)
                vis = draw_detections(vis, detections)

                if self.counter is not None:
                    if isinstance(self.counter, LineCounter):
                        vis = draw_counting_line(vis, self.counter.p1, self.counter.p2)
                    elif isinstance(self.counter, RegionCounter):
                        vis = draw_counting_region(vis, self.counter.polygon)
                    vis = draw_count_panel(vis, self.counter.get_counts())

                writer.write(vis)
                processed_frames += 1
                frame_index += 1

                if display:
                    cv2.imshow("Traffic Counter", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if frame_callback:
                    frame_callback(vis)

        finally:
            cap.release()
            writer.release()
            if display:
                cv2.destroyAllWindows()

        counts = self.counter.get_counts() if self.counter is not None else {}
        return {
            "output_path": str(output_path),
            "processed_frames": processed_frames,
            "counts": counts,
        }
