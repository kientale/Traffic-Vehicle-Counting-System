from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SAMSegmenter:
    """
    Dùng SAM để tạo mask từ bounding box của YOLO.

    Lưu ý: SAM không thay YOLO để detect class. SAM chỉ nhận box/point làm prompt
    rồi tách vùng pixel của object.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_type: str = "vit_b",
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Bạn cần cài segment-anything và torch. Ví dụ: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = str(checkpoint_path)

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def segment(self, image_bgr, detections: list[dict]) -> list[dict]:
        if not detections:
            return detections

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
        boxes_torch = self.torch.as_tensor(boxes, device=self.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_torch,
            image_rgb.shape[:2],
        )

        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks_np = masks.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        for idx, det in enumerate(detections):
            det["mask"] = masks_np[idx, 0]
            det["mask_score"] = float(scores_np[idx, 0])

        return detections
