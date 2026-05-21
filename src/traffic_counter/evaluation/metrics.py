from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def match_detections(
    predictions: list[dict],
    ground_truths: list[dict],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict:
    """
    Greedy matching giữa prediction và ground truth.
    Trả về TP/FP/FN tổng và theo class.
    """
    predictions = sorted(predictions, key=lambda d: d.get("confidence", 0), reverse=True)
    matched_gt = set()

    tp = 0
    fp = 0
    by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for pred in predictions:
        pred_class = pred.get("class_name")
        best_iou = 0.0
        best_gt_idx = None

        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue

            if class_aware and pred_class != gt.get("class_name"):
                continue

            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_gt_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            by_class[pred_class]["tp"] += 1
        else:
            fp += 1
            by_class[pred_class]["fp"] += 1

    fn = len(ground_truths) - len(matched_gt)
    for idx, gt in enumerate(ground_truths):
        if idx not in matched_gt:
            by_class[gt.get("class_name")]["fn"] += 1

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "by_class": dict(by_class),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def counting_error(predicted_count: int, ground_truth_count: int) -> dict:
    absolute_error = abs(predicted_count - ground_truth_count)
    percentage_error = (
        absolute_error / ground_truth_count * 100 if ground_truth_count > 0 else 0.0
    )
    return {
        "predicted_count": predicted_count,
        "ground_truth_count": ground_truth_count,
        "absolute_error": absolute_error,
        "percentage_error": percentage_error,
    }
