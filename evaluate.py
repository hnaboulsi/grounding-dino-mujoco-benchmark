"""IoU matching and counts for the one-target experiment."""

from __future__ import annotations

from collections.abc import Sequence

from detector import Detection

IOU_THRESHOLD = 0.50
CONFIDENCE_THRESHOLD = 0.35


def bbox_iou(
    first: Sequence[float] | None,
    second: Sequence[float] | None,
) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def evaluate(
    detections: list[Detection],
    ground_truth: dict[str, object],
    *,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, object]:
    """Match the highest-scoring qualifying target prediction once."""
    target_label = str(ground_truth["label"]).lower()
    target_box = ground_truth["bbox"]
    saved = [
        (index, detection)
        for index, detection in enumerate(detections)
        if detection.score >= confidence_threshold
    ]
    saved.sort(key=lambda item: item[1].score, reverse=True)
    matched_index = None
    matched_iou = 0.0
    for index, detection in saved:
        if detection.label.lower() != target_label:
            continue
        overlap = bbox_iou(detection.bbox, target_box)
        if overlap >= iou_threshold:
            matched_index = index
            matched_iou = overlap
            break
    true_positives = int(matched_index is not None)
    return {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": len(saved) - true_positives,
        "false_negatives": 1 - true_positives,
        "matched_detection_index": matched_index,
        "matched_iou": matched_iou,
    }
