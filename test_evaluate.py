from detector import Detection
from evaluate import evaluate

TARGET = {"label": "couch", "bbox": [10.0, 10.0, 50.0, 50.0]}


def test_correct_match_counts_one_true_positive():
    result = evaluate([Detection("couch", 0.9, (10, 10, 50, 50))], TARGET)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0


def test_no_detection_counts_one_false_negative():
    result = evaluate([], TARGET)
    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1


def test_localization_failure_is_false_positive_and_false_negative():
    result = evaluate([Detection("couch", 0.9, (60, 60, 90, 90))], TARGET)
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


def test_duplicate_prediction_is_false_positive():
    detections = [
        Detection("couch", 0.9, (10, 10, 50, 50)),
        Detection("couch", 0.8, (10, 10, 50, 50)),
    ]
    result = evaluate(detections, TARGET)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 0
