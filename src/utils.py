import numpy as np

def reference_to_intervals(labels, frame_rate):
    """Convert labels to intervals."""
    intervals = []
    start = None
    for i, label in enumerate(labels):
        if label == 1 and start is None:
            start = i / frame_rate
        elif label == 0 and start is not None:
            intervals.append((start, i / frame_rate))
            start = None
    if start is not None:
        intervals.append((start, len(labels) / frame_rate))
    return intervals

def compute_overlap(interval1, interval2):
    """Compute the overlap between two intervals."""
    return max(0, min(interval1[1], interval2[1]) - max(interval1[0], interval2[0]))

def compute_detection_error_rate_intervals(reference_intervals, predicted_intervals):
    """Compute the Detection Error Rate."""
    total_reference_duration = sum(end - start for start, end in reference_intervals)

    # Handle empty predictions
    if not predicted_intervals:
        return 1.0 if total_reference_duration > 0 else 0.0

    false_alarm = 0
    missed_detection = 0

    # Compute false alarms
    for pred_start, pred_end in predicted_intervals:
        pred_duration = pred_end - pred_start
        overlap = sum(compute_overlap((pred_start, pred_end), ref_interval) 
                      for ref_interval in reference_intervals)
        false_alarm += pred_duration - overlap

    # Compute missed detections
    for ref_start, ref_end in reference_intervals:
        ref_duration = ref_end - ref_start
        overlap = sum(compute_overlap((ref_start, ref_end), pred_interval) 
                      for pred_interval in predicted_intervals)
        missed_detection += ref_duration - overlap

    if total_reference_duration == 0:
        return 1.0 if false_alarm > 0 else 0.0
    else:
        return (false_alarm + missed_detection) / total_reference_duration
