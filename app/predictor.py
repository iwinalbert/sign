from __future__ import annotations

from collections import Counter, deque


class PredictionStabilizer:
    def __init__(
        self,
        window_size: int = 7,
        min_stable_count: int = 5,
        confidence_threshold: float = 0.70,
    ) -> None:
        self.predictions = deque(maxlen=window_size)
        self.min_stable_count = min_stable_count
        self.confidence_threshold = confidence_threshold

        self.stable_label: str | None = None
        self.stable_confidence = 0.0

    def reset(self) -> None:
        self.predictions.clear()
        self.stable_label = None
        self.stable_confidence = 0.0

    def update(
        self,
        label: str,
        confidence: float,
    ) -> tuple[str | None, float]:
        if confidence < self.confidence_threshold:
            self.predictions.clear()
            self.stable_label = None
            self.stable_confidence = 0.0
            return None, 0.0

        self.predictions.append((label, confidence))

        counts = Counter(item[0] for item in self.predictions)

        if not counts:
            return None, 0.0

        candidate, count = counts.most_common(1)[0]

        if count < self.min_stable_count:
            return None, 0.0

        candidate_confidences = [
            confidence
            for predicted_label, confidence in self.predictions
            if predicted_label == candidate
        ]

        average_confidence = sum(candidate_confidences) / len(
            candidate_confidences
        )

        changed = candidate != self.stable_label

        self.stable_label = candidate
        self.stable_confidence = average_confidence

        if changed:
            return candidate, average_confidence

        return candidate, average_confidence