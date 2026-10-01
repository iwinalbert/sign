from __future__ import annotations

from pathlib import Path

import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 1,
        detection_confidence: float = 0.5,
        presence_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = str(model_path)

        base_options = mp.tasks.BaseOptions(
            model_asset_path=self.model_path
        )

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            options
        )

    def detect(self, rgb_frame, timestamp_ms: int):
        """
        Detect hands in an RGB NumPy frame.
        """
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        return self.detector.detect_for_video(
            image,
            timestamp_ms,
        )

    def close(self) -> None:
        self.detector.close()