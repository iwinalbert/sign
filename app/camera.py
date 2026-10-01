from __future__ import annotations

import time

import cv2

from app.config import (
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    JPEG_QUALITY,
    TARGET_FPS,
)
from app.features import normalize_landmarks
from app.predictor import PredictionStabilizer


HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def draw_hand(frame, landmarks) -> None:
    height, width, _ = frame.shape

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (80, 220, 120),
            -1,
            cv2.LINE_AA,
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (80, 220, 120),
            2,
            cv2.LINE_AA,
        )


class CameraEngine:
    def __init__(
        self,
        tracker,
        classifier,
        labels,
        shared_state,
        speech_engine,
    ) -> None:
        self.tracker = tracker
        self.classifier = classifier
        self.labels = labels
        self.shared_state = shared_state
        self.speech_engine = speech_engine

        self.stabilizer = PredictionStabilizer(
            window_size=7,
            min_stable_count=5,
            confidence_threshold=0.70,
        )

        self.running = False
        self.last_timestamp_ms = 0
        self.last_spoken_label = None

    def run(self) -> None:
        cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            self.shared_state.update(
                status="CAMERA ERROR",
            )
            raise RuntimeError("Could not open camera.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self.running = True

        frame_times = []

        while self.running:
            loop_start = time.perf_counter()

            success, frame = cap.read()

            if not success:
                self.shared_state.update(
                    status="CAMERA ERROR",
                    hand_detected=False,
                )
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int(time.monotonic() * 1000)

            if timestamp_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + 1

            self.last_timestamp_ms = timestamp_ms

            inference_start = time.perf_counter()

            result = self.tracker.detect(
                rgb,
                timestamp_ms,
            )

            latency_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            hand_detected = bool(result.hand_landmarks)

            gesture = "—"
            confidence = 0.0
            status = "NO HAND"

            if hand_detected:
                landmarks = result.hand_landmarks[0]

                draw_hand(frame, landmarks)

                features = normalize_landmarks(landmarks)

                class_index, model_confidence = (
                    self.classifier.predict(features)
                )

                if class_index < len(self.labels):
                    predicted_label = self.labels[class_index]

                    stable_label, stable_confidence = (
                        self.stabilizer.update(
                            predicted_label,
                            model_confidence,
                        )
                    )

                    if stable_label:
                        gesture = stable_label
                        confidence = stable_confidence
                        status = "READY"

                        if stable_label != self.last_spoken_label:
                            self.speech_engine.speak(stable_label)
                            self.last_spoken_label = stable_label
                    else:
                        status = "DETECTING"

                else:
                    self.stabilizer.reset()
            else:
                self.stabilizer.reset()
                self.last_spoken_label = None

            frame_time = time.perf_counter() - loop_start

            frame_times.append(frame_time)

            if len(frame_times) > 20:
                frame_times.pop(0)

            average_frame_time = sum(frame_times) / len(frame_times)

            fps = (
                1.0 / average_frame_time
                if average_frame_time > 0
                else 0.0
            )

            # Overlay information on camera stream.
            frame_height, frame_width = frame.shape[:2]
            overlay_x = 12
            overlay_y = 12
            overlay_width = min(300, frame_width - 24)
            overlay_height = min(105, frame_height - 24)

            cv2.rectangle(
                frame,
                (overlay_x, overlay_y),
                (
                    overlay_x + overlay_width,
                    overlay_y + overlay_height,
                ),
                (20, 20, 20),
                -1,
            )

            cv2.putText(
                frame,
                f"SIGN: {gesture}",
                (overlay_x + 10, overlay_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"CONF: {confidence * 100:.1f}%",
                (overlay_x + 10, overlay_y + 57),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "EDGE AI: LOCAL",
                (overlay_x + 10, overlay_y + 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            encode_ok, buffer = cv2.imencode(
                ".jpg",
                frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if encode_ok:
                self.shared_state.update(
                    gesture=gesture,
                    confidence=confidence,
                    status=status,
                    inference_mode="LOCAL",
                    hand_detected=hand_detected,
                    fps=fps,
                    latency_ms=latency_ms,
                    speaking=self.speech_engine.is_speaking,
                    jpeg_frame=buffer.tobytes(),
                )

        cap.release()

    def stop(self) -> None:
        self.running = False
        self.speech_engine.close()