from __future__ import annotations

import json
import threading

from app.camera import CameraEngine
from app.classifier import TinyMLP
from app.config import (
    CLASSIFIER_PATH,
    LABELS_PATH,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
    WEB_HOST,
    WEB_PORT,
)
from app.hand_tracker import HandTracker
from app.speech import SpeechEngine
from app.state import SharedState
from app.web import create_app


def load_labels() -> list[str]:
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    print()
    print("======================================")
    print("       EDGE SIGN - EDGE AI")
    print("======================================")
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Hand model not found: {MODEL_PATH}"
        )

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"Gesture classifier not found: {CLASSIFIER_PATH}\n"
            "Train the model first using:\n"
            "python scripts/train_model.py"
        )

    labels = load_labels()

    shared_state = SharedState()

    tracker = HandTracker(
        model_path=MODEL_PATH,
        num_hands=1,
        detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    classifier = TinyMLP(
        input_size=63,
        hidden1=64,
        hidden2=32,
        output_size=len(labels),
    )

    classifier.load(CLASSIFIER_PATH)

    speech_engine = SpeechEngine()

    camera_engine = CameraEngine(
        tracker=tracker,
        classifier=classifier,
        labels=labels,
        shared_state=shared_state,
        speech_engine=speech_engine,
    )

    inference_thread = threading.Thread(
        target=camera_engine.run,
        daemon=True,
    )

    inference_thread.start()

    app = create_app(shared_state)

    print()
    print("[OK] Model loaded")
    print("[OK] Camera engine starting")
    print("[OK] Edge AI mode: LOCAL")
    print(
        f"[OK] Website: http://127.0.0.1:{WEB_PORT}"
    )
    print()

    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=True,
            debug=False,
        )
    finally:
        camera_engine.stop()
        tracker.close()


if __name__ == "__main__":
    main()