from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2

from app.config import MODEL_PATH
from app.features import normalize_landmarks
from app.hand_tracker import HandTracker


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT / "data" / "dataset"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "landmarks.csv"
)


SAMPLE_EVERY_N_FRAMES = 2


def main() -> None:

    tracker = HandTracker(
        model_path=MODEL_PATH,
        num_hands=1,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_written = 0

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["label"]
            + [
                f"f{i}"
                for i in range(63)
            ]
        )

        video_files = list(
            DATASET_DIR.rglob("*.mp4")
        )

        print()
        print(
            f"Videos found: "
            f"{len(video_files)}"
        )
        print()

        for video_index, video_path in enumerate(
            video_files,
            start=1,
        ):

            label = (
                video_path.parent.name
                .upper()
            )

            print(
                f"[{video_index}/"
                f"{len(video_files)}] "
                f"{label} - "
                f"{video_path.name}"
            )

            cap = cv2.VideoCapture(
                str(video_path)
            )

            if not cap.isOpened():
                print(
                    "[WARN] Cannot open video"
                )
                continue

            frame_index = 0

            while True:

                success, frame = (
                    cap.read()
                )

                if not success:
                    break

                if (
                    frame_index
                    % SAMPLE_EVERY_N_FRAMES
                    != 0
                ):
                    frame_index += 1
                    continue

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                timestamp_ms = int(
                    time.monotonic()
                    * 1000
                )

                try:

                    result = tracker.detect(
                        rgb,
                        timestamp_ms,
                    )

                except Exception:
                    frame_index += 1
                    continue

                if result.hand_landmarks:

                    landmarks = (
                        result.hand_landmarks[0]
                    )

                    features = (
                        normalize_landmarks(
                            landmarks
                        )
                    )

                    writer.writerow(
                        [
                            label
                        ]
                        + features.tolist()
                    )

                    rows_written += 1

                frame_index += 1

            cap.release()

    tracker.close()

    print()
    print(
        f"[DONE] Landmark samples: "
        f"{rows_written}"
    )

    print(
        f"[DONE] Saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()