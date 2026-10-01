from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


# =========================================================
# Make project root importable
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.config import CLASSIFIER_PATH, LABELS_PATH


LANDMARKS_FILE = ROOT / "data" / "landmarks.csv"


# =========================================================
# Load labels
# =========================================================

def load_labels() -> list[str]:
    with LABELS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# Load landmark dataset
# =========================================================

def load_dataset(
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    if not LANDMARKS_FILE.exists():
        raise FileNotFoundError(
            f"\nLandmark dataset not found:\n"
            f"{LANDMARKS_FILE}\n\n"
            f"Run:\n"
            f"python scripts\\extract_landmarks.py"
        )

    label_to_index = {
        label.upper(): index
        for index, label in enumerate(labels)
    }

    features = []
    targets = []

    with LANDMARKS_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            label = (
                row["label"]
                .strip()
                .upper()
            )

            if label not in label_to_index:
                continue

            sample = []

            for i in range(63):

                key = f"f{i}"

                if key not in row:
                    raise ValueError(
                        f"Missing feature: {key}"
                    )

                sample.append(
                    float(row[key])
                )

            features.append(sample)

            targets.append(
                label_to_index[label]
            )

    if not features:
        raise ValueError(
            "No valid landmark samples found."
        )

    return (
        np.asarray(
            features,
            dtype=np.float32,
        ),
        np.asarray(
            targets,
            dtype=np.int64,
        ),
    )


# =========================================================
# Model prediction
# =========================================================

def relu(
    x: np.ndarray,
) -> np.ndarray:
    return np.maximum(x, 0)


def predict(
    x: np.ndarray,
    model,
) -> np.ndarray:

    w1 = model["w1"]
    b1 = model["b1"]

    w2 = model["w2"]
    b2 = model["b2"]

    w3 = model["w3"]
    b3 = model["b3"]

    z1 = x @ w1 + b1
    a1 = relu(z1)

    z2 = a1 @ w2 + b2
    a2 = relu(z2)

    logits = a2 @ w3 + b3

    return np.argmax(
        logits,
        axis=1,
    )


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    overall_accuracy = np.mean(
        y_true == y_pred
    )

    print()
    print("==========================================")
    print("          EDGE SIGN EVALUATION")
    print("==========================================")
    print()

    print(
        f"Total samples:      {len(y_true)}"
    )

    print(
        f"Overall accuracy:   "
        f"{overall_accuracy * 100:.2f}%"
    )

    print()
    print("Per-class accuracy")
    print("-------------------")

    for index, label in enumerate(labels):

        mask = (
            y_true == index
        )

        if not np.any(mask):
            print(
                f"{label:15s} "
                f"NO DATA"
            )
            continue

        class_accuracy = np.mean(
            y_pred[mask] == y_true[mask]
        )

        print(
            f"{label:15s} "
            f"{class_accuracy * 100:7.2f}% "
            f"({np.sum(mask)} samples)"
        )

    print()
    print("==========================================")


# =========================================================
# Main
# =========================================================

def main() -> None:

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"\nClassifier model not found:\n"
            f"{CLASSIFIER_PATH}\n\n"
            f"Run:\n"
            f"python scripts\\train_model.py"
        )

    labels = load_labels()

    x, y = load_dataset(
        labels
    )

    model = np.load(
        CLASSIFIER_PATH
    )

    predictions = predict(
        x,
        model,
    )

    evaluate(
        labels,
        y,
        predictions,
    )


if __name__ == "__main__":
    main()