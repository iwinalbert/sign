from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------
# Make project root importable when running:
# python scripts/train_model.py
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.classifier import TinyMLP
from app.config import CLASSIFIER_PATH, LABELS_PATH


LANDMARKS_FILE = ROOT / "data" / "landmarks.csv"


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

RNG = np.random.default_rng(42)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def load_labels() -> list[str]:
    with LABELS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_landmark_dataset(
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    if not LANDMARKS_FILE.exists():
        raise FileNotFoundError(
            f"\nLandmark dataset not found:\n"
            f"{LANDMARKS_FILE}\n\n"
            f"Run this first:\n"
            f"python scripts/extract_landmarks.py"
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

            label = row["label"].strip().upper()

            if label not in label_to_index:
                continue

            feature_values = []

            for i in range(63):
                key = f"f{i}"

                if key not in row:
                    raise ValueError(
                        f"Missing feature column: {key}"
                    )

                feature_values.append(
                    float(row[key])
                )

            features.append(feature_values)

            targets.append(
                label_to_index[label]
            )

    if not features:
        raise ValueError(
            "\nNo training samples found in "
            f"{LANDMARKS_FILE}\n"
        )

    x = np.asarray(
        features,
        dtype=np.float32,
    )

    y = np.asarray(
        targets,
        dtype=np.int64,
    )

    return x, y


# ---------------------------------------------------------
# Neural network utilities
# ---------------------------------------------------------

def relu(
    x: np.ndarray,
) -> np.ndarray:
    return np.maximum(x, 0)


def softmax(
    x: np.ndarray,
) -> np.ndarray:

    x = x - np.max(
        x,
        axis=1,
        keepdims=True,
    )

    exp_x = np.exp(x)

    return (
        exp_x
        / np.sum(
            exp_x,
            axis=1,
            keepdims=True,
        )
    )


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train_model(
    x: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    epochs: int = 150,
    learning_rate: float = 0.01,
    batch_size: int = 64,
):
    """
    Train:

        63 → 64 → 32 → num_classes
    """

    # Keep every class represented in both splits, then oversample only the
    # training side so rare signs are not drowned out by common signs.
    train_indices = []
    test_indices = []

    for class_index in range(num_classes):
        class_indices = np.flatnonzero(y == class_index)
        class_indices = RNG.permutation(class_indices)
        split = max(1, int(len(class_indices) * 0.80))
        train_indices.extend(class_indices[:split])
        test_indices.extend(class_indices[split:])

    train_indices = np.asarray(train_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)

    train_class_counts = np.bincount(
        y[train_indices],
        minlength=num_classes,
    )
    target_count = int(np.max(train_class_counts))
    balanced_indices = []

    for class_index in range(num_classes):
        class_indices = train_indices[
            y[train_indices] == class_index
        ]
        balanced_indices.extend(
            RNG.choice(
                class_indices,
                size=target_count,
                replace=True,
            )
        )

    train_indices = RNG.permutation(
        np.asarray(balanced_indices, dtype=np.int64)
    )

    x_train = x[
        train_indices
    ]

    y_train = y[
        train_indices
    ]

    x_test = x[
        test_indices
    ]

    y_test = y[
        test_indices
    ]

    input_size = 63
    hidden1 = 64
    hidden2 = 32

    # -----------------------------------------------------
    # Weight initialization
    # -----------------------------------------------------

    w1 = (
        RNG.normal(
            0,
            np.sqrt(
                2.0 / input_size
            ),
            (
                input_size,
                hidden1,
            ),
        )
        .astype(np.float32)
    )

    b1 = np.zeros(
        hidden1,
        dtype=np.float32,
    )

    w2 = (
        RNG.normal(
            0,
            np.sqrt(
                2.0 / hidden1
            ),
            (
                hidden1,
                hidden2,
            ),
        )
        .astype(np.float32)
    )

    b2 = np.zeros(
        hidden2,
        dtype=np.float32,
    )

    w3 = (
        RNG.normal(
            0,
            np.sqrt(
                2.0 / hidden2
            ),
            (
                hidden2,
                num_classes,
            ),
        )
        .astype(np.float32)
    )

    b3 = np.zeros(
        num_classes,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(
        epochs
    ):

        permutation = RNG.permutation(
            len(x_train)
        )

        x_train_epoch = (
            x_train[permutation]
        )

        y_train_epoch = (
            y_train[permutation]
        )

        epoch_loss = 0.0
        batches = 0

        for start in range(
            0,
            len(x_train_epoch),
            batch_size,
        ):

            end = min(
                start + batch_size,
                len(x_train_epoch),
            )

            xb = x_train_epoch[
                start:end
            ]

            yb = y_train_epoch[
                start:end
            ]

            # -------------------------------------------------
            # Forward
            # -------------------------------------------------

            z1 = (
                xb @ w1
                + b1
            )

            a1 = relu(z1)

            z2 = (
                a1 @ w2
                + b2
            )

            a2 = relu(z2)

            logits = (
                a2 @ w3
                + b3
            )

            probabilities = softmax(
                logits
            )

            n = len(xb)

            loss = -np.mean(
                np.log(
                    probabilities[
                        np.arange(n),
                        yb,
                    ]
                    + 1e-8
                )
            )

            epoch_loss += loss
            batches += 1

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------

            d_logits = probabilities.copy()

            d_logits[
                np.arange(n),
                yb,
            ] -= 1

            d_logits /= n

            dw3 = (
                a2.T
                @ d_logits
            )

            db3 = np.sum(
                d_logits,
                axis=0,
            )

            da2 = (
                d_logits
                @ w3.T
            )

            dz2 = (
                da2
                * (z2 > 0)
            )

            dw2 = (
                a1.T
                @ dz2
            )

            db2 = np.sum(
                dz2,
                axis=0,
            )

            da1 = (
                dz2
                @ w2.T
            )

            dz1 = (
                da1
                * (z1 > 0)
            )

            dw1 = (
                xb.T
                @ dz1
            )

            db1 = np.sum(
                dz1,
                axis=0,
            )

            # -------------------------------------------------
            # Gradient descent
            # -------------------------------------------------

            w3 -= (
                learning_rate
                * dw3
            )

            b3 -= (
                learning_rate
                * db3
            )

            w2 -= (
                learning_rate
                * dw2
            )

            b2 -= (
                learning_rate
                * db2
            )

            w1 -= (
                learning_rate
                * dw1
            )

            b1 -= (
                learning_rate
                * db1
            )

        # -----------------------------------------------------
        # Test accuracy
        # -----------------------------------------------------

        test_z1 = (
            x_test @ w1
            + b1
        )

        test_a1 = relu(
            test_z1
        )

        test_z2 = (
            test_a1 @ w2
            + b2
        )

        test_a2 = relu(
            test_z2
        )

        test_logits = (
            test_a2 @ w3
            + b3
        )

        test_predictions = np.argmax(
            test_logits,
            axis=1,
        )

        accuracy = np.mean(
            test_predictions
            == y_test
        )

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):
            average_loss = (
                epoch_loss
                / max(batches, 1)
            )

            print(
                f"Epoch "
                f"{epoch + 1:03d}/{epochs} | "
                f"Loss: "
                f"{average_loss:.4f} | "
                f"Accuracy: "
                f"{accuracy * 100:.2f}%"
            )

    return (
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        x_test,
        y_test,
        test_predictions,
    )


# ---------------------------------------------------------
# Per-class evaluation
# ---------------------------------------------------------

def print_class_accuracy(
    labels: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:

    print()
    print(
        "Per-class accuracy"
    )
    print(
        "-------------------"
    )

    for index, label in enumerate(
        labels
    ):

        mask = (
            y_true == index
        )

        if not np.any(mask):
            print(
                f"{label:15s} "
                f"NO TEST DATA"
            )
            continue

        accuracy = np.mean(
            y_pred[mask]
            == y_true[mask]
        )

        print(
            f"{label:15s} "
            f"{accuracy * 100:7.2f}%"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:

    print()
    print(
        "=========================================="
    )
    print(
        "     EDGE SIGN - MODEL TRAINING"
    )
    print(
        "=========================================="
    )
    print()

    labels = load_labels()

    print(
        f"Classes ({len(labels)}):"
    )

    for label in labels:
        print(
            f"  - {label}"
        )

    print()

    print(
        f"Loading landmark dataset:"
    )

    print(
        LANDMARKS_FILE
    )

    x, y = load_landmark_dataset(
        labels
    )

    print()

    print(
        f"Samples: {len(x)}"
    )

    print(
        f"Features per sample: "
        f"{x.shape[1]}"
    )

    print()

    (
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        x_test,
        y_test,
        predictions,
    ) = train_model(
        x=x,
        y=y,
        num_classes=len(labels),
    )

    # -----------------------------------------------------
    # Build classifier object
    # -----------------------------------------------------

    model = TinyMLP(
        input_size=63,
        hidden1=64,
        hidden2=32,
        output_size=len(labels),
    )

    model.w1 = w1
    model.b1 = b1

    model.w2 = w2
    model.b2 = b2

    model.w3 = w3
    model.b3 = b3

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    CLASSIFIER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(
        CLASSIFIER_PATH
    )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    final_accuracy = np.mean(
        predictions == y_test
    )

    print()
    print(
        "=========================================="
    )

    print(
        f"Final test accuracy: "
        f"{final_accuracy * 100:.2f}%"
    )

    print_class_accuracy(
        labels,
        y_test,
        predictions,
    )

    print()
    print(
        "[OK] Model saved:"
    )

    print(
        CLASSIFIER_PATH
    )

    print(
        "=========================================="
    )
    print()


if __name__ == "__main__":
    main()