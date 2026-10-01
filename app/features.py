from __future__ import annotations

import numpy as np


def landmarks_to_array(landmarks) -> np.ndarray:
    """
    Convert MediaPipe hand landmarks into a 21x3 NumPy array.
    """
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Normalize 21 hand landmarks.

    1. Wrist becomes the origin.
    2. Coordinates are scaled by the maximum distance
       from the wrist to any landmark.
    3. Result is flattened to 63 features.
    """
    points = landmarks_to_array(landmarks)

    wrist = points[0].copy()

    points -= wrist

    distances = np.linalg.norm(points, axis=1)
    scale = float(np.max(distances))

    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    points /= scale

    return points.flatten().astype(np.float32)