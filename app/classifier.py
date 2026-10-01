from __future__ import annotations

from pathlib import Path

import numpy as np


class TinyMLP:
    """
    Small feed-forward neural network:

        63 -> 64 -> 32 -> N

    Implemented entirely with NumPy.
    """

    def __init__(
        self,
        input_size: int = 63,
        hidden1: int = 64,
        hidden2: int = 32,
        output_size: int = 10,
    ) -> None:
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        self.w1 = np.zeros((input_size, hidden1), dtype=np.float32)
        self.b1 = np.zeros(hidden1, dtype=np.float32)

        self.w2 = np.zeros((hidden1, hidden2), dtype=np.float32)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        self.w3 = np.zeros((hidden2, output_size), dtype=np.float32)
        self.b3 = np.zeros(output_size, dtype=np.float32)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        z1 = x @ self.w1 + self.b1
        a1 = self.relu(z1)

        z2 = a1 @ self.w2 + self.b2
        a2 = self.relu(z2)

        z3 = a2 @ self.w3 + self.b3

        return self.softmax(z3)

    def predict(self, x: np.ndarray) -> tuple[int, float]:
        probabilities = self.forward(x)

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        return index, confidence

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
        )

    def load(self, path: str | Path) -> None:
        data = np.load(path)

        self.w1 = data["w1"].astype(np.float32)
        self.b1 = data["b1"].astype(np.float32)

        self.w2 = data["w2"].astype(np.float32)
        self.b2 = data["b2"].astype(np.float32)

        self.w3 = data["w3"].astype(np.float32)
        self.b3 = data["b3"].astype(np.float32)

        self.input_size = self.w1.shape[0]
        self.hidden1 = self.w1.shape[1]
        self.hidden2 = self.w2.shape[1]
        self.output_size = self.w3.shape[1]