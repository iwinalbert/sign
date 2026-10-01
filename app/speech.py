from __future__ import annotations

import platform
import queue
import subprocess
import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class SpeechEngine:
    def __init__(self) -> None:
        self.system = platform.system()

        self._requests: queue.Queue[str | None] = queue.Queue(maxsize=4)
        self._speaking = threading.Event()
        self._stopped = threading.Event()

        self.engine = None

        self._worker = threading.Thread(
            target=self._run,
            name="speech-worker",
            daemon=True,
        )
        self._worker.start()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str) -> None:
        """Queue speech without blocking camera inference."""
        if self._stopped.is_set():
            return

        try:
            self._requests.put_nowait(text)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass

            try:
                self._requests.put_nowait(text)
            except queue.Full:
                pass

    def _run(self) -> None:
        if self.system == "Windows" and pyttsx3 is not None:
            # SAPI objects must be created and used on the same thread.
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)

        while not self._stopped.is_set():
            try:
                text = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                return

            self._speaking.set()
            try:
                if self.system == "Windows":
                    if self.engine is not None:
                        self.engine.say(text)
                        self.engine.runAndWait()
                else:
                    subprocess.run(
                        ["espeak-ng", text],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            finally:
                self._speaking.clear()

    def close(self) -> None:
        self._stopped.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            pass