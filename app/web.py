from __future__ import annotations

import time

from flask import Flask, Response, jsonify, render_template

from app.state import SharedState


def create_app(shared_state: SharedState) -> Flask:
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        def generate():
            while True:
                state = shared_state.snapshot()

                if state.jpeg_frame is None:
                    time.sleep(0.02)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + state.jpeg_frame
                    + b"\r\n"
                )

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/api/status")
    def api_status():
        state = shared_state.snapshot()

        return jsonify(
            {
                "gesture": state.gesture,
                "confidence": state.confidence,
                "status": state.status,
                "inference_mode": state.inference_mode,
                "hand_detected": state.hand_detected,
                "fps": state.fps,
                "latency_ms": state.latency_ms,
                "speaking": state.speaking,
            }
        )

    return app