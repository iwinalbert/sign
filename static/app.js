async function updateStatus() {
    try {
        const response = await fetch(
            "/api/status",
            {
                cache: "no-store"
            }
        );

        if (!response.ok) {
            return;
        }

        const data = await response.json();

        document.getElementById("gesture").textContent =
            data.gesture || "—";

        document.getElementById("confidence").textContent =
            `${(data.confidence * 100).toFixed(1)}%`;

        document.getElementById("status").textContent =
            data.status || "—";

        document.getElementById("ai-mode").textContent =
            data.inference_mode || "LOCAL";

        document.getElementById("latency").textContent =
            `${Number(data.latency_ms).toFixed(0)} ms`;

        document.getElementById("fps").textContent =
            Number(data.fps).toFixed(1);

        document.getElementById("speech").textContent =
            data.speaking
                ? "SPEAKING..."
                : "READY";

    } catch (error) {
        console.error(error);
    }
}


setInterval(updateStatus, 200);

updateStatus();