"""Run a real local HTTP round trip without changing the user's database or secrets."""

import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

from detection.adapters import AdapterConfig
from detection.recording import read_recording
from detection.tamper import save_reference
from ml.artifacts import save_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sentinel-smoke-") as temporary:
        folder = Path(temporary)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        env = os.environ.copy()
        env.update(
            {
                "DATABASE_URL": f"sqlite:///{(folder / 'api.db').as_posix()}",
                "INGEST_API_KEY": secrets.token_urlsafe(32),
                "READ_API_KEY": secrets.token_urlsafe(32),
                "APP_ENV": "development",
                "ALERT_MODE": "mock",
                "GATEWAY_API_URL": f"http://127.0.0.1:{port}",
            }
        )
        env.pop("VERCEL", None)

        def run(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                [sys.executable, "-m", *args],
                cwd=root,
                env=env,
                check=True,
                timeout=30,
                capture_output=True,
                text=True,
            )

        run("backend.db")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (folder / "server.log").open("w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.api.index:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
                creationflags=flags,
            )
            try:
                with httpx.Client(base_url=env["GATEWAY_API_URL"], timeout=2) as client:
                    for _ in range(50):
                        try:
                            if client.get("/health").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        time.sleep(0.1)
                    else:
                        raise RuntimeError("local API failed to become ready")
                    recording = str(folder / "demo.jsonl")
                    background = str(folder / "background.jsonl")
                    calibration = str(folder / "calibration.json")
                    model_path = folder / "test-model.csimodel"
                    adapter = AdapterConfig.model_validate_json(
                        (root / "configs/synthetic.json").read_text()
                    )
                    carriers = len(adapter.subcarrier_indices)
                    save_model(
                        model_path,
                        {
                            "mean": np.zeros(carriers),
                            "scale": np.ones(carriers),
                            "coef": np.zeros((1, carriers * 4)),
                            "intercept": np.zeros(1),
                        },
                        adapter,
                        "synthetic-smoke-test-weights",
                        "logistic_summary_v1",
                        ["nonfall", "fall"],
                        {"task": "fall"},
                    )
                    outbox = str(folder / "outbox.db")
                    run(
                        "simulator.generate",
                        "--output",
                        background,
                        "--cycles",
                        "1",
                        "--calibration-only",
                    )
                    run(
                        "detection.calibrate",
                        "--replay",
                        background,
                        "--room",
                        "studio",
                        "--output",
                        calibration,
                        "--confirm-empty-room",
                    )
                    tamper_path = folder / "anomaly.csimodel"
                    save_reference(
                        tamper_path,
                        np.stack([w.values for w in read_recording(Path(background), adapter)]),
                        adapter,
                        "synthetic smoke normal background",
                    )
                    zone_result = client.post(
                        "/api/zones",
                        headers={"X-API-Key": env["INGEST_API_KEY"]},
                        json={"zone_id": "studio-zone", "room_id": "studio", "armed": True},
                    )
                    assert zone_result.status_code == 200
                    run("simulator.generate", "--output", recording, "--cycles", "1")
                    run(
                        "gateway.run",
                        "--replay",
                        recording,
                        "--calibration",
                        calibration,
                        "--model",
                        str(model_path),
                        "--tamper-model",
                        str(tamper_path),
                        "--sync-zones",
                        "--outbox",
                        outbox,
                        "--allow-local-http",
                    )
                    run("gateway.run", "--flush-only", "--outbox", outbox, "--allow-local-http")
                    headers = {"X-API-Key": env["READ_API_KEY"]}
                    events = client.get("/api/events?limit=200", headers=headers).json()
                    rooms = client.get("/api/rooms/status", headers=headers).json()
                    devices = client.get("/api/devices", headers=headers).json()
                    kinds = {event["type"] for event in events}
                    assert {
                        "occupancy",
                        "motion",
                        "fall_suspected",
                        "fall_confirmed",
                        "zone_entry",
                        "tamper_suspected",
                    } <= kinds
                    assert sum(e["type"] == "fall_confirmed" for e in events) == 1
                    assert sum(e["payload"].get("status") == "cancelled" for e in events) == 1
                    assert all(e["payload"]["zone_schedule_status"] == "synced" for e in events)
                    assert all(
                        e["payload"]["activity_prediction"]["label"] == "nonfall" for e in events
                    )
                    assert len(rooms) == 1 and len(devices) == 1
                    assert len({event["event_id"] for event in events}) == len(events)
                    assert all("data" not in event["payload"] for event in events)
                    assert all("identity_prediction" not in event["payload"] for event in events)
                    run("backend.alert_worker", "--once", "--batch-size", "20")
                    alerts = client.get("/api/alerts/status", headers=headers).json()
                    assert alerts["counts"].get("mock:mock_sent", 0) > 0
                    assert alerts["counts"].get("mock:pending", 0) == 0
                    print(
                        f"PASS: real HTTP replay delivered {len(events)} unique events, "
                        f"{len(rooms)} room and {len(devices)} device; no raw CSI uploaded."
                    )
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    main()
