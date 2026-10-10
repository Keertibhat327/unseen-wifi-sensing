"""Run with python -m gateway.run; network delivery uses a separate durable worker."""

import argparse
import json
import os
import threading
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

from detection.adapters import AdapterConfig
from detection.adapters.esp_csi import EspCSIAdapter
from detection.calibration import Calibration
from detection.core import Detection
from detection.engine import Detector
from detection.inference import ModelPredictor
from detection.recording import read_recording
from detection.settings import DetectorSettings
from detection.tamper import TamperMonitor, TamperSettings
from gateway.buffer import Outbox
from gateway.delivery import Delivery
from gateway.zones import ZoneFeed


def delivery_worker(
    path: Path, url: str, key: str, local: bool, stop: threading.Event, zones: ZoneFeed
) -> None:
    outbox = Outbox(path)
    delivery = Delivery(url, key, local)
    try:
        while not stop.is_set():
            zones.refresh(delivery.client)
            delivery.flush(outbox)
            stop.wait(0.5)
    finally:
        delivery.close()
        outbox.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path)
    source.add_argument("--serial", help="TODO: blocked until real ESP-CSI samples are verified")
    source.add_argument("--flush-only", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.json"))
    parser.add_argument("--outbox", type=Path, default=Path(".gateway/outbox.db"))
    parser.add_argument("--outbox-max-items", type=int, default=100000)
    parser.add_argument("--device", default="simulator-01")
    parser.add_argument("--room", default="studio")
    parser.add_argument("--api-url", default=os.getenv("GATEWAY_API_URL", "https://localhost:8000"))
    parser.add_argument("--allow-local-http", action="store_true")
    parser.add_argument("--requeue", action="store_true")
    parser.add_argument("--settings", type=Path, default=Path("configs/detection.json"))
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--model", type=Path, help="Optional compatible activity model; annotations only"
    )
    parser.add_argument(
        "--tamper-model", type=Path, help="Optional local normal-activity reference"
    )
    parser.add_argument("--tamper-settings", type=Path, default=Path("configs/tamper.json"))
    schedules = parser.add_mutually_exclusive_group()
    schedules.add_argument("--zones", type=Path, help="Local UTC zone schedules")
    schedules.add_argument("--sync-zones", action="store_true", help="Poll API using READ_API_KEY")
    parser.add_argument(
        "--pace", action="store_true", help="pace replay using configured window duration"
    )
    args = parser.parse_args()
    config = AdapterConfig.model_validate_json(args.config.read_text())
    if args.serial:
        # Fail before opening hardware or buffering invented detections.
        EspCSIAdapter(config).convert(None)
    key = os.getenv("INGEST_API_KEY", "")
    detector = None
    if args.replay:
        if not args.calibration:
            parser.error("--replay requires --calibration; run python -m detection.calibrate first")
        detector = Detector(
            args.room,
            config,
            Calibration.model_validate_json(args.calibration.read_text()),
            DetectorSettings.model_validate_json(args.settings.read_text()),
        )
    delivery = Delivery(args.api_url, key, args.allow_local_http)
    zones = ZoneFeed(
        path=args.zones,
        api_url=args.api_url if args.sync_zones else None,
        read_key=os.getenv("READ_API_KEY", ""),
    )
    zones.refresh(delivery.client)
    zones.snapshot()
    delivery.close()  # Validate transport configuration before starting a worker.
    outbox = Outbox(args.outbox, max_items=args.outbox_max_items)
    if args.requeue:
        outbox.requeue()
    stop = threading.Event()
    worker = threading.Thread(
        target=delivery_worker,
        daemon=True,
        args=(args.outbox, args.api_url, key, args.allow_local_http, stop, zones),
    )
    worker.start()
    try:
        if args.flush_only:
            print("Flushing durable queue. Press Ctrl+C to stop.")
            while outbox.counts()["pending"]:
                stop.wait(1)
                if not worker.is_alive():
                    raise RuntimeError("delivery worker stopped; events remain on disk")
        else:
            assert detector is not None
            duration = config.window_samples / config.sample_rate_hz
            predictor = None
            monitor = None
            for window in read_recording(args.replay, config):
                activity = {}
                if args.model:
                    if predictor is None:
                        predictor = ModelPredictor.load(
                            args.model, config, window.values, config.sample_rate_hz
                        )
                    prediction = predictor.predict(window.values, config.sample_rate_hz)
                    activity = {
                        "activity_prediction": {
                            "label": prediction.label,
                            "score": prediction.score,
                            "probabilities": prediction.probabilities,
                            "model_sha256": predictor.metadata.weights_sha256,
                            "validated_for_room": False,
                        }
                    }
                scheduled, schedule_status = zones.snapshot()
                anomaly_events = []
                if args.tamper_model:
                    if monitor is None:
                        monitor = TamperMonitor(
                            args.tamper_model,
                            config,
                            window.values,
                            TamperSettings.model_validate_json(args.tamper_settings.read_text()),
                        )
                    anomaly_events = monitor.process(window.values, window.ts)
                for event in detector.process(window.values, window.ts, scheduled) + anomaly_events:
                    enriched = Detection(
                        event.type,
                        event.confidence,
                        event.payload
                        | {
                            "input_source": window.source,
                            "replay": True,
                            "zone_schedule_status": schedule_status,
                        }
                        | activity,
                    )
                    # Decisions use the whole window; timestamp them at its end.
                    outbox.enqueue(
                        enriched, args.device, args.room, window.ts + timedelta(seconds=duration)
                    )
                if args.pace:
                    stop.wait(duration)
                if not worker.is_alive():
                    raise RuntimeError("delivery worker stopped; events remain on disk")
            # Give the worker one bounded chance; offline data remains for --flush-only.
            stop.wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(timeout=6)
        print(json.dumps(outbox.counts()))
        outbox.close()


if __name__ == "__main__":
    main()
