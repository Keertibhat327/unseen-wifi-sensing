# CSI Sentinel

A camera-free Wi-Fi CSI sensing research platform. **M1–M4 and M5's local implementation are complete**; live deployment and hardware validation remain pending. The detector uses calibrated local heuristics, tested on synthetic signals; optional learned predictions are advisory. This is a student research project, not a validated emergency or medical system.

## What works now

- FastAPI + persistent SQL events, room state, device health and UTC zone intervals.
- Versioned event envelope, database-enforced UUID idempotency and per-device sequence uniqueness.
- React dashboard using polling, with rooms, event timeline, health, connection/error/stale states.
- Seeded synthetic CSI and replay → calibrated local detection → SQLite outbox → API.
- Empty-room calibration, signal cleaning, presence/occupancy hysteresis and post-fall stillness checks.
- Fall confirmation/cancellation, gap handling, room-zone schedules and API schedule synchronization.
- HTTPS transport, retry/backoff, lost-acknowledgement safety and quarantined permanent failures.
- Configurable CSI adapter interface, verified WiFall CSV adapter and explicitly blocked ESP-CSI stub.
- Verified WiFall/WiGesture preparation, recording-grouped evaluation, logistic regression and a small 1D-CNN.
- Atomic trained model bundles with required metadata, NumPy inference and strict input/preprocessing checks.
- Offline privacy/robustness experiments, optional sustained anomaly monitoring and explicit-consent local identity with deletion (disabled by default).
- Tests, linting, CI and deployment configuration scaffolds.
- Transactional alert queue, Telegram transport with mock mode, retries/expiry and a separate worker.
- Allowlisted API deployment packaging, production guards, readiness checks and a Postgres CI job.

## Run locally (PowerShell, from this repository)

Python 3.11+ and Node 22+ are recommended. Use an isolated Python environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env`: set **different** long random `INGEST_API_KEY` and `READ_API_KEY` values. Do not commit it. `python -c "import secrets; print(secrets.token_urlsafe(32))"` generates one key.

Initialize the local database explicitly, then start the API:

```powershell
python -m backend.db
python -m uvicorn backend.api.index:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd dashboard
npm ci
npm run dev -- --host localhost
```

Open http://localhost:5173 and enter `READ_API_KEY`. The dashboard holds it only in tab memory. No credential is embedded in the bundle. Its API URL defaults to http://127.0.0.1:8000; override with `dashboard/.env` using the supplied example if needed.

In another terminal at the repository root, with the Python environment active:

```powershell
python -m simulator.generate --calibration-only --cycles 1 --output data/background.jsonl
python -m detection.calibrate --replay data/background.jsonl --room studio --output .gateway/studio-calibration.json --confirm-empty-room
python -m simulator.generate --cycles 1
python -m gateway.run --replay data/demo.jsonl --calibration .gateway/studio-calibration.json --zones configs/zones.example.json --allow-local-http --pace
```

Generate immediately before replay for current timestamps. The dashboard shows stale/unknown when source data is over 30 seconds old. Event times are UTC; the browser displays local time. Replay retains original timestamps so old recordings never appear as live hardware.

Retry a retained offline queue after starting the API:

```powershell
python -m gateway.run --flush-only --allow-local-http
```

4xx configuration errors are quarantined, not deleted. Fix the cause, then add `--requeue`. Ctrl+C leaves unsent events on disk. Keep `.gateway/outbox.db`: it holds the device sequence counter. After losing it, use a **new logical device ID** (`--device simulator-02`) to avoid reusing old sequence numbers. One producer owns each logical device ID.

## Verify

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m scripts.smoke_local
cd dashboard
npm run build
npm test
```

Tests cover concurrency, timezone normalization, reserved events, out-of-order delivery, persistence, queue retry/acknowledgement semantics, non-default dimensions, preprocessing and model mismatch rejection. An optional smoke test reads `.cache/WiFall.zip` if you download it from the dataset source. CI uses synthetic fixtures and does not fetch research data.

## Interfaces and deployment

See [M5 deployment and alerts](docs/M5_DEPLOYMENT.md), [M4 research and local identity](docs/M4_RESEARCH.md), [research results](reports/m4/RESULTS.md), [M3 training and inference](docs/M3_ML.md), [measured ML results](reports/m3/RESULTS.md), [M2 behavior and calibration](docs/M2_DETECTION.md), [event contract](docs/EVENT_CONTRACT.md), [architecture](docs/ARCHITECTURE.md), [assumptions](docs/ASSUMPTIONS.md), [hardware handoff](firmware/README.md), and [learning guide](LEARNING.md).

Prefer `python -m scripts.prepare_deploy --output .deploy/my-api` to build a minimal Vercel API package containing backend/shared contracts only. Configure `APP_ENV=production`, persistent TLS Postgres `DATABASE_URL`, two distinct random keys of at least 32 characters and exact HTTPS `CORS_ORIGINS`. Run `python -m backend.db` explicitly before deployment; it adds M5 tables without deleting existing events. Request handlers never initialize schema or write local state. SQLite is refused in production. See the M5 guide for local Postgres, packaging, worker setup and the still-pending hosted checks.

Deploy `dashboard/` separately as a static Vite app (Vercel or Netlify). Set `VITE_API_URL` to the API's HTTPS URL. No paid APIs are used; verify hosting providers' current free-tier limits when deploying. Alerts default to off. For local testing set `ALERT_MODE=mock`, restart the API and run `python -m backend.alert_worker`. Real Telegram is opt-in and suppresses all current replay/simulator events.

## Milestone boundary

M2 is complete at the software/synthetic-validation boundary: room calibration, cleaning, stateful presence/occupancy, schedule-aware room zones and fall suspicion → post-fall stillness → confirmation/cancellation. The gateway no longer uses simulator labels. Occupancy is an estimate, not a headcount; quiet CSI alone does not establish an empty room.

M3 is complete with 64 exported fold models, participant-held-out evaluation on both datasets, binary leave-action-out false-alarm diagnostics, a Colab notebook and reproducible reports. WiFall binary mean fold accuracy is 62.5% for the baseline and 60.0% for the CNN; these results do not establish reliable real-world alarms. Install `.[dev,ml]` to train and run the optional ML tests. See the M3 guide for commands and evaluation limitations.

M4 is complete at the offline/synthetic-validation boundary: 15 local privacy attacker exports, 140 robustness evaluations, ten saved statistical references, optional gateway anomaly events and consent/deletion-controlled local identity. Reduced features are not guaranteed anonymous; statistical anomalies do not establish malicious tampering. No real person was enrolled.

M5's code, mock alerts, local packaging/build checks and deployment guide are complete. **Hosted deployment, real Telegram and Postgres execution are not yet validated** because services are not configured. UI references can reshape the dashboard independently of these contracts.
