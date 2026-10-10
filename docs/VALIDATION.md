# Validation history

## M5 local implementation — 2026-10-10

- Python suite: **64 passed, one skipped**. The skip is the explicit Postgres integration test without `TEST_POSTGRES_URL`. Added coverage for transactional event/alert idempotency, concurrent claims, expired leases, mock isolation, replay suppression, Telegram 429 and permanent errors, bounded lost-ack retries, secret redaction, body/future-time limits, readiness, production guards, outbox exhaustion, additive schema preservation and deployment allowlisting.
- `ruff check .` and `ruff format --check .`: passed. A local Python 3.14/older SQLAlchemy nullable-union incompatibility was fixed with explicitly typed nullable columns before the successful run. Existing FastAPI/Starlette asyncio deprecation warnings remain.
- Real HTTP smoke: **64 unique events**, one room/device, expected fall and anomaly behavior, and mock alert queue drained successfully by the separate worker. No Telegram networking or raw CSI upload.
- Dashboard: **two tests passed**, production Vite build passed. The Windows sandbox denied source-path resolution; the same build succeeded outside it. No UI redesign.
- Prepared `.deploy/api-m5` using the source allowlist; imported its app from the staged source and served `/health` successfully. No secrets, raw datasets, local identities or model weights copied.
- Added a Postgres service CI job and local compose setup. Docker CLI was present but its engine was unavailable. No local or hosted Postgres run is claimed, and GitHub CI has not executed here.

Per the user's choice, no hosting or Telegram accounts were configured. Hosted Vercel/Netlify builds, live Postgres, real Telegram delivery, browser CORS against hosting, rate limits and backup restoration remain pending. See [the deployment guide](M5_DEPLOYMENT.md).

## M4 — 2026-10-10

- `python -m pytest -q --tb=short`: **54 passed** with optional ML dependencies. New coverage includes explicit consent/default-off behavior, persistence and deletion, unknown identity results, input mismatch, cloud identity rejection, sustained anomalies/recovery/gaps, deterministic perturbations, privacy split isolation, activity-loader rejection of identity artifacts, clean-score reproduction and provenance refusal.
- `python -m scripts.smoke_local`: **64 unique events**, including statistical tamper warnings, one fall confirmation and one cancellation. Verified model annotations, synced schedules, anonymous payloads, one room and one device through real temporary HTTP API/gateway processes.
- WiFall privacy: five held-out-action folds × three representations, **15 numeric local research model exports**, plus majority controls. WiFall robustness: ten participant folds × two fixed models × seven conditions = **140 evaluations**, with ten saved normal references. Final runs are `artifacts/m4/privacy-wifall-exported` and `artifacts/m4/robustness-wifall-final`.
- Saved [aggregate tables and chart](../reports/m4/RESULTS.md); visually inspected the chart. Matplotlib uses a writable project cache. Existing scratch runs remain local and ignored; the final report names the final provenance directories.
- `ruff check .` and `ruff format --check .` passed. CI includes research tests in its ML job; hosted CI was not executed here. Existing Python 3.14 FastAPI/Starlette deprecation warnings remain.

No real identity enrollment, hardware tamper experiment, biometric accuracy claim, hosted deployment or frontend redesign. Identity storage is local plaintext and deletion cannot erase external backups. M4 research uses WiFall only. Full methodological limitations are in [the guide](M4_RESEARCH.md).

## M3 — 2026-10-10

- `python -m pytest -q --tb=short`: **48 passed** with ML dependencies installed. Covers timestamp gaps/duplicates, recording/content split isolation, train-only scaling, binary/multiclass export parity, Torch/NumPy CNN parity and a miniature training run, alongside prior API/detection tests.
- `python -m ruff check .` and `python -m ruff format --check .`: passed (63 files).
- `python -m scripts.smoke_local`: **62 unique events, one room, one device**, with optional numeric model annotations, exactly one fall confirmation and one cancellation, synced schedules and no raw CSI uploads. Corrected the smoke script's initial misplaced `--model` argument before the successful run.
- Windows sandbox denied Python temporary-directory creation; tests and HTTP smoke passed outside that sandbox. Existing FastAPI/Starlette Python 3.14 deprecation warnings remain.
- Prepared both actual archives and completed **64 fold model exports**: WiFall action LOPO, binary fall LOPO, negative-only binary LOAO, and WiGesture action LOPO. Saved splits, per-class metrics, predictions, dataset audits and model metadata. The initial WiGesture baseline run failed convergence; a separate run completed with a larger iteration ceiling.
- Generated [measured results](../reports/m3/RESULTS.md), CSV tables and the pooled binary confusion figure; visually checked the figure. Report generation succeeded despite a sandbox Matplotlib cache warning.
- Colab notebook parses as JSON and all four code cells compile locally. It was not executed in Colab. Added a dedicated ML CI job; GitHub CI has not been run here.

Models are advisory. Public-dataset recording labels do not measure real-time alarm accuracy; LOAO tests only unseen negative activities. Hardware/room transfer, serial acquisition and deployment remain unvalidated. No dashboard source changed in M3.

## M2 — 2026-10-10

- `python -m pytest -q --tb=short`: **36 passed**, including calibration, cleaning, stationary occupancy, fall confirmation/cancellation/timeout, gap handling, alternative rates/shapes, fractional timestamp rounding, input validation, UTC schedules and schedule-cache expiry.
- `python -m ruff check .`: passed; formatting completed with `ruff format .`.
- `python -m scripts.smoke_local`: **62 unique events, one room, one device** through a real temporary HTTP API and actual calibration/gateway CLIs. Verified exactly one fall confirmation, one cancellation, API-synced schedules and no raw CSI uploads.
- Dashboard `npm test`: **2 passed**, covering overnight UTC boundaries and disarm/absolute interval precedence.
- Dashboard `npm run build`: passed. Windows sandbox initially blocked Vite's source-path resolution; the same build succeeded outside the sandbox.
- Added the HTTP smoke check and dashboard tests to CI. CI itself has not been executed on GitHub here.

No real hardware, real-world detection accuracy, numeric headcount, model training or hosted deployment was validated. FastAPI/Starlette still emit upstream asyncio deprecation warnings under local Python 3.14. The M2 dashboard changes were build-tested; no new authenticated browser session was exercised.

## M1 — 2026-10-09

- `python -m pytest -q --tb=short`: **17 passed**, including a smoke test against the downloaded real WiFall archive. CI skips only that optional test if the archive is absent.
- `python -m ruff check .` and `python -m ruff format --check .`: passed.
- `python -m scripts.smoke_local`: passed. Started an isolated temporary loopback API, generated synthetic CSI, ran the actual gateway CLI, drained the outbox, and verified **13 unique events, one room, one device** and no raw CSI in event payloads. Its database, keys and process were isolated from user data.
- `npm run build` in `dashboard/`: passed with Vite 6.4.4. Dependency installation reported zero vulnerabilities.
- Opened the local dashboard in the browser; inspected its rendered empty/connection state and layout. Authenticated browser interaction and mobile breakpoint testing were not performed. API authentication and event reads were exercised in automated tests and the HTTP smoke test.
- Local runtime: Python 3.14.4, NumPy 2.4.4, installed FastAPI 0.115.5. FastAPI/Starlette emit upstream asyncio deprecation warnings on Python 3.14; no test failures. CI targets Python 3.12 and Node 22.

Not validated: real serial input, calibrated sensing accuracy, model training, Postgres integration, hosted deployment, Telegram delivery, or identity/research modules. Those are explicit later milestones, not implied by passing interface tests.
