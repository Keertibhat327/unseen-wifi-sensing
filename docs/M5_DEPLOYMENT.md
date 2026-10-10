# M5: deployment and operations

Implemented locally on 2026-10-10. The API package, dashboard build, mock alert delivery and operational tests are ready. No hosting accounts, hosted Postgres database or Telegram bot are configured yet, so **live deployment and real Telegram delivery remain unverified**. The user requested local setup and a deployment guide only.

## Local verification

```powershell
python -m pip install -e ".[dev,ml]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m scripts.smoke_local
```

From `dashboard/`, run `npm ci`, `npm test` and `npm run build`. The HTTP smoke starts its own temporary API/database, uses mock alerts, and never contacts Telegram. The Postgres test skips unless `TEST_POSTGRES_URL` is explicitly configured; CI has a disposable Postgres service for it.

## Database setup and schema upgrades

Configure `DATABASE_URL` in an uncommitted `.env`. For production use persistent Postgres with `sslmode=require` or stronger; prefer provider-supported certificate verification. Use separate databases for development, preview and production. SQLite is rejected whenever `APP_ENV=production` or `VERCEL` is set.

Run the explicit schema step from a trusted local terminal before starting/redeploying the API:

```powershell
python -m backend.db
```

M5's additive schema setup creates `alerts` and `schema_revisions` alongside existing `events` and `zones`. Existing rows are preserved, and rerunning it is tested. Revision 1 marks this bootstrap schema. It is **not** a general migration engine: future column changes require reviewed versioned migrations. Back up the database before upgrades and test restores. Request handlers never create schema or write local files.

For an optional local Postgres instance, start Docker Desktop first:

```powershell
docker compose up -d postgres
```

The checked-in compose file binds only `127.0.0.1:5433`, uses a named persistent volume, and has development-only credentials. Its development URL is `postgresql+psycopg://sentinel:local-development-only@127.0.0.1:5433/sentinel`. Keep `APP_ENV=development` for this local non-TLS instance. Do not reuse its password for hosting. `docker compose down` stops it while preserving the volume; deleting the volume deletes the database. Docker's engine was not running during this milestone, so no local Postgres execution is claimed.

## Prepare and deploy the API

The preferred path is an allowlisted deployment directory so training data, local identities, models and secrets cannot accidentally enter the API upload:

```powershell
python -m scripts.prepare_deploy --output .deploy/my-api
```

Use a new directory each time. A prepared, import-tested snapshot already exists at `.deploy/api-m5`; regenerate after code changes. The bundle contains backend and contracts modules, a top-level `app.py`, API-only requirements, Python 3.12 selection and Vercel configuration. It intentionally excludes `.env`, databases, recordings, ML dependencies and frontend files. Its import/health check passed locally; a hosted Vercel build has not been run.

Create a Vercel project using that prepared directory with the Vercel CLI, or put that allowlisted source into its own deployment repository. Configure these server-side variables in the project settings before deploying:

| Variable | Value |
|---|---|
| `APP_ENV` | `production` |
| `DATABASE_URL` | Hosted Postgres URL with explicit TLS mode |
| `INGEST_API_KEY` | Unique random secret, at least 32 characters |
| `READ_API_KEY` | Different random secret, at least 32 characters |
| `CORS_ORIGINS` | Exact dashboard HTTPS origin, no trailing slash; comma-separated if necessary |
| `ALERT_MODE` | Start with `off` or `mock` |

Generate keys with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Keep them out of Git, screenshots, frontend build variables and chat messages. Startup rejects short/example keys, non-TLS production database settings and non-HTTPS production CORS origins. `NullPool` prevents per-instance retained connection pools; use provider-recommended pooling and monitor connection limits.

The root repository also has a FastAPI `vercel.json`, `tool.vercel.entrypoint` and `.vercelignore`, but the prepared API directory is the simpler upload boundary. Vercel supports named FastAPI entrypoints and Python runtime configuration; see the [official FastAPI deployment guide](https://vercel.com/docs/frameworks/backend/fastapi) and [Python runtime documentation](https://vercel.com/docs/functions/runtimes/python). Provider quotas and free-tier eligibility must be checked when creating the account; no paid service is required by the code.

## Deploy the dashboard

Use a separate Netlify or Vercel static project with `dashboard/` as the project base, `npm run build` as the build command and `dist` as output. `dashboard/netlify.toml` specifies Node 22, SPA routing and security headers. Set only `VITE_API_URL` to the deployed API HTTPS URL. Never set an API key as a `VITE_*` variable: those values become public JavaScript. Enter the read key in the dashboard at runtime; it remains in tab memory.

Set the API's `CORS_ORIGINS` to the final dashboard origin, redeploy the API, then verify polling and connection/stale states. See [Netlify file-based configuration](https://docs.netlify.com/build/configure-builds/file-based-configuration/). The dashboard build and existing tests passed; hosted CORS and browser behavior remain live-deployment checks.

## Telegram alerts and the external worker

Alerts are disabled by default. `ALERT_MODE=mock` creates durable intents for recent `fall_confirmed`, `zone_entry` and `tamper_suspected` events. It never contacts Telegram. Restart the API after changing its alert mode, then run a separate worker using the **same database and mode**:

```powershell
python -m backend.alert_worker --once
python -m backend.alert_worker
```

The first command processes one bounded batch and exits; the second polls until Ctrl+C. The worker belongs on an always-on local machine or separately managed worker host, not inside a serverless request. In v1 it can run alongside the laptop gateway. If the machine sleeps, delivery stops; queued alerts remain durable but can expire. The CLI pauses between attempts to avoid bursts to one chat. Run one worker for a single chat; leases protect accidental overlapping workers but do not implement a distributed Telegram rate limiter.

When you have a bot and intended chat, obtain the token through Telegram's bot setup, initiate the chat with that bot, and configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` privately on the API and worker. Set `ALERT_MODE=telegram` only when you intend to send notifications. This implementation uses [Telegram's official `sendMessage` API](https://core.telegram.org/bots/api#sendmessage), plain text and protected content. No real message has been sent during development.

Live delivery requires event payload `input_source=live_csi` and `replay=false`. Today's gateway explicitly marks replay events, so they cannot trigger real Telegram notifications. Do not relabel replay as live to test delivery; use mock mode until a verified live acquisition path exists. Alert eligibility and delivery both expire old events using `ALERT_MAX_AGE_SECONDS` (default 300). Events more than 30 seconds ahead of the server clock do not queue alerts. This prevents historical/offline backlog from becoming a burst of apparent current emergencies.

An event and its one-per-UUID alert intent commit in the same SQL transaction. Identical ingestion retries do not create another alert. The worker claims a 60-second lease, releases the database transaction before networking, and records delivery status afterward. Expired leases are recoverable after a crash. Telegram 429 responses honor `retry_after`; transient errors retry with capped exponential delay; permanent rejection and exhausted attempts become `dead`. The default attempt cap is eight.

Telegram delivery is **at-least-once**, not exactly-once. If Telegram accepts a message but its acknowledgement is lost, retrying can duplicate it. Messages include the event UUID for recognition. Transport errors are stored as fixed categories rather than exception strings containing bot-token URLs. After correcting configuration, `--requeue-dead` retries still-fresh dead alerts; expired alerts are not revived. Mock rows and real rows remain separate and cannot be accidentally switched to real delivery.

`GET /api/alerts/status` with the read key returns counts by mode/status. Watch pending/dead counts and worker output. Messages contain only event type, anonymous room/device IDs, time and UUID; raw CSI and arbitrary event payloads are not copied.

## Operational limits and live acceptance

- API requests have a 32 KiB body cap; payloads retain the 16 KiB contract cap. Private responses use `Cache-Control: no-store`. Pagination is bounded. Add platform rate limits/WAF rules before exposing ingestion publicly; there is no unreliable per-instance in-memory rate limiter.
- Ingestion rejects event times over five minutes in the future. Old events remain historical records, never fresh room state. Maintain gateway/server clocks.
- Gateway outbox capacity defaults to 100,000 retained rows, adjustable with `--outbox-max-items`. Full capacity stops enqueueing explicitly without consuming a sequence or deleting old events. It is a row-count bound, not a disk-byte guarantee; monitor disk space and quarantine counts.
- `/health` is liveness only. Authenticated `/api/ready` checks event/alert tables; it does not prove the worker is running or Telegram is reachable.
- No automatic event/alert deletion is implemented. Preserve UUID/sequence history for idempotency and monitor database growth. Plan retention/tombstones before deleting historical events; backups and raw recordings have separate retention needs.
- Rotate read/ingestion keys separately and redeploy/reconfigure consumers. Current keys are project-wide, not per-device credentials or user accounts. This is a personal research deployment, not a finished multi-tenant security system.

After you create the services, set `READ_API_KEY` locally and run:

```powershell
python -m scripts.check_deployment --url https://your-api.example
```

This is read-only: HTTPS health, anonymous-read refusal, authenticated readiness/read endpoints and cache headers. It sends no alerts. Then verify hosted Postgres duplicate/concurrent ingestion on a dedicated staging database, browser CORS, provider request/connection limits, worker restart/expiry, actual Telegram delivery with an authorized live event, and backup restoration. The live checks remain pending; the project should not claim a validated emergency service.
