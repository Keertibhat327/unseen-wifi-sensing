# Learning the project

**CSI** measures the radio channel across frequency bins called **subcarriers**. A person can change reflected signals, but a signal change does not uniquely identify a fall or the number of people. **Windowing** groups samples over time. **Domain shift** means a new room, device, person or antenna setup can change the data distribution enough to invalidate a model's accuracy.

## contracts
An envelope describes when, where and how an event was produced. UUIDs identify one event; sequence counters describe producer order. Idempotency means repeating the same request has no extra effect.

Exercises: resend the same event; change its payload while keeping its UUID; send a timezone-offset timestamp and inspect UTC output.

## backend
FastAPI validates input and SQLAlchemy persists it. Database uniqueness protects against concurrent retries; a Python set would be lost when a serverless instance restarts. Room status uses source time because network arrival order is unreliable.

Exercises: send an older event after a newer one; restart the API and read events; try a read key on a write route.

## dashboard
React renders snapshots fetched every three seconds. The browser keeps the read key in memory. A failed fetch shows an error instead of turning old values into apparent live data. Signal bars are decorative, not streamed CSI.

Exercises: stop the API and inspect the error state; filter the event list; change the polling interval and observe request volume.

## detection and adapters
Each adapter maps source bytes into one two-dimensional array. Amplitude is the magnitude of a complex pair; phase is its angle. Byte order and carrier ordering matter. M2's calibration estimates the empty-room baseline and noise per carrier. Cleaning removes isolated glitches while preserving coherent changes. Presence uses hysteresis; falls require an impact followed by continuous stillness. See docs/M2_DETECTION.md for modules and state transitions.

Exercises: decode a 3+4j pair; change the carrier selection order; compare phase after reversing complex-pair order.

## gateway
The local SQLite outbox writes an event before networking. If the API commits but the reply is lost, retrying the same UUID is safe. Permanent errors stay quarantined for inspection. A separate thread handles transport so ordinary retry waits do not delay replay collection.

Exercises: replay while the API is offline; restart the gateway and flush; use the wrong key, then correct it and explicitly requeue.

## simulator
A seeded random generator produces quiet windows, motion bursts, a displaced stationary background, an injected spike and stillness. Scenario labels are only ground truth annotations: M2 inference never reads them. Separate quiet recordings provide synthetic calibration. No raw windows are sent to the API.

Exercises: change the random seed; use a different carrier count and matching byte range; inspect the API payload and confirm no CSI array is present.

## ml
Model shape, rate and preprocessing are part of the artifact contract. A same-shaped array with reordered carriers is still incompatible. M3 splits by recording/person; overlapping windows from one recording never cross the train/test boundary. Standardization is fit using training data only. Logistic regression summarizes windows; the CNN learns temporal filters. Both export numeric weights for NumPy inference. See [the M3 guide](docs/M3_ML.md) for preparation, training, splits and evaluation.

Exercises: run one participant-held-out fold; inspect its splits and confusion matrix; compare fold means with pooled scores; explain why negative-only LOAO cannot measure fall recall; change the configured rate and observe refusal; change amplitude to phase while keeping the shape unchanged.

## research (M4)
Privacy leakage measures how much sensitive information an exposed representation reveals. M4 fits a local attacker on known participants and tests held-out actions, comparing raw, summary and averaged lower-rate representations. Macro-F1 matters because the majority participant dominates raw accuracy. Robustness experiments perturb recorded data in software only and evaluate fixed exported models. They do not transmit interference. A statistical anomaly can result from an ordinary environment change; it cannot prove malicious intent.

Exercises: reproduce [M4](docs/M4_RESEARCH.md); explain why reduced features retain identifying information; compare clean and noisy scores; contrast per-window anomaly fractions with sustained gateway events; identify the packet-loss blind spot.

## identity (M4)
Identity is implemented but off by default. Explicit local opt-in and consent are required for enrollment. Numeric templates and consent stay in local SQLite and are deleted together. The matcher can return unknown; its candidate result is experimental, not authentication. Anonymous presence remains usable without enrollment, and gateway/API paths do not invoke identity.

Exercises: inspect the consent text without enabling storage; run synthetic enrollment/deletion tests; confirm metadata mismatch is refused; verify identity fields are rejected by the event contract. Do not enroll another person without their consent.

## firmware and deployment (M5)
Firmware will collect CSI; the gateway runs Python. Serverless APIs cannot rely on local disk or an always-running process. M5 stores alert intents in the same transaction as events; a separate worker delivers them using recoverable leases. API idempotency prevents duplicate intents, but Telegram's lost acknowledgements can still produce duplicate external messages. Mock mode exercises the queue without contacting anyone. Minimal deployment packaging excludes local identities, raw data and trained weights.

Exercises: run the mock alert worker; simulate a timeout and inspect retry/dead status; verify production refuses SQLite; inspect the prepared API file list; distinguish health from readiness. Next, configure staging services using [M5](docs/M5_DEPLOYMENT.md), then test hosted Postgres/CORS and alert delivery. Hardware still needs a real serial header and measured packet cadence.
