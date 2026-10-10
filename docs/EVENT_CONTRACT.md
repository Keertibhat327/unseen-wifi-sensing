# Event contract v1

`POST /api/events`, header `X-API-Key: <INGEST_API_KEY>`:

```json
{
  "event_id": "182b0460-354a-4ab9-9c75-50e8c3cfcfa0",
  "schema_version": 1,
  "seq": 42,
  "sender_version": "gateway/0.1.0",
  "device_id": "room-a-receiver-01",
  "room_id": "room-a",
  "ts": "2026-10-09T06:30:00Z",
  "type": "occupancy",
  "confidence": 0.5,
  "payload": {"occupied": true, "source": "synthetic_demo", "validated": false}
}
```

All producers must supply UUID, schema version, durable per-device counter, sender version and timezone-aware timestamp. UTC is the internal and output representation; non-UTC offsets are normalized, naive timestamps rejected. Schema version 1 is required; missing and unknown versions are rejected. Sequences are nonnegative signed 64-bit integers. Confidence is finite and within [0,1]. The object payload is extensible (strict JSON, 16 KiB cap); unknown envelope fields are rejected so typos cannot silently disappear.

M5 caps complete request bodies at 32 KiB and refuses timestamps more than five minutes in the future. Historical events remain accepted. Private API responses disable caching. Alert intents are committed atomically with eligible events; duplicate UUID ingestion never adds another intent. Alert mode defaults to off. Real delivery requires explicit live-source metadata and freshness; current replay adds `payload.replay=true`. Telegram delivery itself is at-least-once. See [M5](M5_DEPLOYMENT.md).

Types: presence, motion, zone_entry, fall_suspected, fall_confirmed, occupancy and tamper_suspected; **reserved** fire_anomaly. M4 optionally emits `tamper_suspected` for sustained statistical distribution changes, with `intent_inferred: false`, a reference checksum, score and threshold. It does not establish malicious activity. Fire sensing has no validated implementation or claim here.

Identity stays local. Known identity/template fields (including nested `identity_prediction`, `subject`, `person_id` and `embedding`) are rejected. Payload extensibility is for anonymous event metadata; this field check is not a general filter for sensitive free text.

New events return 201. An identical event retry returns 200 with `duplicate: true`. A reused UUID with changed content, or reused `(device_id, seq)` with another UUID, returns 409. UUID and sequence constraints are enforced in SQL, including concurrent requests. The original event is never overwritten. Retry the exact envelope; do not issue another UUID after a timeout.

Sequences may have gaps and arrive out of order. They are not timestamps. Source timestamps determine current room status; late queued data cannot roll it backwards. A logical device counter survives gateway restarts. Multiple gateways must not share one logical device ID. NTP/host clock synchronization and future-clock skew rejection need production validation in M5.

Read-only routes require `READ_API_KEY`: `GET /api/events?room_id=...&limit=50&offset=0`, `/api/rooms/status`, `/api/devices`, `/api/zones`. Event listing has a 200-row limit and a bounded offset. This simple live timeline is not a stable export cursor under concurrent arrivals; M3/M5 exports should use a snapshot/cursor.

`POST /api/zones` requires the ingestion/admin key. Fields: `zone_id`, `room_id`, `armed`, optional aware `starts_at`, `ends_at`, and optional `weekly` entries with `weekdays` (Monday=0), `start`/`end` UTC wall times. Intervals include the start and exclude the end. M2 enforces room-zone schedules locally; `--sync-zones` polls this API with the read key. See [M2 details](M2_DETECTION.md).

M2 cancellation keeps the existing `fall_suspected` type with `payload.status: cancelled` and a reason. Fall events share `payload.incident_id`; consumers must distinguish suspicion, cancellation and confirmation. Occupancy payloads can contain `occupied: null` / `occupancy_status: unknown`; no numeric headcount is inferred. M2 confidence is explicitly an uncalibrated placeholder, with `validated: false`.

Do not send raw CSI, enrollment identifiers, names, embeddings or identity guesses in payloads. The bundled gateway sends only reduced anonymous demo summaries. Extensibility is not a privacy enforcement mechanism: arbitrary authorized third-party producers must be reviewed before use. M4 must add explicit payload policies if identity capabilities are introduced.
