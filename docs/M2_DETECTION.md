# M2: calibrated local detection

Implemented on 2026-10-10. This milestone adds a deterministic, testable heuristic pipeline. It is not a trained classifier or evidence of real-world fall-detection accuracy. Frontend styling remains independent and ready for your references.

## Run the complete M2 flow

Start the API and dashboard as described in README.md. From the repository root, with your Python environment active:

```powershell
python -m simulator.generate --calibration-only --cycles 1 --output data/background.jsonl
python -m detection.calibrate --replay data/background.jsonl --room studio --output .gateway/studio-calibration.json --confirm-empty-room
python -m simulator.generate --cycles 1 --output data/demo.jsonl
python -m gateway.run --replay data/demo.jsonl --calibration .gateway/studio-calibration.json --zones configs/zones.example.json --allow-local-http --pace
```

The first two commands build a reference from synthetic empty-room data. For real recordings, `--confirm-empty-room` is an explicit assertion about the capture, not an automatic label detector. Do not calibrate from an activity/fall recording. Recalibrate after changing room, placement, radio configuration or preprocessing. The gateway refuses the wrong room or adapter fingerprint.

Use `--config` on generator, calibration and gateway to select another adapter configuration. Use `--settings configs/detection.json` on calibration/gateway to change detector thresholds. M2 inference supports amplitude only; phase needs its own preprocessing. A saved calibration is not an ML model.

## Pipeline and behavior

1. **Validate:** require a finite real array of the configured shape, declared sample rate, preprocessing fingerprint and UTC-aware source time. Repeated/overlapping windows are rejected (one microsecond tolerance accommodates timestamp rounding).
2. **Calibrate:** use per-carrier median background and robust noise estimates with a floor. Require at least five windows and reject substantial background drift between windows. This cannot prove the room was empty; capture conditions remain the operator's responsibility.
3. **Clean:** replace isolated temporal outliers on individual carriers. Preserve broad simultaneous transients across carriers so cleaning does not erase the candidate impact. This rule is configurable and synthetic-tested, not a guarantee against radio artifacts.
4. **Extract:** measure within-window motion, persistent displacement from the empty baseline, and the strongest coherent temporal change. Scores are expressed relative to background noise.
5. **Presence/occupancy:** start unknown. Sustained activity or background displacement enters occupied. A quiet but displaced signal retains occupied. Sustained quiet **and** baseline similarity permits estimated empty. Hysteresis avoids rapid state changes. There is no person counter: payload always reports `count: null`, `count_supported: false`.
6. **Fall:** a coherent impact creates `fall_suspected`. Only subsequent complete windows count toward the post-fall stillness check. Continuous stillness confirms once; resumed motion or a timeout cancels. A confirmed state holds occupancy until movement or a data gap resets it. Cooldown limits repeated incidents.
7. **Room zones:** emit `zone_entry` only on a known-empty → occupied transition in the matching armed room. Initial unknown → occupied, reconnects and arming an already-occupied room do not claim an entry. This is room-level sensing, not geometric boundary localization.

State resides at the local gateway. Restart begins unknown and requires fresh evidence; calibration and event sequence/outbox remain durable. A gap larger than the configured tolerance resets occupancy evidence and cancels a pending fall check. Missing windows and EOF never count as stillness. If a stream stops entirely, the API/dashboard marks the last source data stale; no new confirmation is produced.

All durations derive from sample rate and window length. M2 expects complete, regularly sampled windows with start timestamps. The decision event timestamp is the window end. Raw hardware packet assembly, reliable measured cadence and resampling remain future integration work; the serial adapter stays explicitly blocked.

## Events and interpretation

- `occupancy` is sent for every accepted window as a health/status update; `presence` records changes.
- `motion` marks transitions with `payload.active` true/false.
- Fall suspicion, confirmation and cancellation share `payload.incident_id`. Cancellation retains the v1 `fall_suspected` type and sets `status: cancelled` plus `reason: movement_resumed | stillness_timeout | data_gap`. Consumers must inspect status, not alert solely on event type.
- Events include calibration and detector-settings fingerprints, input source, schedule status and `validated: false`.
- `confidence: 0.5` is explicitly an **uncalibrated placeholder**, not a probability. The dashboard labels it accordingly. M3 evaluation must replace this with justified model scores/calibration.

Even an empty-like signal may hide a motionless person. Do not treat an estimated empty room as a verified emergency headcount. Real-room calibration, representative recordings, false-positive/false-negative measurement and retraining are separate acceptance steps.

## UTC schedules and API synchronization

`configs/zones.example.json` is a local authoritative schedule file. The gateway rereads it per window; use atomic file replacement when editing during replay. Invalid files stop processing rather than silently change protection.

The API uses the same `Zone` model. `armed` is the master switch. Optional `starts_at`/`ends_at` set a half-open UTC interval. Optional `weekly` entries are ORed within that interval:

```json
{
  "zone_id": "studio-night",
  "room_id": "studio",
  "armed": true,
  "weekly": [{"weekdays": [0, 1, 2, 3, 4], "start": "22:00:00", "end": "06:00:00"}]
}
```

Weekdays use Monday=0; overnight schedules belong to their start day. Start is inclusive and end exclusive. Schedule evaluation uses the decision's UTC source time. Weekly times are UTC wall times, not local times or DST rules.

To use API-controlled schedules, POST the zone using the ingestion key, then replace `--zones ...` with `--sync-zones`. The gateway reads with `READ_API_KEY`, refreshes in the delivery worker approximately every ten seconds, and keeps successful snapshots for at most sixty seconds. Network backlogs can delay refresh. Expired/unavailable snapshots suppress zone evaluation, while occupancy and fall detection continue; every event includes `zone_schedule_status`. Local schedules are preferable for offline operation. API snapshots are not persisted across restarts. Replaying historical data against today's API configuration is an experiment, not a reconstruction of historical arm/disarm state; use an appropriate saved local schedule.

## Validation and next milestone

Tests cover calibration reuse/refusal, isolated glitches versus coherent impacts, stationary occupancy, hysteresis, confirmed/cancelled/timed-out falls, missing windows, timestamp/shape rejection, alternate rates/carrier counts, schedule boundaries, disarming and stale schedule expiry. A label-scrambling test proves inference does not read simulator scenario names. The real HTTP smoke test exercises generation → calibration CLI → gateway → API with schedule sync and both fall outcomes.

M3 should add verified dataset loading and recording-grouped train/test splits, train-only standardization, baseline/CNN models and reproducible metrics. Real packet assembly and room recordings will still be needed before hardware accuracy can be established.
