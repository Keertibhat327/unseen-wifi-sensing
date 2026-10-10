# Future ESP32-S3 handoff — no firmware yet

V1 path: **ESP32-S3 → USB serial → Python gateway → HTTPS API**. The ESP32 never needs the cloud API key or Python inference. This supersedes the original brief's older “device posts events” wording.

Before enabling `EspCSIAdapter`, record the firmware commit, ESP-IDF version, board, antenna setup, bandwidth, channel and CSI config. Save actual CSV header plus sample lines for every packet/LTF combination emitted, including `first_word_invalid=1`, malformed lines, startup logs and disconnect/reconnect. Keep those samples local until MAC/address metadata is sanitized for test fixtures.

Then implement verified CSV parsing, signed imaginary/real pairing, declared-length checking, metadata-derived LTF mapping and timestamp handling in `detection/adapters/esp_csi.py`. Config determines retained LTFs, selected subcarriers and amplitude/phase. Never assume dataset subcarrier indices identify the same RF bins on the board. The gateway's `--serial` option currently raises `NotImplementedError` before accessing hardware.

M2 now processes complete standardized windows and cancels pending fall checks on missing windows. Hardware integration still needs a packet/window assembler handling partial serial reads, resets, dropped packets, measured cadence and explicit resampling. Reject unexpected shapes/rates; never silently reshape into a model's expected input. Establish a reference capture → adapter → saved window test before inference. Calibrate in the actual room, collect consented local recordings and retrain/evaluate under subject/recording-separated splits.

Keep the gateway outbox on durable local storage. Preserve logical device sequence counters across restarts; re-provision a new device ID if state is lost. Test offline operation, API rejection, reconnect storms and lost acknowledgements. Finally configure HTTPS, set the gateway-only ingestion secret and compare room state with ground truth. Hardware/software compatibility and detection accuracy are separate acceptance checks.

M5 adds a separate database-backed alert worker; it can run alongside the gateway on the laptop. Current replay emits `replay=true`, and real Telegram delivery suppresses it. Only the future verified live serial acquisition path should emit `input_source=live_csi` and `replay=false`; do not set those values on saved data. Confirm clocks and freshness before enabling alerts. Keep bot/database credentials on the worker host, never the ESP32. See [M5 operations](../docs/M5_DEPLOYMENT.md).
