# Decisions

- **M2 uses evidence, not scenario labels.** Inference reads only signal arrays and timestamps. Calibration is explicit; restarts and gaps require fresh occupancy evidence. Confirmed falls require subsequent continuous stillness. Unknown occupancy and unsupported headcount are represented explicitly.
- **Room zones only.** Alert on a known empty-to-occupied transition in an armed room; do not imply geometric localization or invent entry after reconnect. UTC weekly schedules and bounded-age API snapshots are supported.

- **M1 first, compatibility now.** Preserve the milestone stop in the brief while establishing interfaces that otherwise cause rework. No unsupported claim of model accuracy or hardware integration.
- **Gateway owns cloud transport.** Python runs locally on a laptop or edge computer. ESP32 firmware sends serial CSI. Future embedded inference can implement the same event contract independently.
- **Persistent UUID deduplication.** A primary key and `(device_id, seq)` uniqueness provide database-level safety under retry races. A duplicate must have identical canonical content.
- **Read/write credentials are distinct.** The browser only needs read access. Keys are never Vite variables, URL parameters or repository secrets. Per-device provisioning and full user auth are future work.
- **Config is part of the model.** Equal array sizes do not imply equal features. The complete preprocessing fingerprint joins shape, rate and count in compatibility checks.
- **No pickle artifacts.** Numeric NPZ weights avoid executable deserialization. A future sklearn/CNN exporter must explicitly serialize its inference parameters or adopt a validated format with the same metadata contract.
- **Conservative hardware stub.** Raising a clear TODO is safer than decoding plausible-looking data with incorrect LTF offsets. Mixed radio formats need an explicit verified router before they can enter fixed-shape windows.
- **UI is replaceable.** API calls and event types remain stable while visual references can change layout and styling tomorrow.
