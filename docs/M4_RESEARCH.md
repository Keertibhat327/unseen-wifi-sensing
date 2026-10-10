# M4: research, anomaly monitoring and optional identity

Implemented and locally validated 2026-10-10. This milestone uses saved recordings only. It includes no transmission, interference, traffic capture or firmware operations. See [measured results and chart](../reports/m4/RESULTS.md).

## Reproduce the offline experiments

Install `python -m pip install -e ".[dev,ml]"` and prepare WiFall using the [M3 guide](M3_ML.md). The experiments below require new output directories; they do not overwrite earlier runs.

```powershell
python -m research.privacy --data data/wifall.npz --output artifacts/my-privacy --factor 4 --acknowledge-research-use
python -m research.robustness --data data/wifall.npz --run artifacts/m3/wifall-fall-lopo --output artifacts/my-robustness
```

Privacy compares flattened amplitude windows, four summary features per carrier (mean, standard deviation, range, temporal change), and block-averaged lower-rate windows. The factor comes from the command; input rate and dimensions come from the prepared dataset. Averaging by four gives one quarter of the configured sample rate. This is an offline representation experiment, not a validated hardware anti-aliasing filter.

The attacker is a balanced ridge classifier with fixed alpha 10 and training-only feature standardization. The test set holds out an entire action while retaining the same known participants. Recording IDs and duplicate-content hashes cannot cross partitions. This measures transfer across activities, not recognition of unseen people. It refuses datasets where a held-out action does not cover every participant. The actual M4 experiment uses WiFall; WiGesture has not been evaluated in M4.

Each privacy run writes split membership, per-class metrics, confusion matrices and numeric attacker models with metadata, representation settings, UTC training date and source hash. These models stay under ignored local `artifacts/`. The gateway activity loader explicitly refuses their identity task. They are not enrollments in the optional identity module. Dataset identities may reflect session/environment differences, so this is evidence of potentially identifying information, not a validated biometric system. Dataset reuse terms still need clarification before publishing weights or source data.

The majority-class control reaches 67.5% mean accuracy because WiFall ID2 dominates. Raw, reduced and lower-rate macro-F1 are 0.426, 0.522 and 0.465 versus 0.081 for the majority control. Lower dimensionality alone does not guarantee privacy; reduced features perform better by macro-F1 in this experiment. These representations are **not** the production event payload. The API still receives anonymous events, not feature arrays.

Robustness uses the original exported M3 binary models and their recording splits. It checks input metadata, archive hash, windowing and training-recording provenance. `configs/research.json` fixes seed 42 and seven conditions. Noise and drift use the saved training standardizer's carrier scale. Packet loss independently removes frames except the first and holds the last received frame. Noise is clipped at zero for amplitude. No perturbation mutates the original array; models are not retrained or selected on these results.

Results include all ten participant folds for both models: 140 model/condition/fold rows with per-class metrics. Each fold also saves a statistical normal reference fitted only to training windows. Its threshold exceedance fraction is reported separately from classification metrics. Labels in public recordings are weak, recording-level labels; none of these scores measures live fall alarm accuracy.

Create the aggregate report from the completed local runs:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.cache/matplotlib'
python -m research.report --privacy artifacts/m4/privacy-wifall-exported --robustness artifacts/m4/robustness-wifall-final --output reports/m4
```

The report builder is a WiFall M4 snapshot with dataset-specific interpretation. It copies aggregate metrics only; source data and local identity templates remain excluded.

## Statistical anomaly monitoring

`detection/tamper.py` summarizes each window by amplitude mean, spread, mean temporal change and endpoint change. A local representative-normal reference stores median/MAD scales, with explicit numerical floors. The threshold is the larger of six robust-scale units or 1.2 times the training 99th-percentile score. These fixed heuristics are not calibrated probabilities.

Fit a reference from a replay that includes expected normal activities in the actual room:

```powershell
python -m research.calibrate_tamper --replay data/normal-room.jsonl --config configs/synthetic.json --output .gateway/normal.csimodel --confirm-representative-normal
```

`normal-room.jsonl` must be collected or supplied by you. A quiet-only baseline may flag ordinary movement; the synthetic HTTP smoke deliberately uses that contrast to test delivery, not to measure false alarms. Use the real acquisition config when hardware becomes available.

Add `--tamper-model .gateway/normal.csimodel` to the normal calibrated gateway command. This is optional and off unless a model is supplied. `configs/tamper.json` requires three consecutive abnormal windows, emits one `tamper_suspected` per sustained episode, and rearms after three normal windows. Timing gaps reset accumulated evidence; invalid or overlapping windows are rejected. A gap alone does not produce a tamper event.

The payload contains a scalar score, threshold, reference checksum and `intent_inferred: false`. No raw CSI or identity data is attached. Fall and occupancy decisions remain independent. The reference uses the same strict shape/rate/preprocessing artifact checks as other models.

In the offline experiment, clean held-out windows had about 0.1% mean threshold exceedance; strong noise 100%; large drift 57.2%; packet loss about 0.1%. These are window fractions, not sustained gateway alert rates. The monitor has blind spots and cannot distinguish malicious interference from ordinary environment or device changes.

## Optional local identity

The module is disabled by default and has no gateway/API integration. No real person was enrolled while implementing M4; tests use generated numeric data. It is an experimental nearest-template matcher with an unknown result and ambiguity margin, not authentication.

First read the consent statement:

```powershell
python -m identity.cli consent
```

Only the consenting participant's supplied recordings should be enrolled. After explicit consent:

```powershell
python -m identity.cli --enable-local-identity enroll --subject my-local-alias --replay data/my-consented-recording.jsonl --accept-consent local-identity-research-v1
python -m identity.cli --enable-local-identity list
python -m identity.cli --enable-local-identity match --replay data/my-test-recording.jsonl
python -m identity.cli --enable-local-identity delete --subject my-local-alias
```

Use `--config` for a non-default adapter. Matching thresholds are configurable with `--max-distance` and `--min-margin`; defaults are experimental, not measured acceptance/error rates. Mismatched input rate, carrier order, representation, shape or consent version is refused. An enrollment needs at least five windows and never silently replaces an existing alias.

`.identity/local.db` stores the consent statement/version, UTC consent date and numeric feature center/scale plus shape/rate/dataset metadata. It does not copy raw recordings. Deletion removes both consent and derived template, uses SQLite secure deletion and compacts the database; the subject is absent after reopening. It does **not** delete the input recording supplied by you or copies made by backups, and cannot promise forensic erasure on SSDs.

Storage is local plaintext SQLite, protected by the host account, not application encryption. Use an unsynced directory and protect the operating-system account. UNC shares are refused; software cannot reliably identify all cloud-synced folders. Do not publish identity databases or local research weights. CLI match output is local terminal output and should not be redirected into cloud logs.

The event validator rejects known identity/template fields recursively, and the gateway imports no identity module. The field denylist is defense in depth, not a general data-loss prevention system: arbitrary free text can still contain sensitive information, so retain anonymous payload conventions.

## Verification and sources

Tests cover default-off behavior without filesystem writes, explicit consent, persistent enrollment, unknown matches, metadata refusal, deletion after reopening, recursive cloud rejection, anomaly persistence/recovery/gaps, deterministic perturbations, split isolation, research artifact isolation and clean-score reproduction. The HTTP smoke exercises real tamper-event delivery alongside fall confirmation/cancellation.

The attacker uses documented [scikit-learn RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) and [training-only preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html). Dataset provenance and license caveats are in the M3 guide. No claim is made to reproduce a published privacy or robustness benchmark.

M5 remains: hosted deployment, Telegram alerts, operational limits and deployment verification. Real ESP serial input stays explicitly TODO until an actual capture is available.
