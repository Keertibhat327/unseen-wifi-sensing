# M3: reproducible ML pipeline

Implemented 2026-10-10. M3 includes verified WiFall and WiGesture loaders, timestamp-aware windows, training-only per-subcarrier standardization, scikit-learn logistic regression, a small PyTorch 1D-CNN, grouped evaluation, safe numeric exports and optional gateway annotations. It does not turn public-dataset scores into validated fall alarms.

## Setup and preparation

From the repository root with the Python environment active:

```powershell
python -m pip install -e ".[dev,ml]"
python -m ml.prepare --archive .cache/WiFall.zip --dataset wifall --output data/wifall.npz
```

The archive is already downloaded locally in this workspace. On another machine, download `WiFall.zip` from [RS2002/WiFall](https://huggingface.co/datasets/RS2002/WiFall/tree/main). Raw archives and prepared arrays are ignored by Git. No original data is redistributed in the project.

Preparation reads `.csv` members directly inside the ZIP. Labels and participant IDs come from verified paths. Every window retains a recording ID and signal-content hash for leakage checks. Numeric arrays are stored in `.npz` with `allow_pickle=False`; audit metadata is saved alongside them.

`configs/wifall.json` specifies the adapter, target rate and window length. `configs/windowing.json` specifies stride, duplicate handling and gap tolerance. Defaults produce one-second non-overlapping windows. The numerical rate/carrier dimensions live in configuration, not training algorithms. If changing the adapter, also select an appropriate windowing config.

Observed irregular timing is handled explicitly: same-timestamp rows are averaged; gaps over 50 ms split a recording into segments; each segment is linearly interpolated to the configured target grid. No interpolation crosses those gaps and no missing endpoint is extrapolated. Partial windows are dropped. A segment remains part of the same original recording for splitting. The target rate is a resampling choice, not a claim that hardware sampled uniformly at that rate. This is amplitude interpolation, not a phase-calibration or anti-aliasing pipeline.

## Audited data

| Dataset | Included recordings | Source rows | Repeated timestamps | Gaps split (>50 ms) | Prepared windows |
|---|---:|---:|---:|---:|---:|
| WiFall | 928 | 232,941 | 13,180 | 89 | 2,076 |
| WiGesture | 118 | 306,352 | 19,967 | 118 | 3,328 |

WiFall recordings span roughly 1.811–3.976 seconds in the inspected archive, rather than the card's claimed 60 seconds. ID2 contributes 640 recordings / 1,406 windows; other participants have 32 recordings each. Results therefore report unweighted fold means as well as pooled confusion counts.

WiGesture contains 119 CSV files under `dataset/dynamic/ID*/<gesture>.csv` and `dataset/static/ID*/GestureN.csv`. The static `taget` column contains a participant code, so static action labels come from the filename. One dynamic recording has an unresolved contradiction: `ID8/waveright.csv` says `waveleft` in every row. The loader rejects it by default. Our run explicitly excludes it and records that exclusion:

```powershell
python -m ml.prepare --archive .cache/WiGesture.zip --dataset wigesture --exclude-recording dataset/dynamic/ID8/waveright.csv --output data/wigesture.npz
```

The same byte adapter is used only because both actual archives were inspected and share that encoding. No identity target is trained. Dataset licenses were not established from the cards; clarify rights before redistributing recordings or trained derivatives.

Archive SHA-256 values used:

- WiFall: `3a39d71aac6eaf8f37414c5d675323297e461b288bf4e70f8f42a9f6b94b5485`
- WiGesture: `acfa61c3f01a85898291fcafd6af5f5b3583ce1b65b889f077010f93dcd6ca2f`

## Train and evaluate

Each output directory must be new; completed runs are never overwritten.

```powershell
python -m ml.train --data data/wifall.npz --output artifacts/my-action-lopo --task action --protocol lopo
python -m ml.train --data data/wifall.npz --output artifacts/my-fall-lopo --task fall --protocol lopo
python -m ml.train --data data/wifall.npz --output artifacts/my-fall-loao --task fall --protocol loao
python -m ml.train --data data/wigesture.npz --output artifacts/my-gesture-lopo --task action --protocol lopo --settings configs/ml_wigesture.json
```

Use `--models baseline` or `--models cnn` to select one model. Use `--only-fold ID0` for a WiFall smoke run; the summary explicitly marks partial runs. `configs/ml.json` fixes seeds and training parameters. WiGesture's larger class problem required a larger baseline iteration ceiling after the initial solver failed to converge; this changes optimization completion, not selection on test scores. Nonconverged fits fail instead of producing a successful metrics report.

**LOPO:** leave one participant entirely out for testing. Within the other participants, hold out whole recording-content groups for validation, preserving each class. All windows from one recording stay in the same partition. Exact duplicate signal recordings cannot cross partitions. Per-subcarrier mean/scale are fitted only on the inner training windows. Class weights use only their labels. The CNN's best epoch is selected using inner validation loss; the outer test set is untouched until model export.

**LOAO:** multiclass action recognition of an entirely unseen target class would be an invalid closed-set evaluation, so that combination is refused. For binary fall detection, hold out each non-fall action to measure false alarms on an unseen negative activity. These test sets contain no falls: `fall_recall` is null and their accuracy is negative specificity, not fall-detection accuracy. The held-out-fall fold is explicitly skipped because the training set contains no positive class. LOAO shares participants across partitions by design and does not measure unseen-person generalization.

Scoring includes window accuracy, per-class precision/recall/F1/support, confusion matrices and recording-level probability-averaged metrics. Macro-F1 includes the full target label set with zero for unsupported classes; interpret negative-only LOAO accordingly. Windows inherit recording-level actions; precise onset timestamps are not supplied, so these metrics do not measure online fall-alert timing.

Each fold saves:

- `splits.csv`: recording-level train/validation/test membership.
- `baseline.csimodel` / `cnn.csimodel`: safe weights, standardizer and required metadata.
- `*.metrics.json`: metrics, training settings, selected epoch, dataset hash and provenance.
- `*.predictions.csv`: held-out window predictions and class scores.

Each run saves `dataset-audit.json`, `results.csv` and `summary.json`. Full runs live under `artifacts/m3/`. The compact measured-results snapshot is [reports/m3/RESULTS.md](../reports/m3/RESULTS.md). No accuracy-based hyperparameter tuning or selection between these evaluated folds was performed. There is no final all-data model pretending to retain a holdout score; exported models are the actual fold models.

## Models and local inference

The baseline summarizes standardized carrier windows using mean, standard deviation, range and mean absolute temporal change, then fits class-balanced logistic regression. The CNN uses two temporal convolutions, ReLU, global average pooling and a linear output. Its channel count is configurable. Both are original small baselines, not reproductions of KNN-MMD.

Exports use the existing atomic metadata + numeric NPZ bundle. Metadata includes input shape, rate, carrier count, dataset, UTC date, ordered class labels, adapter fingerprint, windowing parameters, split provenance and weights checksum. Standardizer parameters travel with weights. No pickle, sklearn object or executable Torch checkpoint is loaded by the gateway.

`detection.inference.ModelPredictor` runs both models with NumPy. Tests compare its outputs against sklearn and PyTorch. Loading and every subsequent prediction reject incompatible shape/rate/preprocessing; model weight shapes, finite values and positive scales are also checked. The caller still must ensure the declared acquisition rate and preprocessing describe the real input—metadata cannot establish that from a bare array.

```powershell
python -m ml.predict --model artifacts/m3/wifall-fall-lopo/ID0/baseline.csimodel --data data/wifall.npz --index 0
```

The sample command is an inference demonstration; check `splits.csv` for evaluation membership. For a compatible M2 replay, add `--model <artifact>` to the gateway. Predictions become `payload.activity_prediction` annotations, including the uncalibrated score and model hash. They do **not** confirm falls, bypass the stillness check, or change occupancy decisions. Synthetic-default and WiFall configurations differ deliberately: the WiFall artifact must refuse the default synthetic gateway input.

## Colab and reproducibility

Open [ml/notebooks/M3_Colab.ipynb](../ml/notebooks/M3_Colab.ipynb). Copy the project to your Drive, set its path in the notebook, and run the CPU workflow. It verifies the archive hash, prepares data, runs one holdout by default and checks exported inference. The notebook is syntax/structure-checked locally, not executed inside Colab here. It requires no paid API.

The tests and CI cover split isolation, duplicate-content leakage, train-only scaling, timing gaps/duplicates, invalid LOAO requests, export parity and a miniature complete training run. Seeds, versions and training settings are stored with every run. Cross-platform floating-point differences may still occur.

## Sources and limits

- [WiFall archive/card](https://huggingface.co/datasets/RS2002/WiFall) and [WiGesture archive/card](https://huggingface.co/datasets/RS2002/WiGesture): both archives inspected locally; card descriptions label themselves AI-generated, so observed files take precedence.
- [KNN-MMD reference](https://github.com/RS2002/KNN-MMD): cross-domain evaluation context; no source code copied and no claim to reproduce its results.
- [CSI-BERT / WiGesture reference](https://github.com/RS2002/CSI-BERT): dataset provenance.
- [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) and [PyTorch Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html): model API references.

M4 remains separate: software-only robustness/privacy experiments and explicit-consent, local-only identity functionality. Hardware data collection and room-specific validation are still needed regardless of these software checks.
