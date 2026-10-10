# M3 measured results

Completed local experiments; no hardware accuracy claim.

Accuracy and macro-F1 below are **unweighted means across held-out folds**, not pooled-window scores.

| Dataset / task / protocol | Model | Folds | Mean accuracy | Mean macro-F1 | F1 std |
|---|---|---:|---:|---:|---:|
| wifall / action / lopo | baseline | 10 | 35.5% | 0.223 | 0.095 |
| wifall / action / lopo | cnn | 10 | 29.4% | 0.157 | 0.047 |
| wifall / fall / lopo | baseline | 10 | 62.5% | 0.594 | 0.104 |
| wifall / fall / lopo | cnn | 10 | 60.0% | 0.556 | 0.134 |
| wifall / fall / loao | baseline | 4 | 77.6% | 0.430 | 0.066 |
| wifall / fall / loao | cnn | 4 | 77.3% | 0.429 | 0.066 |
| wigesture / action / lopo | baseline | 8 | 23.0% | 0.134 | 0.025 |
| wigesture / action / lopo | cnn | 8 | 20.0% | 0.085 | 0.037 |

## Interpretation

- These are untuned reference models. No test fold was used for scaling, early stopping or model selection.
- WiFall ID2 contributes 1,406 of 2,076 windows; pooled metrics overrepresent ID2.
- **LOAO is a negative-only false-alarm transfer diagnostic.** Its accuracy is specificity on an unseen non-fall action, not fall detection accuracy. Fall recall is null; held-out-fall is skipped because training has no positive class.
- Recording labels are copied to windows; precise fall onset is not annotated. Scores therefore do not measure real-time alarm performance.
- WiGesture excludes `dataset/dynamic/ID8/waveright.csv`: its rows say `waveleft`. Static action labels come from verified filenames, not participant codes in `taget`.
- Models, split assignments, predictions and per-class reports remain in `artifacts/m3/`. Raw data and weights are not included in this report snapshot.
- Models remain advisory. Validate on actual hardware and rooms before using decisions.
