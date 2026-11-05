# Data and protocols

## Input

`run.py prepare` builds an NPZ from MIT-BIH v1.0.0:

| Entry | Content |
|---|---|
| `X` | n × 262: 256 waveform samples and six RR features |
| `signals` | n × 256: waveforms for HFAE |
| `groups` | Record IDs |
| `y` | Base labels |
| `y_aami` / `beat_labels` | AAMI / original annotation labels |
| `feature_names` | Column names |

Use NPZs from a trusted source; the loader reads object-string metadata with `allow_pickle=True`.

Preprocessing uses channel 0, 256-sample windows and per-segment z-score normalization. Incomplete boundary beats are dropped. HFAE uses order 0.2 and depth 3, adding seven features. Its median fallback uses the extraction matrix; scaling uses training statistics, independently for each patient inner fold.

## Evaluation

| Setting | Beat level | Patient independent |
|---|---|---|
| Labels | N, S, V, F, Q | N, A, V, L, R |
| Split | Repeated stratified 8:1:1 | Nested patient-group CV |
| Paper repetitions | 10 splits × 10 runs | 4 feasible outer folds × 10 runs |
| mRMR prior weight | 0.02 | 0.10 |
| Patient-equal utility weight | 0 | 0.25 |

Patient settings request five outer and four inner folds; feasibility reduction gives four outer and three inner folds on the paper dataset. Records 201 and 202 share one patient ID. Paced-record exclusion is disabled in the recorded main configuration.

## Parameters

| Parameter | Value |
|---|---|
| Effective candidate budget | 500 |
| Population / chaotic points | 10 / 5 |
| Fractional order / memory | 0.6 / 4 |
| Chaotic mixing weight | 0.7 |
| Minimum global-search ratio | 0.65 |
| Stagnation trigger / forced refinement ratio | 18 / 0.20 |
| Local patience | 80 |
| BLS feature / enhancement nodes | 20 × 3 / 100 |
| Ridge coefficient | 2^-12 |
| Class-weight power | 0.45 |
| Final ensemble | 9 models |

Objective weights are 0.60 Macro-F1, 0.15 minority F1, 0.10 hard-class F1, 0.10 worst-minority F1 and 0.05 balanced accuracy, with subset-ratio penalty 0.0002. Final prior correction is selected by validation Macro-F1.

One FES evaluates an uncached feature mask. Patient candidates use multiple inner-fold fits; final ensemble fits are counted separately.

The base seed is 42. Fold/run indices start at 1; feature/sample indices start at 0. Full parameters and seed offsets are in [beat_paper.json](configs/beat_paper.json) and [patient_paper.json](configs/patient_paper.json). The `stable_seed` functions define the seed mapping.
