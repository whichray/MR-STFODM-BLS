# Data contract and recorded protocols

## Input NPZ

Use `run.py prepare` to build a trusted local NPZ from the official MIT-BIH records. Required aligned entries for this wrapper are:

| Entry | Shape | Meaning |
|---|---|---|
| `X` | n × 262 | 256 normalized waveform samples followed by six RR features |
| `y` | n | Original base-label field; protocol-specific labels below take precedence |
| `groups` | n | Genuine MIT-BIH record IDs, retained as strings |
| `signals` | n × 256 | Aligned waveform segments used to extract HFAE |
| `y_aami` | n | AAMI labels used by the beat-level engine |
| `beat_labels` | n | Original annotation symbols used by the patient engine |
| `feature_names` | 262 | Feature-column names |

The original loader also saves R-peak indices, RR arrays and provenance metadata. Its NPZ uses object-string entries and is read with `allow_pickle=True`; only load trusted archives created by you or by the supplied loader. Passing arbitrary custom numeric vectors through the real-data command is intentionally rejected to avoid confusing base features with already augmented features. For a custom feature study, call the documented core classes directly and describe its separate data protocol.

Waveform channel 0, 256-sample windows and per-segment z-score normalization are retained. Incomplete boundary beats are dropped. RR preprocessing, invalid interval fallback and label mapping use the supplied original loader. HFAE uses order 0.2 and depth 3, yielding seven components. Main-experiment entropy extraction uses its original matrix-level median fallback for non-finite values; it is not the foldwise-v2 HFAE-control implementation. The main `StandardScaler` is fitted on training samples. Patient inner folds independently fit training-only scaling statistics.

## Two distinct evaluation protocols

| Setting | Beat-level main protocol | Patient-independent main protocol |
|---|---|---|
| Label order | N, S, V, F, Q | N, A, V, L, R (original beat symbols) |
| Split | Repeated random stratified 8:1:1 | Nested patient-group evaluation |
| Repetitions | 10 splits × 10 runs | 4 feasible outer folds × 10 runs on the paper data |
| Requested folds in config | 10 splits | 5 outer / 4 inner; original feasibility reduction retained |
| mRMR prior weight | 0.02 | 0.10 |
| Patient-equal utility weight | 0 | 0.25 |

These label systems are different and their class-wise results cannot be treated as the same task. Records 201 and 202 are grouped as the same patient in the patient engine. The historical `PATIENT_EXCLUDED_RECORDS` list alone does not imply that this main protocol excludes those records: the corresponding exclusion flag is false in the recorded main settings. The pipeline applies the original protocol-specific behavior rather than silently imposing another evaluation protocol.

The candidate objective weights are 0.60 Macro-F1, 0.15 mean minority F1, 0.10 hard-class F1, 0.10 worst-minority F1, and 0.05 balanced accuracy, with selected-ratio penalty 0.0002. The patient engine also blends patient-equal utility according to its recorded formula. Read the retained engine functions for the exact aggregation and zero-support class rules.

## Common optimizer and BLS parameters

| Parameter | Recorded value |
|---|---|
| Effective candidate budget | 500 |
| Population / chaotic points | 10 / 5 |
| Fractional order / memory | 0.6 / 4 |
| Chaotic mixing weight | 0.7 |
| Minimum global-search ratio | 0.65 |
| Stagnation trigger | 18 effective evaluations |
| Forced refinement ratio | 0.20 |
| Local patience | 80 |
| BLS feature nodes | 20 × 3 |
| BLS enhancement nodes | 100 |
| Ridge coefficient | 2^-12 |
| Class weighting | sqrt-balanced, power 0.45 |
| Final ensemble members | 9 |
| Final prior correction grid | 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5 |

The prior correction coefficient is selected using validation Macro-F1. Test labels do not select masks, ensemble parameters or the correction coefficient. The final head uses the reference training portion under `HOLDOUT_REFIT_ON_TRAIN_VAL=False`, matching the retained source.

Beat initialization uses ranking prefixes at ratios 0.1/0.2/0.3/0.4 and perturbed companions. Patient initialization uses six seeded individuals (60% of a population of ten), cycling over these ratios with per-subset perturbations. The implementations are deliberately kept separate.

## Random seeds and indexing

The base seed is 42. `stable_seed` joins its arguments with `|`, takes the first eight MD5 hexadecimal digits and reduces modulo `2^31-1`. The retained recipes are:

- Ranking: `stable_seed(42, fold, "MRMR", "ranking")`.
- BLS mapping: `stable_seed(42, "bls", fold, run)`.
- Beat optimizer: `stable_seed(42, "optimizer", fold, run, "ST_FAMILY")` under the recorded shared-family setting.
- Patient optimizer: `stable_seed(42, "optimizer", fold, run, "ST_FODM_MRMR_PATIENT")`.

The patient objective's soft-prior ranking is computed according to its own original function and run seed; do not replace it with the cached initialization ranking merely to simplify the implementation. Final BLS seed offsets are recorded in the JSON configuration. Fold and run numbers start at 1; feature and saved sample indices start at 0.

Version differences in libraries, input record order, annotation files or data filtering can change splits or numerical results. The release records per-run software versions and the input NPZ hash. The original paper comparison and the later mechanism ablations had their own seed-matching conventions; this package implements the proposed main method only.
