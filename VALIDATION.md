# Validation

Checked on 2026-11-05 using synthetic inputs and MIT-BIH data slices.

| Check | Result |
|---|---|
| Source-body hashes | Matched |
| Rankings and selected masks | Matched original code |
| Fitness trajectories and prediction scores | Matched within 1e-12 |
| Deterministic reruns, cache accounting and partition isolation | Passed |
| Beat-level slice | 1470 rows, 269 features, 500 FES |
| Patient-level slice | 1137 rows, 269 features, 500 FES |

Run the source-integrity and synthetic checks with `python -B verify.py`.

Validation covered small-input equivalence and slice integration; full-dataset experiment batches remain untested in this package.
