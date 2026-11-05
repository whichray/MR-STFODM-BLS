# Validation performed

Validation date: 2026-11-05. This report concerns the compact code candidate, not a new scientific experiment or a full reproduction of the paper.

| Check | Outcome |
|---|---|
| Retained function/class bodies versus organized source | SHA256 match |
| Shared BLS, entropy, raw loader and IO modules | Byte-identical copies across both source collections |
| mRMR ranking versus original runner on the same artificial data/seed | Identical, both protocols |
| Selected mask versus original runner | Identical, both protocols |
| Complete search fitness trajectory versus original runner | Match within absolute tolerance 1e-12 |
| Final labels and score matrices versus original runner | Labels identical; scores match within 1e-12 |
| Cache hits consume no extra effective FES | Passed |
| Deterministic fresh-directory demo reruns | Passed, both protocols |
| Train/validation/test sample separation and patient/inner-fold isolation | Passed on the exercised splits |
| Real MIT-BIH slice: beat-level | 1470 rows, 269 features; 500/500 FES, 500 search BLS fits, 9 final heads |
| Real MIT-BIH slice: patient-level | 1137 rows after label filtering, 269 features; 500/500 FES, 1500 search BLS fits, 9 final heads |

The real integration data were privately sampled by record and annotation symbol (at most 12 beats per record/symbol), retaining examples from the available classes. These slices and their prediction outputs are not included. Neither integration test estimates or replaces the manuscript's full-dataset performance. Full-data 100-run beat-level and 40-run patient-level reproduction, the twelve-method comparison, external-cohort validation, and cross-platform testing were not performed.

The source-body integrity and synthetic checks can be rerun with `python -B verify.py`. Private checks against the original local research directories were performed separately; those directories are not needed for the public package. The two artificial demonstrations use reduced settings and do not validate clinical performance.
