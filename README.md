# MR-STFODM–BLS core code

Companion code prepared for the manuscript **Minimum Redundancy Maximum Relevance with a Chaos-Driven Broad Learning System for Electrocardiogram Classification**.

This compact package contains the proposed method, the ECG feature-building dependencies, recorded main-experiment parameters, and independent execution commands. It is a local release candidate. It does not contain the complete research working directory or a published repository URL.

## What is included

- Training-only mRMR ranking, multi-scale ranking seeds and the soft ranking prior.
- Fractional-order directional memory and state-triggered binary local refinement, sharing an effective evaluation budget.
- The BLS classifier, class weighting, subset objective, caching and evaluation metrics.
- Waveform/RR preprocessing and seven HFAE features, with separate beat-level and patient-independent engines.
- The original final BLS ensemble and validation-selected prior correction.
- Main-experiment parameter files, split generation and random-seed rules.
- Artificial-data demonstrations and an integrity/reproducibility check.

The package excludes the other comparison methods, matched mechanism ablations, HFAE control studies, budget-sweep orchestration, plotting scripts, historical logs, predictions, models, and the MIT-BIH database. It therefore supports inspection and execution of the proposed pipeline; it is not a one-command reproduction of every table and figure.

## Installation and a quick check

Python 3.11 or later is recommended. Validation of this package used Python 3.14.6, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0, pandas 3.0.3, threadpoolctl 3.6.0 and WFDB 4.3.1. The requirements give compatible ranges except for scikit-learn, whose version is pinned because split and ranking behavior matter. Other environments have not all been tested.

From the package directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B run.py demo --protocol beat --output outputs/demo_beat
.\.venv\Scripts\python.exe -B run.py demo --protocol patient --output outputs/demo_patient
.\.venv\Scripts\python.exe -B verify.py
```

On Linux/macOS, substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`. Existing output folders are rejected: choose a new name when repeating a run.

The demonstrations use artificial five-class feature vectors, 24 FES, fewer BLS nodes and fewer split candidates. They are not ECG experiments and their scores must not be cited as manuscript results. The verification command checks retained source bodies against their recorded hashes, deterministic reruns, partition isolation and budget accounting.

## Running on MIT-BIH

Obtain the MIT-BIH Arrhythmia Database v1.0.0 separately from its official source:

https://physionet.org/content/mitdb/1.0.0/

Retain the original record names and the `.hea`, `.dat` and `.atr` files together in a local directory. The code does not download or redistribute these files. Follow the database's own usage and citation terms.

```powershell
python -B run.py prepare --raw-dir data/mitdb --output data/mitbih_base.npz
python -B run.py run --protocol beat --features data/mitbih_base.npz --fold 1 --run 1 --output outputs/beat_F01_R01
python -B run.py run --protocol patient --features data/mitbih_base.npz --fold 1 --run 1 --output outputs/patient_F01_R01
```

`prepare` produces 262 base columns and aligned signal/label metadata. `run` appends seven HFAE columns, yielding 269 candidates. The same base NPZ can be used by both protocols because their label filtering is applied by their respective engines. For the NPZ contract and preprocessing boundaries, read [DATA_AND_PROTOCOLS.md](DATA_AND_PROTOCOLS.md).

The default real-data execution runs one specified fold and run, using the recorded main settings, including 500 FES. To run all configured folds and ten repetitions of the proposed method:

```powershell
python -B run.py run --protocol beat --features data/mitbih_base.npz --all --output outputs/beat_all
python -B run.py run --protocol patient --features data/mitbih_base.npz --all --output outputs/patient_all
```

These are sequential commands and can require substantial computation. The paper used ten beat-level splits × ten runs, and four feasible patient outer folds × ten runs. The patient configuration requests five outer folds and four inner folds; the original feasibility logic reduces these to four outer and three inner folds on the paper dataset. Do not change the configuration numbers to four/three merely to reproduce the displayed counts: that can change split generation. The actual counts depend on class-to-patient support. `--all` does not run the twelve-method comparison.

## Outputs and budget meaning

Each run saves the selected columns, validation fitness, held-out test metrics, class metrics, effective FES, actual search-stage BLS fits, global/local switch information, final-head fit count, prediction scores, evaluation history, effective parameters and software versions. Fold files save sample indices and the training-only mRMR ranking. Indices refer to the engine's filtered feature matrix, not necessarily to unfiltered input rows.

One FES is one previously uncached candidate-mask evaluation. A cache hit consumes no additional FES. A patient candidate can fit a BLS in each inner fold; those actual fits are counted separately. Frozen-mask final ensemble training is outside the search FES and is reported separately. The seed recipe and protocol differences are documented in [DATA_AND_PROTOCOLS.md](DATA_AND_PROTOCOLS.md).

## Verification and limitations

The release was checked against the original organized source collections on small artificial inputs: rankings, selected masks, fitness trajectories and final predictions matched. Real MIT-BIH slices also completed one 500-FES run per protocol. These are integration checks, not a rerun of the paper's 100/40 full-dataset runs. Exact full-dataset metric reproduction is not claimed.

The main-experiment HFAE implementation is retained. Its median fallback uses the matrix passed to extraction. This is distinct from the later training-fold-only imputation in the HFAE control experiments; those control experiments are not part of this package. The original patient runner has also changed since its historical run hash, as recorded in [provenance.json](provenance.json). The extracted function/class bodies are preserved verbatim from the organized sources, and the new command-line entry point replaces historical orchestration only.

Read [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before external distribution. No new open-source license has been assigned in this local candidate. The authors must resolve distribution rights for the entropy implementation and choose the intended code license before publishing it. No paper DOI or repository URL has been invented.
