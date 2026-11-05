# MR-STFODM–BLS

Code for **Minimum Redundancy Maximum Relevance with a Chaos-Driven Broad Learning System for Electrocardiogram Classification**.

The pipeline combines mRMR-guided feature selection, fractional-order directional memory, state-triggered local refinement and BLS classification. It includes ECG feature construction and beat-level and patient-independent evaluation.

## Quick start

Use Python 3.11 or later, preferably in a virtual environment:

```bash
python -m pip install -r requirements.txt
python -B run.py demo --protocol beat --output outputs/demo_beat
python -B run.py demo --protocol patient --output outputs/demo_patient
python -B verify.py
```

The demos use synthetic data and reduced settings. Choose a new output directory for each run.

## MIT-BIH data

Download the [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/) and place its `.hea`, `.dat` and `.atr` files in `data/mitdb`.

```bash
python -B run.py prepare --raw-dir data/mitdb --output data/mitbih_base.npz
python -B run.py run --protocol beat --features data/mitbih_base.npz --output outputs/beat_F01_R01
python -B run.py run --protocol patient --features data/mitbih_base.npz --output outputs/patient_F01_R01
```

Preparation creates 262 waveform/RR features; execution adds seven HFAE features. Real-data runs use 500 effective evaluations. The default is fold 1, run 1; use `--fold`, `--run` or `--all` to select tasks.

## Documentation

Results include selected features, validation/test metrics, predictions, search history, splits and parameters.

- [Data and protocols](DATA_AND_PROTOCOLS.md)
- [Validation](VALIDATION.md)
- [Source provenance](provenance.json)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [中文说明](使用说明_中文.md)
