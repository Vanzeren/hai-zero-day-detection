# Unsupervised Zero-Day Attack Detection in an ICS

Reproduction of Group 88's HAI 21.03 → 23.05 study: matched-capacity Deep SVDD and convolutional VAE, a PCA reconstruction baseline, per-sensor attribution, and SPOT/DSPOT decision layers under plant-version shift.

## Protocol

- Train representations only on normal HAI 21.03 data; never use attack labels for training, model selection, or threshold calibration.
- Start with 59 tags shared by HAI 21.03 and 23.05, then retain 41 tags with non-zero variance on the source training partitions. This source-only rule prevents incompatible constant encodings from dominating max fusion. Fit normalization on the chronological training partitions only.
- Evaluate unseen attacks in-domain (21.03, 50 events) and cross-version (23.05, 52 events).
- Score causal 60-second windows every 10 seconds, aligned to the window endpoint. Fuse per-sensor scores by maximum.
- Calibrate each model/seed on source normal data at 10 false-alarm points/day. Compare frozen SPOT with test-stream DSPOT.
- Report PR-AUC, ROC-AUC, unadjusted point metrics, eTaPR, event detection/delay, false alarms/day, and per-attack target attribution. No point adjustment is used.
- Default experiment: three seeds (11, 23, 37).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e .
```

For a GPU environment, install the appropriate PyTorch build instead. The supplied experiments run on CPU.

## Reproduce

```bash
# Verify/download the exact 341 MB raw dataset and technical documentation.
hai-download

# Align tags, parse both label formats, fit source normalization, and cache streams.
hai-prepare

# Optional fast end-to-end check; writes only to results/smoke.
hai-train --model all --seed 11 --smoke

# Full grid: PCA, Deep SVDD, and VAE × three seeds, followed by all stream scores.
hai-train --model all

# Source-only EVT calibration and both deployment arms on both test domains.
hai-evaluate

# Aggregate statistics, figures, final report, and poster (Markdown/PDF/PNG).
hai-report

# Behavioral regression tests.
pytest
```

Individual models or seeds can be rerun with `--model {pca,svdd,vae}` and `--seed N` on the training/evaluation commands.

## Outputs

- `data/processed/metadata.json`: row counts, attack counts, label alignment, source SHA-256.
- `results/checkpoints/`: model state and training configuration/history.
- `results/scores/`: per-sensor and fused scores for every stream.
- `results/metrics.csv`, `per_attack.csv`, `calibration.csv`: unaggregated experimental evidence.
- `results/metrics_summary.csv`, `degradation_recovery.csv`, `recovery_summary.csv`: three-seed analysis.
- `figures/`: degradation, PR-AUC, false-alarm, delay, attribution, and per-attack plots.
- `reports/final_report.{md,pdf}` and `reports/poster.{md,pdf,png}`.

## Data and references

HAI is distributed under CC BY-SA 4.0 at <https://github.com/icsdataset/hai>. The downloader pins the current official files by SHA-256. HAI 23.05 `label-test2.csv` has minute-resolution timestamps; the loader accepts them only after verifying every value against the corresponding floored telemetry timestamp.

The report cites Deep SVDD (ICML 2018), SPOT/DSPOT (KDD 2017), TaPR (CIKM 2019), eTaPR (SAC 2022), and the time-series evaluation protocol of Kim et al. (AAAI 2022).
