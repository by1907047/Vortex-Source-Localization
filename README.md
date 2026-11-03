# Vortex source localization

Code, data and models for cylinder, fixed robotic-fish and swimming robotic-fish localization using velocity and differential-pressure signals.

| Directory | Task and model |
|---|---|
| [`mlp/`](mlp/) | Cylinder and fixed robotic fish: MLP, five configurations, 20 runs each |
| [`swimming_fish/`](swimming_fish/) | Swimming robotic fish: Bi-GRU using FFT features |

The swimming-fish package also includes time-domain training and test tables, six G-series CSVs, and a saved Bi-GRU checkpoint. The training command below uses the FFT tables and creates a new checkpoint.

### Original experiment files

- [Cylinder localization](Cylinder%20localization/)
- [Fixed robotic fish localization](Fixed%20robotic%20fish%20localization/)
- [Swimming robotic fish localization](Swimming%20robotic%20fish%20localization/)

## 1. System requirements

Environment: Windows 11 (64-bit), Python 3.12.10, PyTorch 2.13.0, NumPy 2.3.5, pandas 2.2.3 and scikit-learn 1.6.1.

Recommended hardware: a desktop CPU with at least four cores, 16 GB RAM and 4 GB free storage. No GPU or non-standard hardware is required.

## 2. Installation

With Python 3.12 installed, open Windows PowerShell in the directory containing this README and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\mlp\requirements.txt scikit-learn==1.6.1
```

Estimated installation time: 5-20 minutes on the recommended desktop with broadband internet.

## 3. Demos

### Cylinder and fixed robotic fish

```powershell
.\.venv\Scripts\python.exe .\mlp\code\demo_predict.py
```

The default demo predicts 147 cylinder test samples using 20 saved `V6+P12` models, without retraining. Expected prediction shape: `(20, 147, 2)`; full-field errors are:

| Statistic across 20 runs | X MAE (cm) | Y MAE (cm) | Mean distance error (cm) |
|---|---:|---:|---:|
| Mean | 0.644574 | 0.180909 | 0.702160 |
| Sample SD | 0.168909 | 0.019942 | 0.157418 |

Estimated runtime: 10-60 seconds. Select `--task fish` for the fixed robotic-fish test set. No file is written unless `--output` is specified.

### Swimming robotic fish

The training and test FFT tables are in `swimming_fish/signal_features2/`. Run:

```powershell
.\.venv\Scripts\python.exe .\swimming_fish\bi_gru_traj.py
```

This training demo uses 36 FFT features and sequences of five rows to train a two-layer Bi-GRU for 80 epochs. It prints epoch losses and saves `signal_features2/bi_GRU_results_<timestamp>/run/bi_GRU_model.pth` under `swimming_fish/`. Model outputs are a flag probability and two position components (`dx`, `dy`). Estimated CPU runtime: 5-30 minutes.

## 4. Instructions for use

**MLP:** prepare a feature CSV with the columns listed in the selected model's `scalers.json`, retaining the original feature units and preprocessing. Optional `x`, `y` columns provide true coordinates in centimetres.

```powershell
.\.venv\Scripts\python.exe .\mlp\code\demo_predict.py --task cylinder --config "V6+P12" --input my_features.csv --output my_predictions.csv
```

Tasks: `cylinder`, `fish`. Configurations: `V6`, `P12`, `V6+P12`, `P6`, `V4+P2`. Predictions are exported in centimetres using the stored scaling parameters; existing output files are not overwritten.

The three MLP scripts are in `mlp/code/`; data and results remain in `mlp/data/` and `mlp/results/`. The MLP files include training/test features, saved weights and scalers, the seed mapping, reference metrics and validation records. Use `--output` to export per-run predictions.

**Bi-GRU:** provide training and test CSVs in the configured input directory, with the 36 columns in `feature_cols` and the labels `flag`, `dx`, `dy`. Retain chronological row order and the original units. The script fits the input scaler on training data and applies it to test data.

Optional full MLP retraining:

```powershell
.\.venv\Scripts\python.exe .\mlp\code\reproduce_selected.py
```

The 200 models and associated results are saved in `mlp/reproduced_results/`. The program stops if that directory already exists. Estimated runtime: 10-60 minutes.

## 5. Licenses

- Code: MIT License; see `mlp/LICENSE`.
- The supplied dataset documentation identifies the datasets as CC BY 4.0.
