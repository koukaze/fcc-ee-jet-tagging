# Running the analysis

Run commands from the repository root. Python 3.11 or later is required; the pinned dependencies have been checked with Python 3.13.7 on macOS.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. The requirements file pins direct dependencies; it is not a complete transitive lockfile. GPU training requires a PyTorch installation compatible with the machine's CUDA runtime.

## Saved-result analysis

```bash
python scripts/reproduce_results.py --output-dir outputs/results
```

This uses the prediction arrays and ablation table included in `results/`. It needs neither a GPU nor external input files.

| Output | Contents |
| --- | --- |
| `final_comparison.png` / `.svg` | Per-class recall and light-flavour ROC curves |
| `pair_feature_ablation.png` / `.svg` | Recorded pair-feature comparisons |
| `accuracy_summary.csv` | Overall, macro and per-class recall |
| `working_points.csv` | Signal efficiency at fixed background efficiency |

## Notebooks

```bash
python -m pip install -r requirements-dev.txt
python -m ipykernel install --user --name fccee-jet-tagging --display-name "Python (FCC-ee jet tagging)"
jupyter lab notebooks/01_final_comparison.ipynb
```

Select **Python (FCC-ee jet tagging)** and run all cells from a fresh kernel. `02_pair_feature_ablation.ipynb` uses the recorded study table. Both notebooks run using the included files.

## Train the baseline

Place compatible simulation ROOT files under `data/`, following the [data specification](docs/data.md). The following commands use the settings for the saved baseline and final `no_kaon` specialist. Keep the same files, file order, seed and split fractions across both models and their evaluation.

```bash
python scripts/train_baseline.py \
  --root-files data/chunk_*.root \
  --output-dir runs/baseline090 \
  --epochs 60 --batch-size 4000 --lr 0.005 \
  --d-model 128 --nhead 8 --n-encoder-layers 3 \
  --dropout 0.1 --weight-decay 0 \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --save-every 0 --device cuda
```

The shell must expand the ROOT wildcard for training. On shells that do not expand wildcards, supply the input filenames explicitly. The loader reads all selected jets into memory. Use `--device cpu` for small local checks; `--limit` limits the total number of input rows but changes the resulting split.

## Train the UDS specialist

```bash
python scripts/train_uds.py \
  --root-files data/chunk_*.root \
  --output-dir runs/uds_no_kaon \
  --pair-feature-set no_kaon --pair-hidden-dim 32 \
  --epochs 60 --batch-size 1024 --lr 0.005 \
  --d-model 128 --nhead 8 --n-encoder-layers 3 \
  --dropout 0.1 --weight-decay 0 \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --save-every 0 --device cuda
```

Both trainers write `checkpoint_latest.pth`, `checkpoint_best_loss.pth` and epoch metrics in their output directory. The UDS specialist splits the full sample first, then selects the true `s/u/d` jets within each split. Training uses evaluation loss for checkpoint selection.

Use `--resume` with the same output directory to continue a run. Use `--resume --resume-path PATH` to select a checkpoint explicitly. A requested resume checkpoint must exist; otherwise the command exits without starting training. Checkpoint files contain model and optimizer state; load only checkpoints from a trusted source.

## Evaluate checkpoints

Evaluate the baseline on the test split:

```bash
python scripts/evaluate_baseline.py \
  --root-files data/chunk_*.root \
  --checkpoint runs/baseline090/checkpoint_best_loss.pth \
  --plot-dir outputs/checkpoint_evaluation --name baseline090_6cls \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --split test --save-predictions --device cuda
```

Evaluate the specialist on the true UDS subset:

```bash
python scripts/evaluate_uds.py \
  --root-files data/chunk_*.root \
  --checkpoint runs/uds_no_kaon/checkpoint_best_loss.pth \
  --plot-dir outputs/checkpoint_evaluation --name no_kaon_uds \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --split test --save-predictions --device cuda
```

Combine both models on every jet in the six-class test sample:

```bash
python scripts/evaluate_soft_fusion.py \
  --root-files data/chunk_*.root \
  --baseline-checkpoint runs/baseline090/checkpoint_best_loss.pth \
  --uds-checkpoint runs/uds_no_kaon/checkpoint_best_loss.pth \
  --plot-dir outputs/checkpoint_evaluation --name A_no_kaon_baseline_softfusion \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --batch-size 1024 --device cuda
```

Soft-fusion evaluation always uses the test split and writes a prediction archive, text summary and diagnostic plots. Baseline and specialist evaluation default to the validation split, so their commands above explicitly select `--split test`.

The saved arrays reproduce the reported figures directly. Reproducing predictions from training also requires the original simulation inputs and the corresponding checkpoints; training from scratch can vary with software, hardware and nondeterministic GPU operations.

## Slurm jobs

The `slurm/` examples run the same baseline and specialist commands. Activate the project environment before submission and set the data directory:

```bash
export FCCEE_DATA_DIR="/absolute/path/to/root-files"
mkdir -p logs
sbatch slurm/train_baseline.slurm
sbatch slurm/train_uds.slurm
```

Supply the account, partition and QoS required by your cluster as `sbatch` options. The examples request one GPU. Set `FCCEE_PYTHON` to an absolute Python executable if the compute environment does not inherit the desired interpreter.

## Tests

```bash
python -m pytest -q
```

The tests cover model forward/backward passes, padding masks, feature-group selection, soft-fusion probability conservation, deterministic splitting, ROOT loading, optimizer checkpoint round-trips and agreement with the saved result tables. Synthetic ROOT fixtures are used for software checks.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named djt` | Activate the new environment and install this package. |
| ROOT wildcard is treated as a filename | Let the shell expand it for training, or supply explicit filenames. |
| Missing ROOT branches | Compare the file schema with `docs/data.md`. |
| Model/checkpoint size mismatch | Use the matching architecture and checkpoint metadata. |
| Out-of-memory during loading or training | The loader materialises the full selected sample; use smaller software checks locally and sufficient host/GPU memory for full runs. |
| Different prediction rows | Preserve input files, file order, split seed, fractions and row limit. |
