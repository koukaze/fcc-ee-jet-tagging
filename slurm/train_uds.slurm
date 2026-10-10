#!/bin/bash
#SBATCH --job-name=fccee-uds
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

set -euo pipefail
: "${FCCEE_DATA_DIR:?Set FCCEE_DATA_DIR to the simulation ROOT directory}"
cd "${SLURM_SUBMIT_DIR}"

"${FCCEE_PYTHON:-python}" scripts/train_uds.py \
  --root-files "${FCCEE_DATA_DIR}"/chunk_*.root \
  --output-dir runs/uds_no_kaon \
  --epochs 60 --batch-size 1024 --lr 0.005 \
  --d-model 128 --nhead 8 --n-encoder-layers 3 \
  --dropout 0.1 --weight-decay 0 --pair-feature-set no_kaon --pair-hidden-dim 32 \
  --seed 1123 --eval-fraction 0.15 --test-fraction 0.15 \
  --save-every 0 --device cuda
