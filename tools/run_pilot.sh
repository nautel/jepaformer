#!/usr/bin/env bash
# Train one configuration on a shared server: 1 GPU, low CPU/IO priority.
# Usage: tools/run_pilot.sh GPU SEQ_MODEL JEPA_MODE [extra overrides...]
# Env: CSV_DIR (Libri2Mix CSVs), RESULTS, PYTHON, EPOCHS, BS, TAG
set -euo pipefail
GPU=$1; SEQ=$2; MODE=$3; shift 3
REPO=$(cd "$(dirname "$0")/.." && pwd)
CSV=${CSV_DIR:?set CSV_DIR to the folder with libri2mix_{train-100,dev,test}.csv}
RESULTS=${RESULTS:-$REPO/results}
PYTHON=${PYTHON:-python}
EPOCHS=${EPOCHS:-20}
BS=${BS:-2}
STEPS_PER_EPOCH=$(( 13900 / BS ))
LAYERS=$([ "$SEQ" = mamba ] && echo 4 || echo 8)  # ~23M vs ~26M params
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
cd "$REPO"
exec nice -n 10 ionice -c3 "$PYTHON" train.py hparams/jepa2-libri2mix.yaml \
  --seq_model="$SEQ" --jepa_mode="$MODE" --sep_layers="$LAYERS" \
  --output_folder="$RESULTS/pilot-$SEQ-$MODE${TAG:-}" \
  --train_data="$CSV/libri2mix_train-100.csv" --valid_data="$CSV/libri2mix_dev.csv" \
  --test_data="$CSV/libri2mix_test.csv" --skip_prep=True \
  --N_epochs="$EPOCHS" --batch_size="$BS" --training_signal_len=32000 \
  --ema_anneal_steps=$(( EPOCHS * STEPS_PER_EPOCH )) --noprogressbar "$@"
