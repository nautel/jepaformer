#!/usr/bin/env bash
# Pilot run on a shared server: 1 GPU, low CPU/IO priority.
# Usage: tools/run_pilot.sh GPU SEQ_MODEL JEPA_MODE [extra overrides...]
set -euo pipefail
GPU=$1; SEQ=$2; MODE=$3; shift 3
ROOT=$HOME/code/jepa2
CSV=$ROOT/csv
EPOCHS=${EPOCHS:-20}
BS=${BS:-2}
STEPS_PER_EPOCH=$(( 13900 / BS ))
LAYERS=$([ "$SEQ" = mamba ] && echo 4 || echo 8)
export SPEECHBRAIN_ROOT=$ROOT/sb PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=4
cd "$ROOT/sepformer-jepa"
exec nice -n 10 ionice -c3 "$HOME/envs/jepa2/bin/python" train.py hparams/jepa2-libri2mix.yaml \
  --seq_model="$SEQ" --jepa_mode="$MODE" --sep_layers="$LAYERS" \
  --output_folder="$ROOT/results/pilot-$SEQ-$MODE${TAG:-}" \
  --train_data="$CSV/libri2mix_train-100.csv" --valid_data="$CSV/libri2mix_dev.csv" \
  --test_data="$CSV/libri2mix_test.csv" --skip_prep=True \
  --N_epochs="$EPOCHS" --batch_size="$BS" --training_signal_len=32000 \
  --ema_anneal_steps=$(( EPOCHS * STEPS_PER_EPOCH )) --noprogressbar=True "$@"
