#!/usr/bin/env bash
# Launch the {none, masked} pair on the two least-used GPUs.
# Env: same as run_pilot.sh (CSV_DIR, PYTHON, EPOCHS, RESULTS, TAG, DP_SIZE).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEQ=${SEQ:-dpmamba}
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
  sort -t, -k2 -n | head -2 | cut -d, -f1)
mkdir -p "$REPO/logs"
i=0
for MODE in none masked; do
  GPU=${GPUS[$i]}; i=$((i + 1))
  echo "$(date '+%F %T') launching $SEQ $MODE on GPU $GPU"
  setsid nohup "$REPO/tools/run_pilot.sh" "$GPU" "$SEQ" "$MODE" \
    > "$REPO/logs/$SEQ-$MODE${TAG:-}.log" 2>&1 < /dev/null &
  sleep 5
done
