#!/usr/bin/env bash
# Run a queue of configurations, at most MAX_JOBS of mine on GPUs at a time.
# Each line of the queue file: TAG|SEQ|MODE|extra --overrides...
# Env: QUEUE (file), MAX_JOBS (default 2), plus run_pilot.sh's variables.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
QUEUE=${QUEUE:?set QUEUE to the queue file}
MAX_JOBS=${MAX_JOBS:-2}
mkdir -p "$REPO/logs"

running() { pgrep -u "$USER" -f "hparams/jepa2-libri2mix.yaml" | wc -l; }
free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    sort -t, -k2 -n | head -1 | cut -d, -f1
}

while IFS='|' read -r TAG SEQ MODE EXTRA; do
  [ -z "${TAG// }" ] && continue
  case "$TAG" in \#*) continue ;; esac
  while [ "$(running)" -ge "$MAX_JOBS" ]; do sleep 120; done
  GPU=$(free_gpu)
  echo "$(date '+%F %T') start $TAG on GPU $GPU: $SEQ $MODE $EXTRA"
  TAG="-$TAG" setsid nohup "$REPO/tools/run_pilot.sh" "$GPU" "$SEQ" "$MODE" $EXTRA \
    > "$REPO/logs/$TAG.log" 2>&1 < /dev/null &
  sleep 180  # let it allocate before measuring free memory again
done < "$QUEUE"
echo "$(date '+%F %T') queue submitted"
