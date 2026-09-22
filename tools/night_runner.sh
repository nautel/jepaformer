#!/usr/bin/env bash
# Run the training plan only between START_H and END_H, and only on GPUs that
# other users leave free. Jobs are killed at the end of the window and resume
# from their checkpoint on the next night (same output folder = same TAG).
#
# Plan file lines: TAG|SEQ|MODE|extra --overrides...   ('#' comments allowed)
# Env: PLAN, MAX_JOBS (default 2), FREE_MB (GPU counts as free below this,
#      default 12000), START_H (0), END_H (6), plus run_pilot.sh's variables.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
PLAN=${PLAN:?set PLAN to the plan file}
MAX_JOBS=${MAX_JOBS:-2}
FREE_MB=${FREE_MB:-12000}
START_H=${START_H:-0}
END_H=${END_H:-6}
EPOCHS=${EPOCHS:-20}
RESULTS=${RESULTS:-$REPO/results}
mkdir -p "$REPO/logs"

in_window() {
  local h=$((10#$(date +%H)))
  if [ "$START_H" -lt "$END_H" ]; then
    [ "$h" -ge "$START_H" ] && [ "$h" -lt "$END_H" ]
  else  # window crosses midnight
    [ "$h" -ge "$START_H" ] || [ "$h" -lt "$END_H" ]
  fi
}
mine() { pgrep -u "$USER" -f "train[.]py hparams/jepa2" | wc -l; }
finished() {  # $1 = tag -> done when the log already has EPOCHS epochs
  local log="$RESULTS/pilot-$2-$3$1/train_log.txt"
  [ -f "$log" ] && [ "$(grep -c '^epoch:' "$log")" -ge "$EPOCHS" ]
}
free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
    sort -t, -k2 -n | awk -F', *' -v m="$FREE_MB" '$2 < m {print $1; exit}'
}

echo "$(date '+%F %T') night runner up (window ${START_H}-${END_H}h, max $MAX_JOBS jobs)"
while in_window; do
  launched=0
  while IFS='|' read -r TAG SEQ MODE EXTRA; do
    [ -z "${TAG// }" ] && continue
    case "$TAG" in \#*) continue ;; esac
    in_window || break
    finished "-$TAG" "$SEQ" "$MODE" && continue
    pgrep -u "$USER" -f "pilot-$SEQ-$MODE-$TAG" > /dev/null && continue
    [ "$(mine)" -ge "$MAX_JOBS" ] && continue
    GPU=$(free_gpu)
    if [ -z "$GPU" ]; then
      echo "$(date '+%F %T') no free GPU, waiting"
      break
    fi
    echo "$(date '+%F %T') start $TAG on GPU $GPU: $SEQ $MODE $EXTRA"
    TAG="-$TAG" EPOCHS="$EPOCHS" RESULTS="$RESULTS" \
      setsid nohup "$REPO/tools/run_pilot.sh" "$GPU" "$SEQ" "$MODE" $EXTRA \
      >> "$REPO/logs/$SEQ-$MODE-$TAG.log" 2>&1 < /dev/null &
    launched=1
    sleep 180
  done < "$PLAN"
  [ "$launched" -eq 0 ] && sleep 300
done

echo "$(date '+%F %T') window closed, stopping my jobs"
pkill -u "$USER" -f "train[.]py hparams/jepa2"
