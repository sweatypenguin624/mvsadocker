#!/bin/bash
# Overnight batch run for one folder of camera videos.
#   ./run_pipeline.sh "<folder name | Drive path | Drive folder ID | local dir>" [options]
# Examples:
#   ./run_pipeline.sh "tvc-5"
#   ./run_pipeline.sh "tvc-5" --map "Cam 3=tvc5_cam1_eb" --map "Cam 4=tvc5_cam2_wb"
#   ./run_pipeline.sh /data/tvc5_videos --config tvc5_cam2_wb
# Shows the plan (each camera -> config) first and starts only if every camera
# has a config. Runs inside tmux, survives SSH disconnects, restarts after a
# crash and resumes where it stopped. All options: python historical-processor/pipeline.py -h
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PY="${PYTHON:-$DIR/env/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

if [ "${1:-}" = "__run" ]; then
    shift
    LOG="$1"; shift
    for i in $(seq 1 "$MAX_RESTARTS"); do
        "$PY" historical-processor/pipeline.py "$@" 2>&1 | tee -a "$LOG"
        rc=${PIPESTATUS[0]}
        if [ "$rc" -eq 0 ]; then echo "Pipeline finished." | tee -a "$LOG"; break; fi
        if [ "$rc" -eq 2 ]; then echo "Pipeline stopped: needs attention (exit 2)." | tee -a "$LOG"; break; fi
        echo "$(date '+%F %T') pipeline exited with code $rc; restart $i/$MAX_RESTARTS in 60s" | tee -a "$LOG"
        sleep 60
    done
    exec bash  # keep the tmux window open for inspection
fi

if [ $# -lt 1 ]; then sed -n '2,11p' "$0"; exit 1; fi

"$PY" historical-processor/pipeline.py "$@" --dry-run || exit $?

NAME=$(basename -- "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g')
SESSION="pipe_${NAME:0:40}"
LOG_DIR="$DIR/logs/pipeline"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${NAME}_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Already running in tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
    exit 1
fi
# %q-quote so folder names with spaces survive tmux's shell.
tmux new-session -d -s "$SESSION" "$(printf '%q ' "$DIR/run_pipeline.sh" __run "$LOG" "$@")"
echo "Started in tmux session '$SESSION'."
echo "  watch:    tmux attach -t $SESSION   (detach: Ctrl-b d)"
echo "  log:      $LOG"
echo "  progress: cat pipeline_runs/*/status.txt"
