#!/bin/bash
# L40S host setup for running many camera processes on one GPU.
#   ./scripts/l40s/gpu_setup.sh start   # persistence mode + MPS daemon
#   ./scripts/l40s/gpu_setup.sh stop
#   ./scripts/l40s/gpu_setup.sh status
# MPS lets kernels from separate processes run concurrently instead of
# time-slicing the GPU; it does not change any computation. Start it before
# launching the run_*.sh scripts, and stop it only when no process is using the GPU.
set -e
GPU="${GPU:-0}"

case "${1:-status}" in
  start)
    nvidia-smi -i "$GPU" -pm 1
    if pgrep -x nvidia-cuda-mps-control >/dev/null; then
      echo "MPS already running"
    else
      nvidia-cuda-mps-control -d
      echo "MPS started"
    fi
    ;;
  stop)
    echo quit | nvidia-cuda-mps-control || true
    echo "MPS stopped"
    ;;
  status)
    nvidia-smi -i "$GPU" --query-gpu=name,persistence_mode,compute_mode,clocks.sm,clocks.max.sm,memory.used,memory.total,utilization.gpu,utilization.decoder --format=csv
    pgrep -a nvidia-cuda-mps || echo "MPS not running"
    ;;
  *)
    echo "usage: $0 start|stop|status"; exit 1 ;;
esac
