#!/bin/bash
# Usage: qrsh -g hp190122 -l gpu_1=1 -l h_rt=0:30:00
#   then: bash run_bench.sh <target1> [target2] ...
#   or:   bash run_bench.sh all                  # run every built target
# Output: tee'd to bench_logs/<target>_<timestamp>.log

set -eu

cd "$(dirname "$0")"

module load cuda/12.8.0 >/dev/null 2>&1 || true

mkdir -p bench_logs
TS=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date +%H:%M:%S)] $*"; }

print_env() {
  log "=== Environment ==="
  hostname
  nvidia-smi -L || true
  nvcc --version | tail -2
  echo
}

run_one() {
  local target=$1
  local logfile="bench_logs/${target}_${TS}.log"
  log ">>> ${target} -> ${logfile}"
  {
    print_env
    log "=== Build ${target} ==="
    make -B "${target}"
    log "=== Run ${target} ==="
    "./${target}"
  } 2>&1 | tee "${logfile}"
  echo
}

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <target> [target...]  |  $0 all"
  exit 1
fi

if [ "$1" = "all" ]; then
  TARGETS=$(ls -1 *.cu 2>/dev/null | sed 's/\.cu$//' | grep -E '^(13_|14_|15_|16_|17_|18_|19_)' | sort)
  for t in $TARGETS; do run_one "$t"; done
else
  for t in "$@"; do run_one "$t"; done
fi
