#!/usr/bin/env bash
set -uo pipefail
cd /storage/users/oauser/mvsa
source env/bin/activate

GOKUL="Data Corp - Dataset/DC646KA01 - Traffic & Transport Primary Surveys Hubballi-Dharwad in the state of Karnataka/TVC - 7 Days and 24 Hour CTVC/TVC 5 - Gokul Rd Near Airport"
OUT_ROOT="results/gokul_batch"
mkdir -p "$OUT_ROOT"
FFPROBE="tools/ffprobe"

find "$GOKUL" -iname "*.mp4" | sort | while read -r vid; do
  base=$(basename "$vid" .mp4)
  # base like 20260715_053640_tp00023
  date_part=${base:0:8}
  time_part=${base:9:6}
  yyyy=${date_part:0:4}; mm=${date_part:4:2}; dd=${date_part:6:2}
  hh=${time_part:0:2}; mi=${time_part:2:2}; ss=${time_part:4:2}
  start_clock="${hh}:${mi}:${ss}"
  day="${yyyy}-${mm}-${dd}"

  if [[ "$vid" == *"Cam 3"* ]]; then cam="cam3_EB"; else cam="cam4_WB"; fi

  run_id="${cam}_${day}_${hh}${mi}${ss}"
  out_dir="${OUT_ROOT}/${run_id}"

  if [[ -f "${out_dir}/interval_counts_3class.csv" ]]; then
    echo "[SKIP] ${run_id} already done"
    continue
  fi

  echo "[RUN] ${run_id}  video=${vid}  start=${start_clock}"
  mkdir -p "$out_dir"

  fps=$($FFPROBE -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=nk=1:nw=1 "$vid" | head -1)
  fps_val=$(python3 -c "n,d=('$fps'.split('/')+['1'])[:2]; print(float(n)/float(d))")

  python vehicle-counting/pipeline/counting/main.py \
    --video "$vid" \
    --output_dir "$out_dir" \
    > "${out_dir}/stdout.log" 2>&1
  status=$?

  if [[ $status -ne 0 ]]; then
    echo "[FAIL] ${run_id} exit=$status -- see ${out_dir}/stdout.log"
    continue
  fi

  python vehicle-counting/pipeline/counting/aggregate_intervals_3class.py \
    "$out_dir" --fps "$fps_val" --start "$start_clock" --interval-min 15 \
    >> "${out_dir}/stdout.log" 2>&1

  echo "$day,$start_clock,$cam,$vid" >> "${OUT_ROOT}/manifest.csv"
  echo "[DONE] ${run_id}"
done

echo "ALL DONE"
