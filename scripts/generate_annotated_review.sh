#!/bin/bash
set -e

DIR="/home/users/oauser/mvsa"
cd "$DIR"

echo "Starting parallel 10-minute annotated runs for all 3 cameras..."
mkdir -p vehicle-count-output/annotated_review/tvc5_wb
mkdir -p vehicle-count-output/annotated_review/tvc1_nb
mkdir -p vehicle-count-output/annotated_review/tvc1_sb

/home/users/oauser/mvsa/env/bin/python vehicle-counting/pipeline/counting/main.py \
  --video vehicle-count-output/test_videos/tvc5_wb_sample.mp4 \
  --output_dir vehicle-count-output/annotated_review/tvc5_wb \
  --config vehicle-counting/config/vehicle_count_config_tvc5_cam2_wb.yaml \
  --duration_minutes 10 \
  --save_annotated_video \
  --start_time 09:00:00 > vehicle-count-output/annotated_review/tvc5_wb/stdout.log 2>&1 &
PID_WB=$!

/home/users/oauser/mvsa/env/bin/python vehicle-counting/pipeline/counting/main.py \
  --video vehicle-count-output/test_videos/tvc1_nb_sample.mp4 \
  --output_dir vehicle-count-output/annotated_review/tvc1_nb \
  --config vehicle-counting/config/vehicle_count_config_tvc1_cam1_nb.yaml \
  --duration_minutes 10 \
  --save_annotated_video \
  --start_time 10:00:00 > vehicle-count-output/annotated_review/tvc1_nb/stdout.log 2>&1 &
PID_NB=$!

/home/users/oauser/mvsa/env/bin/python vehicle-counting/pipeline/counting/main.py \
  --video vehicle-count-output/test_videos/tvc1_sb_sample.mp4 \
  --output_dir vehicle-count-output/annotated_review/tvc1_sb \
  --config vehicle-counting/config/vehicle_count_config_tvc1_cam2_sb.yaml \
  --duration_minutes 10 \
  --save_annotated_video \
  --start_time 14:00:00 > vehicle-count-output/annotated_review/tvc1_sb/stdout.log 2>&1 &
PID_SB=$!

echo "Processes launched in parallel: WB (PID $PID_WB), NB (PID $PID_NB), SB (PID $PID_SB)"
wait $PID_WB
echo "[1/3] TVC 5 WB 10-min annotated run complete."
wait $PID_NB
echo "[2/3] TVC 1 NB 10-min annotated run complete."
wait $PID_SB
echo "[3/3] TVC 1 SB 10-min annotated run complete."

echo "All 3 annotated runs completed successfully!"
